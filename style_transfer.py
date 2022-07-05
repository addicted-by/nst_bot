import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from os import getenv
import torchvision.transforms as transforms
import torchvision.models as models
import io
import copy
import numpy as np
import asyncio 
from utils.image_methods import *
from torch.nn import Module, Sequential, Upsample, ReflectionPad2d, Conv2d, InstanceNorm2d, ReLU, MaxPool2d
from collections.abc import Iterable
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature: torch.Tensor):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # normalize img
        return (img - self.mean) / self.std



class StyleTransformer():
    def __init__(self, content_img: torch.Tensor, style_img: torch.Tensor, imsize: int=150):
        super(StyleTransformer, self).__init__()
        print("Start model")
        if os.path.exists("./data/models/vgg19.pth"):
            self.cnn = models.vgg19()
            self.cnn.load_state_dict(torch.load("./data/models/vgg19.pth"))
        else:
            self.cnn = models.vgg19(pretrained=True)
        print("Model is ready")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imsize = imsize
        self.cnn = self.cnn.features.to(self.device).eval()
        sizes = Image.open(io.BytesIO(content_img)).size
        self.content_img = self.image_loader(content_img, sizes)
        self.style_img = self.image_loader(style_img, sizes)
        del sizes
        self.input_img = self.content_img.clone()
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        print(self.content_img.size(), self.style_img.size())


    # def image_loader(self, image_bytes: bytes, sizes: tuple[int, int]):
    def image_loader(self, image_bytes: bytes, sizes: tuple) -> torch.Tensor:
        loader = transforms.Compose([
                            transforms.Resize(self.imsize),  # scale imported image
                            transforms.ToTensor()])  # transform it into a torch tensor
        self.unloader = transforms.ToPILImage()  # reconvert into PIL image
        
        image = Image.open(io.BytesIO(image_bytes)).resize(sizes)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def image_loader_file(self, image_name: str) -> torch.Tensor:
        loader = transforms.Compose([
                            transforms.Resize(self.imsize),  # scale imported image
                            transforms.ToTensor()])  # transform it into a torch tensor
        self.unloader = transforms.ToPILImage()  # reconvert into PIL image
        
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)



    def get_input_optimizer(self, input_img: torch.tensor):
    # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

        
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    


    def get_style_model_and_losses(self) -> tuple:
        content_layers = StyleTransformer.content_layers_default
        style_layers = StyleTransformer.style_layers_default
        cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = Normalization(self.normalization_mean,
                                 self.normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses


    async def run_style_transfer(self, 
                        num_steps=300, style_weight=100000, content_weight=1) -> torch.Tensor:
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer(self.input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            await asyncio.sleep(0)
            def closure():
                # correct the values of updated input image
                self.input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)

        return self.input_img

    # def image_to_byte_array(self, input_img: Image) -> bytes:
    #     image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    #     image = image.squeeze(0)      # remove the fake batch dimension
    #     image = self.unloader(image)
    #     imgByteArr = io.BytesIO()
    #     image.save(imgByteArr, format='JPEG')
    #     imgByteArr = imgByteArr.getvalue()
    #     return imgByteArr




class ConvBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int, kernel_size: int, 
                 stride: int=1, upsample: bool=False, 
                 norm: bool=True, relu: bool=True):
        super().__init__()

        self.upsample = Upsample(scale_factor=2) if upsample else None
        self.conv_block = Sequential(ReflectionPad2d(kernel_size // 2), Conv2d(in_channels, out_channels, kernel_size, stride))
        self.norm = InstanceNorm2d(out_channels, affine=True) if norm else None
        self.relu = ReLU(inplace=True) if relu else None

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        conv = self.conv_block(x)
        if self.norm:
            conv = self.norm(conv)
        if self.relu:
            conv = self.relu(conv)
        return conv


class ResBlock(Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv_blocks = Sequential(ConvBlock(channels, channels, kernel_size=3),
                                      ConvBlock(channels, channels, kernel_size=3, relu=False))

    def forward(self, x):
        return self.conv_blocks(x) + x


class ImageTransformationNetwork(Module):
    def __init__(self, num_res_blocks: int=5):
        super().__init__()

        self.transnet = Sequential(ConvBlock(3, 32, kernel_size=9, stride=1),
                                   ConvBlock(32, 64, kernel_size=3, stride=2),
                                   ConvBlock(64, 128, kernel_size=3, stride=2),
                                   *[ResBlock(128) for i in range(num_res_blocks)],
                                   ConvBlock(128, 64, kernel_size=3, upsample=True),
                                   ConvBlock(64, 32, kernel_size=3, upsample=True),
                                   ConvBlock(32, 3, kernel_size=9, norm=False, relu=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transnet(x)



class DefaultStylesTransformer():
    def __init__(self, content: bytes, stylization_type: str):
        super(DefaultStylesTransformer, self).__init__()
        assert stylization_type in {'picasso', 'vincent'}, "Don't know this style"
        self.stylization_type = stylization_type
        self.dir = './data/models/van_gogh.pth' if stylization_type == 'vincent' \
                            else './data/models/picasso.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.content = prep(content).to(self.device)
        self.transnet = ImageTransformationNetwork().to(self.device)
        self.transnet.load_state_dict(torch.load(self.dir, map_location=self.device))

    def transfer_style(self) -> Image:
        self.transnet.eval()
        with torch.no_grad():
            output = post(self.transnet(self.content))

        return output        
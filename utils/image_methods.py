import warnings
warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import io


# def image_resize(input_img: Image, sizes: tuple[int, int]) -> Image:
def image_resize(input_img: Image, sizes: tuple) -> Image:
    return transforms.ToPILImage()(input_img).resize(sizes)


def image_to_byte_array(input_img: Image, cpu: bool=False) -> bytes:
    if not cpu:
        image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = transforms.ToPILImage()(image)
    else:
        image = input_img
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


SIZE = 256

# ImageNet statistics
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Pre-processing
def prep(image: bytes, size=SIZE, normalize=True, mean=MEAN, std=STD, device='cpu') -> torch.Tensor:
    image = Image.open(io.BytesIO(image))
    resize = transforms.Compose([transforms.Resize(size, Image.Resampling.LANCZOS),
                                 transforms.CenterCrop(size)])
    image = resize(image.convert('RGB'))
    if normalize:
        norm = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
        return norm(image).unsqueeze(0).to(device)
    else:
        return image

# Post-processing
def post(tensor, mean=MEAN, std=STD) -> Image:
    mean, std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    tensor = transforms.Lambda(lambda x: x * std + mean)(tensor.cpu().clone().squeeze(0))
    return transforms.ToPILImage()(tensor.clamp_(0, 1))



# def image_resize_to_byte_array(input_img: Image, sizes: tuple[int, int]) -> bytes:
def image_resize_to_byte_array(input_img: Image, sizes: tuple) -> bytes:
    image = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    image = image.resize(sizes)
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

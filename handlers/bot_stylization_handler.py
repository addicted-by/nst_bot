import aiogram
import requests
from utils.image_methods import * 
from style_transfer import *
from aiogram.dispatcher import Dispatcher
import emoji
import sys
sys.path.append("..")
from aiogram.utils import executor
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import Message
from aiogram.types.bot_command import BotCommand
import logging
from aiogram.dispatcher.filters import Text
import numpy as np
from os import getenv
import os
import asyncio
from utils.db import DBHandler
from aiogram.utils.markdown import text, bold, italic, code, pre
from menus.menus import *
from loader import *
import time 


class BotStatesStylization(StatesGroup):
    waiting_for_type = State()
    waiting_for_first_image = State()
    waiting_for_second_image = State()
    waiting_for_swap_decision = State()

class BotStatesDefaultStyles(StatesGroup):
    waiting_for_image = State()


async def first_type_chosen(message: types.Message, state: FSMContext):
    if message.content_type != 'text':
        await message.answer("Please, choose the photo type first")
        return
    if message.text.lower() not in USERS_CONTENT[message.from_user.id]:
        if message.text.lower() in {'content', 'style'}:
            msg_str = "You have already uploaded {uploaded_type}\. Now, please, upload {notuploaded_type}".format(
            uploaded_type=message.text.lower(),
            notuploaded_type=USERS_CONTENT[message.from_user.id]-set(message.text.lower()))
            await message.answer(msg_str)
            return
        await message.answer("Please, upload the content/style photo or get back\.")
        return
    USERS_CONTENT[message.from_user.id] -= {message.text.lower()}
    await state.update_data(chosen_type=message.text.lower())
    message_text = text(f"You have chosen the {message.text.lower()} image.",
                        bold("Upload now it please"))
    await message.answer(message_text, reply_markup=CANCEL, parse_mode="Markdown")  
    await BotStatesStylization.waiting_for_first_image.set()  

async def first_image_uploaded(message: types.Message, state: FSMContext):
    global bot, bot_token, db
    if message.content_type != 'photo' and message.content_type != 'document':
        await message.answer(text="Please send me a photo or get back")
    elif message.content_type == 'photo':
        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.photo[-1]['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, data['chosen_type'])
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
            return
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded {chosen_type} image:check_mark::check_mark::check_mark:.
It is a time to upload {to_upload}'''.format(
                chosen_type=data['chosen_type'],
                to_upload=(USERS_CONTENT[message.from_user.id] - {data['chosen_type']}).pop()))
        await message.answer(message_str, parse_mode="")
        message_str = text("Don't worry if you made a mistake with your choice.",
                                    "You can swap photos after it's uploading...")
        await message.answer(message_str, parse_mode="")
        print("FIRST IMAGE UPLOADED")
        await state.update_data(chosen_type=data['chosen_type'],
                                      file_path=file_path,
                                      need_type=next(iter(USERS_CONTENT[message.from_user.id])))
        await BotStatesStylization.waiting_for_second_image.set()
    else: 
        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.document.thumb['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, data['chosen_type'])
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
            return
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded {chosen_type} image:check_mark::check_mark::check_mark:.
It is a time to upload {to_upload}'''.format(
                chosen_type=data['chosen_type'],
                to_upload=(USERS_CONTENT[message.from_user.id] - {data['chosen_type']}).pop()))
        await message.answer(message_str, parse_mode="")
        print("FIRST IMAGE UPLOADED")
        await state.update_data(chosen_type=data['chosen_type'],
                                      file_path=file_path,
                                      need_type=next(iter(USERS_CONTENT[message.from_user.id])))
        await BotStatesStylization.waiting_for_second_image.set()

async def second_image_uploaded(message: types.Message, state: FSMContext):
    if message.content_type != 'photo' and message.content_type != 'document':
        await message.answer(text="Please send me a photo or get back")
    elif message.content_type == 'photo':
        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.photo[-1]['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, data['need_type'])
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded both images. Congrat:thumbs_up_dark_skin_tone:''')
        await message.answer(message_str, parse_mode="")
        print("SECOND IMAGE UPLOADED")
        await state.update_data(images={data["chosen_type"]: data['file_path'],
                                                data["need_type"]: file_path})
        message_str = '''Check if your decision correct. Maybe you want to swap images'''
        await message.answer(message_str, reply_markup=SWAP_MENU, parse_mode="")
        await BotStatesStylization.waiting_for_swap_decision.set()
    else:

        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.document.thumb['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, data['need_type'])
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded both images. Congrat:fire:''')
        await message.answer(message_str, parse_mode="")
        print("SECOND IMAGE UPLOADED")
        await state.update_data(images={data["chosen_type"]: data['file_path'],
                                                data["need_type"]: file_path})
        message_str = '''Check if your decision correct. Maybe you want to swap images'''
        await message.answer(message_str, reply_markup=SWAP_MENU, parse_mode="")
        await BotStatesStylization.waiting_for_swap_decision.set()


async def swap_or_not(message: types.Message, state: FSMContext):
    if message.text.lower() not in {'continue', 'swap'}:
        await message.answer("Please change continue or swap images")
    elif message.text.lower() == 'continue':
        data = await state.get_data()
        print(data['stylization_type'])
        await message.answer('Starting stylization...', reply_markup=MAIN_MENU, parse_mode="")
        try:
            content, style = db.extract_picture(message.from_user.id, both=True)
            print("Extracted")
        except:
            await message.answer("Something went wrong. Try once again", 
                        parse_mode="", reply_markup=MAIN_MENU)
        sizes = Image.open(io.BytesIO(content)).size
        transformer = StyleTransformer(content, style)
        print("Model loaded")
        start = time.time()
        output_pil_image = await transformer.run_style_transfer()
        finish = time.time()
        if transformer.device == 'cuda':
            torch.cuda.empty_cache()   
        del content, style, transformer
        output_image = image_resize_to_byte_array(output_pil_image, sizes)
        try:
            db.insert_stats(message.from_user.id, 
                                bytes("Last section time {:.4} ms. Last section type: {st_type}".format(
                                    finish-start,
                                    st_type="Slow"
                                ), 'utf8'))
        except:
            print("Can't insert stats into the db")
        try: 
            db.insert_stylized(message.from_user.id, output_image)
        except:
            print("Can't insert stylized photo in db")
        await message.answer_photo(output_image, caption="Styled image", reply_markup=MAIN_MENU)
        del output_image
        await state.finish()
        return
    elif message.text.lower() == 'swap':
        data = await state.get_data()
        images = data['images']
        images['content'], images['style'] = images['style'], images['content']
        for key in images.keys():
            file_path = images[key]
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, key)
    try:
        content, style = db.extract_picture(message.from_user.id, both=True)
        await message.answer_photo(photo=content, caption="Content Photo")
        await message.answer_photo(photo=style, caption="Style Photo")
    except:
        print("Something went wrong")
    await state.update_data(images={'content': images['content'],
                                                'style': images['style']})

@dp.message_handler(commands='cancel')
@dp.message_handler(commands='Cancel')
async def cancel(message: Message, state: FSMContext):
    if message.text.lower() == "cancel":
        await state.finish()
        await message.answer(text="Turning you back", reply_markup=MAIN_MENU)

async def image_uploaded(message: types.Message, state: FSMContext):
    if message.content_type != 'photo' and message.content_type != 'document':
        await message.answer(text="Please send me a photo or get back")
    elif message.content_type == 'photo':
        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.photo[-1]['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, 'Content')
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded image. Congrat:thumbs_up_dark_skin_tone:''')
        await message.answer(message_str, parse_mode="")
        print(data['stylization_type'])
        await message.answer('Starting stylization...', parse_mode="")
        try:
            content = db.extract_picture(message.from_user.id, both=False)
        except:
            print("Can't extract data")
        transformer = DefaultStylesTransformer(content, data['stylization_type'])
        start = time.time()
        output = transformer.transfer_style()
        finish = time.time()
        output = image_to_byte_array(output, cpu=True)
        try:
            db.insert_stats(message.from_user.id, 
                                bytes("Last section time {:.4} ms. Last section type: {st_type}".format(
                                    finish-start,
                                    st_type=data['stylization_type'].upper()
                                ), 'utf8'))
        except:
            print("Can't insert stats into the db")
        try: 
            db.insert_stylized(message.from_user.id, output)
        except:
            print("Can't insert stylized photo in db")
        await message.answer_photo(output, caption="Stylized photo", reply_markup=MAIN_MENU)
        await state.finish()
        return
    else:
        data = await state.get_data()
        try:
            file_path = (await bot.get_file(message.document.thumb['file_id']))['file_path']
            response = requests.get('https://api.telegram.org/file/bot{bot_token}/{file_path}'.format(
                bot_token=bot_token,
                file_path=file_path
            ))
            db.insert_picture_by_bytes(message.from_user.id, response.content, 'Content')
        except:
            print("Can't download photo.")
            await message.answer('Sorry, something went wrong\. Try again')
        print('Adding image to db have done sucessfully')
        message_str = emoji.emojize('''You have uploaded image. Congrat:fire:''')
        await message.answer(message_str, parse_mode="")
        print(data['stylization_type'])
        await message.answer('Starting stylization...', parse_mode="")
        try:
            content = db.extract_picture(message.from_user.id, both=False)
        except:
            print("Can't extract data")
        
        transformer = DefaultStylesTransformer(content, data['stylization_type'])
        start = time.time()
        output = transformer.transfer_style()
        finish = time.time()
        try:
            db.insert_stats(message.from_user.id, 
                                bytes("Last section time {:.4} ms. Last section type: {st_type}".format(
                                    finish-start,
                                    st_type=data['stylization_type'].upper()
                                ), 'utf8'))
        except:
            print("Can't insert stats into the db")
        try: 
            db.insert_stylized(message.from_user.id, output)
        except:
            print("Can't insert stylized photo in db")
        output = image_to_byte_array(output, cpu=True)
        await message.answer_photo(output, caption="Stylized photo", reply_markup=MAIN_MENU)
        await state.finish()
        return


def register_handlers_images_stylization(dp: Dispatcher):
    dp.register_message_handler(cancel, Text(equals=['Cancel']), state=[
        BotStatesStylization.waiting_for_type,
        BotStatesStylization.waiting_for_first_image,
        BotStatesStylization.waiting_for_second_image,
        BotStatesStylization.waiting_for_swap_decision
    ])
    dp.register_message_handler(first_type_chosen, content_types=["any"], state=BotStatesStylization.waiting_for_type)
    dp.register_message_handler(first_image_uploaded, content_types=["any"], state=BotStatesStylization.waiting_for_first_image)
    dp.register_message_handler(second_image_uploaded, content_types=["any"], state=BotStatesStylization.waiting_for_second_image)
    dp.register_message_handler(swap_or_not, state=BotStatesStylization.waiting_for_swap_decision)

def register_handlers_images_default_stylization(dp: Dispatcher):
    dp.register_message_handler(cancel, Text(equals=['Cancel']), state=[
        BotStatesDefaultStyles.waiting_for_image
    ])
    dp.register_message_handler(image_uploaded, content_types=["any"], state=BotStatesDefaultStyles.waiting_for_image)

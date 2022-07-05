import aiogram
import requests
from style_transfer import *
from aiogram.dispatcher import Dispatcher
import emoji
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
from menus import *
from handlers.bot_stylization_handler import *


greeting_stickers = ['CAACAgIAAxkBAAEFLFZiv5SYdqkrNaegGd6GCfk4wZx6tAAC2A8AAkjyYEsV-8TaeHRrmCkE',
                            'CAACAgIAAxkBAAEFLFhiv5TgndKYrRqNT3Q9axKgfRJ7EgACMwADwZxgDMvuV12UjE-BKQQ',
                            'CAACAgIAAxkBAAEFLFpiv5T4JAeRJKwQ84axYF65BhNlowAC7w0AAj0hKUoILjTaq1uy4ikE',
                            'CAACAgIAAxkBAAEFLFxiv5UMInz-Z7PlMfLu2oGCZPfYmQACoxUAAio0wEsKxlfuq9WvDikE',
                            'CAACAgIAAxkBAAEFLF5iv5UTWLfDCk0VREBVBSBmbu4dJQACbwAD9wLID-kz_ZsHgo4yKQQ',
                            'CAACAgIAAxkBAAEFLGBiv5UdAdzBnxexodOVi8Qa7zi3uwACxgEAAhZCawpKI9T0ydt5RykE',
                            'CAACAgIAAxkBAAEFLGJiv5UnFUCR5JtSyYZs3NKkGWHlRAACWAADDbbSGZ3CcZkZ26gpKQQ',
                            'CAACAgIAAxkBAAEFLGRiv5UwVzy-mmPrpoOhtuc_1CrJhQACRQADeKjmD8U-FA5dAz7LKQQ']

bot_token = getenv("TELEGRAM_TOKEN")
if not bot_token:
    exit("Error: no token provided")

USERS_CONTENT = {}


bot = Bot(token=bot_token, parse_mode=types.ParseMode.MARKDOWN_V2)
dp = Dispatcher(bot, storage=MemoryStorage())
db = DBHandler()

logging.basicConfig(level=logging.INFO)

isEmptySet = lambda x:(x == set())

logger = logging.getLogger(__name__)

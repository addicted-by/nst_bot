from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup



MAIN_MENU = ReplyKeyboardMarkup(resize_keyboard=True, row_width=1).add(
    KeyboardButton(text="Use your own style (neural transfer)"),
    KeyboardButton(text="Picasso style"),
    KeyboardButton(text="Vincent Van Gogh style")
)

CANCEL = ReplyKeyboardMarkup(resize_keyboard=True).add(
    InlineKeyboardButton(text="Cancel")
)

PHOTO_TYPE_MENU = ReplyKeyboardMarkup(resize_keyboard=True).add(
    InlineKeyboardButton(text="Content"),
    InlineKeyboardButton(text="Style"),
    InlineKeyboardButton(text="Cancel")   
)

SWAP_MENU = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2).add(
    InlineKeyboardButton(text="Swap"),
    InlineKeyboardButton(text="Continue"),
    InlineKeyboardButton(text="Cancel")
)

REFERENCES = InlineKeyboardMarkup(row_width=1).add(
    InlineKeyboardButton(text="GitHub", url='https://github.com/addicted-by'),
    InlineKeyboardButton(text="GitHub page with the project", url="https://github.com/addicted-by/nst_bot")
)
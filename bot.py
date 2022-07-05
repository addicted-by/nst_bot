from menus.menus import *
from handlers.bot_stylization_handler import *
from loader import *

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    global greeting_stickers
    message_text = text("I am very glad to see you,", bold(message.from_user.first_name))
    await message.reply(message_text)
    await message.answer_sticker(np.random.choice(greeting_stickers))
    message_text = text("I was made to style images. And today I can do it for you.")
    await message.answer(message_text, parse_mode="")
    message_text = text("Type\choose in menu <code>/help</code> to give some advices of the usage of my tools.")
    await message.answer(message_text, parse_mode="HTML", 
                reply_markup=MAIN_MENU
                    )
    db.insert_id(message.from_user.id)
    print(message.from_user.id)
    

@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    msg = text(
        "<strong>My features:</strong>\n\n",
        "<u>Style transfer</u> — a tool that allows you to transfer style from one image to another.\n",
        "You need to upload two photos: <i>content</i> — photo you need to transform and <i>style</i> -- the photo means style for content.\n\n",
        "<u>Picasso style</u> — a pretrained tool that allow you to transfer a Picasso modern style to your photo. You need to upload only one photo.\n\n",
        emoji.emojize("<u>Van Gogh style</u> — the same tool as previous but to transfer Van Gogh stylistic to your images.\n\n Good Luck in usage!:fire::fire::fire:")
    )
    await message.answer(msg, parse_mode="HTML")
    await message.answer("Try my features:", reply_markup=MAIN_MENU)


@dp.message_handler(lambda message: message.text == "Use your own style (neural transfer)")
async def fast_own_style(message: types.Message, state: FSMContext):
    await state.finish()
    await message.reply("You have chosen fast stylization")
    message_text = text("You need to upload two photos:\n\n",
                         code("content:"), "\nthe image, that I will transform and\n\n",
                         code("style:"), "\nthe image that will be the transformation you need")
    USERS_CONTENT[message.from_user.id] = {'content', 'style'}
    await message.answer(message_text)
    message_text = text("Let's <s>play</s> start. Сhoose which photo to upload first!")
    await message.answer(message_text, parse_mode='HTML', reply_markup=PHOTO_TYPE_MENU)
    await state.update_data(stylization_type='fast')
    await BotStatesStylization.waiting_for_type.set()


@dp.message_handler(lambda message: message.text == "Picasso style")
@dp.message_handler(lambda message: message.text == "Vincent Van Gogh style")
async def default_style(message: types.Message, state: FSMContext):
    await state.finish()
    await message.reply(f"You have chosen {message.text}")
    message_text = text(f"You need to upload a picture to style it into the style of the *{' '.join(message.text.split()[:-1])}*")
    await message.answer(message_text)
    message_text = text("Let's start. Upload your photo please")
    await message.answer(message_text, reply_markup=CANCEL, parse_mode="")
    await state.update_data(stylization_type=message.text.split()[0].lower())
    await BotStatesDefaultStyles.waiting_for_image.set()


@dp.message_handler(commands='cancel')
@dp.message_handler(commands='Cancel')
async def cancel(message: Message):
    if message.text.lower() == "cancel":
        await message.answer(text="Turning you back to the main menu", reply_markup=MAIN_MENU)

@dp.message_handler(commands=['stat'])
async def send_stats(message: types.Message):
    try:
        stat = db.extract_field(message.from_user.id, 'stats')
        stat = str(db.extract_field(message.from_user.id, 'STATS'), 'utf8')
        await message.answer("*Last Usage Statistics*", parse_mode="Markdown")
        await message.answer(stat, parse_mode="")
        try:
            stylized = db.extract_field(message.from_user.id, "stylized")
            await message.answer_photo(stylized, caption="Last stylized photo", reply_markup=MAIN_MENU)
        except:
            print("Can't extract photo")
    except:
        print("Can't extract stats")
        await message.answer("Can't extract statistics from the database, try later.", parse_mode="", reply_markup=MAIN_MENU)


@dp.message_handler(commands=['ref'])
async def send_references(message: types.Message):
    await message.answer("There are some links for you", parse_mode="", reply_markup=REFERENCES)


# Регистрация команд, отображаемых в интерфейсе Telegram
async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/help", description="HELP"),
        BotCommand(command='/ref', description="References"),
        BotCommand(command='/stat', description="Last usage statistics")
     ] # ]
    await bot.set_my_commands(commands)

async def main():
    global bot, dp
    # Настройка логирования в stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger.error("Starting bot")

    # Регистрация хэндлеров
    # register_handlers_common(dp)
    register_handlers_images_stylization(dp)
    register_handlers_images_default_stylization(dp)

    # Установка команд бота
    await set_commands(bot)

    # Запуск поллинга
    # await dp.skip_updates()  # пропуск накопившихся апдейтов (необязательно)
    await dp.start_polling()

if __name__ == '__main__':
    asyncio.run(main())
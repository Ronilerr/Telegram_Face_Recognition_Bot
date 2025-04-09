import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# Load environment variables from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Define the custom keyboard
keyboard_buttons = [
    [KeyboardButton("Add face"), KeyboardButton("Recognize faces")],
    [KeyboardButton("Reset faces")]
]
reply_markup = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Choose an option below:",
        reply_markup=reply_markup
    )

# Message handler for any of the buttons
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text in ["Hello", "World", "Telegram", "Bot"]:
        await update.message.reply_text(text)
    else:
        await update.message.reply_text("Please choose one of the options from the keyboard.")

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variable
BOT_TOKEN = os.getenv('BOT_TOKEN')

# This function handles the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey! I'm a basic Telegram bot :)")

# This function echoes back any text message sent to the bot
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    await update.message.reply_text(f"You said: {text}")

def main():
    # Create the bot application
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Add command and message handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Bot is running... Press Ctrl+C to stop")
    app.run_polling()

if __name__ == "__main__":
    main()

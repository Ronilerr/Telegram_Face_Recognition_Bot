import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import face_recognition
from PIL import Image
import numpy as np
from io import BytesIO
# Load environment variables from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Define the custom keyboard
keyboard_buttons = [
    [KeyboardButton("Add face"), KeyboardButton("Recognize faces")],
    [KeyboardButton("Reset faces")]
]
reply_markup = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)
# In-memory face database
known_faces = []
known_names = []

# State tracking for users
user_states = {}

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Choose an option below:",
        reply_markup=reply_markup
    )

# Message handler for any of the buttons
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    # User pressed "Add face"
    if text == "Add face":
        await update.message.reply_text("Upload an image with a single face")
        user_states[user_id] = "awaiting_face"

    # User is now sending a name after face image
    elif user_states.get(user_id) == "awaiting_name":
        name = text
        encoding = context.user_data.get("temp_face")

        if encoding is not None:
            known_faces.append(encoding)
            known_names.append(name)
            await update.message.reply_text(f"Great. I will now remember {name}.")
        else:
            await update.message.reply_text("Something went wrong. Try again.")

        user_states[user_id] = None
        await start(update, context)

    # All other messages
    else:
        await update.message.reply_text("Please choose one of the options from the keyboard.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id  # Get the unique ID of the user

    # If we are expecting a face image from this user
    if user_states.get(user_id) == "awaiting_face":
        # Download the image sent by the user
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Load the image and extract face encodings
        img = face_recognition.load_image_file(BytesIO(photo_bytes))
        encodings = face_recognition.face_encodings(img)

        # Make sure exactly one face was found
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return

        # Temporarily store the encoding and update user state
        context.user_data["temp_face"] = encodings[0]
        user_states[user_id] = "awaiting_name"

        # Ask for the person's name
        await update.message.reply_text("Great. What’s the name of the person in this image?")

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()

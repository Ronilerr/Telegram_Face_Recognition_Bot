import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import face_recognition
from PIL import Image, ImageDraw
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

    # Recognize face flow
    elif text == "Recognize faces":
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in this image.")
        user_states[user_id] = "awaiting_recognition"

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
    user_id = update.effective_user.id

    # Handle photo for "Add face"
    if user_states.get(user_id) == "awaiting_face":
        # Get photo file and convert it to bytes
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Load image and extract face encoding
        img = face_recognition.load_image_file(BytesIO(photo_bytes))
        encodings = face_recognition.face_encodings(img)

        # Make sure there's exactly one face
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return

        # Temporarily store the face encoding and move to name step
        context.user_data["temp_face"] = encodings[0]
        user_states[user_id] = "awaiting_name"
        await update.message.reply_text("Great. What’s the name of the person in this image?")

    # Handle photo for "Recognize faces"
    elif user_states.get(user_id) == "awaiting_recognition":
        # Download the image from Telegram
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Load the image as numpy array
        img_np = face_recognition.load_image_file(BytesIO(photo_bytes))

        # Get face encodings and face locations
        encodings = face_recognition.face_encodings(img_np)
        locations = face_recognition.face_locations(img_np)

        if not encodings:
            await update.message.reply_text("I couldn't find any faces in the image.")
            return

        recognized_names = []

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)

            if any(matches):
                best_match_index = face_distances.argmin()
                recognized_names.append(known_names[best_match_index])
            else:
                recognized_names.append("Unknown")

        # Convert numpy image to PIL for drawing
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)

        # Draw blue boxes around each face
        for (top, right, bottom, left), name in zip(locations, recognized_names):
            draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
            draw.text((left, top - 10), name, fill="blue")

        # Save image to memory
        output_buffer = BytesIO()
        img_pil.save(output_buffer, format="JPEG")
        output_buffer.seek(0)

        # Send result photo
        await update.message.reply_photo(photo=output_buffer)

        # Send text summary
        if all(name == "Unknown" for name in recognized_names):
            await update.message.reply_text("I don’t recognize anyone in this image.")
        else:
            names_str = ", ".join(recognized_names)
            await update.message.reply_text(f"I found {len(encodings)} face(s): {names_str}")

        # Reset state and return to main menu
        user_states[user_id] = None
        await start(update, context)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()


if __name__ == "__main__":
    main()

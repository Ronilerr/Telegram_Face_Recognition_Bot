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
    [KeyboardButton("Reset faces"), KeyboardButton("Similar celebs")], [KeyboardButton("Map")]
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
    
    # Reset faces
    elif text == "Reset faces":
        known_faces.clear()
        known_names.clear()
        await start(update, context)
    # Similar celebs
    elif text == "Similar celebs":
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person.")
        user_states[user_id] = "awaiting_celebrity_comparison"

    # User is now sending a name after face image
    elif user_states.get(user_id) == "awaiting_name":
        name = text
        encoding = context.user_data.get("temp_face")

        if encoding is not None:
            known_faces.append(encoding)
            known_names.append(name)
            await update.message.reply_text(f"Great. I will now remember this face")
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
            await update.message.reply_text(f"I found {len(encodings)} faces in this images and the people are {names_str}")

        # Reset state and return to main menu
        user_states[user_id] = None
        await start(update, context)

    # Handle photo for "Similar celebs"
    elif user_states.get(user_id) == "awaiting_celebrity_comparison":
        # Download image from user
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Load image and get face encodings
        img = face_recognition.load_image_file(BytesIO(photo_bytes))
        encodings = face_recognition.face_encodings(img)

        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return

        uploaded_encoding = encodings[0]
        best_match_name = None
        best_match_image_path = None
        best_distance = float("inf")

        # Go over celeb folders
        celeb_dir = "celebs"  # Path to your celeb library
        for celeb_name in os.listdir(celeb_dir):
            celeb_path = os.path.join(celeb_dir, celeb_name)
            if not os.path.isdir(celeb_path):
                continue

            for img_name in os.listdir(celeb_path):
                img_path = os.path.join(celeb_path, img_name)
                try:
                    celeb_image = face_recognition.load_image_file(img_path)
                    celeb_encodings = face_recognition.face_encodings(celeb_image)
                    if not celeb_encodings:
                        continue
                    celeb_encoding = celeb_encodings[0]
                    distance = face_recognition.face_distance([celeb_encoding], uploaded_encoding)[0]

                    if distance < best_distance:
                        best_distance = distance
                        best_match_name = celeb_name
                        best_match_image_path = img_path
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
                    continue

        if best_match_name:
            # Send image of best match
            with open(best_match_image_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"The celeb that the person is most similar to is {best_match_name}.")
        else:
            await update.message.reply_text("Sorry, I couldn’t find a similar celeb.")

        # Reset state and show main menu
        user_states[user_id] = None
        await start(update, context)   

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()


if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load environment variables from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Define the custom keyboard
keyboard_buttons = [
    [KeyboardButton("Add face"), KeyboardButton("Recognize faces")],
    [KeyboardButton("Reset faces"), KeyboardButton("Similar celebs")],
    [KeyboardButton("Map")]
]
reply_markup = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

# In-memory face database: list of dicts with encoding, name, image
known_faces = []

# State tracking for users
user_states = {}

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option below:", reply_markup=reply_markup)

# Function to generate the similarity map using TSNE
def generate_tsne_map(known_faces, celeb_dir="celebs", output_path="celebs_tsne_map.png"):
    encodings = []
    labels = []
    images = []

    for entry in known_faces:
        encodings.append(entry["encoding"])
        labels.append(entry["name"])
        img = entry.get("image")
        if img:
            images.append(img)
        else:
            images.append(Image.new("RGB", (45, 45), color=(150, 150, 150)))

    for celeb in os.listdir(celeb_dir):
        celeb_path = os.path.join(celeb_dir, celeb)
        if not os.path.isdir(celeb_path):
            continue
        for filename in os.listdir(celeb_path):
            path = os.path.join(celeb_path, filename)
            try:
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    encodings.append(encoding[0])
                    labels.append(celeb)
                    images.append(Image.fromarray(image))
            except Exception:
                continue

    if not encodings:
        return False

    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    reduced = tsne.fit_transform(np.array(encodings))

    norm_x = (reduced[:, 0] - np.min(reduced[:, 0])) / (np.max(reduced[:, 0]) - np.min(reduced[:, 0]))
    norm_y = (reduced[:, 1] - np.min(reduced[:, 1])) / (np.max(reduced[:, 1]) - np.min(reduced[:, 1]))
    norm_y *= 0.9

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_title("Face Similarity Map", fontsize=14, pad=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    for x, y, img, label in zip(norm_x, norm_y, images, labels):
        thumb = img.resize((45, 45))
        im = OffsetImage(thumb, zoom=1)
        ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.3)
        ax.add_artist(ab)
        ax.text(x, y - 0.035, label, fontsize=6, ha='center', va='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    if text == "Add face":
        await update.message.reply_text("Upload an image with a single face")
        user_states[user_id] = "awaiting_face"

    elif text == "Recognize faces":
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in this image.")
        user_states[user_id] = "awaiting_recognition"

    elif text == "Reset faces":
        known_faces.clear()
        await start(update, context)

    elif text == "Similar celebs":
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person.")
        user_states[user_id] = "awaiting_celebrity_comparison"

    elif text == "Map":
        await update.message.reply_text("Generating face similarity map...")
        success = generate_tsne_map(known_faces)
        if success:
            with open("celebs_tsne_map.png", "rb") as photo:
                await update.message.reply_photo(photo=photo, caption="Here is the similarity map 🎯")
        else:
            await update.message.reply_text("Sorry, I couldn't generate the map.")

        # Show menu again after sending the map
        await update.message.reply_text("Choose an option below:", reply_markup=reply_markup)

    elif user_states.get(user_id) == "awaiting_name":
        name = text
        encoding = context.user_data.get("temp_face")
        img = context.user_data.get("temp_image")

        if encoding is not None and img is not None:
            known_faces.append({
                "encoding": encoding,
                "name": name,
                "image": Image.fromarray(img)
            })
            await update.message.reply_text(f"Great. I will now remember this face")
        else:
            await update.message.reply_text("Something went wrong. Try again.")

        user_states[user_id] = None
        await start(update, context)

    else:
        await update.message.reply_text("Please choose one of the options from the keyboard.")

# Handler for uploaded photos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    img = face_recognition.load_image_file(BytesIO(photo_bytes))

    if user_states.get(user_id) == "awaiting_face":
        encodings = face_recognition.face_encodings(img)
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return

        context.user_data["temp_face"] = encodings[0]
        context.user_data["temp_image"] = img
        user_states[user_id] = "awaiting_name"
        await update.message.reply_text("Great. What’s the name of the person in this image?")

    elif user_states.get(user_id) == "awaiting_recognition":
        encodings = face_recognition.face_encodings(img)
        locations = face_recognition.face_locations(img)

        if not encodings:
            await update.message.reply_text("I couldn't find any faces in the image.")
            return

        recognized_names = []

        for face_encoding in encodings:
            matches = [face_recognition.compare_faces([face["encoding"]], face_encoding)[0] for face in known_faces]
            distances = [face_recognition.face_distance([face["encoding"]], face_encoding)[0] for face in known_faces]

            if any(matches):
                best_index = np.argmin(distances)
                recognized_names.append(known_faces[best_index]["name"])
            else:
                recognized_names.append("Unknown")

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        for (top, right, bottom, left), name in zip(locations, recognized_names):
            draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
            draw.text((left, top - 10), name, fill="blue")

        output_buffer = BytesIO()
        img_pil.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        await update.message.reply_photo(photo=output_buffer)

        if all(name == "Unknown" for name in recognized_names):
            await update.message.reply_text("I don’t recognize anyone in this image.")
        else:
            await update.message.reply_text(f"I found {len(encodings)} faces and the people are: {', '.join(recognized_names)}")

        user_states[user_id] = None
        await start(update, context)

    elif user_states.get(user_id) == "awaiting_celebrity_comparison":
        encodings = face_recognition.face_encodings(img)
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return

        uploaded_encoding = encodings[0]
        best_match_name = None
        best_match_image_path = None
        best_distance = float("inf")

        celeb_dir = "celebs"
        for celeb_name in os.listdir(celeb_dir):
            celeb_path = os.path.join(celeb_dir, celeb_name)
            if not os.path.isdir(celeb_path):
                continue

            for img_name in os.listdir(celeb_path):
                img_path = os.path.join(celeb_path, img_name)
                try:
                    celeb_image = face_recognition.load_image_file(img_path)
                    celeb_encodings = face_recognition.face_encodings(celeb_image)
                    if celeb_encodings:
                        distance = face_recognition.face_distance([celeb_encodings[0]], uploaded_encoding)[0]
                        if distance < best_distance:
                            best_distance = distance
                            best_match_name = celeb_name
                            best_match_image_path = img_path
                except:
                    continue

        if best_match_name:
            with open(best_match_image_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"The celeb that the person is most similar to is {best_match_name}.")
        else:
            await update.message.reply_text("Sorry, I couldn’t find a similar celeb.")

        user_states[user_id] = None
        await start(update, context)

# Main function
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
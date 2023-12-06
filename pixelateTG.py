from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import cv2
import numpy as np
from io import BytesIO
import os

def detect_eyes(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    eyes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes_data = eye_cascade.detectMultiScale(roi_gray)
        eyes.extend([{"box": (ex + x, ey + y, ew, eh)} for (ex, ey, ew, eh) in eyes_data])

    return eyes

def replace_eyes(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    eyes = detect_eyes(image)

    for eye in eyes:
        eye_x, eye_y, eye_w, eye_h = eye["box"]
        print(f"Eye detected at ({eye_x}, {eye_y}), width: {eye_w}, height: {eye_h}")

        # Replace the following line with your code to add the lasereye
        # Example: cv2.rectangle(image, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)

    processed_path = f"processed/{user_id}_lasereyes.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def process_image(photo_path, user_id, mode, bot):
    if mode == "pixelated":
        processed_path = pixelate(photo_path, user_id, bot)
    elif mode == "lasereyes":
        processed_path = replace_eyes(photo_path, user_id, bot)
    else:
        processed_path = photo_path

    return processed_path

def pixelate(photo_path, user_id, bot):
    # Placeholder function for pixelation, replace with your code
    # Example: cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # You can use OpenCV functions to pixelate the detected faces

    processed_path = f"processed/{user_id}_pixelated.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def button_callback(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id

    if "photo" in update.message.effective_attachment:
        photo = update.message.effective_attachment["photo"][-1]
        file_id = photo.file_id
        file = bot.get_file(file_id)
        photo_path = f"downloads/{user_id}_input.jpg"
        file.download(photo_path)

        context.user_data["photo_path"] = photo_path

        keyboard = [["Pixelate", "Laser Eyes"]]
        reply_markup = {"keyboard": keyboard, "one_time_keyboard": True}
        update.message.reply_text("Choose an option:", reply_markup=reply_markup)
    else:
        update.message.reply_text("Please send a photo.")

def handle_text(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    mode = update.message.text.lower()

    if "photo_path" in context.user_data:
        photo_path = context.user_data["photo_path"]
        processed_path = process_image(photo_path, user_id, mode, context.bot)
        context.user_data.clear()
        with open(processed_path, "rb") as photo:
            update.message.reply_photo(photo)
        os.remove(processed_path)
    else:
        update.message.reply_text("Please start with sending a photo.")

def main() -> None:
    TOKEN = "your_bot_token"
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
    dp.add_handler(CommandHandler("start", button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

# pixelateTG.py

import os
import cv2
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from concurrent.futures import ThreadPoolExecutor

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def pixelate_faces(update: Update, context: CallbackContext):
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = context.bot.get_file(file_id)
        photo_path = f"downloads/{file.file_path}"
        file.download(photo_path)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future = executor.submit(process_image, photo_path, update.message.chat_id, context.bot)
            future.add_done_callback(lambda fut: context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=open(fut.result(), 'rb')
            ))
    else:
        update.message.reply_text('Please send a photo.')

def process_image(photo_path, chat_id, bot):
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        pixelated_face = cv2.resize(face, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    processed_path = f"processed/{chat_id}_pixelated.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dp.add_handler(MessageHandler(Filters.command & Filters.text & ~Filters.update.edited_message, start))

    updater.start_polling()

    updater.idle()
    
if __name__ == '__main__':
    main()

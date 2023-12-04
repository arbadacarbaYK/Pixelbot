# pixelateTG.py

import os
import cv2
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5

def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Pixelate Faces", callback_data='pixelate')],
        [InlineKeyboardButton("Overlay with Liotta", callback_data='overlay')],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

def button_callback(update: Update, context: CallbackContext):
    query = update.callback_query
    option = query.data

    if option == 'pixelate':
        pixelate_faces(update, context)
    elif option == 'overlay':
        overlay_faces(update, context)

def pixelate_faces(update: Update, context: CallbackContext):
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = context.bot.get_file(file_id)

        # Extract the file name from the file path
        file_name = file.file_path.split('/')[-1]

        # Construct the local file path
        photo_path = f"downloads/{file_name}"
        file.download(photo_path)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future = executor.submit(process_image, photo_path, update.message.chat_id, file_id, context.bot)
            future.add_done_callback(lambda fut: context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=open(fut.result(), 'rb')
            ))
    else:
        update.message.reply_text('Please send a photo.')

def overlay_faces(update: Update, context: CallbackContext):
    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = context.bot.get_file(file_id)

        # Extract the file name from the file path
        file_name = file.file_path.split('/')[-1]

        # Construct the local file path
        photo_path = f"downloads/{file_name}"
        file.download(photo_path)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future = executor.submit(process_image_with_liotta, photo_path, update.message.chat_id, file_id, context.bot)
            future.add_done_callback(lambda fut: context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=open(fut.result(), 'rb')
            ))
    else:
        update.message.reply_text('Please send a photo.')

def process_image(photo_path, chat_id, file_id, bot):
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the more lightweight MTCNN model for face detection
    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        pixelated_face = cv2.resize(face, (0, 0), fx=0.03, fy=0.03, interpolation=cv2.INTER_NEAREST)
        image[y:y + h, x:x + w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    processed_path = f"processed/{chat_id}_{file_id}_pixelated.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def process_image_with_liotta(photo_path, chat_id, file_id, bot):
    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the more lightweight MTCNN model for face detection
    faces = detect_faces(image)

    liotta_path = "liotta.png"
    liotta = cv2.imread(liotta_path, cv2.IMREAD_UNCHANGED)

    for (x, y, w, h) in faces:
        liotta_resized = cv2.resize(liotta, (w, h), interpolation=cv2.INTER_AREA)

        alpha_liotta = liotta_resized[:, :, 3] / 255.0
        alpha_image = 1.0 - alpha_liotta

        for c in range(0, 3):
            image[y:y + h, x:x + w, c] = (alpha_liotta * liotta_resized[:, :, c] +
                                           alpha_image * image[y:y + h, x:x + w, c])

    processed_path = f"processed/{chat_id}_{file_id}_with_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def detect_faces(image):
    # Use MTCNN for face detection
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    return [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]

def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, start))
    dp.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()


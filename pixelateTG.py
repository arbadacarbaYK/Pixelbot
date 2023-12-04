import os
import cv2
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5

def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Pixelate Faces", callback_data='pixelate')],
        [InlineKeyboardButton("Liotta Overlay", callback_data='liotta')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Send me a picture, and choose an option:', reply_markup=reply_markup)

def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    option = query.data
    file_id = query.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    if option == 'pixelate':
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future = executor.submit(process_image, photo_path, update.message.chat_id, file_id, context.bot)
            future.add_done_callback(lambda fut: context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=open(fut.result(), 'rb')
            ))
    elif option == 'liotta':
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future = executor.submit(liotta_overlay, photo_path, update.message.chat_id, context.bot)
            future.add_done_callback(lambda fut: context.bot.send_photo(
                chat_id=update.message.chat_id,
                photo=open(fut.result(), 'rb')
            ))

def process_image(photo_path, chat_id, file_id, bot):
    # Your pixelation logic here

def liotta_overlay(photo_path, chat_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png')

    # Resize Liotta to fit the detected face
    faces = detect_faces(image)
    if faces:
        (x, y, w, h) = faces[0]  # Assuming only one face is detected
        liotta_resized = cv2.resize(liotta, (w, h))
        image[y:y+h, x:x+w] = liotta_resized

    processed_path = f"processed/{chat_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def detect_faces(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    bounding_boxes = [face['box'] for face in faces]
    return bounding_boxes

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, start))
    dp.add_handler(CallbackQueryHandler(button_click))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()


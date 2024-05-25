import os
import cv2
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN
from uuid import uuid4
from threading import Timer
import time

SESSION_TIMEOUT = 300  # 5 minutes in seconds

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5
PIXELATION_FACTOR = 0.03

RESIZE_FACTORS = {
    'liotta': 1.7,
    'skull': 1.7,
    'cats': 1.7,
    'pepe': 1.7,
    'chad': 1.7,
    'clowns': 1.7
}

executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]
    return head_boxes

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())  # Generate a unique session ID
    context.user_data[session_id] = {
        'state': 'waiting_for_photo',
        'timestamp': time.time(),
        'chat_id': update.message.chat_id,
        'message_id': update.message.message_id
    }

    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    # Check if any faces are detected
    image = cv2.imread(photo_path)
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    if not faces:
        # No faces detected, do nothing
        return

    keyboard = [
        [InlineKeyboardButton("Pixelate", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("Liotta Overlay", callback_data=f'liotta_{session_id}')],
        [InlineKeyboardButton("Skull of Satoshi", callback_data=f'skull_of_satoshi_{session_id}')],
        [InlineKeyboardButton("Cats (press until happy)", callback_data=f'cats_overlay_{session_id}')],
        [InlineKeyboardButton("Pepe (press until happy)", callback_data=f'pepe_overlay_{session_id}')],
        [InlineKeyboardButton("Chad (press until happy)", callback_data=f'chad_overlay_{session_id}')],
        [InlineKeyboardButton("Clowns (press until happy)", callback_data=f'clowns_overlay_{session_id}')],
        [InlineKeyboardButton("Cancel", callback_data=f'cancel_{session_id}')],  # Add Cancel button
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    context.user_data[session_id]['photo_path'] = photo_path
    context.user_data[session_id]['user_id'] = update.message.from_user.id
    # Delete the original picture from the chat
    update.message.delete()

    # Schedule a cleanup for this session after 5 minutes
    Timer(SESSION_TIMEOUT, clean_up_sessions, [context]).start()



def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_heads(image)

    def process_face(x, y, w, h):
        face = image[y:y+h, x:x+w]
        pixelated_face = cv2.resize(face, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in faces]
    wait(futures)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def overlay_image(image, overlay, heads, resize_factor):
    for (x, y, w, h) in heads:
        original_aspect_ratio = overlay.shape[1] / overlay.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * resize_factor * w) - int(0.1 * resize_factor * w)
        overlay_y = int(center_y - 0.5 * resize_factor * h) - int(0.1 * resize_factor * w)
        new_width = int(resize_factor * w)
        new_height = int(new_width / original_aspect_ratio)
        overlay_resized = cv2.resize(overlay, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            overlay_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (overlay_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - overlay_resized[:, :, 3:] / 255.0)
        )

    return image

def apply_overlay(photo_path, user_id, bot, overlay_name):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)
    
    if overlay_name == 'liotta':
        overlay = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)
    elif overlay_name == 'skull':
        overlay = cv2.imread('skullofsatoshi.png', cv2.IMREAD_UNCHANGED)
    elif overlay_name == 'cats':
        num_cats = len([name for name in os.listdir() if name.startswith('cat_')])
        overlay = cv2.imread(f'cat_{random.randint(1, num_cats)}.png', cv2.IMREAD_UNCHANGED)
    elif overlay_name == 'pepe':
        num_pepes = len([name for name in os.listdir() if name.startswith('pepe_')])
        overlay = cv2.imread(f'pepe_{random.randint(1, num_pepes)}.png', cv2.IMREAD_UNCHANGED)
    elif overlay_name == 'chad':
        num_chads = len([name for name in os.listdir() if name.startswith('chad_')])
        overlay = cv2.imread(f'chad_{random.randint(1, num_chads)}.png', cv2.IMREAD_UNCHANGED)
    elif overlay_name == 'clowns':
        num_clowns = len([name for name in os.listdir() if name.startswith('clown_')])
        overlay = cv2.imread(f'clown_{random.randint(1, num_clowns)}.png', cv2.IMREAD_UNCHANGED)
    
    resize_factor = RESIZE_FACTORS[overlay_name]
    image = overlay_image(image, overlay, heads, resize_factor)

    processed_path = f"processed/{user_id}_{overlay_name}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    user_data = context.user_data.get(session_id)

    if user_data and user_data['state'] == 'waiting_for_photo':
        photo_path = user_data.get('photo_path')
        user_id = user_data.get('user_id')
        user_data['timestamp'] = time.time()  # Update the timestamp on each interaction

        if query.data.startswith('cancel'):
            del context.user_data[session_id]  # Delete session data
            query.message.delete()  # Remove the message containing the keyboard
            return

        processed_path = None

        if query.data.startswith('pixelate'):
            processed_path = process_image(photo_path, user_id, query.id, context.bot)
        elif query.data.startswith('liotta'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'liotta')
        elif query.data.startswith('cats_overlay'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'cats')
        elif query.data.startswith('skull_of_satoshi'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'skull')
        elif query.data.startswith('pepe_overlay'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'pepe')
        elif query.data.startswith('chad_overlay'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'chad')
        elif query.data.startswith('clowns_overlay'):
            processed_path = apply_overlay(photo_path, user_id, context.bot, 'clowns')

        if processed_path:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))
            # Keep the keyboard visible by editing the original message's markup
            query.edit_message_reply_markup(reply_markup=query.message.reply_markup)
            return  # Exit the function here to prevent further execution



def clean_up_sessions(context: CallbackContext) -> None:
    current_time = time.time()
    user_data = context.user_data
    sessions_to_remove = []

    for session_id, data in user_data.items():
        if 'timestamp' in data and current_time - data['timestamp'] > SESSION_TIMEOUT:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        del user_data[session_id]

def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

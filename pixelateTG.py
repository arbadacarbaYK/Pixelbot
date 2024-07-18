import os
import cv2
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN
from uuid import uuid4

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5  # Common resize factor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(RESIZE_FACTOR * face['box'][2]), int(RESIZE_FACTOR * face['box'][3])) for face in faces]
    return head_boxes

def overlay(photo_path, user_id, overlay_type, resize_factor, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        overlay_files = [name for name in os.listdir() if name.startswith(f'{overlay_type}_')]
        if not overlay_files:
            continue
        random_overlay = random.choice(overlay_files)
        overlay_image = cv2.imread(random_overlay, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * resize_factor * w) - int(0.1 * resize_factor * w)
        overlay_y = int(center_y - 0.5 * resize_factor * h) - int(0.1 * resize_factor * w)
        new_width = int(resize_factor * w)
        new_height = int(new_width / original_aspect_ratio)
        overlay_image_resized = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            overlay_image_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (overlay_image_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - overlay_image_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_{overlay_type}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# looking for one straight file

def liotta_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'liotta', RESIZE_FACTOR, bot)

def skull_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'skullofsatoshi', RESIZE_FACTOR, bot)

# looking for a random file out of a similar naming

def pepe_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'pepe', RESIZE_FACTOR, bot)

def chad_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'chad', RESIZE_FACTOR, bot)

def cats_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'cat', RESIZE_FACTOR, bot)

def clowns_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'clown', RESIZE_FACTOR, bot)

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())  # Generate a unique session ID
    context.user_data[session_id] = {'state': 'waiting_for_photo'}

    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

# Check if any faces are detected
    image = cv2.imread(photo_path)
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)

    chat_id = update.effective_chat.id
    if chat_id > 0:  # Direct message
        if not faces:
            update.message.reply_text("No faces detected.")
            return

        keyboard = [
            [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
             InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_{session_id}'),
             InlineKeyboardButton("☠️ Skull", callback_data=f'skull_of_satoshi_{session_id}')],
            [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}'),
             InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
            [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),
             InlineKeyboardButton("CANCEL", callback_data=f'cancel_{session_id}')],  # Add Cancel button
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Press until happy', reply_markup=reply_markup)

        context.user_data[session_id]['photo_path'] = photo_path
        context.user_data[session_id]['user_id'] = update.message.from_user.id
        # Delete the original picture from the chat
        update.message.delete()
    else:  # Group chat
        update.message.reply_text('Reply to an image with /pixel to pixelate faces.')


def pixel_command(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())  # Generate a unique session ID
    context.user_data[session_id] = {'state': 'waiting_for_photo'}

    message = update.message.reply_to_message
    if not message or not message.photo:
        update.message.reply_text("Please reply to an image with /pixel to pixelate faces.")
        return

    file_id = message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    keyboard = [
        [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),
         InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_{session_id}')],
        [InlineKeyboardButton("☠️ Skull", callback_data=f'skull_of_satoshi_{session_id}'),
         InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}')],
        [InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),
         InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
        [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
         InlineKeyboardButton("CANCEL", callback_data=f'cancel_{session_id}')],  # Add Cancel button
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Press until happy', reply_markup=reply_markup)

    context.user_data[session_id]['photo_path'] = photo_path
    context.user_data[session_id]['user_id'] = update.message.from_user.id


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

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    user_data = context.user_data.get(session_id)

if user_data and user_data['state'] == 'waiting_for_photo':
        photo_path = user_data.get('photo_path')
        user_id = user_data.get('user_id')

        if query.data.startswith('cancel'):
            del context.user_data[session_id]  # Delete session data
            query.message.delete()  # Remove the message containing the keyboard
            return

        processed_path = None

        if query.data.startswith('pixelate'):
            processed_path = process_image(photo_path, user_id, query.id, context.bot)
        elif query.data.startswith('liotta'):
            processed_path = liotta_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('cats_overlay'):
            processed_path = cats_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('skull_of_satoshi'):
            processed_path = skull_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('pepe_overlay'):
            processed_path = pepe_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('chad_overlay'):
            processed_path = chad_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('clowns_overlay'):
            processed_path = clowns_overlay(photo_path, user_id, context.bot)

        if processed_path:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))
            # Keep the keyboard visible by editing the original message's markup
            query.edit_message_reply_markup(reply_markup=query.message.reply_markup)

def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("pixel", lambda update, context: pixel_command(update, context, dispatcher)))
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if name == 'main':
    main()

import os
import logging
from dotenv import load_dotenv
import cv2
import random
import imageio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN
from uuid import uuid4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Constants
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5

# Thread pool
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture or a GIF, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(RESIZE_FACTOR * face['box'][2]), int(RESIZE_FACTOR * face['box'][3])) for face in faces]
    return head_boxes

def overlay(photo_path, user_id, overlay_type, resize_factor, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)
    heads.sort(key=lambda box: box[1])

    for (x, y, w, h) in heads:
        overlay_files = [name for name in os.listdir() if name.startswith(f'{overlay_type}_')]
        if not overlay_files:
            continue
        random_overlay = random.choice(overlay_files)
        overlay_image = cv2.imread(random_overlay, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]

        new_width = int(resize_factor * w)
        new_height = int(new_width / original_aspect_ratio)
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * resize_factor * w) - int(0.1 * resize_factor * w)
        overlay_y = int(center_y - 0.5 * resize_factor * h) - int(0.1 * resize_factor * w)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        overlay_image_resized = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        roi_start_x = overlay_x
        roi_start_y = overlay_y
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)

        try:
            overlay_part = overlay_image_resized[:roi_end_y - roi_start_y, :roi_end_x - roi_start_x]
            alpha_mask = overlay_part[:, :, 3] / 255.0
            for c in range(3):
                image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, c] = (
                    alpha_mask * overlay_part[:, :, c] +
                    (1 - alpha_mask) * image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, c]
                )
        except ValueError as e:
            logger.error(f"Error blending overlay: {e}")
            continue

    processed_path = f"processed/{user_id}_{overlay_type}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return processed_path

def liotta_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'liotta', RESIZE_FACTOR, bot)

def skull_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'skullofsatoshi', RESIZE_FACTOR, bot)

def pepe_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'pepe', RESIZE_FACTOR, bot)

def chad_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'chad', RESIZE_FACTOR, bot)

def cats_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'cat', RESIZE_FACTOR, bot)

def clowns_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'clown', RESIZE_FACTOR, bot)

def process_gif(gif_path, session_id, user_id, bot):
    try:
        frames = imageio.mimread(gif_path)
        logger.info(f"Number of frames in GIF: {len(frames)}")

        processed_frames = []
        for frame in frames:
            temp_image_path = f"temp_frame_{session_id}.jpg"
            imageio.imwrite(temp_image_path, frame)
            processed_frame_path = process_image(temp_image_path, user_id, session_id, bot)
            processed_frame = imageio.imread(processed_frame_path)
            processed_frames.append(processed_frame)
            os.remove(temp_image_path)  # Clean up temporary frame image

        processed_gif_path = f"processed/{user_id}_{session_id}.gif"
        imageio.mimsave(processed_gif_path, processed_frames)
        logger.info(f"Processed GIF saved at: {processed_gif_path}")
        return processed_gif_path
    except Exception as e:
        logger.error(f"Error processing GIF: {e}")
        raise

def process_image(photo_path, user_id, session_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_heads(image)

    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        pixelation_size = max(1, int(PIXELATION_FACTOR * min(w, h)))

        pixelated_roi = cv2.resize(roi, (pixelation_size, pixelation_size), interpolation=cv2.INTER_NEAREST)
        pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        if pixelated_roi.shape[0] != h or pixelated_roi.shape[1] != w:
            pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        if pixelated_roi.shape == (h, w, 3):
            image[y:y+h, x:x+w] = pixelated_roi
        else:
            logger.error(f"Dimension mismatch: pixelated_roi shape {pixelated_roi.shape} != roi shape {(h, w, 3)}")

    processed_path = f"processed/{user_id}_{session_id}_pixelated.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return processed_path

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    user_data = context.user_data
    chat_data = context.chat_data
    data = user_data.get(session_id) or chat_data.get(session_id)

    if data:
        photo_path = data.get('photo_path')
        user_or_chat_id = data.get('user_id') or data.get('chat_id')

        if query.data.startswith('cancel'):
            if session_id in user_data:
                del user_data[session_id]
            if session_id in chat_data:
                del chat_data[session_id]
            query.message.delete()
            return

        processed_path = None
        if query.data.startswith('pixelate'):
            processed_path = process_image(photo_path, user_or_chat_id, query.id, context.bot)
        elif query.data.startswith('liotta'):
            processed_path = liotta_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('cats_overlay'):
            processed_path = cats_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('chad'):
            processed_path = chad_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('clowns'):
            processed_path = clowns_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('pepe'):
            processed_path = pepe_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('skull'):
            processed_path = skull_overlay(photo_path, user_or_chat_id, context.bot)

        if processed_path:
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(processed_path, 'rb'))
            if os.path.exists(processed_path):
                os.remove(processed_path)
            del user_data[session_id]
            del chat_data[session_id]
    else:
        query.edit_message_text(text="Sorry, I lost the image. Please send it again.")

def handle_photo(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    file = context.bot.getFile(update.message.photo[-1].file_id)
    session_id = uuid4().hex
    download_path = f'downloads/{session_id}.jpg'
    file.download(download_path)

    buttons = [
        [InlineKeyboardButton("Pixelate", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("Ray Liotta", callback_data=f'liotta_{session_id}'),
         InlineKeyboardButton("Cats", callback_data=f'cats_overlay_{session_id}')],
        [InlineKeyboardButton("Chad", callback_data=f'chad_{session_id}'),
         InlineKeyboardButton("Clowns", callback_data=f'clowns_{session_id}')],
        [InlineKeyboardButton("Pepe", callback_data=f'pepe_{session_id}'),
         InlineKeyboardButton("Skull of Satoshi", callback_data=f'skull_{session_id}')],
        [InlineKeyboardButton("Cancel", callback_data=f'cancel_{session_id}')],
    ]

    user_data = context.user_data
    chat_data = context.chat_data
    user_data[session_id] = {'photo_path': download_path, 'user_id': user_id}
    chat_data[session_id] = {'photo_path': download_path, 'chat_id': chat_id}

    update.message.reply_text(
        'Choose an action:',
        reply_markup=InlineKeyboardMarkup(buttons)
    )

def handle_gif(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    file = context.bot.getFile(update.message.document.file_id)
    session_id = uuid4().hex
    download_path = f'downloads/{session_id}.gif'
    file.download(download_path)

    buttons = [
        [InlineKeyboardButton("Pixelate", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("Ray Liotta", callback_data=f'liotta_{session_id}'),
         InlineKeyboardButton("Cats", callback_data=f'cats_overlay_{session_id}')],
        [InlineKeyboardButton("Chad", callback_data=f'chad_{session_id}'),
         InlineKeyboardButton("Clowns", callback_data=f'clowns_{session_id}')],
        [InlineKeyboardButton("Pepe", callback_data=f'pepe_{session_id}'),
         InlineKeyboardButton("Skull of Satoshi", callback_data=f'skull_{session_id}')],
        [InlineKeyboardButton("Cancel", callback_data=f'cancel_{session_id}')],
    ]

    user_data = context.user_data
    chat_data = context.chat_data
    user_data[session_id] = {'gif_path': download_path, 'user_id': user_id}
    chat_data[session_id] = {'gif_path': download_path, 'user_id': user_id}

    update.message.reply_text(
        'Choose an action:',
        reply_markup=InlineKeyboardMarkup(buttons)
    )

def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
    dispatcher.add_handler(MessageHandler(Filters.document.gif, handle_gif))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

import os 
from dotenv import load_dotenv
import cv2
import random
import imageio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN
from uuid import uuid4
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5
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
            print(f"Error blending overlay: {e}")
            continue

    processed_path = f"processed/{user_id}_{overlay_type}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return processed_path

# Overlay functions
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
        processed_frames = [process_image(frame, user_id, session_id, bot) for frame in frames]
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
        pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)  # Ensure same dimensions
        image[y:y+h, x:x+w] = pixelated_roi

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
        elif query        .data.startswith('cats_overlay'):
            processed_path = cats_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('skull_overlay'):
            processed_path = skull_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('pepe_overlay'):
            processed_path = pepe_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('chad_overlay'):
            processed_path = chad_overlay(photo_path, user_or_chat_id, context.bot)
        elif query.data.startswith('clowns_overlay'):
            processed_path = clowns_overlay(photo_path, user_or_chat_id, context.bot)

        if processed_path:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())
    user_data = context.user_data

    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = context.bot.get_file(file_id)
        file_name = file.file_path.split('/')[-1]

        # Process the image
        photo_path = f"downloads/{file_name}"
        file.download(photo_path)
        
        logger.info(f"Photo downloaded to {photo_path}")

        image = cv2.imread(photo_path)
        faces = detect_heads(image)

        if not faces:
            update.message.reply_text('No faces detected in the image.')
            return

        keyboard = [
            [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
             InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_overlay_{session_id}'),
             InlineKeyboardButton("☠️ Skull", callback_data=f'skull_overlay_{session_id}')],
            [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}'),
             InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
            [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),
             InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        user_data[session_id] = {'photo_path': photo_path, 'user_id': update.message.from_user.id}

        update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)
        update.message.delete()
    elif update.message.document and update.message.document.mime_type == 'image/gif':
        file_id = update.message.document.file_id
        file = context.bot.get_file(file_id)
        file_name = file.file_path.split('/')[-1]

        # Download the GIF file
        gif_path = f"downloads/{file_name}"
        file.download(gif_path)
        
        logger.info(f"GIF downloaded to {gif_path}")

        # Process the GIF
        try:
            processed_gif_path = process_gif(gif_path, session_id, update.message.from_user.id, context.bot)
            context.bot.send_animation(chat_id=update.message.from_user.id, animation=open(processed_gif_path, 'rb'))
        except Exception as e:
            logger.error(f"Error processing GIF: {e}")

        # Clean up temporary files
        os.remove(gif_path)
    else:
        update.message.reply_text('Please send either a photo or a GIF.')


def pixelate_command(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())
    chat_data = context.chat_data

    if update.message.reply_to_message and (update.message.reply_to_message.photo or update.message.reply_to_message.document.mime_type == 'image/gif'):
        if update.message.reply_to_message.photo:
            file_id = update.message.reply_to_message.photo[-1].file_id
            file = context.bot.get_file(file_id)
            file_name = file.file_path.split('/')[-1]
            photo_path = f"downloads/{file_name}"
            file.download(photo_path)
            image = cv2.imread(photo_path)
            faces = detect_heads(image)
        else: # GIF handling
            file_id = update.message.reply_to_message.document.file_id
            file = context.bot.get_file(file_id)
            file_name = file.file_path.split('/')[-1]
            photo_path = f"downloads/{file_name}"
            file.download(photo_path)
            # For GIFs, no need to detect faces at this step

        if not faces and not update.message.reply_to_message.document.mime_type == 'image/gif':
            update.message.reply_text('No faces detected in the image.')
            return

        keyboard = [
            [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
             InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_overlay_{session_id}'),
             InlineKeyboardButton("☠️ Skull", callback_data=f'skull_overlay_{session_id}')],
            [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}'),
             InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
            [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),
             InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        chat_data[session_id] = {'photo_path': photo_path, 'chat_id': update.message.chat_id}

        update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)
    elif update.message.reply_to_message and update.message.reply_to_message.document.mime_type == 'image/gif':
        file_id = update.message.reply_to_message.document.file_id
        file = context.bot.get_file(file_id)
        file_name = file.file_path.split('/')[-1]

        # Download the GIF file
        gif_path = f"downloads/{file_name}"
        file.download(gif_path)
        
        logger.info(f"GIF downloaded to {gif_path}")

        # Process the GIF
        try:
            processed_gif_path = process_gif(gif_path, session_id, update.message.chat_id, context.bot)
            context.bot.send_animation(chat_id=update.message.chat_id, animation=open(processed_gif_path, 'rb'))
        except Exception as e:
            logger.error(f"Error processing GIF: {e}")

        # Clean up temporary files
        os.remove(gif_path)
    else:
        update.message.reply_text('Please reply to a photo or a GIF.')


def main() -> None:
    updater = Updater(TOKEN)

    # Dispatchers
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("pixelate", pixelate_command))

    # Callbacks
    dp.add_handler(CallbackQueryHandler(button_callback))

    # Message handler for photos and GIFs
    dp.add_handler(MessageHandler(Filters.photo | Filters.document.mime_type("image/gif"), pixelate_faces))

    # Start the Bot
    updater.start_polling()
    logger.info("Bot started polling...")

    # Run the bot until you press Ctrl-C
    updater.idle()


if __name__ == '__main__':
    main()


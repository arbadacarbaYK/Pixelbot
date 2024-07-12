import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
from dotenv import load_dotenv
import cv2
import random
import imageio
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import moviepy.editor as mp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')  # Get the Telegram bot token from the environment variable
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5  # Common resize factor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    """Handles the /start command to welcome the user. Applicable for both DMs and groups."""
    update.message.reply_text('Send me a picture, GIF, or MP4 video, and I will process faces in it!')

def detect_heads(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    return faces

def process_image(image, overlay_type=None):
    faces = detect_heads(image)
    for face in faces:
        x, y, w, h = face['box']
        roi = image[y:y+h, x:x+w]
        pixelation_size = max(1, int(PIXELATION_FACTOR * min(w, h)))
        pixelated_roi = cv2.resize(roi, (pixelation_size, pixelation_size), interpolation=cv2.INTER_NEAREST)
        pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated_roi
    if overlay_type:
        image = overlay(image, overlay_type)
    return image

def process_gif(gif_path, session_id, user_id):
    frames = imageio.mimread(gif_path)
    processed_frames = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = process_image(frame_rgb)
        processed_frames.append(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
    processed_gif_path = f"processed/{user_id}_{session_id}_pixelated.gif"
    imageio.mimsave(processed_gif_path, processed_frames)
    return processed_gif_path

def process_video(video_path, session_id, user_id):
    clip = mp.VideoFileClip(video_path)
    frames = [frame for frame in clip.iter_frames()]
    processed_frames = [process_image(frame) for frame in frames]
    processed_clip = mp.ImageSequenceClip(processed_frames, fps=clip.fps)
    processed_video_path = f"processed/{user_id}_{session_id}.mp4"
    processed_clip.write_videofile(processed_video_path, codec='libx264')
    return processed_video_path

def handle_photo(update: Update, context: CallbackContext) -> None:
    """Handles photo messages for both DMs and groups."""
    session_id = str(uuid4())
    user_data = context.user_data

    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

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
        [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    user_data[session_id] = {'photo_path': photo_path, 'user_id': update.message.from_user.id}

    update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)
    update.message.delete()

def handle_gif_or_video(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())
    user_data = context.user_data

    if update.message.document:
        mime_type = update.message.document.mime_type
        if mime_type in ['image/gif', 'video/mp4']:
            file_id = update.message.document.file_id
            file = context.bot.get_file(file_id)
            file_name = file.file_path.split('/')[-1]
            media_path = f"downloads/{file_name}"
            file.download(media_path)

            if mime_type == 'image/gif':
                processed_gif_path = process_gif(media_path, session_id, user_data['user_id'])
                context.bot.send_animation(chat_id=update.message.chat_id, animation=open(processed_gif_path, 'rb'))
            elif mime_type == 'video/mp4':
                processed_video_path = process_video(media_path, session_id, user_data['user_id'])
                context.bot.send_video(chat_id=update.message.chat_id, video=open(processed_video_path, 'rb'))
    else:
        update.message.reply_text('Please send either a GIF or a video.')

def pixelate_command(update: Update, context: CallbackContext) -> None:
    """Handles the /pixel command to pixelate faces in a photo, GIF, or video. Applicable for both DMs and groups."""
    if update.message.reply_to_message and (update.message.reply_to_message.photo or update.message.reply_to_message.document):
        context.args = ['pixelate']
        pixelate_faces(update, context)
    else:
        update.message.reply_text('Please reply to a photo, GIF, or video to pixelate faces.')

def button_callback(update: Update, context: CallbackContext) -> None:
    """Handles button presses for selecting overlays or pixelation. Applicable for both DMs and groups."""
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    user_data = context.user_data

    if session_id not in user_data:
        query.edit_message_text('Session expired. Please try again.')
        return

    photo_path = user_data[session_id]['photo_path']
    user_id = user_data[session_id]['user_id']

    if 'pixelate' in query.data:
        processed_path = process_image(cv2.imread(photo_path))
    elif 'liotta_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'liotta')
    elif 'skull_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'skullofsatoshi')
    elif 'pepe_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'pepe')
    elif 'chad_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'chad')
    elif 'cats_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'cat')
    elif 'clowns_overlay' in query.data:
        processed_path = overlay(cv2.imread(photo_path), 'clown')
    else:
        query.edit_message_text('Invalid option. Please try again.')
        return

    processed_file_path = f"processed/{user_id}_{session_id}_processed.jpg"
    cv2.imwrite(processed_file_path, processed_path, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_file_path, 'rb'))

def main() -> None:
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("pixel", pixelate_command))
    dispatcher.add_handler(MessageHandler(Filters.photo & Filters.private, handle_photo))
    dispatcher.add_handler(MessageHandler(Filters.document.mime_type(['image/gif', 'video/mp4']), handle_gif_or_video))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

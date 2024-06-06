import os
from dotenv import load_dotenv
import cv2
import random
import imageio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN
from uuid import uuid4
import moviepy.editor as mpy

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    """Handles the /start command to welcome the user. Applicable for both DMs and groups."""
    update.message.reply_text('Send me a picture, GIF, or MP4 video, and I will process faces in it!')

def detect_heads(image):
    """Detects faces in an image using MTCNN. Used for processing images (photos). Applicable for both DMs and groups."""
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(RESIZE_FACTOR * face['box'][2]), int(RESIZE_FACTOR * face['box'][3])) for face in faces]
    head_boxes.sort(key=lambda box: box[1])
    return head_boxes

def overlay(photo_path, user_id, overlay_type, resize_factor, bot):
    """Applies the specified overlay to detected faces in the photo. Used for processing images (photos). Applicable for both DMs and groups."""
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

def process_video(video_path, user_id, bot, overlay_type):
    """Processes each frame of the video with the specified overlay. Used for processing MP4 videos. Applicable for both DMs and groups."""
    video = mpy.VideoFileClip(video_path)
    processed_frames = []

    for progress in range(0, 101, 10):
        print(f"Processing video frame {progress}...")
        frames = video.snap(progress / 100)
        for frame in frames:
            processed_frame_path = f"processed/{user_id}_{overlay_type}_{progress}_frame.jpg"
            frame.save_frame(processed_frame_path, quality_percent(progress))
            processed_frames.append(processed_frame_path)

    processed_video_path = f"processed/{user_id}_{overlay_type}.mp4"
    video.write_videoclips(processed_video_path, fps=video.fps, audio=False)

    return processed_video_path

# Overlay functions for different types
def liotta_overlay(photo_path, user_id, bot):
    """Applies Liotta overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'liotta', RESIZE_FACTOR, bot)

def skull_overlay(photo_path, user_id, bot):
    """Applies Skull overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'skullofsatoshi', RESIZE_FACTOR, bot)

def pepe_overlay(photo_path, user_id, bot):
    """Applies Pepe overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'pepe', RESIZE_FACTOR, bot)

def chad_overlay(photo_path, user_id, bot):
    """Applies Chad overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'chad', RESIZE_FACTOR, bot)

def cats_overlay(photo_path, user_id, bot):
    """Applies Cats overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'cat', RESIZE_FACTOR, bot)

def clowns_overlay(photo_path, user_id, bot):
    """Applies Clowns overlay to the photo. Used for processing images (photos). Applicable for both DMs and groups."""
    return overlay(photo_path, user_id, 'clown', RESIZE_FACTOR, bot)

def process_gif(gif_path, session_id, user_id, bot):
    """Processes each frame of the GIF with the specified overlay. Used for processing GIFs. Applicable for both DMs and groups."""
    frames = imageio.mimread(gif_path)
    processed_frames = [process_image(frame, user_id, session_id, bot) for frame in frames]
    processed_gif_path = f"processed/{user_id}_{session_id}.gif"
    imageio.mimsave(processed_gif_path, processed_frames)
    return processed_gif_path

def process_video(video_path, session_id, user_id, bot, overlay_type):
    """Processes each frame of the video with the specified overlay. Used for processing MP4 videos. Applicable for both DMs and groups."""
    video = mpy.VideoFileClip(video_path)
    processed_frames = []

    for progress in range(0, 101, 10):
        print(f"Processing video frame {progress}...")
        frames = video.subclip(progress / 100, (progress + 10) / 100)
        for frame in frames.iter_frames():
            processed_frame = overlay(frame, overlay_type)
            processed_frames.append(processed_frame)

    processed_video_path = f"processed/{user_id}_{session_id}_{overlay_type}.mp4"
    mpy.ImageSequenceClip(processed_frames, fps=video.fps).write_videofile(processed_video_path, codec='libx264', fps=video.fps)
    return processed_video_path

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    """Main handler function to process photos, GIFs, and MP4 videos, detecting faces and applying overlays or pixelating faces."""
    session_id = str(uuid4())
    user_data = context.user_data

    if update.message.photo:
        # Handles photo messages for both DMs and groups
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
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')]
        ]
        
        if update.message.chat.type == 'private':
            keyboard.append([InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}')])

        keyboard.append([InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        user_data[session_id] = {'photo_path': photo_path, 'user_id': update.message.from_user.id}

        update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)
        update.message.delete()

    elif update.message.document and update.message.document.mime_type == 'image/gif':
        # Handles GIF messages for both DMs and groups
        file_id = update.message.document.file_id
        file = context.bot.get_file(file_id)
        file_name = file.file_path.split('/')[-1]
        gif_path = f"downloads/{file_name}"
        file.download(gif_path)

        processed_gif_path = process_gif(gif_path, session_id, str(uuid4()), context.bot)
        context.bot.send_animation(chat_id=update.message.chat_id, animation=open(processed_gif_path, 'rb'))

    elif update.message.document and update.message.document.mime_type == 'video/mp4':
        # Handles MP4 video messages for both DMs and groups
        file_id = update.message.document.file_id
        file = context.bot.get_file(file_id)
        file_name = file.file_path.split('/')[-1]
        video_path = f"downloads/{file_name}"
        file.download(video_path)

        overlay_type = 'default_overlay'  # Set a default overlay type
        processed_video_path = process_video(video_path, str(uuid4()), context.bot, overlay_type)
        context.bot.send_video(chat_id=update.message.chat_id, video=open(processed_video_path, 'rb'))

    else:
        update.message.reply_text('Please send either a photo, GIF, or MP4 video.')

def pixelate_command(update: Update, context: CallbackContext) -> None:
    """Handles the /pixel command to pixelate faces in a photo, GIF, or video. Applicable for both DMs and groups."""
    if update.message.reply_to_message and (update.message.reply_to_message.photo or update.message.reply_to_message.document):
        context.args = ['pixelate']
        pixelate_faces(update, context)
    else:
        update.message.reply_text('Please reply to a photo, GIF, or video to pixelate faces.')

def button(update: Update, context: CallbackContext) -> None:
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
        # Pixelate functionality - handle accordingly
        pass
    else:
        overlay_type = query.data.split('_')[0]  # Extract overlay type
        if photo_path.endswith('.mp4'):
            processed_path = process_video(photo_path, session_id, user_id, context.bot, overlay_type)
        else:
            processed_path = overlay(photo_path, user_id, overlay_type, RESIZE_FACTOR, context.bot)

    if processed_path:
        if photo_path.endswith('.mp4'):
            context.bot.send_video(chat_id=query.message.chat_id, video=open(processed_path, 'rb'))
        else:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))
    else:
        query.edit_message_text('Error processing the media.')


def main() -> None:
    """Main entry point for the bot, setting up the command and message handlers."""
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("pixel", pixelate_command))
    dispatcher.add_handler(CallbackQueryHandler(button))
    dispatcher.add_handler(MessageHandler(Filters.photo | Filters.document, pixelate_faces))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

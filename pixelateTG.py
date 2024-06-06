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

def pixelate(photo_path):
    """Pixelates detected faces in the photo."""
    image = cv2.imread(photo_path)
    heads = detect_heads(image)
    
    for (x, y, w, h) in heads:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (int(w * PIXELATION_FACTOR), int(h * PIXELATION_FACTOR)), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = face
    
    processed_path = f"processed/{uuid4()}_pixelated.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return processed_path

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
            keyboard.append([InlineKeyboardButton("🟦 Pixelate", callback_data=f'pixelate_{session_id}')])

        user_data[session_id] = photo_path
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Choose an overlay or pixelate option:', reply_markup=reply_markup)
    else:
        update.message.reply_text('Please send a photo for processing.')

def handle_button_click(update: Update, context: CallbackContext) -> None:
    """Handles the button click for selecting overlays and pixelation. Applicable for both DMs and groups."""
    query = update.callback_query
    user_id = query.from_user.id
    session_id = query.data.split('_')[-1]
    action = query.data.rsplit('_', 1)[0]

    if session_id not in context.user_data:
        query.answer(text='Session expired. Please send a new photo.')
        return

    photo_path = context.user_data[session_id]

    if action == 'pixelate':
        processed_path = pixelate(photo_path)
    elif action == 'clowns_overlay':
        processed_path = clowns_overlay(photo_path, user_id, context.bot)
    elif action == 'liotta_overlay':
        processed_path = liotta_overlay(photo_path, user_id, context.bot)
    elif action == 'skull_overlay':
        processed_path = skull_overlay(photo_path, user_id, context.bot)
    elif action == 'cats_overlay':
        processed_path = cats_overlay(photo_path, user_id, context.bot)
    elif action == 'pepe_overlay':
        processed_path = pepe_overlay(photo_path, user_id, context.bot)
    elif action == 'chad_overlay':
        processed_path = chad_overlay(photo_path, user_id, context.bot)
    else:
        query.answer(text='Invalid action.')
        return

    context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))
    query.answer()

def main():
    """Main function to start the bot and define handlers. Applicable for both DMs and groups."""
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(handle_button_click))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

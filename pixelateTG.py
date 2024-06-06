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

def pixelate(image):
    """Pixelates the faces in the image. Used for processing images (photos). Applicable for both DMs and groups."""
    faces = detect_heads(image)
    for (x, y, w, h) in faces:
        # Define the region of interest (ROI)
        roi = image[y:y+h, x:x+w]

        # Apply pixelation to the ROI
        pixelated_roi = cv2.resize(roi, (PIXEL_SIZE, PIXEL_SIZE), interpolation=cv2.INTER_NEAREST)
        pixelated_roi = cv2.resize(pixelated_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        # Replace the original face region with the pixelated ROI
        image[y:y+h, x:x+w] = pixelated_roi

    return image

def process_image(photo_path, user_id, session_id, bot):
    """Processes the image, applying pixelation to faces. Used for processing images (photos). Applicable for both DMs and groups."""
    image = cv2.imread(photo_path)
    pixelated_image = pixelate(image)
    processed_path = f"processed/{user_id}_{session_id}_pixelated.jpg"
    cv2.imwrite(processed_path, pixelated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
            keyboard.append([InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}')])

        keyboard.append([InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        user_data[session_id] = {'photo_path': photo_path, 'user_id': update.message.from_user.id}

        update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)


def pixelate_command(update: Update, context: CallbackContext) -> None:
    """Handles the /pixel command to pixelate faces in a photo, GIF, or video. Applicable for both DMs and groups."""
    if update.message.reply_to_message and (update.message.reply_to_message.photo or update.message.reply_to_message.document):
        context.args = ['pixelate']
        pixelate_faces(update, context)
    else:
        update.message.reply_text('Please reply to a photo, GIF, or video to pixelate faces.')

def button(update: Update, context: CallbackContext) -> None:
    """Handles button clicks."""
    query = update.callback_query
    query.answer()

    # Split query data into overlay type and session ID
    data_parts = query.data.split('_')
    print("Data parts:", data_parts)  # Debugging
    if len(data_parts) != 2:
        query.edit_message_text('Invalid button data. Please try again.')
        return

    overlay_type, session_id = data_parts
    user_id = query.from_user.id

    # Debugging
    print("Overlay type:", overlay_type)
    print("Session ID:", session_id)

    # Retrieve photo path from user data
    user_data = context.user_data
    if session_id not in user_data:
        query.edit_message_text('Session expired. Please try again.')
        return
    photo_path = user_data[session_id]['photo_path']

    # Apply overlay or pixelation based on button clicked
    if overlay_type == 'pixelate':
        executor.submit(process_image, photo_path, user_id, session_id, context.bot)
        query.edit_message_text('Pixelating faces. Please wait...')
    else:
        executor.submit(overlay, photo_path, user_id, overlay_type, RESIZE_FACTOR, context.bot)
        query.edit_message_text('Applying overlay. Please wait...')


def main():
    """Starts the bot."""
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    # Register handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(button))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()


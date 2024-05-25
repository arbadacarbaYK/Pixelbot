from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from mtcnn.mtcnn import MTCNN
from threading import Timer
import os
import cv2
import random
import time

SESSION_TIMEOUT = 1800  # 30 minutes in seconds

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5
PIXELATION_FACTOR = 0.03

# Dictionary to store session data using message IDs as keys
session_data = {}

# Dictionary to store message IDs of original pictures sent by users
original_pictures = {}

# Dictionary to store user options selected for each picture
user_options = {}

# Dictionary to store the timer for each session
session_timers = {}

# Dictionary to store the resize factors for different overlays
RESIZE_FACTORS = {
    'liotta': 2.1,
    'skull': 2.1,
    'cats': 2.1,
    'pepe': 2.1,
    'chad': 2.3,
    'clowns': 2.1
}

# Function to start the bot
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

# Function to detect faces in an image
def detect_faces(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]
    return head_boxes

# Function to pixelate faces in an image
def pixelate_faces(update: Update, context: CallbackContext) -> None:
    # Generate a unique session ID
    session_id = update.message.message_id
    
    # Store the session data using the message ID as the key
    session_data[session_id] = {
        'state': 'waiting_for_photo',
        'timestamp': time.time(),
        'chat_id': update.message.chat_id
    }
    
    # Store the message ID of the original picture
    original_pictures[session_id] = update.message.message_id

    # Download the photo sent by the user
    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    # Check if any faces are detected
    image = cv2.imread(photo_path)
    faces = detect_faces(image)
    if not faces:
        # No faces detected, do nothing
        return

    # Display options for the user to choose
    keyboard = [
        [InlineKeyboardButton("Pixelate", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("Liotta Overlay", callback_data=f'liotta_{session_id}')],
        [InlineKeyboardButton("Skull of Satoshi", callback_data=f'skull_of_satoshi_{session_id}')],
        [InlineKeyboardButton("Cats (press until happy)", callback_data=f'cats_{session_id}')],
        [InlineKeyboardButton("Pepe (press until happy)", callback_data=f'pepe_{session_id}')],
        [InlineKeyboardButton("Chad (press until happy)", callback_data=f'chad_{session_id}')],
        [InlineKeyboardButton("Clowns (press until happy)", callback_data=f'clowns_{session_id}')],
        [InlineKeyboardButton("Cancel", callback_data=f'cancel_{session_id}')],  # Add Cancel button
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    # Store the photo path and user ID in the session data
    session_data[session_id]['photo_path'] = photo_path
    session_data[session_id]['user_id'] = update.message.from_user.id

    # Schedule a cleanup for this session after 30 minutes
    session_timers[session_id] = Timer(SESSION_TIMEOUT, clean_up_session, [session_id])
    session_timers[session_id].start()

# Function to process the selected option
def process_option(update: Update, context: CallbackContext, option: str, session_id: int) -> None:
    photo_path = session_data[session_id]['photo_path']
    user_id = session_data[session_id]['user_id']
    session_data[session_id]['timestamp'] = time.time()  # Update the timestamp on each interaction

    processed_path = None

    if option.startswith('pixelate'):
        processed_path = process_image(photo_path, user_id, option, context.bot)
    else:
        overlay_name = option.split('_')[0]
        processed_path = apply_overlay(photo_path, user_id, context.bot, overlay_name)

    if processed_path:
        context.bot.send_photo(chat_id=session_data[session_id]['chat_id'], photo=open(processed_path, 'rb'))
        # Keep the keyboard visible by editing the original message's markup
        update.callback_query.edit_message_reply_markup(reply_markup=update.callback_query.message.reply_markup)

# Function to process the image and pixelate faces
def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_faces(image)

    def process_face(x, y, w, h):
        face = image[y:y+h, x:x+w]
        pixelated_face = cv2.resize(face, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    for (x, y, w, h) in faces:
        process_face(x, y, w, h)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# Function to apply an overlay to faces
def apply_overlay(photo_path, user_id, bot, overlay_name):
    image = cv2.imread(photo_path)
    faces = detect_faces(image)
    
    # Load the overlay image
    overlay = cv2.imread(f'{overlay_name}.png', cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return None  # or handle the error in an appropriate way
    
    resize_factor = RESIZE_FACTORS[overlay_name]

    for (x, y, w, h) in faces:
        original_aspect_ratio = overlay.shape[1] / overlay.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * resize_factor * w) - int(0.0 * resize_factor * w)  
        overlay_y = int(center_y - 0.5 * resize_factor * h) - int(0.0 * resize_factor * w)
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

    processed_path = f"processed/{user_id}_{overlay_name}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path


# Function to handle button clicks
def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = int(query.data.split('_')[-1])

    if session_id not in session_data:
        return

    if query.data.startswith('cancel'):
        query.message.delete()
        cancel_session(session_id)
    else:
        process_option(update, context, query.data, session_id)

# Function to cancel a session
def cancel_session(session_id: int) -> None:
    if session_id in session_data:
        del session_data[session_id]
    if session_id in original_pictures:
        del original_pictures[session_id]
    if session_id in user_options:
        del user_options[session_id]
    if session_id in session_timers:
        session_timers[session_id].cancel()
        del session_timers[session_id]

# Function to clean up an expired session
def clean_up_session(session_id: int) -> None:
    cancel_session(session_id)

# Function to start the bot
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

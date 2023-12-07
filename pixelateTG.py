import os
import cv2
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5
PIXELATION_FACTOR = 0.03
LIOTTA_RESIZE_FACTOR = 1.5
SKULL_RESIZE_FACTOR = 1.9  # Adjust the resize factor for Skull of Satoshi

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    
    # Extracting bounding boxes from the faces
    head_boxes = [(face['box'][0], face['box'][1], int(LIOTTA_RESIZE_FACTOR * face['box'][2]), int(LIOTTA_RESIZE_FACTOR * face['box'][3])) for face in faces]
    
    return head_boxes

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    
    # Extract the file name from the file path
    file_name = file.file_path.split('/')[-1]
    
    # Construct the local file path
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    keyboard = [
        [InlineKeyboardButton("Pixelate", callback_data='pixelate')],
        [InlineKeyboardButton("Liotta Overlay", callback_data='liotta')],
        [InlineKeyboardButton("Skull of Satoshi", callback_data='skull_of_satoshi')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    # Save photo_path and user_id in context for later use in button callback
    context.user_data['photo_path'] = photo_path
    context.user_data['user_id'] = update.message.from_user.id

# Inside liotta_overlay function
def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)

    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Calculate aspect ratio of the original liotta image
        original_aspect_ratio = liotta.shape[1] / liotta.shape[0]

        # Adjusting starting position for better alignment
        overlay_x = max(0, x - int(0.15 * w))
        overlay_y = max(0, y - int(0.15 * h))

        # Resize Liotta to match the width and height of the face
        new_width = int(LIOTTA_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)

        liotta_resized = cv2.resize(liotta, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Blend Liotta and ROI using alpha channel
        image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] = (
            liotta_resized[:, :, :3] * (liotta_resized[:, :, 3:] / 255.0) +
            image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] *
            (1.0 - liotta_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# Inside skull_overlay function
def skull_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    skull = cv2.imread('skullofsatoshi.png', cv2.IMREAD_UNCHANGED)

    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Calculate aspect ratio of the original skull image
        original_aspect_ratio = skull.shape[1] / skull.shape[0]

        # Adjusting starting position for better alignment
        overlay_x = max(0, x - int(0.2 * w))
        overlay_y = max(0, y - int(0.2 * h))

        # Resize Skull of Satoshi to match the width and height of the face
        new_width = int(SKULL_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)

        skull_resized = cv2.resize(skull, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Blend Skull of Satoshi and ROI using alpha channel
        image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] = (
            skull_resized[:, :, :3] * (skull_resized[:, :, 3:] / 255.0) +
            image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] *
            (1.0 - skull_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_skull_of_satoshi.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()

    # Check if 'photo_path' key is present in context.user_data
    if 'photo_path' in context.user_data:
        option = query.data
        photo_path = context.user_data['photo_path']
        user_id = context.user_data['user_id']

        if option == 'pixelate':
            processed_path = process_image(photo_path, user_id, 'pixelated', context.bot)
        elif option == 'liotta':
            processed_path = liotta_overlay(photo_path, user_id, context.bot)
        elif option == 'skull_of_satoshi':
            processed_path = skull_overlay(photo_path, user_id, context.bot)

        context.bot.send_photo(chat_id=update.callback_query.message.chat_id, photo=open(processed_path, 'rb'))
    else:
        # Handle the case when 'photo_path' key is not present
        update.message.reply_text("Error: 'photo_path' not found in user data.")

def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_heads(image)

    def process_face(x, y, w, h):
        # Extract the face
        face = image[y:y+h, x:x+w]

        # Apply pixelation directly to the face
        pixelated_face = cv2.resize(face, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)

        # Replace the face in the original image with the pixelated version
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in faces]
    wait(futures)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def main() -> None:
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
    main()

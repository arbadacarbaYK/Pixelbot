import os
import cv2
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5
PIXELATION_FACTOR = 0.03
LIOTTA_RESIZE_FACTOR = 1.5
SKULL_RESIZE_FACTOR = 1.9  # Adjust the resize factor for Skull of Satoshi
CATS_RESIZE_FACTOR = 1.5  # Adjust the resize factor for cats

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
        [InlineKeyboardButton("Cats (press until happy)", callback_data='cats_overlay')],
        [InlineKeyboardButton("Use own file", callback_data='swap_face')],
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

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Adjusting starting position based on the center for better alignment
        overlay_x = int(center_x - 0.5 * LIOTTA_RESIZE_FACTOR * w) - int(0.1 * LIOTTA_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * LIOTTA_RESIZE_FACTOR * h)

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

# Inside cats_overlay function
def cats_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Calculate aspect ratio of the original cat image
        num_cats = len([name for name in os.listdir() if name.startswith('cat_')])
        random_cat = f'cat_{random.randint(1, num_cats)}.png'
        cat = cv2.imread(random_cat, cv2.IMREAD_UNCHANGED)

        original_aspect_ratio = cat.shape[1] / cat.shape[0]

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate the overlay position to center the cat on the face
        overlay_x = int(center_x - 0.5 * CATS_RESIZE_FACTOR * w) - int(0.1 * CATS_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * CATS_RESIZE_FACTOR * h) - int(0.1 * CATS_RESIZE_FACTOR * w)

        # Resize the cat image
        new_width = int(CATS_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        cat_resized = cv2.resize(cat, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Ensure the overlay position is within the image boundaries
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)

        # Calculate the region of interest (ROI) for blending
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)

        # Blend cats and ROI using alpha channel
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            cat_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (cat_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - cat_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_cats.jpg"
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

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Adjusting starting position based on the center for better alignment
        overlay_x = max(0, center_x - int(0.5 * SKULL_RESIZE_FACTOR * w)) - int(0.1 * SKULL_RESIZE_FACTOR * w)
        overlay_y = max(0, center_y - int(0.5 * SKULL_RESIZE_FACTOR * h))

        # Resize Skull of Satoshi to match the width and height of the face
        new_width = int(SKULL_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)

        # Ensure the overlay image has a valid size
        if new_height <= 0 or new_width <= 0:
            continue

        skull_resized = cv2.resize(skull, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Blend Skull of Satoshi and ROI using alpha channel
        mask = skull_resized[:, :, 3] / 255.0
        mask_inv = 1.0 - mask

        roi = image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3]

        for c in range(3):
            roi[:, :, c] = (mask * skull_resized[:, :, c] + mask_inv * roi[:, :, c])

        image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] = roi

    processed_path = f"processed/{user_id}_skull_of_satoshi.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# Swap face function
def swap_face(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    heads = detect_heads(update.message.photo[-1].file_id)

    # Check if there is at least one face in the image
    if not heads:
        update.message.reply_text("No faces detected in the provided image.")
        return

    # Assuming user's own picture is a PNG file
    user_picture_id = update.message.photo[-1].file_id
    user_picture = context.bot.get_file(user_picture_id)

    if user_picture is None:
        update.message.reply_text("Error: Failed to retrieve the user picture.")
        return

    # Construct the local file path for the user's picture
    user_picture_path = f"user_{user_id}_picture.png"
    user_picture.download(user_picture_path)

    # Read the user's picture
    user_picture_image = cv2.imread(user_picture_path)

    # Process each detected face
    for (x, y, w, h) in heads:
        # Rest of the code for face swap...

    # Save the processed image in the same format as the user's picture
    _, user_picture_extension = os.path.splitext(user_picture_path)
    processed_path = f"processed/{user_id}_face_swap{user_picture_extension}"
    cv2.imwrite(processed_path, user_picture_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Send the processed image back to the user
    with open(processed_path, 'rb') as photo:
        update.message.reply_photo(photo)

    # Optionally, you can delete the user's picture file after processing
    os.remove(user_picture_path)

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
        elif option == 'cats_overlay':
            processed_path = cats_overlay(photo_path, user_id, context.bot)
        elif option == 'swap_face':
            processed_path = swap_face(photo_path, user_id, context.bot)

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

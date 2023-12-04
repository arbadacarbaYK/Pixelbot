import os
import cv2
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

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
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    # Save photo_path and user_id in context for later use in button callback
    context.user_data['photo_path'] = photo_path
    context.user_data['user_id'] = update.message.from_user.id

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()

    option = query.data
    photo_path = context.user_data['photo_path']
    user_id = context.user_data['user_id']

    if option == 'pixelate':
        processed_path = process_image(photo_path, user_id, 'pixelated', context.bot)
    elif option == 'liotta':
        processed_path = liotta_overlay(photo_path, user_id, context.bot)

    context.bot.send_photo(chat_id=update.callback_query.message.chat_id, photo=open(processed_path, 'rb'))

def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        # Extract the face
        face = image[y:y+h, x:x+w]

        # Apply pixelation directly to the face
        pixelated_face = cv2.resize(face, (0, 0), fx=0.03, fy=0.03, interpolation=cv2.INTER_NEAREST)

        # Replace the face in the original image with the pixelated version
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)

    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        # Calculate a fixed percentage (e.g., 30%) of the face size
        percentage = 0.3
        liotta_size = (int(w * percentage), int(h * percentage))

        # Resize Liotta to the calculated size
        liotta_resized = cv2.resize(liotta, liotta_size, interpolation=cv2.INTER_AREA)

        # Calculate position for the Liotta overlay
        x_pos = x - int(0.5 * (liotta_resized.shape[1] - w))
        y_pos = y - int(0.5 * (liotta_resized.shape[0] - h))

        # Make sure the Liotta overlay does not go beyond the image boundaries
        x_pos = max(0, x_pos)
        y_pos = max(0, y_pos)

        # Calculate the region of interest (ROI) for Liotta overlay
        roi_liotta = liotta_resized[max(0, -y_pos):min(liotta_resized.shape[0], image.shape[0] - y_pos),
                                    max(0, -x_pos):min(liotta_resized.shape[1], image.shape[1] - x_pos)]

        # Calculate the region of interest (ROI) in the original image
        roi_image = image[y_pos:y_pos + roi_liotta.shape[0], x_pos:x_pos + roi_liotta.shape[1]]

        # Extract alpha channel
        alpha_channel = roi_liotta[:, :, 3] / 255.0

        # Create a mask and inverse mask for Liotta image
        mask = alpha_channel
        mask_inv = 1.0 - mask

        # Blend Liotta and ROI using the mask
        for c in range(0, 3):
            roi_image[:, :, c] = (mask * roi_liotta[:, :, c] +
                                  mask_inv * roi_image[:, :, c])

        # Update the original image with the blended ROI
        image[y_pos:y_pos + roi_liotta.shape[0], x_pos:x_pos + roi_liotta.shape[1]] = roi_image

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def detect_faces(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    bounding_boxes = [face['box'] for face in faces]
    return bounding_boxes

def main() -> None:
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, pixelate_faces))
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()


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

try:
def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)

    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        print(f"Processing face at ({x}, {y}), width: {w}, height: {h}")
    # Resize Liotta to match the width of the detected face
        liotta_resized = cv2.resize(liotta, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_AREA)
    

        # Extract alpha channel
        alpha_channel = liotta_resized[:, :, 3] / 255.0

        # Create a mask and inverse mask for Liotta image
        mask = alpha_channel
        mask_inv = 1.0 - mask

        # Region of interest (ROI) in the original image
        roi = image[y:y+h, x:x+w]

        # Blend Liotta and ROI using the mask
        for c in range(0, 3):
            roi[:, :, c] = (mask * liotta_resized[:, :, c] +
                            mask_inv * roi[:, :, c])

        # Update the original image with the blended ROI
        image[y:y+h, x:x+w] = roi

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path
except Exception as e:
    print(f"Error in liotta_overlay function: {e}")


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


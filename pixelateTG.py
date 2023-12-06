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
LASER_EYE_PATH = 'lasereye.png'

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_faces(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    return faces

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
        [InlineKeyboardButton("Laser Eyes", callback_data='lasereyes')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    # Save photo_path and user_id in context for later use in button callback
    context.user_data['photo_path'] = photo_path
    context.user_data['user_id'] = update.message.from_user.id

def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)

    faces = detect_faces(image)

    def process_face(face):
        x, y, w, h = face['box']
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Region of interest (ROI) in the original image
        roi = image[y:y + h, x:x + w]

        # Resize Liotta to match the width and height of the original bounding box
        liotta_resized = cv2.resize(liotta, (w, h), interpolation=cv2.INTER_AREA)

        # Extract alpha channel
        alpha_channel = liotta_resized[:, :, 3] / 255.0

        # Resize mask arrays to match the shape of roi[:, :, c]
        mask = cv2.resize(alpha_channel, (w, h), interpolation=cv2.INTER_AREA)
        mask_inv = 1.0 - mask

        # Blend Liotta and ROI using the resized mask
        for c in range(0, 3):
            roi[:, :, c] = (mask * liotta_resized[:, :, c] +
                            mask_inv * roi[:, :, c])

        # Update the original image with the blended ROI
        image[y:y + h, x:x + w] = roi

    for face in faces:
        process_face(face)

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def replace_eyes(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    lasereye = cv2.imread(LASER_EYE_PATH, cv2.IMREAD_UNCHANGED)

    faces = detect_faces(image)

    def process_eye(eye):
        if isinstance(eye, dict) and 'box' in eye:  # Check if the output contains a 'box' key
            box = eye['box']
            print(f"Processing eye at ({box[0]}, {box[1]}), width: {box[2]}, height: {box[3]}")

            # Adjusting starting position for better alignment
            overlay_x = max(0, box[0] - int(0.25 * box[2]))
            overlay_y = max(0, box[1] - int(0.25 * box[3]))

            # Region of interest (ROI) in the original image
            roi = image[overlay_y:overlay_y + box[3], overlay_x:overlay_x + box[2]]

            # Resize lasereye to match the width and height of the ROI
            lasereye_resized = cv2.resize(lasereye, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)

            # Extract alpha channel
            alpha_channel = lasereye_resized[:, :, 3] / 255.0

            # Resize mask arrays to match the shape of roi[:, :, c]
            mask = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            mask_inv = 1.0 - mask

            # Blend lasereye and ROI using the resized mask
            for c in range(0, 3):
                roi[:, :, c] = (mask * lasereye_resized[:, :, c] +
                                mask_inv * roi[:, :, c])

            # Update the original image with the blended ROI
            image[overlay_y:overlay_y + box[3], overlay_x:overlay_x + box[2]] = roi
        else:
            print("Invalid eye structure. Skipping.")

    for face in faces:
        if isinstance(face, dict) and 'keypoints' in face:  # Check if the output contains a 'keypoints' key
            keypoints = face['keypoints']
            for eye in keypoints.values():
                process_eye(eye)
        else:
            print("Invalid face structure. Skipping.")

    processed_path = f"processed/{user_id}_lasereyes.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

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
    elif option == 'lasereyes':
        processed_path = replace_eyes(photo_path, user_id, context.bot)

    context.bot.send_photo(chat_id=update.callback_query.message.chat_id, photo=open(processed_path, 'rb'))

def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_faces(image)

    def process_face(x, y, w, h):
        # Extract the face
        face = image[y:y + h, x:x + w]

        # Apply pixelation directly to the face
        pixelated_face = cv2.resize(face, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)

        # Replace the face in the original image with the pixelated version
        image[y:y + h, x:x + w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

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

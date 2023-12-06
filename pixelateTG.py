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

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(LIOTTA_RESIZE_FACTOR * face['box'][2]), int(LIOTTA_RESIZE_FACTOR * face['box'][3])) for face in faces]
    return head_boxes

def detect_eyes(image):
    mtcnn = MTCNN()
    eyes = mtcnn.detect_faces(image)
    return eyes

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    keyboard = [
        [InlineKeyboardButton("Pixelate", callback_data='pixelate')],
        [InlineKeyboardButton("Liotta Overlay", callback_data='liotta')],
        [InlineKeyboardButton("Laser Eyes", callback_data='lasereyes')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose an option:', reply_markup=reply_markup)

    context.user_data['photo_path'] = photo_path
    context.user_data['user_id'] = update.message.from_user.id

def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_heads(image)

    def process_face(face):
        if 'box' in face:
            x, y, w, h = face['box']
            face_image = image[y:y + h, x:x + w]
            pixelated_face = cv2.resize(face_image, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)
            image[y:y + h, x:x + w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            print("Invalid face structure. Skipping.")

    futures = [executor.submit(process_face, face) for face in faces]
    wait(futures)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)
    heads = detect_heads(image)

    def process_face(x, y, w, h):
        overlay_x = max(0, x - int(0.25 * w))
        overlay_y = max(0, y - int(0.25 * h))
        roi = image[overlay_y:overlay_y + h, overlay_x:overlay_x + w]
        liotta_resized = cv2.resize(liotta, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
        alpha_channel = liotta_resized[:, :, 3] / 255.0
        mask = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
        mask_inv = 1.0 - mask
        for c in range(0, 3):
            roi[:, :, c] = (mask * liotta_resized[:, :, c] + mask_inv * roi[:, :, c])
        image[overlay_y:overlay_y + h, overlay_x:overlay_x + w] = roi

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in heads]
    wait(futures)

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def replace_eyes(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    eyes = detect_eyes(image)

    def process_eye(eye):
        if 'box' in eye:
            eye_x, eye_y, eye_w, eye_h = eye['box']
            print(f"Eye detected at ({eye_x}, {eye_y}), width: {eye_w}, height: {eye_h}")
            eye_image = image[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
            print(f"Eye image shape: {eye_image.shape}")
            
            lasereye_resized = cv2.resize(lasereye, (eye_image.shape[1], eye_image.shape[0]), interpolation=cv2.INTER_AREA)
            alpha_channel = lasereye_resized[:, :, 3] / 255.0
            mask = cv2.resize(alpha_channel, (eye_image.shape[1], eye_image.shape[0]), interpolation=cv2.INTER_AREA)
            mask_inv = 1.0 - mask
            for c in range(0, 3):
                eye_image[:, :, c] = (mask * lasereye_resized[:, :, c] + mask_inv * eye_image[:, :, c])
            image[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w] = eye_image
        else:
            print("Invalid eye structure. Skipping.")

    futures = [executor.submit(process_eye, eye) for eye in eyes]
    wait(futures)

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
    lasereye = cv2.imread('lasereye.png', cv2.IMREAD_UNCHANGED)
    main()

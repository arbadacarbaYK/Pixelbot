import os
import cv2
import random
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler, CallbackQueryHandler
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN
from uuid import uuid4

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5
PIXELATION_FACTOR = 0.04
LIOTTA_RESIZE_FACTOR = 1.5
SKULL_RESIZE_FACTOR = 1.9
CATS_RESIZE_FACTOR = 1.5
PEPE_RESIZE_FACTOR = 1.5
CHAD_RESIZE_FACTOR = 1.7
CLOWNS_RESIZE_FACTOR = 1.7

executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(LIOTTA_RESIZE_FACTOR * face['box'][2]), int(LIOTTA_RESIZE_FACTOR * face['box'][3])) for face in faces]
    return head_boxes

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    session_id = str(uuid4())  # Generate a unique session ID
    context.user_data[session_id] = {'state': 'waiting_for_photo'}

    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_name = file.file_path.split('/')[-1]
    photo_path = f"downloads/{file_name}"
    file.download(photo_path)

    # Check if any faces are detected
    image = cv2.imread(photo_path)
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    if not faces:
        # No faces detected, do nothing
        return

    keyboard = [
        [InlineKeyboardButton("Pixel ⚔️", callback_data=f'pixelate_{session_id}')],
        [InlineKeyboardButton("Liotta 🤣", callback_data=f'liotta_{session_id}')],
        [InlineKeyboardButton("Skull of Satoshi ☠️☠", callback_data=f'skull_of_satoshi_{session_id}')],
        [InlineKeyboardButton("Cats 🐈‍⬛", callback_data=f'cats_overlay_{session_id}')],
        [InlineKeyboardButton("Pepe 🐸", callback_data=f'pepe_overlay_{session_id}')],
        [InlineKeyboardButton("Chad 🏆", callback_data=f'chad_overlay_{session_id}')],
        [InlineKeyboardButton("Clowns 🤡", callback_data=f'clowns_overlay_{session_id}')],
        [InlineKeyboardButton("CANCEL", callback_data=f'cancel_{session_id}')],  # Add Cancel button
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Press until happy', reply_markup=reply_markup)

    context.user_data[session_id]['photo_path'] = photo_path
    context.user_data[session_id]['user_id'] = update.message.from_user.id
    # Delete the original picture from the chat
    update.message.delete()


def process_image(photo_path, user_id, file_id, bot):
    image = cv2.imread(photo_path)
    faces = detect_heads(image)

    def process_face(x, y, w, h):
        face = image[y:y+h, x:x+w]
        pixelated_face = cv2.resize(face, (0, 0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR, interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in faces]
    wait(futures)

    processed_path = f"processed/{user_id}_{file_id}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        original_aspect_ratio = liotta.shape[1] / liotta.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * LIOTTA_RESIZE_FACTOR * w) - int(0.1 * LIOTTA_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * LIOTTA_RESIZE_FACTOR * h)
        new_width = int(LIOTTA_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        liotta_resized = cv2.resize(liotta, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] = (
            liotta_resized[:, :, :3] * (liotta_resized[:, :, 3:] / 255.0) +
            image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] *
            (1.0 - liotta_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def cats_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        num_cats = len([name for name in os.listdir() if name.startswith('cat_')])
        random_cat = f'cat_{random.randint(1, num_cats)}.png'
        cat = cv2.imread(random_cat, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = cat.shape[1] / cat.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * CATS_RESIZE_FACTOR * w) - int(0.1 * CATS_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * CATS_RESIZE_FACTOR * h) - int(0.1 * CATS_RESIZE_FACTOR * w)
        new_width = int(CATS_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        cat_resized = cv2.resize(cat, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
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

def skull_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    skull = cv2.imread('skullofsatoshi.png', cv2.IMREAD_UNCHANGED)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        original_aspect_ratio = skull.shape[1] / skull.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = max(0, center_x - int(0.5 * SKULL_RESIZE_FACTOR * w)) - int(0.1 * SKULL_RESIZE_FACTOR * w)
        overlay_y = max(0, center_y - int(0.5 * SKULL_RESIZE_FACTOR * h))
        new_width = int(SKULL_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        if new_height <= 0 or new_width <= 0:
            continue
        skull_resized = cv2.resize(skull, (new_width, new_height), interpolation=cv2.INTER_AREA)
        mask = skull_resized[:, :, 3] / 255.0
        mask_inv = 1.0 - mask
        roi = image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3]
        for c in range(3):
            roi[:, :, c] = (mask * skull_resized[:, :, c] + mask_inv * roi[:, :, c])
        image[overlay_y:overlay_y + new_height, overlay_x:overlay_x + new_width, :3] = roi

    processed_path = f"processed/{user_id}_skull_of_satoshi.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def pepe_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        num_pepes = len([name for name in os.listdir() if name.startswith('pepe_')])
        random_pepe = f'pepe_{random.randint(1, num_pepes)}.png'
        pepe = cv2.imread(random_pepe, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = pepe.shape[1] / pepe.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * PEPE_RESIZE_FACTOR * w) - int(0.1 * PEPE_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * PEPE_RESIZE_FACTOR * h) - int(0.1 * PEPE_RESIZE_FACTOR * w)
        new_width = int(PEPE_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        pepe_resized = cv2.resize(pepe, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            pepe_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (pepe_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - pepe_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_pepe.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def chad_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        num_chads = len([name for name in os.listdir() if name.startswith('chad_')])
        random_chad = f'chad_{random.randint(1, num_chads)}.png'
        chad = cv2.imread(random_chad, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = chad.shape[1] / chad.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * CHAD_RESIZE_FACTOR * w) - int(0.1 * CHAD_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * CHAD_RESIZE_FACTOR * h) - int(0.1 * CHAD_RESIZE_FACTOR * w)
        new_width = int(CHAD_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        chad_resized = cv2.resize(chad, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            chad_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (chad_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - chad_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_chad.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def clowns_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        num_clowns = len([name for name in os.listdir() if name.startswith('clown_')])
        random_clown = f'clown_{random.randint(1, num_clowns)}.png'
        clown = cv2.imread(random_clown, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = clown.shape[1] / clown.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * CLOWNS_RESIZE_FACTOR * w) - int(0.1 * CLOWNS_RESIZE_FACTOR * w)
        overlay_y = int(center_y - 0.5 * CLOWNS_RESIZE_FACTOR * h) - int(0.1 * CLOWNS_RESIZE_FACTOR * w)
        new_width = int(CLOWNS_RESIZE_FACTOR * w)
        new_height = int(new_width / original_aspect_ratio)
        clown_resized = cv2.resize(clown, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            clown_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (clown_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - clown_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_clowns.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

    

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    user_data = context.user_data.get(session_id)

    if user_data and user_data['state'] == 'waiting_for_photo':
        photo_path = user_data.get('photo_path')
        user_id = user_data.get('user_id')

        if query.data.startswith('cancel'):
            del context.user_data[session_id]  # Delete session data
            query.message.delete()  # Remove the message containing the keyboard
            return

        processed_path = None

        if query.data.startswith('pixelate'):
            processed_path = process_image(photo_path, user_id, query.id, context.bot)
        elif query.data.startswith('liotta'):
            processed_path = liotta_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('cats_overlay'):
            processed_path = cats_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('skull_of_satoshi'):
            processed_path = skull_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('pepe_overlay'):
            processed_path = pepe_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('chad_overlay'):
            processed_path = chad_overlay(photo_path, user_id, context.bot)
        elif query.data.startswith('clowns_overlay'):
            processed_path = clowns_overlay(photo_path, user_id, context.bot)

        if processed_path:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))
            # Keep the keyboard visible by editing the original message's markup
            query.edit_message_reply_markup(reply_markup=query.message.reply_markup)


def main() -> None:
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

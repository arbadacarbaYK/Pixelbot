import os
import cv2
import random
import imageio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor, wait
from mtcnn.mtcnn import MTCNN
from uuid import uuid4

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 1.5  # Common resize factor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me a picture or a GIF, and I will pixelate faces in it!')

def detect_heads(image):
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image)
    head_boxes = [(face['box'][0], face['box'][1], int(RESIZE_FACTOR * face['box'][2]), int(RESIZE_FACTOR * face['box'][3])) for face in faces]
    return head_boxes

def overlay(photo_path, user_id, overlay_type, resize_factor, bot):
    image = cv2.imread(photo_path)
    heads = detect_heads(image)

    for (x, y, w, h) in heads:
        overlay_files = [name for name in os.listdir() if name.startswith(f'{overlay_type}_')]
        if not overlay_files:
            continue
        random_overlay = random.choice(overlay_files)
        overlay_image = cv2.imread(random_overlay, cv2.IMREAD_UNCHANGED)
        original_aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]
        center_x = x + w // 2
        center_y = y + h // 2
        overlay_x = int(center_x - 0.5 * resize_factor * w) - int(0.1 * resize_factor * w)
        overlay_y = int(center_y - 0.5 * resize_factor * h) - int(0.1 * resize_factor * w)
        new_width = int(resize_factor * w)
        new_height = int(new_width / original_aspect_ratio)
        overlay_image_resized = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        overlay_x = max(0, overlay_x)
        overlay_y = max(0, overlay_y)
        roi_start_x = max(0, overlay_x)
        roi_start_y = max(0, overlay_y)
        roi_end_x = min(image.shape[1], overlay_x + new_width)
        roi_end_y = min(image.shape[0], overlay_y + new_height)
        image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] = (
            overlay_image_resized[
                roi_start_y - overlay_y : roi_end_y - overlay_y,
                roi_start_x - overlay_x : roi_end_x - overlay_x,
                :3
            ] * (overlay_image_resized[:, :, 3:] / 255.0) +
            image[roi_start_y:roi_end_y, roi_start_x:roi_end_x, :3] *
            (1.0 - overlay_image_resized[:, :, 3:] / 255.0)
        )

    processed_path = f"processed/{user_id}_{overlay_type}.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# Overlay functions
def liotta_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'liotta', RESIZE_FACTOR, bot)

def skull_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'skullofsatoshi', RESIZE_FACTOR, bot)

def pepe_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'pepe', RESIZE_FACTOR, bot)

def chad_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'chad', RESIZE_FACTOR, bot)

def cats_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'cat', RESIZE_FACTOR, bot)

def clowns_overlay(photo_path, user_id, bot):
    return overlay(photo_path, user_id, 'clown', RESIZE_FACTOR, bot)

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

def process_gif(gif_path, session_id, user_id, bot):
    # Extract frames from GIF and process each frame
    frames = imageio.mimread(gif_path)
    processed_frames = [process_image(frame, user_id, session_id, bot) for frame in frames]
    # Reconstruct GIF
    processed_gif_path = f"processed/{user_id}_{session_id}.gif"
    imageio.mimsave(processed_gif_path, processed_frames)
    return processed_gif_path

def pixelate_faces(update: Update, context: CallbackContext) -> None:
    chat_type = update.effective_chat.type

    if chat_type == "private":
        session_id = str(uuid4())  # Generate a unique session ID
        context.user_data[session_id] = {'state': 'waiting_for_photo'}

        if update.message.photo:  # If the message contains a photo
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
                update.message.reply_text('No faces detected in the image.')
                return

            keyboard = [
                [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
                 InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_{session_id}'),
                 InlineKeyboardButton("☠️ Skull", callback_data=f'skull_of_satoshi_overlay_{session_id}')],
                [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}'),                 
                 InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),                 
                 InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
                [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),                 
                 InlineKeyboardButton("CANCEL", callback_data=f'cancel_{session_id}')],  # Add Cancel button
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            context.user_data[session_id]['photo_path'] = photo_path
            context.user_data[session_id]['user_id'] = update.message.from_user.id

            update.message.reply_text('Press buttons until happy', reply_markup=reply_markup)

            # Delete the original picture from the chat
            update.message.delete()

        elif update.message.document and update.message.document.mime_type == 'image/gif':  # If the message contains a GIF
            file_id = update.message.document.file_id
            file = context.bot.get_file(file_id)
            file_name = file.file_path.split('/')[-1]
            gif_path = f"downloads/{file_name}"
            file.download(gif_path)

            # Process GIF
            processed_gif_path = process_gif(gif_path, session_id, str(uuid4()), context.bot)

            # Send the processed GIF
            context.bot.send_animation(chat_id=update.message.chat_id, animation=open(processed_gif_path, 'rb'))

        else:
            update.message.reply_text('Please send either a photo or a GIF.')

def pixelate_command(update: Update, context: CallbackContext) -> None:
    if update.message.reply_to_message and update.message.reply_to_message.photo:
        session_id = str(uuid4())  # Generate a unique session ID
        context.chat_data[session_id] = {'state': 'waiting_for_photo'}

        file_id = update.message.reply_to_message.photo[-1].file_id
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
            update.message.reply_text('No faces detected in the image.')
            return

        keyboard = [
            [InlineKeyboardButton("🤡 Clowns", callback_data=f'clowns_overlay_{session_id}'),
             InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_{session_id}'),
             InlineKeyboardButton("☠️ Skull", callback_data=f'skull_of_satoshi_{session_id}')],
            [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cats_overlay_{session_id}'),
             InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_overlay_{session_id}'),
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_overlay_{session_id}')],
            [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}'),
             InlineKeyboardButton("CANCEL", callback_data=f'cancel_{session_id}')],  # Add Cancel button
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        context.chat_data[session_id]['photo_path'] = photo_path
        context.chat_data[session_id]['chat_id'] = update.message.chat.id

        update.message.reply_text('Choose an overlay or pixelate the faces:', reply_markup=reply_markup)
    else:
        update.message.reply_text('This only works as a reply to a picture.')

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    session_id = query.data.split('_')[-1]
    chat_data = context.chat_data.get(session_id)

    if chat_data and chat_data['state'] == 'waiting_for_photo':
        photo_path = chat_data.get('photo_path')
        chat_id = chat_data.get('chat_id')

        if query.data.startswith('cancel'):
            del context.chat_data[session_id]  # Delete session data
            query.message.delete()  # Remove the message containing the keyboard
            return

        processed_path = None

        if query.data.startswith('pixelate'):
            processed_path = process_image(photo_path, chat_id, query.id, context.bot)
        elif query.data.startswith('liotta'):
            processed_path = liotta_overlay(photo_path, chat_id, context.bot)
        elif query.data.startswith('cats_overlay'):
            processed_path = cats_overlay(photo_path, chat_id, context.bot)
        elif query.data.startswith('skull_of_satoshi'):
            processed_path = skull_overlay(photo_path, chat_id, context.bot)
        elif query.data.startswith('pepe_overlay'):
            processed_path = pepe_overlay(photo_path, chat_id, context.bot)
        elif query.data.startswith('chad_overlay'):
            processed_path = chad_overlay(photo_path, chat_id, context.bot)
        elif query.data.startswith('clowns_overlay'):
            processed_path = clowns_overlay(photo_path, chat_id, context.bot)

        if processed_path:
            context.bot.send_photo(chat_id=query.message.chat_id, photo=open(processed_path, 'rb'))

def main() -> None:
    updater = Updater(TOKEN)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("pixel", pixelate_command))
    dispatcher.add_handler(MessageHandler(Filters.photo & Filters.private, pixelate_faces))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

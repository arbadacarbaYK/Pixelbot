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

# ... (other functions)

def liotta_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    liotta = cv2.imread('liotta.png', cv2.IMREAD_UNCHANGED)

    heads = detect_heads(image)

    def process_face(x, y, w, h):
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Similar processing logic as in liotta_overlay function
        overlay_x = max(0, x - int(0.25 * w))
        overlay_y = max(0, y - int(0.25 * h))

        roi = image[overlay_y:overlay_y + h, overlay_x:overlay_x + w]

        # Resize Skull of Satoshi with aspect ratio maintained
        aspect_ratio = liotta.shape[1] / liotta.shape[0]
        new_height = int(w / aspect_ratio)
        liotta_resized = cv2.resize(liotta, (w, new_height), interpolation=cv2.INTER_AREA)

        alpha_channel = liotta_resized[:, :, 3] / 255.0

        mask = cv2.resize(alpha_channel, (w, new_height), interpolation=cv2.INTER_AREA)
        mask_inv = 1.0 - mask

        for c in range(0, 3):
            roi[:, :, c] = (mask * liotta_resized[:, :, c] +
                            mask_inv * roi[:, :, c])

        image[overlay_y:overlay_y + h, overlay_x:overlay_x + w] = roi

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in heads]
    wait(futures)

    processed_path = f"processed/{user_id}_liotta.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

def skull_overlay(photo_path, user_id, bot):
    image = cv2.imread(photo_path)
    skull = cv2.imread('skullofsatoshi.png', cv2.IMREAD_UNCHANGED)

    heads = detect_heads(image)

    def process_face(x, y, w, h):
        print(f"Processing head at ({x}, {y}), width: {w}, height: {h}")

        # Similar processing logic as in liotta_overlay function
        overlay_x = max(0, x - int(0.25 * w))
        overlay_y = max(0, y - int(0.25 * h))

        roi = image[overlay_y:overlay_y + h, overlay_x:overlay_x + w]

        # Resize Skull of Satoshi with aspect ratio maintained
        aspect_ratio = skull.shape[1] / skull.shape[0]
        new_height = int(w / aspect_ratio)
        skull_resized = cv2.resize(skull, (w, new_height), interpolation=cv2.INTER_AREA)

        alpha_channel = skull_resized[:, :, 3] / 255.0

        mask = cv2.resize(alpha_channel, (w, new_height), interpolation=cv2.INTER_AREA)
        mask_inv = 1.0 - mask

        for c in range(0, 3):
            roi[:, :, c] = (mask * skull_resized[:, :, c] +
                            mask_inv * roi[:, :, c])

        image[overlay_y:overlay_y + h, overlay_x:overlay_x + w] = roi

    futures = [executor.submit(process_face, x, y, w, h) for (x, y, w, h) in heads]
    wait(futures)

    processed_path = f"processed/{user_id}_skull_of_satoshi.jpg"
    cv2.imwrite(processed_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return processed_path

# ... (rest of the code)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
    main()

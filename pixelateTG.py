import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
from dotenv import load_dotenv
import cv2
import random
import imageio
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor, wait
from uuid import uuid4
import time
import logging
import traceback
import socket
import urllib3
from telegram.utils.request import Request
import psutil
import glob
import logging.handlers
from gif_processor import process_telegram_gif
from constants import PIXELATION_FACTOR, detect_heads
from gif_processor import GifProcessor
# Configure DNS settings
socket.setdefaulttimeout(20)
urllib3.disable_warnings()
        
# Configure logging
logging.basicConfig(
    filename='pixelbot_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('telegram').setLevel(logging.INFO)
logging.getLogger('telegram.ext.dispatcher').setLevel(logging.INFO)
logging.getLogger('telegram.bot').setLevel(logging.INFO)

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MAX_THREADS = 15
RESIZE_FACTOR = 2.0
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

# Cache for overlay files
overlay_cache = {}

overlay_image_cache = {}

overlay_adjustments = {
    'clown': {'x_offset': -0.15, 'y_offset': -0.25, 'size_factor': 1.66},
    'liotta': {'x_offset': -0.12, 'y_offset': -0.2, 'size_factor': 1.5},
    'skull': {'x_offset': -0.25, 'y_offset': -0.5, 'size_factor': 1.65},
    'cat': {'x_offset': -0.15, 'y_offset': -0.45, 'size_factor': 1.5}, 
    'pepe': {'x_offset': -0.05, 'y_offset': -0.2, 'size_factor': 1.4},
    'chad': {'x_offset': -0.15, 'y_offset': -0.15, 'size_factor': 1.6}  
}

face_detection_cache = {}

rotated_overlay_cache = {}

# Dictionary to store active sessions
active_sessions = {}

def verify_permissions():
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Verifying permissions for directories...")
    for directory in ['processed', 'downloads']:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Test write permissions
            test_file = os.path.join(directory, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            logger.info(f"Verified write permissions for {directory}")
            
        except Exception as e:
            logger.error(f"Permission error for directory {directory}: {str(e)}")
            return False
    return True

def get_file_path(directory, id_prefix, session_id, suffix):
    """Generate a file path with the given parameters"""
    # Make sure we don't add .jpg to GIF files
    if suffix.endswith('.gif.jpg'):
        suffix = suffix.replace('.gif.jpg', '.gif')
    elif suffix.endswith('.gif'):
        # Keep as is
        pass
    elif not suffix.endswith('.jpg'):
        suffix = f"{suffix}.jpg"
        
    return os.path.join(directory, f"{id_prefix}_{session_id}_{suffix}")

def cleanup_temp_files():
    """Clean up temporary files in downloads and processed directories"""
    for directory in ['downloads', 'processed']:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                os.remove(os.path.join(directory, f))
            logger.info(f"Cleaned up {directory} directory")

def start(update: Update, context: CallbackContext) -> None:
    """Handle /start command"""
    update.message.reply_text("Send me a photo to get started!")

def get_overlay_files(overlay_type):
    """Get all overlay files for a specific type"""
    # Look for overlays directly in the root directory
    overlay_files = glob.glob(f"{overlay_type}_*.png")
    
    if not overlay_files:
        logger.error(f"No overlay files found matching pattern: {overlay_type}_*.png")
        logger.error(f"Searched in directory: {os.getcwd()}")
    
    return overlay_files

def get_cached_overlay(overlay_path):
    if overlay_path in overlay_image_cache:
        return overlay_image_cache[overlay_path].copy()
    
    overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay_img is not None:
        overlay_image_cache[overlay_path] = overlay_img
        logger.debug(f"Cached overlay image: {overlay_path}")
    return overlay_img.copy() if overlay_img is not None else None

def get_id_prefix(update):
    """Generate a consistent ID prefix for a user"""
    return f"user_{update.effective_user.id}"

def process_image(photo_path, output_path, effect_type, selected_overlay=None, faces=None):
    try:
        image = cv2.imread(photo_path)
        if image is None:
            logger.error(f"Failed to read image: {photo_path}")
            return False
            
        height, width = image.shape[:2]
        
        if faces is None:
            faces = detect_heads(image)
        
        output = image.copy()
        
        # Ensure an overlay is selected if effect_type is not 'pixelate'
        if effect_type != 'pixelate' and not selected_overlay:
            overlay_files = glob.glob(f"{effect_type}_*.png")
            if overlay_files:
                selected_overlay = random.choice(overlay_files)
                logger.info(f"Selected overlay: {selected_overlay}")
            else:
                logger.error("No overlay files found for the selected effect type.")
                return False
        
        for face in faces:
            x, y, w, h = face['rect']
            angle = face['angle']
            
            if effect_type == 'pixelate':
                face_roi = image[y:y+h, x:x+w]
                small = cv2.resize(face_roi, (0, 0), fx=1.0/PIXELATION_FACTOR, fy=1.0/PIXELATION_FACTOR)
                pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                output[y:y+h, x:x+w] = pixelated_face
            else:
                overlay_img = cv2.imread(selected_overlay, cv2.IMREAD_UNCHANGED)
                if overlay_img is None:
                    logger.error(f"Failed to read overlay: {selected_overlay}")
                    continue
                
                adjust = overlay_adjustments.get(effect_type, {'x_offset': 0, 'y_offset': 0, 'size_factor': 1.0})
                
                overlay_width = int(w * adjust['size_factor'])
                overlay_height = int(h * adjust['size_factor'])
                x_pos = int(x + w * adjust['x_offset'])
                y_pos = int(y + h * adjust['y_offset'])
                
                overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))
                
                center = (overlay_width // 2, overlay_height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_overlay = cv2.warpAffine(overlay_resized, M, (overlay_width, overlay_height))
                
                roi_x = max(0, x_pos)
                roi_y = max(0, y_pos)
                roi_w = min(overlay_width, width - roi_x)
                roi_h = min(overlay_height, height - roi_y)
                
                if roi_w <= 0 or roi_h <= 0:
                    continue
                
                roi = output[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                overlay_roi = rotated_overlay[0:roi_h, 0:roi_w]
                
                if overlay_roi.shape[2] == 4:
                    alpha = overlay_roi[:, :, 3] / 255.0
                    for c in range(3):
                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_roi[:, :, c] * alpha
                else:
                    output[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = overlay_roi
        
        cv2.imwrite(output_path, output)
        return True
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def apply_overlay_to_image(image, overlay, x, y):
    """Apply an overlay to an image at the specified position"""
    try:
        h, w = overlay.shape[:2]
        img_h, img_w = image.shape[:2]
        
        # Calculate the region where the overlay will be placed
        roi_x = max(0, x)
        roi_y = max(0, y)
        roi_w = min(w, img_w - roi_x)
        roi_h = min(h, img_h - roi_y)
        
        if roi_w <= 0 or roi_h <= 0:
            return
        
        # Get the region of interest in the image
        roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Get the region of the overlay to use
        overlay_roi = overlay[0:roi_h, 0:roi_w]
        
        # Apply the overlay with alpha blending if it has an alpha channel
        if overlay_roi.shape[2] == 4:  # With alpha channel
            alpha = overlay_roi[:, :, 3] / 255.0
            for c in range(3):  # Apply for each color channel
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_roi[:, :, c] * alpha
        else:  # No alpha channel
            image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = overlay_roi
            
    except Exception as e:
        logger.error(f"Error applying overlay: {str(e)}")
        logger.error(traceback.format_exc())

def get_random_overlay_file(overlay_type):
    try:
        overlay_files = get_overlay_files(overlay_type)
        if not overlay_files:
            return None
        return random.choice(overlay_files)
    except Exception as e:
        logger.error(f"Error in get_random_overlay_file: {str(e)}")
        return None

def overlay(input_path, overlay_type, output_path, faces=None):
    try:
        logger.debug(f"Starting overlay process for {overlay_type}")
        
        # Read input image
        image = cv2.imread(input_path)
        if image is None:
            logger.error(f"Failed to read input image: {input_path}")
            return False
            
        # Only detect faces if not provided
        if faces is None:
            faces = detect_heads(image)
            
        logger.debug(f"Processing {len(faces)} faces")
        
        if len(faces) == 0:
            logger.error("No faces detected in image")
            return False

        overlay_files = get_overlay_files(overlay_type)
        if not overlay_files:
            logger.error(f"No overlay files found for type: {overlay_type}")
            return False
            
        for face in faces:
            overlay_file = random.choice(overlay_files)
            overlay_path = os.path.join(os.getcwd(), overlay_file)
            overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
            
            if overlay_img is None:
                logger.error(f"Failed to read overlay: {overlay_path}")
                continue
                
            rect = face['rect']
            angle = face['angle']
            x, y, w, h = rect
            
            adjust = overlay_adjustments.get(overlay_type, {
                'x_offset': 0, 'y_offset': 0, 'size_factor': 1.0
            })
            
            # Calculate size and position
            overlay_width = int(w * adjust['size_factor'])
            overlay_height = int(h * adjust['size_factor'])
            x_pos = int(x + w * adjust['x_offset'])
            y_pos = int(y + h * adjust['y_offset'])
            
            # Resize overlay
            overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))
            
            # Create rotation matrix around center of overlay
            center = (x_pos + overlay_width//2, y_pos + overlay_height//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Create a larger canvas for rotation to prevent cropping
            canvas = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            canvas[y_pos:y_pos+overlay_height, x_pos:x_pos+overlay_width] = overlay_resized
            
            # Apply rotation
            rotated_canvas = cv2.warpAffine(canvas, M, (image.shape[1], image.shape[0]))
            
            # Blend with original image
            alpha = rotated_canvas[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            overlay_rgb = rotated_canvas[:, :, :3]
            
            image = image * (1 - alpha) + overlay_rgb * alpha
            
        image = image.astype(np.uint8)
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        logger.error(f"Error in overlay function: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Overlay functions
def clown_overlay(photo_path, output_path):
    logger.info("Starting clowns overlay")
    return process_image(photo_path, output_path, 'clown')

def liotta_overlay(photo_path, output_path):
    logger.info("Starting liotta overlay")
    return process_image(photo_path, output_path, 'liotta')

def skull_overlay(photo_path, output_path):
    logger.info("Starting skull overlay")
    return process_image(photo_path, output_path, 'skull')

def cat_overlay(photo_path, output_path):
    logger.info("Starting cats overlay")
    return process_image(photo_path, output_path, 'cat')

def pepe_overlay(photo_path, output_path):
    logger.info("Starting pepe overlay")
    return process_image(photo_path, output_path, 'pepe')

def chad_overlay(photo_path, output_path):
    logger.info("Starting chad overlay")
    return process_image(photo_path, output_path, 'chad')

def process_gif(gif_path, session_id, id_prefix, bot, action):
    try:
        # Get output path first - make sure it has .gif extension
        processed_gif_path = get_file_path('processed', id_prefix, session_id, f'{action}.gif')
        
        # Use the existing process_telegram_gif function
        success = process_telegram_gif(
            gif_path,
            processed_gif_path,
            process_image,  # This is the same function used for photos
            action=action   # Pass the action (pixelate/overlay type)
        )
        
        if success and os.path.exists(processed_gif_path):
            # Verify it's actually a GIF file
            if os.path.getsize(processed_gif_path) > 0:
                logger.info(f"Successfully processed GIF: {processed_gif_path}")
                return processed_gif_path
            else:
                logger.error(f"Processed GIF file is empty: {processed_gif_path}")
        
        logger.error("Failed to process GIF")
        return None
            
    except Exception as e:
        logger.error(f"Error in process_gif: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def handle_message(update: Update, context: CallbackContext, photo=None) -> None:
    try:
        logger.debug("Function called")
        message = photo if photo else update.message
        chat_id = message.chat_id
        
        session_id = str(uuid4())
        id_prefix = f"user_{chat_id}"

        session_data = {
            'chat_id': chat_id,
            'id_prefix': id_prefix,
            'session_id': session_id,
            'is_gif': False
        }

        # Check if this is a GIF
        is_gif = False
        file_id = None
        
        # Handle photos
        if message.photo:
            file_id = message.photo[-1].file_id
        # Handle GIFs as documents
        elif message.document and message.document.mime_type == 'image/gif':
            file_id = message.document.file_id
            is_gif = True
        # Handle GIFs as animations
        elif message.animation:
            file_id = message.animation.file_id
            is_gif = True
            
        if not file_id:
            return
            
        # Download the file
        file_extension = 'gif.jpg' if is_gif else 'original.jpg'
        file_path = get_file_path('downloads', id_prefix, session_id, file_extension)
        
        file = context.bot.get_file(file_id)
        file.download(file_path)
        logger.info(f"Downloaded {'GIF' if is_gif else 'photo'} to {file_path}")
        
        # Store session data
        session_data['input_path'] = file_path
        session_data['is_gif'] = is_gif
        context.user_data[session_id] = session_data
        logger.debug(f"Created new session: {session_id}")
        
        # Create keyboard with effect options
        keyboard = [
            [
                InlineKeyboardButton("🧩 Pixelate", callback_data=f"pixelate:{session_id}")
            ],
            [
                InlineKeyboardButton("🤡 Clown", callback_data=f"clown:{session_id}"),
                InlineKeyboardButton("😎 Liotta", callback_data=f"liotta:{session_id}"),
                InlineKeyboardButton("💀 Skull", callback_data=f"skull:{session_id}")
            ],
            [
                InlineKeyboardButton("🐱 Cat", callback_data=f"cat:{session_id}"),
                InlineKeyboardButton("🐸 Pepe", callback_data=f"pepe:{session_id}"),
                InlineKeyboardButton("👨 Chad", callback_data=f"chad:{session_id}")
            ],
            [
                InlineKeyboardButton("❌ Close", callback_data=f"close:{session_id}")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send the keyboard
        message.reply_text('Choose an effect:', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(traceback.format_exc())

def cleanup_before_start(bot):
    """Clean up before starting the bot"""
    logger.info("Cleanup before start completed")
    return True

def error_handler(update: Update, context: CallbackContext) -> None:
    """Log Errors caused by Updates."""
    logger.error(f'Update "{update}" caused error "{context.error}"')

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    
    try:
        # Parse the callback data
        if ':' in query.data:
            action, session_id = query.data.split(':')
        else:
            # Handle the old format for backward compatibility
            for effect in ['pixelate', 'clown', 'liotta', 'skull', 'cat', 'pepe', 'chad', 'close']:
                if query.data.startswith(f"{effect}_"):
                    action = effect
                    session_id = query.data[len(effect)+1:]
                    break
            else:
                query.answer("Invalid action")
                return
        
        # Handle close action separately
        if action == 'close':
            # Simply delete the message with the keyboard
            query.delete_message()
            return
            
        # Get session data
        session_data = context.user_data.get(session_id)
        if not session_data:
            logger.error(f"No session data found for {action}")
            query.answer("Session expired, please try again")
            return
        
        # Process the image with the selected effect
        chat_id = session_data.get('chat_id')
        input_path = session_data.get('input_path')
        is_gif = session_data.get('is_gif', False)
        id_prefix = session_data.get('id_prefix')
        
        if not input_path or not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            query.answer("Original file not found, please send a new one")
            return
            
        # Show processing message
        query.answer(f"Processing with {action} effect...")
        query.edit_message_text(f"Processing with {action} effect...")
        
        # Process the image
        if is_gif:
            # Process GIF
            output_path = process_gif(input_path, session_id, id_prefix, context.bot, action)
            if output_path and os.path.exists(output_path):
                # Send the processed GIF without caption
                with open(output_path, 'rb') as f:
                    context.bot.send_animation(
                        chat_id=chat_id,
                        animation=f
                    )
                # Clean up
                os.remove(output_path)
                logger.debug(f"Cleaned up {output_path}")
            else:
                context.bot.send_message(
                    chat_id=chat_id,
                    text="Failed to process GIF. Please try again."
                )
        else:
            # Process photo
            output_path = get_file_path('processed', id_prefix, session_id, action)
            if process_image(input_path, output_path, action):
                # Send the processed photo without caption
                with open(output_path, 'rb') as f:
                    context.bot.send_photo(
                        chat_id=chat_id,
                        photo=f
                    )
                # Clean up
                os.remove(output_path)
            else:
                context.bot.send_message(
                    chat_id=chat_id,
                    text="Failed to process image. Please try again."
                )
                
        # Restore the keyboard after processing
        keyboard = [
            [
                InlineKeyboardButton("🧩 Pixelate", callback_data=f"pixelate:{session_id}")
            ],
            [
                InlineKeyboardButton("🤡 Clown", callback_data=f"clown:{session_id}"),
                InlineKeyboardButton("😎 Liotta", callback_data=f"liotta:{session_id}"),
                InlineKeyboardButton("💀 Skull", callback_data=f"skull:{session_id}")
            ],
            [
                InlineKeyboardButton("🐱 Cat", callback_data=f"cat:{session_id}"),
                InlineKeyboardButton("🐸 Pepe", callback_data=f"pepe:{session_id}"),
                InlineKeyboardButton("👨 Chad", callback_data=f"chad:{session_id}")
            ],
            [
                InlineKeyboardButton("❌ Close", callback_data=f"close:{session_id}")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        query.edit_message_text('Choose an effect:', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in button_callback: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            query.answer("An error occurred")
        except:
            pass

def get_last_update_id() -> int:
    try:
        with open('pixelbot_last_update.txt', 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def save_last_update_id(update_id: int) -> None:
    with open('pixelbot_last_update.txt', 'w') as f:
        f.write(str(update_id))

def cleanup_old_files():
    """Cleanup files older than 24 hours"""
    current_time = time.time()
    for directory in ['processed', 'downloads']:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                filepath = os.path.join(directory, f)
                if os.path.getmtime(filepath) < (current_time - 86400):  # 24 hours
                    try:
                        os.remove(filepath)
                        logger.debug(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to remove {filepath}: {e}")

def get_rotated_overlay(overlay_img, angle, size):
    """Cache and return rotated overlays"""
    cache_key = f"{id(overlay_img)}_{angle}_{size}"
    if cache_key in rotated_overlay_cache:
        return rotated_overlay_cache[cache_key]
        
    rotated = cv2.warpAffine(
        overlay_img,
        cv2.getRotationMatrix2D((size[0]//2, size[1]//2), angle, 1.0),
        size
    )
    rotated_overlay_cache[cache_key] = rotated
    return rotated

def photo_command(update: Update, context: CallbackContext) -> None:
    # Check if this is a reply to a photo
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        return
    # Process the replied-to photo
    handle_message(update, context, photo=update.message.reply_to_message)

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Send me a photo or GIF with faces, and I'll pixelate them or add fun overlays!\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n\n"
        "Just send a photo or GIF with faces, and I'll process it!"
    )
    update.message.reply_text(help_text)

def handle_photo(update: Update, context: CallbackContext) -> None:
    """Handle photos sent to the bot"""
    # Check if this is a direct message or a group
    chat_type = update.effective_chat.type
    
    # In groups, only respond to explicit /pixel commands as replies
    if chat_type in ['group', 'supergroup']:
        return
    
    # In direct messages, process automatically
    process_media(update, context)

def handle_pixel_command(update: Update, context: CallbackContext) -> None:
    """Handle /pixel command - must be a reply to a photo/GIF in groups"""
    try:
        # Check if this is a reply to a message
        if not update.message.reply_to_message:
            update.message.reply_text("Please use this command as a reply to a photo or GIF.")
            return
        
        # Check if the replied message contains a photo or document (GIF)
        replied_msg = update.message.reply_to_message
        
        # Check for photos
        has_photo = bool(replied_msg.photo)
        
        # Check for GIFs - they can be in document or animation field
        has_gif = (replied_msg.document and 
                  (replied_msg.document.mime_type == 'image/gif' or 
                   replied_msg.document.file_name.lower().endswith('.gif'))) or bool(replied_msg.animation)
        
        if not (has_photo or has_gif):
            # Send message without reply_to_message_id to avoid errors
            update.message.chat.send_message("Please reply to a photo or GIF.")
            return
        
        # Process the media from the replied message
        process_media(update, context, replied_msg)
        
    except Exception as e:
        logger.error(f"Error in handle_pixel_command: {str(e)}")
        logger.error(traceback.format_exc())
        # Send error message without reply_to_message_id
        try:
            update.message.chat.send_message("An error occurred while processing your request.")
        except:
            pass

def process_media(update: Update, context: CallbackContext, replied_msg=None) -> None:
    """Process media (photos or GIFs) and show the effect keyboard"""
    try:
        # Use either the replied message or the current message
        message = replied_msg if replied_msg else update.message
        
        # Handle the message with our existing function
        handle_message(update, context, photo=message)
        
    except Exception as e:
        logger.error(f"Error in process_media: {str(e)}")
        logger.error(traceback.format_exc())

def main() -> None:
    try:
        # Kill any existing instances
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['pid'] != current_pid:
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'pixelateTG.py' in cmdline[0]:
                        proc.kill()
                        logger.info(f"Killed existing bot instance with PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Get environment from ENV var, default to 'development'
        env = os.getenv('BOT_ENV', 'development')
        logger.info(f"Starting bot in {env} environment")
        
        # Initialize as before
        if not verify_permissions():
            logger.error("Failed to verify directory permissions")
            return
            
        for directory in ['processed', 'downloads']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
                
        cleanup_temp_files()
        socket.setdefaulttimeout(20)
        urllib3.disable_warnings()
        
        # Use token from environment
        if not TOKEN:
            logger.error("No TELEGRAM_BOT_TOKEN found in environment")
            return
        token = TOKEN
        
        # Initialize the updater with proper timeouts
        updater = Updater(
            token=token,
            use_context=True,
            request_kwargs={
                'connect_timeout': 20,
                'read_timeout': 20
            }
        )
        
        cleanup_before_start(updater.bot)
        
        # Register handlers
        logger.info("Registering handlers...")
        dispatcher = updater.dispatcher
        
        # Register error handler
        dispatcher.add_error_handler(error_handler)
        
        # Command handlers - register pixel command with explicit filters for ALL chat types
        dispatcher.add_handler(CommandHandler(
            "pixel", 
            handle_pixel_command,
            filters=Filters.command & (Filters.chat_type.groups | Filters.chat_type.private)
        ))
        
        # Other command handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        
        # Media handlers - photos and GIFs
        dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
        dispatcher.add_handler(MessageHandler(Filters.document.category("image/gif") | 
                                            Filters.animation, handle_photo))
        
        # Button callback handler
        dispatcher.add_handler(CallbackQueryHandler(button_callback))
        
        # Start the Bot with clean state
        logger.info(f"Starting bot in {env} mode...")
        updater.start_polling(drop_pending_updates=True)
        
        # Run the bot until the user presses Ctrl-C
        updater.idle()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

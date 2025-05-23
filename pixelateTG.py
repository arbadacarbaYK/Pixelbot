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
from keyboard import get_main_keyboard, get_pixelation_keyboard, get_full_pixelation_keyboard
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

# Full pixelation levels
FULL_PIXELATION_LEVELS = {
    'very_fine': 0.2,
    'fine': 0.15,
    'rough': 0.09,
    'very_rough': 0.08,
    'distorted': 0.06
}

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
    try:
        for directory in ['processed']:  # Only clean processed directory, keep downloads
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    try:
                        os.remove(os.path.join(directory, f))
                        logger.debug(f"Cleaned up {f} from {directory}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {f}: {e}")
                logger.info(f"Cleaned up {directory} directory")
    except Exception as e:
        logger.error(f"Error in cleanup_temp_files: {e}")

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

def process_full_image(input_path: str, output_path: str, pixelation_factor: float) -> bool:
    """Process the entire image with pixelation"""
    try:
        image = cv2.imread(input_path)
        if image is None:
            logger.error("Failed to read image")
            return False
        
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * pixelation_factor), int(h * pixelation_factor)), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(output_path, pixelated)
        return True
    except Exception as e:
        logger.error(f"Error processing full image: {str(e)}")
        return False

def process_image(input_path, output_path, overlay_type, selected_overlay=None):
    try:
        image = cv2.imread(input_path)
        if image is None:
            return False
            
        faces = detect_heads(image)
        if not faces:
            return False
            
        if overlay_type == 'pixelate':
            for face in faces:
                x, y, w, h = face['rect']
                face_region = image[y:y+h, x:x+w]
                h_face, w_face = face_region.shape[:2]
                w_small = int(w_face * PIXELATION_FACTOR)
                h_small = int(h_face * PIXELATION_FACTOR)
                temp = cv2.resize(face_region, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(temp, (w_face, h_face), interpolation=cv2.INTER_NEAREST)
                image[y:y+h, x:x+w] = pixelated
        elif overlay_type.startswith('full_'):
            # Handle full pixelation
            factor = float(overlay_type.split('_')[1])
            h, w = image.shape[:2]
            small = cv2.resize(image, (int(w * factor), int(h * factor)), 
                             interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # If a specific overlay was provided (for GIFs), use it for all faces
            if selected_overlay:
                overlay_files = [selected_overlay]
                use_same_overlay = True
            else:
                # For static images, get all overlays of this type
                overlay_files = get_overlay_files(overlay_type)
                use_same_overlay = False
                
            if not overlay_files:
                return False
                
            # For static images, we'll pick a random overlay for each face
            # For GIFs, we'll use the provided overlay for all faces
            for face in faces:
                x, y, w, h = face['rect']
                angle = face.get('angle', 0)
                
                # For static images, choose a random overlay for each face
                # For GIFs, use the pre-selected overlay for all faces
                if use_same_overlay:
                    overlay_path = overlay_files[0]
                else:
                    overlay_path = random.choice(overlay_files)
                
                overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                if overlay_img is None:
                    continue
                    
                adjustment = overlay_adjustments.get(overlay_type, {'x_offset': 0, 'y_offset': 0, 'size_factor': 1.0})
                overlay_width = int(w * adjustment['size_factor'])
                overlay_height = int(overlay_width * overlay_img.shape[0] / overlay_img.shape[1])
                
                x_pos = max(0, min(image.shape[1] - overlay_width, x + int(w * adjustment['x_offset'])))
                y_pos = max(0, min(image.shape[0] - overlay_height, y + int(h * adjustment['y_offset'])))
                
                # Ensure overlay fits within image bounds
                overlay_width = min(overlay_width, image.shape[1] - x_pos)
                overlay_height = min(overlay_height, image.shape[0] - y_pos)
                
                overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))
                
                # Apply overlay with proper shape matching
                alpha = overlay_resized[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)
                
                for c in range(3):
                    image[y_pos:y_pos+overlay_height, x_pos:x_pos+overlay_width, c] = \
                        image[y_pos:y_pos+overlay_height, x_pos:x_pos+overlay_width, c] * (1 - alpha[:,:,0]) + \
                        overlay_resized[:, :, c] * alpha[:,:,0]
                        
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

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
            logger.error(f"Processed GIF file is empty: {processed_gif_path}")
            return None
            
        logger.error("Failed to process GIF")
        return None
            
    except Exception as e:
        logger.error(f"Error in process_gif: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def handle_message(update: Update, context: CallbackContext, photo=None) -> None:
    try:
        message = photo if photo else update.message
        chat_id = message.chat_id
        chat_type = message.chat.type
        session_id = str(uuid4())
        id_prefix = f"user_{chat_id}"

        # In groups, only process messages that are replies with /pixel command
        if chat_type in ['group', 'supergroup']:
            # If this is not a command-triggered call, ignore
            if not photo:  # photo parameter is None for natural message flow
                return
            
        session_data = {
            'chat_id': chat_id,
            'id_prefix': id_prefix,
            'session_id': session_id,
            'is_gif': False,
            'chat_type': chat_type,  # Store chat type for later use
            'original_message_id': message.message_id  # Store original message ID for reference
        }

        # Handle animations (GIFs)
        if message.animation or (message.document and message.document.mime_type in ['image/gif', 'video/mp4']):
            is_gif = True
            if message.animation:
                file = message.animation.get_file()
                logger.debug("Processing animation")
            else:
                file = message.document.get_file()
                logger.debug(f"Processing document with mime type: {message.document.mime_type}")

            # Always save with .gif extension for GIF processor
            input_path = get_file_path('downloads', id_prefix, session_id, 'animation.gif')
            file.download(input_path)
            logger.info(f"Downloaded GIF/Animation to {input_path}")
            
            if not os.path.exists(input_path):
                logger.error(f"Failed to download GIF to {input_path}")
                message.reply_text("Sorry, I couldn't download that GIF. Please try again!")
                return
                
            session_data['is_gif'] = True
            session_data['input_path'] = input_path

        # Handle photos
        elif message.photo:
            input_path = get_file_path('downloads', id_prefix, session_id, 'original.jpg')
            file = context.bot.get_file(message.photo[-1].file_id)
            file.download(input_path)
            logger.info(f"Downloaded photo to {input_path}")
            session_data['input_path'] = input_path
            session_data['photo_path'] = input_path  # Store photo path for full pixelation
        else:
            message.reply_text("Please send me a photo or GIF!")
            return

        # Store session data in appropriate context
        if chat_type in ['group', 'supergroup']:
            if 'sessions' not in context.chat_data:
                context.chat_data['sessions'] = {}
            context.chat_data['sessions'][session_id] = session_data
            logger.debug(f"Stored session {session_id} in chat_data")
        else:
            if 'sessions' not in context.user_data:
                context.user_data['sessions'] = {}
            context.user_data['sessions'][session_id] = session_data
            logger.debug(f"Stored session {session_id} in user_data")
            
        logger.debug(f"Created new session: {session_id}")
        
        # Create keyboard - show full pixelation only for photos, not GIFs
        keyboard = [
            [
                InlineKeyboardButton("⚔️ Pixelate", callback_data=f"{session_id}:pixelate"),
            ]
        ]
        
        # Only add full pixelation for photos, not GIFs
        if not session_data['is_gif']:
            keyboard[0].append(InlineKeyboardButton("✂️ Full Pixelate", callback_data=f"{session_id}:full_pixelate"))
            
        keyboard.extend([
            [
                InlineKeyboardButton("🤡 Clown", callback_data=f"{session_id}:clown"),
                InlineKeyboardButton("😎 Ray Liotta", callback_data=f"{session_id}:liotta")
            ],
            [
                InlineKeyboardButton("💀 Skull", callback_data=f"{session_id}:skull"),
                InlineKeyboardButton("😺 Cat", callback_data=f"{session_id}:cat"),
                InlineKeyboardButton("🐸 Pepe", callback_data=f"{session_id}:pepe")
            ],
            [
                InlineKeyboardButton("👨 Chad", callback_data=f"{session_id}:chad")
            ]
        ])

        # Add cancel button with special group handling
        if chat_type in ['group', 'supergroup']:
            keyboard.append([InlineKeyboardButton("❌ Cancel (Anyone)", callback_data=f"{session_id}:cancel_group")])
        else:
            keyboard.append([InlineKeyboardButton("❌ Close", callback_data=f"{session_id}:cancel")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        message.reply_text('Choose an effect:', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            message.reply_text("Sorry, something went wrong. Please try again!")
        except:
            pass

def cleanup_before_start(bot):
    """Clean up before starting the bot"""
    logger.info("Cleanup before start completed")
    return True

def error_handler(update: Update, context: CallbackContext) -> None:
    """Log Errors caused by Updates."""
    logger.error(f'Update "{update}" caused error "{context.error}"')

def button_callback(update: Update, context: CallbackContext) -> None:
    try:
        query = update.callback_query
        session_id, action = query.data.split(':')
        chat_type = query.message.chat.type
        
        # Handle navigation actions first
        if action == 'cancel':
            # In DMs, only the original sender can cancel
            if chat_type == 'private':
                query.message.delete()
            query.answer()
            return
        elif action == 'cancel_group':
            # In groups, anyone can cancel
            query.message.delete()
            query.answer()
            return
        elif action == 'back':
            keyboard = get_main_keyboard(session_id)
            query.edit_message_reply_markup(reply_markup=keyboard)
            query.answer()
            return
            
        # Get session data from the appropriate context
        session_data = None
        if chat_type in ['group', 'supergroup']:
            if 'sessions' in context.chat_data:
                session_data = context.chat_data['sessions'].get(session_id)
                logger.debug(f"Got session {session_id} from chat_data: {session_data is not None}")
        else:
            if 'sessions' in context.user_data:
                session_data = context.user_data['sessions'].get(session_id)
                logger.debug(f"Got session {session_id} from user_data: {session_data is not None}")
                
        if not session_data:
            logger.error(f"No session data found for {session_id}")
            query.message.delete()
            query.answer()
            return
            
        input_path = session_data['input_path']
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            query.message.delete()
            query.answer()
            return

        # Handle GIF processing
        if session_data.get('is_gif'):
            output_path = process_gif(
                session_data['input_path'],
                session_id,
                session_data['id_prefix'],
                context.bot,
                action=action  # Pass the selected action
            )
            
            if output_path and os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    context.bot.send_animation(
                        chat_id=session_data['chat_id'],
                        animation=f,
                        reply_to_message_id=session_data['original_message_id']  # Reply to original message
                    )
            else:
                query.answer("Failed to process GIF!")
            return
                
        # Handle Photo processing
        else:
            output_path = get_file_path('processed', session_data['id_prefix'], session_id, action)
            logger.debug(f"Processing from {input_path} to {output_path}")
            
            success = False
            if action == 'pixelate':
                logger.debug("Starting pixelation...")
                success = process_image(input_path, output_path, 'pixelate')
            elif action == 'full_pixelate':
                # Show full pixelation keyboard
                keyboard = get_full_pixelation_keyboard(session_id)
                query.edit_message_reply_markup(reply_markup=keyboard)
                query.answer()
                return
            elif action.startswith('full_'):
                # Handle full pixelation with specific factor
                factor = float(action.split('_')[1])
                success = process_full_image(input_path, output_path, factor)
            elif action in ['clown', 'liotta', 'skull', 'cat', 'pepe', 'chad']:
                logger.debug(f"Starting {action} overlay...")
                success = process_image(input_path, output_path, action)
            
            if success:
                with open(output_path, 'rb') as f:
                    # Keep the same keyboard for pixelation options
                    keyboard = None
                    if action.startswith('pixelate_'):
                        keyboard = get_pixelation_keyboard(session_id)
                    elif action.startswith('full_'):
                        keyboard = get_full_pixelation_keyboard(session_id)
                    context.bot.send_photo(
                        chat_id=session_data['chat_id'],
                        photo=f,
                        reply_to_message_id=session_data['original_message_id'],  # Reply to original message
                        reply_markup=keyboard
                    )
                query.answer()
            else:
                query.answer(f"Failed to process {action}!")
            return

        # Just acknowledge the button press
        query.answer()
        
    except Exception as e:
        logger.error(f"Error in button_callback: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            query.answer()
            query.message.delete()
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
    """Cleanup files older than 6 hours"""
    try:
        current_time = time.time()
        for directory in ['processed', 'downloads']:
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    filepath = os.path.join(directory, f)
                    # Keep files around for 6 hours to ensure they're available for group chats
                    if os.path.getmtime(filepath) < (current_time - 21600):  # 6 hours
                        try:
                            os.remove(filepath)
                            logger.debug(f"Removed old file: {filepath}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {e}")

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
    """Handle /pixel command when used as a reply to a photo or GIF"""
    message = update.message
    
    # Check if this is a reply to a message
    if not message.reply_to_message:
        if message.chat.type in ['group', 'supergroup']:
            message.reply_text("Please use this command as a reply to a photo or GIF!")
        return
        
    replied_msg = message.reply_to_message
    
    # Check if the replied message contains a photo or GIF
    if replied_msg.photo:
        handle_message(update, context, photo=replied_msg)
    elif replied_msg.animation or (replied_msg.document and 
          replied_msg.document.mime_type in ['image/gif', 'video/mp4']):
        handle_message(update, context, photo=replied_msg)
    else:
        if message.chat.type in ['group', 'supergroup']:
            message.reply_text("Please reply to a photo or GIF with this command!")

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    chat_type = update.message.chat.type
    
    if chat_type in ['group', 'supergroup']:
        help_text = (
            "To use this bot in groups:\n"
            "1. Reply to any photo or GIF with /pixel\n"
            "2. Choose an effect from the menu\n"
            "3. Anyone in the group can cancel the effect menu\n\n"
            "Commands:\n"
            "/pixel - Reply to a photo/GIF with this command\n"
            "/help - Show this help message"
        )
    else:
        help_text = (
            "In private chats:\n"
            "Just send me a photo or GIF with faces, and I'll pixelate them or add fun overlays!\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n\n"
            "Just send a photo or GIF with faces, and I'll process it!"
        )
    update.message.reply_text(help_text)

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

        # Initialize directories
        if not verify_permissions():
            logger.error("Failed to verify directory permissions")
            return
            
        for directory in ['processed', 'downloads']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
                
        # Only clean processed files on startup, keep downloads
        cleanup_temp_files()
        
        socket.setdefaulttimeout(20)
        urllib3.disable_warnings()
        
        # Use token from environment
        if not TOKEN:
            logger.error("No TELEGRAM_BOT_TOKEN found in environment")
            return
        token = TOKEN
        
        updater = Updater(
            token=token,
            use_context=True,
            request_kwargs={
                'connect_timeout': 20,
                'read_timeout': 20
            }
        )
        
        cleanup_before_start(updater.bot)
        dispatcher = updater.dispatcher
        
        # Register handlers
        logger.info("Registering handlers...")
        
        # Create a combined filter for all GIF types
        gif_filter = (
            Filters.animation |
            (Filters.document.mime_type("image/gif") |
             Filters.document.mime_type("video/mp4"))
        )
        
        # Add handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(CommandHandler("pixel", photo_command))
        
        # Add media handlers - these will only work in private chats due to handle_message logic
        dispatcher.add_handler(MessageHandler(Filters.photo, handle_message))
        dispatcher.add_handler(MessageHandler(gif_filter, handle_message))
        
        # Add callback handler - only use button_callback, not handle_callback
        dispatcher.add_handler(CallbackQueryHandler(button_callback))
        
        # Add error handler
        dispatcher.add_error_handler(error_handler)
        
        logger.info("Starting bot...")
        updater.start_polling(drop_pending_updates=True)
        updater.idle()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

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
            logger.error("No file_id found in message")
            return
            
        # Download the file
        file_extension = 'gif.jpg' if is_gif else 'original.jpg'
        file_path = get_file_path('downloads', id_prefix, session_id, file_extension)
        
        file = context.bot.get_file(file_id)
        file.download(file_path)
        
        if is_gif:
            logger.info(f"Downloaded GIF to {file_path}")
        else:
            logger.info(f"Downloaded photo to {file_path}")
            
        # Store file path in session data
        session_data['input_path'] = file_path
        session_data['is_gif'] = is_gif

        # Store session data
        context.user_data[session_id] = session_data
        logger.debug(f"Created new session: {session_id}")

        # Create keyboard with effect options
        keyboard = [
            [
                InlineKeyboardButton("🧩 Pixelate", callback_data=f"pixelate:{session_id}"),
                InlineKeyboardButton("🤡 Clown", callback_data=f"clown:{session_id}")
            ],
            [
                InlineKeyboardButton("😎 Liotta", callback_data=f"liotta:{session_id}"),
                InlineKeyboardButton("💀 Skull", callback_data=f"skull:{session_id}")
            ],
            [
                InlineKeyboardButton("🐱 Cat", callback_data=f"cat:{session_id}"),
                InlineKeyboardButton("🐸 Pepe", callback_data=f"pepe:{session_id}")
            ],
            [
                InlineKeyboardButton("👨 Chad", callback_data=f"chad:{session_id}"),
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
        
        # Only delete message if cancel is pressed
        if action == 'cancel':
            query.message.delete()
            return
            
        # Get session data
        session_data = context.user_data.get(session_id)
        if not session_data:
            logger.error(f"No session data found for {session_id}")
            query.edit_message_text(text="Session expired, please send a new photo!")
            return
            
        input_path = session_data['input_path']
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            query.edit_message_text(text="Original photo not found, please send a new one!")
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
                        reply_to_message_id=query.message.message_id
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
            elif action in ['clown', 'liotta', 'skull', 'cat', 'pepe', 'chad']:
                logger.debug(f"Starting {action} overlay...")
                success = process_image(input_path, output_path, action)
            
            if success:
                with open(output_path, 'rb') as f:
                    context.bot.send_photo(
                        chat_id=session_data['chat_id'],
                        photo=f,
                        reply_to_message_id=query.message.message_id
                    )
            else:
                query.answer(f"Failed to process {action}!")
                return

        # Just acknowledge the button press
        query.answer()
        
        # Cleanup processed file
        try:
            os.remove(output_path)
            logger.debug(f"Cleaned up {output_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {output_path}: {e}")
            
    except Exception as e:
        logger.error(f'Error in button callback: {str(e)}')
        logger.error(traceback.format_exc())
        if query:
            query.answer("Sorry, there was an error processing your request")
        cleanup_temp_files()

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
        
        # Command handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(CommandHandler("pixel", handle_pixel_command))
        
        # Media handlers - photos and GIFs
        dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
        dispatcher.add_handler(MessageHandler(Filters.document.category("image/gif") | 
                                             Filters.animation, handle_photo))
        
        # Button callback handler
        dispatcher.add_handler(CallbackQueryHandler(button_callback))
        
        # Start the Bot
        logger.info(f"Starting bot in {env} mode...")
        updater.start_polling()
        
        # Run the bot until the user presses Ctrl-C
        updater.idle()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()


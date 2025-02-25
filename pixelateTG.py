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
from mtcnn.mtcnn import MTCNN
from uuid import uuid4
import time
import logging
import traceback
import socket
import urllib3
from telegram.utils.request import Request

# Configure DNS settings
socket.setdefaulttimeout(20)
urllib3.disable_warnings()

# Add this after the other imports at the top (around line 12)
logging.basicConfig(
    filename='pixelbot_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MAX_THREADS = 15
PIXELATION_FACTOR = 0.04
RESIZE_FACTOR = 2.0
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

# Global MTCNN detector
mtcnn_detector = MTCNN()

# Cache for overlay files
overlay_cache = {}

# At the top with other globals
overlay_image_cache = {}

# Add this with the other globals at the top
overlay_adjustments = {
    'clown': {'x_offset': -0.15, 'y_offset': -0.25, 'size_factor': 1.4},
    'liotta': {'x_offset': -0.05, 'y_offset': -0.15, 'size_factor': 1.2},
    'skull': {'x_offset': -0.05, 'y_offset': -0.15, 'size_factor': 1.3},
    'cat': {'x_offset': -0.15, 'y_offset': -0.25, 'size_factor': 1.4},
    'pepe': {'x_offset': -0.05, 'y_offset': -0.2, 'size_factor': 1.3},
    'chad': {'x_offset': -0.05, 'y_offset': -0.15, 'size_factor': 1.2}
}

# Add at the top with other globals
OVERLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overlays')

def verify_permissions():
    """Verify write permissions for required directories"""
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

def get_file_path(directory, id_prefix, session_id, action_type):
    """Generate consistent file paths with proper ID prefix"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory, f'{id_prefix}_{session_id}_{action_type}.jpg')

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

def detect_heads(image):
    """Detect faces using MTCNN"""
    try:
        faces = mtcnn_detector.detect_faces(image)
        return [(f['box'][0], f['box'][1], f['box'][2], f['box'][3]) for f in faces]
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []

def get_overlay_files(overlay_type):
    """Get list of overlay files for a given type"""
    try:
        if overlay_type in overlay_cache:
            return overlay_cache[overlay_type]
            
        # Get current working directory (root)
        current_dir = os.getcwd()
        logger.debug(f"Looking for overlays in: {current_dir}")
        
        # Search for overlay files
        files = []
        for f in os.listdir(current_dir):
            if f.startswith(f'{overlay_type}_') and f.endswith('.png'):
                full_path = os.path.join(current_dir, f)
                if os.path.isfile(full_path):
                    files.append(full_path)
                    logger.debug(f"Found overlay file: {full_path}")
                    
        if not files:
            logger.error(f"No overlay files found for type: {overlay_type}")
            return []
            
        overlay_cache[overlay_type] = files
        return files
        
    except Exception as e:
        logger.error(f"Error getting overlay files: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def get_cached_overlay(overlay_file):
    """Get overlay image from cache or load it"""
    try:
        if overlay_file in overlay_image_cache:
            return overlay_image_cache[overlay_file]
            
        overlay_img = cv2.imread(overlay_file, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            logger.error(f"Failed to read overlay file: {overlay_file}")
            return None
            
        overlay_image_cache[overlay_file] = overlay_img
        logger.debug(f"Cached overlay image: {overlay_file}")
        return overlay_img
        
    except Exception as e:
        logger.error(f"Error loading overlay image: {str(e)}")
        return None

def get_id_prefix(update):
    """Generate a consistent ID prefix for a user"""
    return f"user_{update.effective_user.id}"

def process_image(input_path, output_path):
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            logger.error(f"Failed to read image: {input_path}")
            return False
            
        # Detect faces
        faces = mtcnn_detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not faces:
            logger.warning("No faces detected in image")
            return False
            
        # Process each face
        for face in faces:
            x, y, width, height = [int(v) for v in face['box']]
            face_region = image[y:y+height, x:x+width]
            
            # Pixelate face
            if face_region.size > 0:  # Check if region is valid
                small = cv2.resize(face_region, (0,0), fx=PIXELATION_FACTOR, fy=PIXELATION_FACTOR)
                pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
                image[y:y+height, x:x+width] = pixelated
            
        # Save result
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_random_overlay_file(overlay_type):
    """Get a random overlay file of the given type"""
    try:
        # Find all files matching pattern overlay_type_N.png
        matching_files = [f for f in os.listdir('.') if f.startswith(f'{overlay_type}_') and f.endswith('.png')]
        if not matching_files:
            logger.error(f"No overlay files found for type: {overlay_type}")
            return None
        return random.choice(matching_files)
    except Exception as e:
        logger.error(f"Error finding overlay files: {str(e)}")
        return None

def overlay(input_path, overlay_type, output_path):
    """Apply overlay to detected faces"""
    try:
        logger.debug(f"Starting overlay process for {overlay_type}")
        
        # Read input image
        image = cv2.imread(input_path)
        if image is None:
            logger.error(f"Failed to read input image: {input_path}")
            return False
            
        # Convert for face detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(rgb_image)
        
        logger.debug(f"Detected {len(faces)} faces")
        
        if not faces:
            logger.error("No faces detected in image")
            return False

        # Get overlay files
        overlay_files = get_overlay_files(overlay_type)
        if not overlay_files:
            logger.error(f"No overlay files found for type: {overlay_type}")
            return False
            
        # Pick random overlay
        overlay_file = random.choice(overlay_files)
        logger.debug(f"Selected overlay: {overlay_file}")
        
        # Get full path to overlay file
        overlay_path = os.path.join(os.getcwd(), overlay_file)
        logger.debug(f"Full overlay path: {overlay_path}")
        
        # Read overlay with alpha channel
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            logger.error(f"Failed to read overlay: {overlay_path}")
            return False
            
        # Check if overlay has alpha channel
        has_alpha = overlay_img.shape[2] == 4 if len(overlay_img.shape) > 2 else False
        if not has_alpha:
            logger.error(f"Overlay image doesn't have alpha channel: {overlay_path}")
            # Create a dummy alpha channel (fully opaque)
            if len(overlay_img.shape) == 3:
                b, g, r = cv2.split(overlay_img)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                overlay_img = cv2.merge((b, g, r, alpha))
            else:
                # Grayscale image
                gray = overlay_img
                alpha = np.ones(gray.shape, dtype=gray.dtype) * 255
                overlay_img = cv2.merge((gray, gray, gray, alpha))
        
        logger.debug(f"Overlay shape: {overlay_img.shape}")
        
        # Process each face
        for face_idx, face in enumerate(faces):
            try:
                x, y, width, height = face['box']
                
                # Get adjustments
                adjust = overlay_adjustments.get(overlay_type, {
                    'x_offset': 0,
                    'y_offset': 0, 
                    'size_factor': 1.0
                })
                
                # Calculate overlay size
                overlay_width = int(width * adjust['size_factor'])
                overlay_height = int(height * adjust['size_factor'])
                
                # Calculate position
                x_pos = max(0, x + int(width * adjust['x_offset']))
                y_pos = max(0, y + int(height * adjust['y_offset']))
                
                # Make sure overlay doesn't go out of bounds
                if x_pos + overlay_width > image.shape[1]:
                    overlay_width = image.shape[1] - x_pos
                if y_pos + overlay_height > image.shape[0]:
                    overlay_height = image.shape[0] - y_pos
                
                if overlay_width <= 0 or overlay_height <= 0:
                    logger.warning(f"Overlay dimensions invalid: {overlay_width}x{overlay_height}")
                    continue
                
                # Resize overlay
                overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))
                
                # Get the region of the image where we'll place the overlay
                roi = image[y_pos:y_pos+overlay_height, x_pos:x_pos+overlay_width]
                
                # Create a mask from the alpha channel
                alpha_channel = overlay_resized[:,:,3] / 255.0
                alpha_3channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
                
                # Apply the overlay
                foreground = overlay_resized[:,:,:3] * alpha_3channel
                background = roi * (1 - alpha_3channel)
                result = foreground + background
                
                # Place the result back in the image
                image[y_pos:y_pos+overlay_height, x_pos:x_pos+overlay_width] = result
                
                logger.debug(f"Applied overlay to face {face_idx+1}")
                
            except Exception as e:
                logger.error(f"Error processing face {face_idx+1}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Save result
        cv2.imwrite(output_path, image)
        logger.debug(f"Saved processed image to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error in overlay function: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Overlay functions
def clown_overlay(photo_path, output_path):
    logger.info("Starting clowns overlay")
    return overlay(photo_path, 'clown', output_path)

def liotta_overlay(photo_path, output_path):
    logger.info("Starting liotta overlay")
    return overlay(photo_path, 'liotta', output_path)

def skull_overlay(photo_path, output_path):
    logger.info("Starting skull overlay")
    return overlay(photo_path, 'skull', output_path)

def cat_overlay(photo_path, output_path):
    logger.info("Starting cats overlay")
    return overlay(photo_path, 'cat', output_path)

def pepe_overlay(photo_path, output_path):
    logger.info("Starting pepe overlay")
    return overlay(photo_path, 'pepe', output_path)

def chad_overlay(photo_path, output_path):
    logger.info("Starting chad overlay")
    return overlay(photo_path, 'chad', output_path)

def process_gif(gif_path, session_id, id_prefix, bot):
    reader = imageio.get_reader(gif_path)
    processed_frames = []
    
    for frame in reader:
        # Convert frame to temporary image file
        temp_frame_path = get_file_path('downloads', id_prefix, session_id, 'frame')
        temp_output_path = get_file_path('processed', id_prefix, session_id, 'frame')
        imageio.imwrite(temp_frame_path, frame)
        
        process_image(temp_frame_path, temp_output_path)
        processed_frame = imageio.imread(temp_output_path)
        processed_frames.append(processed_frame)
        
        # Cleanup temp files
        os.remove(temp_frame_path)
        os.remove(temp_output_path)
    
    processed_gif_path = get_file_path('processed', id_prefix, session_id, 'gif')
    imageio.mimsave(processed_gif_path, processed_frames)
    return processed_gif_path

def handle_message(update: Update, context: CallbackContext) -> None:
    try:
        # Generate session ID
        session_id = str(uuid4())
        id_prefix = get_id_prefix(update)
        
        # Get the largest photo
        photo = max(update.message.photo, key=lambda x: x.file_size)
        photo_file = context.bot.get_file(photo.file_id)
        
        # Download photo
        output_path = get_file_path('downloads', id_prefix, session_id, 'original')
        photo_file.download(output_path)
        logger.info(f"Downloaded photo to {output_path}")
        
        # Store session data
        context.chat_data[session_id] = {
            'chat_id': update.effective_chat.id,
            'id_prefix': id_prefix,
            'photo_path': output_path
        }
        logger.debug(f"Created new session: {session_id}")
        
        # Create keyboard and send menu
        keyboard = [
            [InlineKeyboardButton("🤡 Clowns", callback_data=f'clown_{session_id}'),
             InlineKeyboardButton("😂 Liotta", callback_data=f'liotta_{session_id}'),
             InlineKeyboardButton("☠️ Skull", callback_data=f'skull_{session_id}')],
            [InlineKeyboardButton("🐈‍⬛ Cats", callback_data=f'cat_{session_id}'),
             InlineKeyboardButton("🐸 Pepe", callback_data=f'pepe_{session_id}'),
             InlineKeyboardButton("🏆 Chad", callback_data=f'chad_{session_id}')],
            [InlineKeyboardButton("⚔️ Pixel", callback_data=f'pixelate_{session_id}')],
            [InlineKeyboardButton("CLOSE ME", callback_data=f'cancel_{session_id}')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("Choose an option:", reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(traceback.format_exc())

def cleanup_before_start(bot):
    """Clean up any pending messages or files before starting"""
    cleanup_temp_files()
    logger.info("Cleanup before start completed")

def button_callback(update: Update, context: CallbackContext) -> None:
    try:
        query = update.callback_query
        logger.debug(f"Received callback query: {query.data}")
        query.answer()
        
        callback_data = query.data
        action, session_id = callback_data.split('_', 1)
        logger.debug(f"Processing action: {action} for session: {session_id}")
        
        # Get session data
        session_data = context.chat_data.get(session_id)
        if not session_data:
            logger.error(f"No session data found for {session_id}")
            query.edit_message_text(text="Session expired, please send a new photo!")
            return
            
        input_path = session_data['photo_path']
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            query.edit_message_text(text="Original photo not found, please send a new one!")
            return
            
        output_path = get_file_path('processed', session_data['id_prefix'], session_id, action)
        logger.debug(f"Processing from {input_path} to {output_path}")
        
        # Handle cancel action
        if action == 'cancel':
            query.edit_message_text(text="Closed!")
            return
            
        try:
            # Process based on action
            success = False
            if action == 'pixelate':
                logger.debug("Starting pixelation...")
                success = process_image(input_path, output_path)
            elif action in ['clown', 'liotta', 'skull', 'cat', 'pepe', 'chad']:
                # Check overlay files
                overlay_files = get_overlay_files(action)
                if not overlay_files:
                    logger.error(f"No overlay files found for {action}")
                    query.edit_message_text(text=f"No {action} overlays available!")
                    return
                    
                logger.debug(f"Starting {action} overlay...")
                success = overlay(input_path, action, output_path)
            else:
                logger.error(f"Unknown action: {action}")
                query.edit_message_text(text="Invalid action!")
                return
                
            if not success:
                logger.error(f"Processing failed for {action}")
                query.edit_message_text(text=f"Failed to process {action}!")
                return
                
            # Verify output file exists
            if not os.path.exists(output_path):
                logger.error(f"Output file not created: {output_path}")
                query.edit_message_text(text="Failed to create output file!")
                return
                
            # Send processed image
            logger.debug(f"Sending processed image: {output_path}")
            with open(output_path, 'rb') as f:
                context.bot.send_photo(
                    chat_id=session_data['chat_id'],
                    photo=f,
                    reply_to_message_id=query.message.message_id
                )
            query.message.delete()
            
            # Cleanup
            try:
                os.remove(output_path)
                logger.debug(f"Cleaned up {output_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {output_path}: {e}")
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            query.edit_message_text(text=error_msg)
            
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            query.edit_message_text(text=f"Error: {str(e)}")
        except:
            pass

def main() -> None:
    # Add permission check at the start
    if not verify_permissions():
        logger.error("Failed to verify directory permissions")
        return
        
    # Ensure temp directories exist
    for directory in ['processed', 'downloads']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
            
    # Initial cleanup
    cleanup_temp_files()
    
    # Configure DNS settings
    socket.setdefaulttimeout(20)
    urllib3.disable_warnings()
    
    # Initialize Updater with correct parameters
    updater = Updater(
        token=TOKEN,
        use_context=True,
        request_kwargs={
            'connect_timeout': 20,
            'read_timeout': 20
        }
    )
    
    cleanup_before_start(updater.bot)
    dispatcher = updater.dispatcher

    # Add debug logging for handler registration
    logger.info("Registering handlers...")
    
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_message))
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    # Add this right after the directory creation
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Available files: {os.listdir('.')}")

    logger.info("Checking overlay files...")
    for overlay_type in ['clown', 'liotta', 'skull', 'cat', 'pepe', 'chad']:
        files = get_overlay_files(overlay_type)
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.error(f"Cannot read {f}")
            else:
                logger.info(f"Successfully read {f} shape: {img.shape}")

    logger.info("Starting polling...")
    try:
        updater.start_polling()
        logger.info("Bot is now running")
        updater.idle()
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

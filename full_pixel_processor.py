import os
import cv2
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
from uuid import uuid4
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

# Pixelation levels
PIXELATION_LEVELS = {
    'very_fine': 0.2,
    'fine': 0.15,
    'rough': 0.09,
    'very_rough': 0.08,
    'distorted': 0.06
}

def process_full_image(photo_path: str, output_path: str, pixelation_factor: float) -> bool:
    try:
        image = cv2.imread(photo_path)
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
        logger.error(f"Error processing image: {str(e)}")
        return False

def handle_full_pixel_command(update: Update, context: CallbackContext) -> None:
    """Handle /fullpixel command"""
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        update.message.reply_text("This special feature only works on pics.")
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    photo = update.message.reply_to_message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    
    user_dir = f"user_{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    
    input_path = f"{user_dir}/original_{uuid4()}.jpg"
    file.download(input_path)
    
    session_id = str(uuid4())
    
    # Store both chat_id and input_path under the session_id
    if 'sessions' not in context.user_data:
        context.user_data['sessions'] = {}
    context.user_data['sessions'][session_id] = {
        'chat_id': chat_id,
        'input_path': input_path,
        'id_prefix': f"user_{user_id}",
        'session_id': session_id
    }
    
    keyboard = create_full_pixel_keyboard(session_id)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(
        "Choose your pixelation level for the full image:",
        reply_markup=reply_markup
    )

def handle_full_pixel_text(update: Update, context: CallbackContext) -> None:
    """Handle text responses for full pixelation level selection"""
    user_id = update.effective_user.id
    text = update.message.text.lower().replace(' ', '_')
    
    if 'photo_path' not in context.user_data:
        update.message.reply_text("This only works as reply to a pic.")
        return
        
    if text not in PIXELATION_LEVELS:
        update.message.reply_text("Choose a pixelation.")
        return
        
    input_path = context.user_data['photo_path']
    output_path = f"user_{user_id}/full_pixelated_{text}.jpg"
    pixelation_factor = PIXELATION_LEVELS[text]
    
    if process_full_image(input_path, output_path, pixelation_factor):
        with open(output_path, 'rb') as f:
            update.message.reply_photo(photo=f)
    else:
        update.message.reply_text("Sorry, failed to process the image. Please try again.")

def create_full_pixel_keyboard(session_id: str) -> List[List[InlineKeyboardButton]]:
    return [
        [
            InlineKeyboardButton("🧵 Very Fine 🧵", callback_data=f"very_fine:{session_id}"),
            InlineKeyboardButton("🧶 Fine 🧶", callback_data=f"fine:{session_id}")
        ],
        [
            InlineKeyboardButton("🪵 Rough 🪵", callback_data=f"rough:{session_id}"),
            InlineKeyboardButton("🪨 Very Rough 🪨", callback_data=f"very_rough:{session_id}"),
            InlineKeyboardButton("🔮 Distorted 🔮", callback_data=f"distorted:{session_id}")
        ],
        [
            InlineKeyboardButton("❌ Close ❌", callback_data=f"close:{session_id}")
        ]
    ]
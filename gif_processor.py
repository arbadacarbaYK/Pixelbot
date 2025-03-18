import os
import cv2
import imageio
import numpy as np
from uuid import uuid4
import logging
import traceback
from typing import List, Tuple, Optional
from PIL import Image
from constants import PIXELATION_FACTOR, detect_heads
import glob
import random

logger = logging.getLogger(__name__)

# Import overlay_adjustments from pixelateTG or define it here
overlay_adjustments = {
    'clown': {'x_offset': -0.15, 'y_offset': -0.25, 'size_factor': 1.66},
    'liotta': {'x_offset': -0.12, 'y_offset': -0.2, 'size_factor': 1.5},
    'skull': {'x_offset': -0.25, 'y_offset': -0.5, 'size_factor': 1.65},
    'cat': {'x_offset': -0.15, 'y_offset': -0.45, 'size_factor': 1.5}, 
    'pepe': {'x_offset': -0.05, 'y_offset': -0.2, 'size_factor': 1.4},
    'chad': {'x_offset': -0.15, 'y_offset': -0.15, 'size_factor': 1.6}  
}

def get_overlay_files(overlay_type):
    """Get all overlay files for a specific type"""
    # Look for overlays directly in the root directory
    overlay_files = glob.glob(f"{overlay_type}_*.png")
    
    if not overlay_files:
        logger.error(f"No overlay files found matching pattern: {overlay_type}_*.png")
        logger.error(f"Searched in directory: {os.getcwd()}")
    
    return overlay_files

def get_random_overlay_file(overlay_type):
    """Get a random overlay file for the given type"""
    try:
        overlay_files = get_overlay_files(overlay_type)
        if not overlay_files:
            return None
        return random.choice(overlay_files)
    except Exception as e:
        logger.error(f"Error in get_random_overlay_file: {str(e)}")
        return None

def get_file_path(directory: str, id_prefix: str, session_id: str, suffix: str) -> str:
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

class GifProcessor:
    """Class for processing GIFs with various effects"""
    
    @staticmethod
    def validate_gif(input_path: str) -> bool:
        """Validate if the input file is a valid GIF"""
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            logger.error(f"Invalid input file: {input_path}")
            return False

        if not input_path.lower().endswith('.gif'):
            logger.error(f"Input file is not a GIF: {input_path}")
            return False
        return True

    @staticmethod
    def extract_frames(input_path: str) -> Tuple[List[np.ndarray], float]:
        """Extract frames from a GIF file"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open GIF with OpenCV: {input_path}")
            return [], 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        return frames, fps

    @staticmethod
    def process_frames(frames: List[np.ndarray], process_func, session_id: str, id_prefix: str, action: str) -> List[np.ndarray]:
        """Process each frame with the given function"""
        processed_frames = []
        
        for i, frame in enumerate(frames):
            # Use .jpg for temporary files since OpenCV can't write GIFs
            temp_in = get_file_path('downloads', id_prefix, session_id, f'temp_{i}.jpg')
            temp_out = get_file_path('processed', id_prefix, session_id, f'temp_{i}.jpg')
            
            try:
                cv2.imwrite(temp_in, frame)
                success = process_func(temp_in, temp_out, action)  # Removed unnecessary session_id and id_prefix
                
                if success and os.path.exists(temp_out):
                    processed = cv2.imread(temp_out)
                    if processed is not None:
                        processed_frames.append(processed)
                    else:
                        processed_frames.append(frame)
                else:
                    processed_frames.append(frame)
            finally:
                # Clean up temporary files
                for temp in [temp_in, temp_out]:
                    if os.path.exists(temp):
                        try:
                            os.remove(temp)
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {temp}: {e}")
                        
        return processed_frames

    @staticmethod
    def save_gif(frames: List[np.ndarray], output_path: str, fps: float) -> bool:
        """Save processed frames as a GIF"""
        try:
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            imageio.mimsave(output_path, rgb_frames, format='GIF', fps=fps)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            logger.error(f"Error saving GIF: {str(e)}")
            return False

    @staticmethod
    def process_gif(input_path: str, output_path: str, process_func, **kwargs) -> bool:
        """Process a GIF using the provided function"""
        try:
            if not GifProcessor.validate_gif(input_path):
                return False

            frames, fps = GifProcessor.extract_frames(input_path)
            if not frames:
                logger.error("No frames extracted from GIF")
                return False

            session_id = kwargs.get('session_id')
            id_prefix = kwargs.get('id_prefix')
            action = kwargs.get('action', 'pixelate')

            # Get a random overlay file for the entire GIF if using overlays
            selected_overlay = None
            if action != 'pixelate':
                selected_overlay = get_random_overlay_file(action)
                if not selected_overlay:
                    logger.error(f"No overlay files found for type: {action}")
                    return False

            # Create a wrapper function that matches what process_frames expects
            def frame_processor(temp_in: str, temp_out: str, action: str) -> bool:
                # Call process_image with the correct parameters, including the selected overlay
                return process_func(temp_in, temp_out, action, selected_overlay)

            processed_frames = GifProcessor.process_frames(frames, frame_processor, session_id, id_prefix, action)
            return GifProcessor.save_gif(processed_frames, output_path, fps)

        except Exception as e:
            logger.error(f"Error processing GIF: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Keep this for backward compatibility
def process_telegram_gif(input_path: str, output_path: str, process_func, **kwargs) -> bool:
    """Legacy function that uses GifProcessor class"""
    return GifProcessor.process_gif(input_path, output_path, process_func, **kwargs)

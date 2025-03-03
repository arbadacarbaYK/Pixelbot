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

class GifProcessor:
    """Class for processing GIFs with various effects"""
    
    @staticmethod
    def process_gif(input_path, output_path, process_func, **kwargs):
        """Process a GIF using the provided function"""
        return process_telegram_gif(input_path, output_path, process_func, **kwargs)

def process_telegram_gif(input_path: str, output_path: str, process_func, **kwargs) -> bool:
    try:
        # Verify input file
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            logger.error(f"Invalid input file: {input_path}")
            return False

        # Use OpenCV to read the GIF
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open GIF with OpenCV: {input_path}")
            return False

        # Get basic video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10  # Default FPS if not detected

        # Extract frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            logger.error("No frames extracted from GIF")
            return False

        # Process each frame
        processed_frames = []
        overlay_type = kwargs.get('action', 'pixelate')
        
        # For overlays, pre-select one random overlay to use for all frames
        selected_overlay = None
        if overlay_type != 'pixelate':
            # Get all overlays of this type
            overlay_files = glob.glob(f"{overlay_type}_*.png")
            if overlay_files:
                selected_overlay = random.choice(overlay_files)
                logger.info(f"Selected overlay for all frames: {selected_overlay}")
        
        for i, frame in enumerate(frames):
            temp_in = f"downloads/temp_{uuid4()}.jpg"
            temp_out = f"processed/temp_{uuid4()}.jpg"
            
            try:
                cv2.imwrite(temp_in, frame)
                
                # Detect faces in each frame to follow the movement
                current_faces = detect_heads(frame)
                
                if process_func(temp_in, temp_out, overlay_type, selected_overlay=selected_overlay, faces=current_faces):
                    processed = cv2.imread(temp_out)
                    if processed is not None:
                        processed_frames.append(processed)
                    else:
                        processed_frames.append(frame)
                else:
                    processed_frames.append(frame)
            finally:
                for temp in [temp_in, temp_out]:
                    if os.path.exists(temp):
                        os.remove(temp)
        
        # Convert to RGB for imageio
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames]
        
        # Write GIF using imageio
        imageio.mimsave(output_path, rgb_frames, format='GIF', fps=fps)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        logger.error(f"Error processing GIF: {str(e)}")
        logger.error(traceback.format_exc())
        return False
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

logger = logging.getLogger(__name__)

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
        overlay_type = kwargs.pop('action', 'pixelate')
        
        for i, frame in enumerate(frames):
            temp_in = f"downloads/temp_{uuid4()}.jpg"
            temp_out = f"processed/temp_{uuid4()}.jpg"
            
            try:
                # Save frame as JPEG
                cv2.imwrite(temp_in, frame)
                
                # Process with the provided function
                if process_func(temp_in, temp_out, overlay_type):
                    processed = cv2.imread(temp_out)
                    if processed is not None:
                        processed_frames.append(processed)
                    else:
                        processed_frames.append(frame)
                else:
                    processed_frames.append(frame)
            finally:
                # Clean up temp files
                for temp in [temp_in, temp_out]:
                    if os.path.exists(temp):
                        os.remove(temp)

        # Convert to RGB for imageio
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames]
        
        # Write GIF using imageio with simple parameters
        try:
            imageio.mimsave(output_path, rgb_frames, format='GIF', fps=fps)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            logger.error(f"Failed to save GIF with imageio: {str(e)}")
            
            # Fallback: try saving with PIL
            try:
                pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    optimize=False,
                    duration=int(1000/fps),
                    loop=0
                )
                return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            except Exception as e2:
                logger.error(f"Failed to save GIF with PIL: {str(e2)}")
                return False
        
    except Exception as e:
        logger.error(f"Error processing GIF: {str(e)}")
        logger.error(traceback.format_exc())
        return False
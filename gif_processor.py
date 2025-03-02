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
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames
        
    def read_gif(self, input_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Read GIF and return frames with durations"""
        try:
            reader = imageio.get_reader(input_path, format='GIF')
            frames = []
            durations = []
            for frame in reader:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                durations.append(frame.meta.get('duration', 100) / 1000.0)
            return frames, durations
        except Exception as e:
            logger.error(f"Error reading GIF: {str(e)}")
            return [], []

    def save_gif(self, frames: List[np.ndarray], output_path: str, duration: float) -> bool:
        """Save processed frames as GIF"""
        try:
            # Convert frames back to RGB
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            
            imageio.mimsave(
                output_path,
                rgb_frames,
                duration=duration,
                loop=0
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving GIF: {str(e)}")
            return False

    def process_frames(self, frames: List[np.ndarray], process_func, **kwargs) -> List[np.ndarray]:
        """Process each frame using the provided processing function"""
        processed_frames = []
        
        for frame in frames:
            try:
                # Create temporary paths for single frame processing
                temp_frame_path = f"temp_frame_{uuid4()}.jpg"
                temp_output_path = f"temp_output_{uuid4()}.jpg"
                
                # Save frame temporarily
                cv2.imwrite(temp_frame_path, frame)
                
                # Process frame using existing photo processing function
                if process_func(temp_frame_path, temp_output_path, **kwargs):
                    processed_frame = cv2.imread(temp_output_path)
                    if processed_frame is not None:
                        processed_frames.append(processed_frame)
                
                # Cleanup temp files
                os.remove(temp_frame_path)
                os.remove(temp_output_path)
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue
                
        return processed_frames

def process_telegram_gif(input_path: str, output_path: str, process_func, **kwargs) -> bool:
    processor = GifProcessor()
    frames, durations = processor.read_gif(input_path)
    
    if not frames:
        return False
        
    processed_frames = []
    for frame in frames:
        # Create temp files for frame processing
        temp_in = f"downloads/temp_{uuid4()}.jpg"
        temp_out = f"processed/temp_{uuid4()}.jpg"
        
        cv2.imwrite(temp_in, frame)
        
        if process_func(temp_in, temp_out, **kwargs):
            processed = cv2.imread(temp_out)
            if processed is not None:
                processed_frames.append(processed)
            else:
                processed_frames.append(frame)
        else:
            processed_frames.append(frame)
            
        # Cleanup temps
        for temp in [temp_in, temp_out]:
            if os.path.exists(temp):
                os.remove(temp)
    
    if not processed_frames:
        return False
        
    return processor.save_gif(processed_frames, output_path, durations[0] if durations else 0.1)
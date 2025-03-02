import cv2
import random
import numpy as np

PIXELATION_FACTOR = 0.04

def detect_heads(image):
    height, width = image.shape[:2]
    scale = min(1.0, 800 / max(height, width))
    if scale < 1.0:
        small = cv2.resize(image, (int(width * scale), int(height * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    faces_with_angles = []
    for (x, y, w, h) in faces:
        if scale < 1.0:
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        angle = random.uniform(-15, 15)
        faces_with_angles.append({
            'rect': (x, y, w, h),
            'angle': angle
        })
    
    return faces_with_angles 
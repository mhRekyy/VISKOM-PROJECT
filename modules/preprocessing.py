import cv2
import imutils
import numpy as np  # Kita butuh numpy untuk memperbaiki format data

def prepare_image(frame, width=450):
    """
    Melakukan preprocessing gambar.
    """
    # 1. Resize gambar (Materi PDF 07)
    frame = imutils.resize(frame, width=width)
    
    # 2. Convert ke Grayscale (Materi PDF 03)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- PERBAIKAN ERROR ---
    # Paksa agar data menjadi 8-bit integer dan memorinya 'contiguous' (rapi)
    # Ini wajib dilakukan agar Dlib mau membacanya.
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    
    return frame, gray
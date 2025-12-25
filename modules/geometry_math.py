import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye):
    """
    Menghitung Eye Aspect Ratio (EAR) menggunakan Euclidean Distance.
    Rumus ini menerapkan materi: Math in Computer Vision (Vektor & Jarak).
    """
    # Hitung jarak vertikal antara kelopak mata (p2-p6 dan p3-p5)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Hitung jarak horizontal (p1-p4)
    C = dist.euclidean(eye[0], eye[3])

    # Hitung rasio
    ear = (A + B) / (2.0 * C)
    return ear

def get_3d_model_points():
    """
    Titik-titik model wajah generik dalam ruang 3D (X, Y, Z).
    Digunakan untuk Head Pose Estimation (3D Reconstruction).
    """
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Hidung
        (0.0, -330.0, -65.0),        # Dagu
        (-225.0, 170.0, -135.0),     # Ujung mata kiri
        (225.0, 170.0, -135.0),      # Ujung mata kanan
        (-150.0, -150.0, -125.0),    # Ujung mulut kiri
        (150.0, -150.0, -125.0)      # Ujung mulut kanan
    ], dtype="double")
    return model_points
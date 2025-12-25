import cv2
import time
import dlib
import numpy as np
import winsound
from imutils import face_utils
from modules import geometry_math

# --- KONFIGURASI ---
THRESH_EAR = 0.20
FRAME_CHECK = 50
PATH_HAAR = "assets/haarcascade_frontalface_default.xml"
PATH_PREDICTOR = "assets/shape_predictor_68_face_landmarks.dat"
PATH_ALARM = "assets/crowing.wav"

# --- INISIALISASI MODEL ---
print("[INFO] Memuat model detektor...")
try:
    detector_haar = cv2.CascadeClassifier(PATH_HAAR)
    if detector_haar.empty():
        raise IOError("File Haar Cascade xml tidak ditemukan!")
    
    predictor = dlib.shape_predictor(PATH_PREDICTOR)
except Exception as e:
    print(f"[ERROR] {e}")
    print("Cek folder 'assets' Anda!")
    exit()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] Memulai kamera...")
cap = cv2.VideoCapture(0)
counter_drowsy = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret: break # Jika kamera disconnect, berhenti

    # --- PENANGANAN FRAME (Safety First) ---
    try:
        # 1. Resize Frame
        height, width = frame.shape[:2]
        new_width = 450
        ratio = new_width / float(width)
        new_height = int(height * ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        # 2. Convert ke Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. PERBAIKAN EXTRA: Paksa memori menjadi 'Contiguous' (Rapi)
        # Ini obat manjur untuk error "Unsupported image type"
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        # --- DETEKSI WAJAH ---
        rects_haar = detector_haar.detectMultiScale(gray, scaleFactor=1.1, 
                                                    minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects_haar:
            # --- TAMBAHAN 1: Gambar Kotak Wajah ---
            # Agar Anda tahu kalau wajah sudah terdeteksi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            # Buat kotak dlib
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            # --- "SHOCKBREAKER" (Try-Except Block) ---
            try:
                shape = predictor(gray, rect) 
                shape = face_utils.shape_to_np(shape)
            except RuntimeError as e: # --- TAMBAHAN 2: Tangkap pesan errornya
                print(f"[DEBUG] Dlib Error: {e}") # Cetak ke terminal biar ketahuan
                continue

            # Ambil mata
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Hitung EAR
            leftEAR = geometry_math.calculate_ear(leftEye)
            rightEAR = geometry_math.calculate_ear(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # Gambar Mata
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # --- LOGIKA KANTUK BARU ---
            if avgEAR < THRESH_EAR:
                counter_drowsy += 1

                # Jika waktu sudah terlewati
                if counter_drowsy >= FRAME_CHECK:
                    # Tampilkan Teks
                    cv2.putText(frame, "BAHAYA: MENGANTUK!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Cek: Apakah alarm sudah nyala? 
                    if not alarm_on:
                        try:
                            # Putar suara dengan mode LOOP (berulang terus) + ASYNC (background)
                            winsound.PlaySound(PATH_ALARM, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
                            alarm_on = True # Tandai bahwa alarm sedang bunyi
                        except:
                            # Fallback jika gagal (Beep windows tidak bisa di-loop asinkronus dengan mudah)
                            winsound.Beep(2500, 100) 
            
            else:
                # Jika mata terbuka (EAR Normal)
                counter_drowsy = 0
                
                # Jika alarm sebelumnya sedang nyala, MATIKAN SEKARANG
                if alarm_on:
                    winsound.PlaySound(None, winsound.SND_PURGE) # Stop suara
                    alarm_on = False # Reset status

            # --- HEAD POSE (3D Reconstruction) + ALARM DISTRAKSI ---
            try:
                # 1. Ambil titik model & image (Sama seperti sebelumnya)
                model_points = geometry_math.get_3d_model_points()
                image_points = np.array([
                    shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
                ], dtype="double")

                # 2. Parameter Kamera
                size = frame.shape
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype = "double")
                dist_coeffs = np.zeros((4,1))

                # 3. Hitung PnP
                (success, rot_vec, trans_vec) = cv2.solvePnP(model_points, 
                                                            image_points, 
                                                            camera_matrix, 
                                                            dist_coeffs, 
                                                            flags=cv2.SOLVEPNP_ITERATIVE)

                # 4. Proyeksi
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                                 rot_vec, trans_vec, 
                                                                 camera_matrix, dist_coeffs)

                # 5. Koordinat Garis
                p1 = ( int(image_points[0][0]), int(image_points[0][1])) 
                point_3d = nose_end_point2D.ravel() 
                p2 = ( int(point_3d[0]), int(point_3d[1]))
                
                # Gambar Garis Biru (Visual)
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # --- LOGIKA BARU: ALARM DISTRAKSI ---
                # Hitung jarak horizontal (X) antara ujung hidung dan pangkal hidung
                # Jika jaraknya > 100 pixel, berarti menoleh tajam
                dist_x = p2[0] - p1[0]
                
                if abs(dist_x) > 70: # <-- Angka sensitivitas 
                    cv2.putText(frame, "FOKUS KE JALAN!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # Bunyikan Alarm (Sama seperti logika ngantuk)
                    if not alarm_on:
                        try:
                            winsound.PlaySound(PATH_ALARM, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
                            alarm_on = True
                        except:
                            winsound.Beep(3000, 100) # Nada beda (lebih tinggi) biar ketahuan bedanya
                
                # Matikan alarm jika TIDAK menoleh DAN TIDAK ngantuk
                elif counter_drowsy == 0 and alarm_on:
                     # Cek double: pastikan mata juga sedang melek (counter_drowsy == 0)
                     winsound.PlaySound(None, winsound.SND_PURGE)
                     alarm_on = False

            except Exception as e:
                print(f"[ERROR 3D]: {e}") 

    except Exception as e:
        # Jika ada error aneh lain di luar Dlib, print saja jangan crash
        print(f"Skipping bad frame: {e}")
        continue

    cv2.imshow("Driver Vigilance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
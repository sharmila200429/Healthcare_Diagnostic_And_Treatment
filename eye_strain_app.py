import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import mediapipe as mp
import numpy as np
import time

# --------- EAR calculation ---------
def calculate_EAR(eye_points, landmarks, img_w, img_h):
    def get_point(index):
        return np.array([landmarks[index].x * img_w, landmarks[index].y * img_h])

    p1 = get_point(eye_points[0])
    p2 = get_point(eye_points[1])
    p3 = get_point(eye_points[2])
    p4 = get_point(eye_points[3])
    p5 = get_point(eye_points[4])
    p6 = get_point(eye_points[5])

    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# --------- Landmark indices ---------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --------- Mediapipe Setup ---------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# --------- Webcam ---------
cap = cv2.VideoCapture(0)

# --------- Variables ---------
blink_count = 0
blinks_in_minute = 0
CLOSED_FRAMES = 0
CLOSED_FRAME_DURATION = 0
EAR_THRESHOLD = 0.21

start_time = time.time()
blink_start_time = start_time

blink_rate_history = []
ear_history = []
closed_duration_history = []

# --------- Alert Function ---------
def show_alert():
    messagebox.showinfo("Eye Strain Alert", "Take a break! \U0001F4A1 Follow the 20-20-20 rule!")

# --------- Improved stress level calculation ---------
def calculate_stress_level(blinks_per_minute, ear, avg_eye_closed_duration):
    stress_score = 0

    # Blink Rate contribution
    if blinks_per_minute < 10:
        stress_score += 20
    elif 10 <= blinks_per_minute <= 18:
        stress_score += 0
    elif 18 < blinks_per_minute <= 25:
        stress_score += 20
    else:
        stress_score += 30

    # EAR contribution
    if ear >= 0.25:
        stress_score += 0
    elif 0.21 <= ear < 0.25:
        stress_score += 15
    else:
        stress_score += 30

    # Eye closure duration
    if avg_eye_closed_duration > 0.4:
        stress_score += 30
    elif avg_eye_closed_duration > 0.25:
        stress_score += 15

    return min(stress_score, 100)

# --------- Function to calculate average stress over time ---------
def calculate_average_stress():
    global blink_rate_history, ear_history, closed_duration_history

    if blink_rate_history and ear_history and closed_duration_history:
        avg_blink_rate = np.mean(blink_rate_history)
        avg_ear = np.mean(ear_history)
        avg_closed_duration = np.mean(closed_duration_history)

        stress_level = calculate_stress_level(avg_blink_rate, avg_ear, avg_closed_duration)

        blink_rate_history = []
        ear_history = []
        closed_duration_history = []

        return stress_level
    return 0

# --------- Function to run the eye strain detector ---------
def run_eye_strain_detector():
    global blink_count, blinks_in_minute, CLOSED_FRAMES, blink_start_time, start_time, CLOSED_FRAME_DURATION

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                img_h, img_w, _ = frame.shape
                left_ear = calculate_EAR(LEFT_EYE, landmarks.landmark, img_w, img_h)
                right_ear = calculate_EAR(RIGHT_EYE, landmarks.landmark, img_w, img_h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    CLOSED_FRAMES += 1
                else:
                    if CLOSED_FRAMES > 3:
                        blink_count += 1
                        blinks_in_minute += 1
                        closed_duration = CLOSED_FRAMES / fps
                        CLOSED_FRAME_DURATION += closed_duration
                    CLOSED_FRAMES = 0

                current_time = time.time()
                elapsed_min = current_time - blink_start_time
                total_elapsed = current_time - start_time

                if elapsed_min > 60:
                    blink_rate_history.append(blinks_in_minute)
                    ear_history.append(avg_ear)
                    closed_duration_history.append(CLOSED_FRAME_DURATION / max(blinks_in_minute, 1))

                    stress_level = calculate_average_stress()
                    update_stress_label(stress_level)

                    blink_start_time = current_time
                    blinks_in_minute = 0
                    CLOSED_FRAME_DURATION = 0

                if total_elapsed > 1 * 60:
                    show_alert()
                    start_time = current_time

                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Blinks: {blink_count}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Eye Strain Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# --------- GUI Setup ---------
def start_detector():
    messagebox.showinfo("Info", "Eye Strain Detector Started!")
    threading.Thread(target=run_eye_strain_detector, daemon=True).start()

def stop_detector():
    messagebox.showinfo("Info", "Eye Strain Detector Stopped!")
    cap.release()

def update_stress_label(stress_level):
    if stress_level < 30:
        stress_label.config(text=f"Stress Level: Low ({stress_level}%)", fg="green")
    elif stress_level < 60:
        stress_label.config(text=f"Stress Level: Moderate ({stress_level}%)", fg="orange")
    else:
        stress_label.config(text=f"Stress Level: High ({stress_level}%)", fg="red")

root = tk.Tk()
root.title("Eye Strain Detector")
root.geometry("400x400")
root.config(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

label = tk.Label(root, text="Eye Strain Detector", font=("Arial", 18, "bold"), bg="#f0f0f0")
label.pack(pady=20)

status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 12), bg="#f0f0f0")
status_label.pack(pady=10)

stress_label = tk.Label(root, text="Stress Level: Low", font=("Arial", 14), bg="#f0f0f0")
stress_label.pack(pady=10)

start_button = tk.Button(frame, text="Start", width=20, height=2, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", command=start_detector)
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(frame, text="Stop", width=20, height=2, font=("Arial", 12, "bold"), bg="#f44336", fg="white", command=stop_detector)
stop_button.grid(row=0, column=1, padx=10)

root.mainloop()

# unified_atm_system.py

import cv2
import numpy as np
import time
import threading
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

# -------------------------
# CONFIG (update paths)
# -------------------------
FACE_DNN_PROTO = "face_detector/deploy.prototxt"
FACE_DNN_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MASK_MODEL_PATH = "mask_detector.model"
AGE_PROTO = "age_models/age_deploy.prototxt"
AGE_MODEL = "age_models/age_net.caffemodel"
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
TTS_COOLDOWN = 5.0
EVAL_INTERVAL = 3.0  # seconds between access checks

# -------------------------
# TTS SETUP
# -------------------------
tts_engine = None
tts_lock = threading.Lock()
last_tts_time = 0.0

def init_tts(rate=150):
    global tts_engine
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', rate)

def speak_async(text):
    global last_tts_time
    now = time.time()
    if now - last_tts_time < TTS_COOLDOWN:
        return
    last_tts_time = now
    threading.Thread(target=_speak, args=(text,), daemon=True).start()

def _speak(text):
    with tts_lock:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

# -------------------------
# MODEL LOADING
# -------------------------
def load_face_dnn(proto, model):
    if os.path.exists(proto) and os.path.exists(model):
        return cv2.dnn.readNetFromCaffe(proto, model)
    return None

def load_face_haar(path):
    return cv2.CascadeClassifier(path)

def load_mask_model(path):
    return load_model(path)

def load_age_model(proto, model):
    if os.path.exists(proto) and os.path.exists(model):
        return cv2.dnn.readNetFromCaffe(proto, model)
    return None

# -------------------------
# FACE DETECTION
# -------------------------
def detect_faces_dnn(frame, net, conf_thresh=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = box.astype("int")
            sx, sy, ex, ey = max(0, sx), max(0, sy), min(w-1, ex), min(h-1, ey)
            faces.append((sx, sy, ex, ey))
    return faces

def detect_faces_haar(gray, face_cascade):
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    faces = [(x, y, x+w, y+h) for (x, y, w, h) in rects]
    return faces

# -------------------------
# MASK PREDICTION
# -------------------------
def predict_mask(face_img, mask_model):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    preds = mask_model.predict(face)
    mask, no_mask = preds[0]
    return mask > no_mask, max(mask, no_mask)

# -------------------------
# AGE PREDICTION
# -------------------------
def predict_age(face_img, age_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    idx = preds[0].argmax()
    age_bucket = AGE_BUCKETS[idx]
    nums = age_bucket.strip('()').split('-')
    est_age = (float(nums[0]) + float(nums[1])) / 2.0
    return age_bucket, est_age

def map_age_to_group(age):
    if age < 18: return "child"
    elif age >= 60: return "senior"
    else: return "adult"

# -------------------------
# ACCESS LOGIC
# -------------------------
def evaluate_access(detections):
    if any(d['mask'] for d in detections):
        return False, "Access denied: face covering detected. Please remove mask."
    total = len(detections)
    if total == 0:
        return False, "No person detected. Please step in front of the ATM."
    if total == 1:
        return True, "Access granted. Please proceed."
    if total == 2:
        groups = [d['age_group'] for d in detections]
        if "adult" in groups and ("child" in groups or "senior" in groups):
            return True, "Access granted for adult and child/senior."
        return False, "Access denied: more than one unauthorized adult present."
    return False, "Access denied: too many people at the ATM."

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    init_tts()

    face_net = load_face_dnn(FACE_DNN_PROTO, FACE_DNN_MODEL)
    face_cascade = None if face_net else load_face_haar(HAAR_PATH)
    mask_model = load_mask_model(MASK_MODEL_PATH)
    age_net = load_age_model(AGE_PROTO, AGE_MODEL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    last_eval = 0

    while True:
        ret, frame = cap.read()
        if not ret: continue

        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_dnn(frame, face_net) if face_net else detect_faces_haar(gray, face_cascade)
        detections = []

        for (sx, sy, ex, ey) in faces:
            face_img = orig[sy:ey, sx:ex]
            if face_img.size == 0: continue

            mask, mask_prob = predict_mask(face_img, mask_model)
            age_bucket, age_est = (None, 30.0)
            if age_net:
                try:
                    age_bucket, age_est = predict_age(face_img, age_net)
                except: pass
            age_group = map_age_to_group(age_est)

            detections.append({
                'box': (sx, sy, ex, ey),
                'mask': mask,
                'mask_prob': mask_prob,
                'age_bucket': age_bucket,
                'age_est': age_est,
                'age_group': age_group
            })

            label = f"{age_group} {'MASK' if mask else 'NO-MASK'}"
            color = (0, 0, 255) if mask else (0, 255, 0)
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            cv2.putText(frame, label, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        now = time.time()
        if now - last_eval > EVAL_INTERVAL:
            allow, message = evaluate_access(detections)
            print("[EVAL]", message)
            speak_async(message)
            last_eval = now
            if not allow:
                ts = int(now)
                cv2.imwrite(f"evidence_{ts}.jpg", orig)

        cv2.imshow("ATM Access Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

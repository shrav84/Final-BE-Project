# unified_system.py
import cv2
import numpy as np
import time
import argparse
import threading
import os
from collections import deque
import pyttsx3

# -------------------------
# CONFIG (update paths)
# -------------------------
DEFAULT_FACE_DNN_PROTO = "face_detector\deploy.prototxt"
DEFAULT_FACE_DNN_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
DEFAULT_HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Mask model (Keras .h5 or saved_model)
DEFAULT_MASK_MODEL = "mask_detector.model"  # update to your actual file

# Age model - Caffe (common pattern)
DEFAULT_AGE_PROTO = "age_models/age_deploy.prototxt"
DEFAULT_AGE_MODEL = "age_models/age_net.caffemodel"


# Age buckets (common Caffe model mapping)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# TTS throttle (seconds)
TTS_COOLDOWN = 5.0

# -------------------------
# UTILS: TTS (non-blocking)
# -------------------------
tts_engine = None
tts_lock = threading.Lock()
last_tts_time = 0.0

def init_tts(rate=150):
    global tts_engine
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', rate)

def speak_async(text):
    """Spawn a thread to speak text, but throttle to avoid repeated alerts"""
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
# MODEL LOADING HELPERS
# -------------------------
def load_mask_model(mask_model_path):
    # Try load via Keras (works for .h5 or SavedModel)
    try:
        from tensorflow.keras.models import load_model
        model = load_model(mask_model_path)
        print("[INFO] Loaded mask model (Keras) from", mask_model_path)
        return ("keras", model)
    except Exception as e:
        print("[WARN] Could not load Keras model:", e)
        return (None, None)

def load_age_caffe(proto_path, model_path):
    if os.path.exists(proto_path) and os.path.exists(model_path):
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        print("[INFO] Loaded age model (Caffe)")
        return net
    print("[WARN] Age Caffe files not found:", proto_path, model_path)
    return None

# If you converted to TFLITE, use tflite interpreter:
def load_tflite_interpreter(tflite_path):
    try:
        import tflite_runtime.interpreter as tflite
    except Exception:
        try:
            from tensorflow.lite import Interpreter as tflite
        except Exception as e:
            print("[WARN] tflite runtime not available:", e)
            return None
    interp = tflite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    print("[INFO] Loaded TFLite model:", tflite_path)
    return interp

# -------------------------
# PREDICTION HELPERS
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
            (startX, startY, endX, endY) = box.astype("int")
            # clamp
            startX = max(0, startX); startY = max(0, startY)
            endX = min(w-1, endX); endY = min(h-1, endY)
            faces.append((startX, startY, endX, endY, confidence))
    return faces

def detect_faces_haar(gray, face_cascade):
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = []
    for (x, y, w, h) in rects:
        faces.append((x, y, x+w, y+h, None))
    return faces

def predict_mask_on_face(face_img, mask_model_tuple):
    model_type, model = mask_model_tuple
    if model_type == "keras":
        # Typical preprocessing for many mask models: resize to 224x224, scale /255
        blob = cv2.resize(face_img, (224, 224))
        blob = blob.astype("float32") / 255.0
        blob = np.expand_dims(blob, axis=0)
        preds = model.predict(blob)
        # many mask models output: [mask_prob, no_mask_prob] or vice versa. Try common pattern:
        # If shape (1,2) assume [mask, no_mask]
        if preds.shape[-1] == 2:
            mask_prob = float(preds[0][0])
            no_mask_prob = float(preds[0][1])
            mask_present = mask_prob > no_mask_prob
            return mask_present, mask_prob
        # fallback: if single scalar, interpret >0.5 as masked
        val = preds.ravel()[0]
        return (val > 0.5), float(val)
    else:
        return False, 0.0

def predict_age_with_caffe(face_img, age_net):
    # preprocessing: resize to 227x227, mean subtraction as original models expect
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    i = preds[0].argmax()
    age_bucket = AGE_BUCKETS[i]
    # Convert bucket to midpoint approximate age
    # e.g. '(25-32)' -> 28.5
    nums = age_bucket.strip('()').split('-')
    est_age = (float(nums[0]) + float(nums[1])) / 2.0
    return age_bucket, est_age

# -------------------------
# ACCESS LOGIC
# -------------------------
def map_age_to_group(age_est):
    # simple thresholds
    if age_est < 18:
        return "child"
    elif age_est >= 60:
        return "senior"
    else:
        return "adult"

def evaluate_access(detections):
    """
    detections: list of dicts with keys: {'mask':bool, 'age_est':float, 'age_group':str}
    Returns: (allow:bool, message:str)
    Rules:
      - Deny if anyone is wearing a mask or has a head covering (head_covering detection optional)
      - Only one person allowed unless the second person is a child or a senior (60+)
    """
    # Deny if any mask present
    for d in detections:
        if d.get('mask', False):
            return False, "Access denied: face covering detected. Please remove mask or head covering."

    total = len(detections)
    # If no people detected -> deny (or wait)
    if total == 0:
        return False, "No person detected. Please step in front of the ATM."

    if total == 1:
        return True, "Access granted. Please proceed."

    # total >= 2
    # if total == 2 and one is child or senior -> allow (assuming they accompanied an adult)
    if total == 2:
        groups = [d['age_group'] for d in detections]
        # If at least one adult present and the other is child/senior -> allow
        if ("adult" in groups) and (("child" in groups) or ("senior" in groups)):
            return True, "Access granted for adult and child/senior."
        else:
            return False, "Access denied: more than one unauthorized adult present."

    # more than 2 people — deny
    return False, "Access denied: too many people at the ATM."

# -------------------------
# MAIN
# -------------------------
def main(args):
    init_tts()
    print("[INFO] Starting unified system")
    # Load face detector (DNN preferred)
    face_net = None
    use_dnn = False
    if os.path.exists(args.face_proto) and os.path.exists(args.face_model):
        try:
            face_net = cv2.dnn.readNetFromCaffe(args.face_proto, args.face_model)
            use_dnn = True
            print("[INFO] Using OpenCV DNN face detector")
        except Exception as e:
            print("[WARN] Face DNN load failed:", e)

    # Haar fallback
    face_cascade = None
    if not use_dnn:
        face_cascade = cv2.CascadeClassifier(args.haar)
        print("[INFO] Using Haar Cascade face detector (fallback)")

    # Load mask model
    mask_model_tuple = load_mask_model(args.mask_model)

    # Load age model (Caffe)
    age_net = load_age_caffe(args.age_proto, args.age_model)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Could not open camera index", args.camera)
        return

    last_eval_time = 0
    eval_interval = args.eval_interval  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame grab failed")
            time.sleep(0.1)
            continue

        orig = frame.copy()
        (h, w) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = []
        if use_dnn and face_net is not None:
            faces = detect_faces_dnn(frame, face_net, conf_thresh=0.5)
        else:
            faces = detect_faces_haar(gray, face_cascade)

        detections = []
        for (sx, sy, ex, ey, conf) in faces:
            face = orig[sy:ey, sx:ex]
            if face.size == 0:
                continue
            # mask prediction
            mask_present, mask_prob = predict_mask_on_face(face, mask_model_tuple)
            # age prediction
            age_bucket, age_est = (None, None)
            if age_net is not None:
                try:
                    age_bucket, age_est = predict_age_with_caffe(face, age_net)
                except Exception as e:
                    print("Age predict error:", e)
            # fallback if age not available
            if age_est is None:
                age_est = 30.0
            age_group = map_age_to_group(age_est)

            detections.append({
                'box': (sx, sy, ex, ey),
                'mask': mask_present,
                'mask_prob': mask_prob,
                'age_bucket': age_bucket,
                'age_est': age_est,
                'age_group': age_group
            })

            # draw boxes & text
            label = f"{age_group} {'MASK' if mask_present else 'NO-MASK'}"
            color = (0, 0, 255) if mask_present else (0, 255, 0)
            cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
            cv2.putText(frame, label, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Evaluate access at intervals to avoid per-frame flipping
        now = time.time()
        if now - last_eval_time > eval_interval:
            allow, message = evaluate_access(detections)
            print("[EVAL]", message)
            # Trigger TTS alert if deny or informational
            speak_async(message)
            last_eval_time = now
            # Optionally save image evidence on violations
            if not allow:
                ts = int(now)
                evidence_path = f"evidence_{ts}.jpg"
                cv2.imwrite(evidence_path, orig)
                print("[INFO] Saved evidence:", evidence_path)

        cv2.imshow("ATM Access Control", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------
# ARGPARSE & RUN
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--face-proto", default=DEFAULT_FACE_DNN_PROTO)
    ap.add_argument("--face-model", default=DEFAULT_FACE_DNN_MODEL)
    ap.add_argument("--haar", default=DEFAULT_HAAR)
    ap.add_argument("--mask-model", default=DEFAULT_MASK_MODEL)
    ap.add_argument("--age-proto", default=DEFAULT_AGE_PROTO)
    ap.add_argument("--age-model", default=DEFAULT_AGE_MODEL)
    ap.add_argument("--eval-interval", type=float, default=3.0,
                    help="seconds between applying access logic and speaking")
    args = ap.parse_args()
    main(args)

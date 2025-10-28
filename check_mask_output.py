# check_age_cropped.py
import cv2, numpy as np, os

# paths (adjust if your repo uses different names)
FACE_PROTO = "face_detector/deploy.prototxt"
FACE_MODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "age_models/age_deploy.prototxt"
AGE_MODEL = "age_models/age_net.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

TEST_IMAGE = "oldwoman.webp"   # <--- change to the file you used (oldwoman.webp) or any other

print("cwd:", os.getcwd())
print("Checking files exist:", os.path.exists(FACE_PROTO), os.path.exists(FACE_MODEL), os.path.exists(AGE_PROTO), os.path.exists(AGE_MODEL))
# Load nets
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net  = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
print("Loaded face and age nets.")

img = cv2.imread(TEST_IMAGE)
if img is None:
    raise SystemExit(f"Couldn't load test image: {TEST_IMAGE}")

(h, w) = img.shape[:2]

# Detect faces using the same DNN you use in unified_system
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
                             (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

# pick the detection with max confidence
best_conf = 0.0
best_box = None
for i in range(detections.shape[2]):
    conf = float(detections[0, 0, i, 2])
    if conf > 0.4 and conf > best_conf:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (sx, sy, ex, ey) = box.astype("int")
        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(w - 1, ex), min(h - 1, ey)
        best_conf = conf
        best_box = (sx, sy, ex, ey)

if best_box is None:
    raise SystemExit("No face found by DNN in the test image (try a different image).")

sx, sy, ex, ey = best_box
face_crop = img[sy:ey, sx:ex].copy()
if face_crop.size == 0:
    raise SystemExit("Face crop is empty — check bounding box.")

# Save the crop so you can inspect it
cv2.imwrite("debug_face_crop.jpg", face_crop)
print("Saved face crop to debug_face_crop.jpg  (open this to visually inspect)")

# Preprocess for Caffe age model (BGR, resize 227x227, mean subtraction, swapRB=False)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
age_net.setInput(age_blob)
preds = age_net.forward()
preds = preds.flatten()
idx = int(preds.argmax())
print("Age softmax vector:", preds)
print("Chosen index:", idx, "bucket:", AGE_BUCKETS[idx], "confidence:", float(preds[idx]))

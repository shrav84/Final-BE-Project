# age_demo.py
import cv2
import numpy as np

AGE_PROTO = "age_models/age_deploy.prototxt"
AGE_MODEL = "age_models/age_net.caffemodel"
AGE_BUCKETS = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']

# load models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
# choose camera 0 — change if your webcam index differs
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # use OpenCV face detector if you have it (or simple haar cascade)
    # For speed we use a simple Haar cascade fallback if face_detector not present
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x,y,wf,hf) in faces:
        face = frame[y:y+hf, x:x+wf].copy()
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        idx = int(np.argmax(preds))
        label = AGE_BUCKETS[idx]
        prob = float(preds[0][idx])
        cv2.rectangle(frame, (x,y), (x+wf, y+hf), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        break  # ATM single-person case: just the largest face

    cv2.imshow("Age Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

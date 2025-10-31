import tensorflow as tf
import os

model_path = 'mask_detector.model'

if os.path.isdir(model_path):
    print("📁 Detected a directory — checking for TensorFlow SavedModel format...")
    if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
        print("✅ This is a TensorFlow SavedModel.")
    else:
        print("❌ Directory found, but not a valid SavedModel (missing saved_model.pb).")

elif os.path.isfile(model_path):
    print("📄 Detected a file — trying to load as Keras model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Successfully loaded as a Keras model (HDF5 or .model format).")
        print(model.summary())
    except Exception as e:
        print("❌ Not a valid Keras model file.")
        print("Error message:", e)
else:
    print("❌ Path not found:", model_path)

import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model('mask_detector.model')

# Create a converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ Apply optimization settings before converting
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert to TensorFlow Lite model (Float16 optimized)
tflite_fp16_model = converter.convert()

# Save the converted model to file
with open('mask_detector_fp16.tflite', 'wb') as f:
    f.write(tflite_fp16_model)

print("✅ Converted to mask_detector_fp16.tflite (optimized for Raspberry Pi)")

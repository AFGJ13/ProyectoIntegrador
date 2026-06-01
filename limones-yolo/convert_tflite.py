import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

# Optimización (IMPORTANTE para Raspberry)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite listo")
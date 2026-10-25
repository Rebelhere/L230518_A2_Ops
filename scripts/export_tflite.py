import shutil, tempfile
import tensorflow as tf
from tensorflow import keras

# champion.keras (from export_champion.py) -> champion.tflite for the Vercel API (TensorFlow does not fit there)
model = keras.models.load_model("models/champion.keras")
tmp = tempfile.mkdtemp()
model.export(tmp, format="tf_saved_model")                 # preprocess_input ([-1, 1]) stays inside the graph
conv = tf.lite.TFLiteConverter.from_saved_model(tmp)       # float32, no quantisation: keep parity with Keras
open("models/champion.tflite", "wb").write(conv.convert())
shutil.rmtree(tmp, ignore_errors=True)
print("saved models/champion.tflite")

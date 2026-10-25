import glob, os, sys
import tensorflow as tf
from mlflow import MlflowClient
from tensorflow import keras
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
c = MlflowClient()
mv = c.get_model_version_by_alias(NAME, "champion")
thr = float(mv.tags["threshold"])
os.makedirs("models/champion_dl", exist_ok=True)
folder = c.download_artifacts(mv.run_id, "model_file", "models/champion_dl")
model = keras.models.load_model(glob.glob(folder + "/**/*.keras", recursive=True)[0])
print(f"champion = version {mv.version} (run {mv.tags['run_name']}), registry threshold {thr:.3f}")
img = tf.io.decode_image(tf.io.read_file(sys.argv[1]), channels=3, expand_animations=False)
img = tf.image.resize(img, (224, 224))
p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0][0])
print(f"{sys.argv[1]}\np(AI) = {p:.3f} -> {'AI-generated' if p >= thr else 'Real'}")
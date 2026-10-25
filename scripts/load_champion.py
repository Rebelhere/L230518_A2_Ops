import sys
import mlflow, tensorflow as tf
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
run = MlflowClient().get_run(mv.run_id)
thr = run.data.metrics["threshold"]
print(f"champion = version {mv.version} (run {run.info.run_name}), threshold {thr:.3f}")
model = mlflow.keras.load_model(f"models:/{NAME}@champion")
img = tf.io.decode_image(tf.io.read_file(sys.argv[1]), channels=3, expand_animations=False)
img = tf.image.resize(img, (224, 224))
p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0][0])
print(f"p(AI) = {p:.3f} -> {'AI-generated' if p >= thr else 'Real'}")
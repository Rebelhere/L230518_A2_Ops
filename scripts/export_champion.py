import json
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
thr = MlflowClient().get_run(mv.run_id).data.metrics["threshold"]
mlflow.keras.load_model(f"models:/{NAME}@champion").save("models/champion.keras")
json.dump({"version": mv.version, "run_id": mv.run_id, "threshold": thr}, open("models/champion_meta.json", "w"))
print("saved models/champion.keras, threshold", round(thr, 3))
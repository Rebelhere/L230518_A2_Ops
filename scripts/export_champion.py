import glob, json, os, shutil
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
c = MlflowClient()
mv = c.get_model_version_by_alias(NAME, "champion")
os.makedirs("models/champion_dl", exist_ok=True)
folder = c.download_artifacts(mv.run_id, "model_file", "models/champion_dl")
shutil.copy(glob.glob(folder + "/**/*.keras", recursive=True)[0], "models/champion.keras")
json.dump({"version": mv.version, "run_id": mv.run_id, "registry_threshold": float(mv.tags["threshold"])},
          open("models/champion_meta.json", "w"))
print("saved models/champion.keras from registry version", mv.version)
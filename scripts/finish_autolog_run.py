import mlflow
from dotenv import load_dotenv

load_dotenv()
RUN_ID = "ad6817fee12f4d4cb4708111d1648184"
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_artifact("models/mobilenet_autolog_s2.keras", artifact_path="model_file")
    mlflow.set_tag("note", "log_model was rejected by DagsHub after autolog; best .keras checkpoint attached instead")
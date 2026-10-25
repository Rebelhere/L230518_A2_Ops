import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
client = MlflowClient()
VERSIONS = [  # (run_id, best checkpoint, alias)
    ("ead5585546b84621a726c3350d3906fe", "models/mobilenet_manual_s2.keras", "challenger"),
    ("d2d11145ea5643348d792283c8a0320b", "models/mobilenet_ft_lr1e-4_s2.keras", "champion"),
]
for run_id, path, alias in VERSIONS:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path, artifact_path="model_file")
    run = client.get_run(run_id)
    mv = client.create_model_version(
        NAME, source=run.info.artifact_uri + "/model_file", run_id=run_id,
        tags={"run_name": run.info.run_name, "threshold": str(run.data.metrics["threshold"])})
    client.set_registered_model_alias(NAME, alias, mv.version)
    print(f"version {mv.version} <- {run.info.run_name}  alias = {alias}")
client.update_model_version(NAME, client.get_model_version_by_alias(NAME, "champion").version,
                            description="Best of 7 runs by validation ROC-AUC and F1 (mobilenet_ft_lr1e-4)")
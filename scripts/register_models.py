import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()
NAME = "L230518_A2_Ops_VisionGuard"
client = MlflowClient()
v1 = mlflow.register_model("runs:/ead5585546b84621a726c3350d3906fe/model", NAME)   # mobilenet_manual
v2 = mlflow.register_model("runs:/d2d11145ea5643348d792283c8a0320b/model", NAME)   # mobilenet_ft_lr1e-4
client.set_registered_model_alias(NAME, "challenger", v1.version)
client.set_registered_model_alias(NAME, "champion", v2.version)
client.set_model_version_tag(NAME, v2.version, "reason", "best val ROC-AUC and F1 of 7 runs")
print("registered versions:", v1.version, v2.version, "-> champion =", v2.version)
import json
import numpy as np, pandas as pd, mlflow
from dotenv import load_dotenv
from tensorflow import keras
from sklearn.metrics import roc_curve, confusion_matrix
from src.data import make_ds

load_dotenv()
RUN_ID = "d2d11145ea5643348d792283c8a0320b"      # mobilenet_ft_lr1e-4
MAX_FPR = 0.20                                    # at most 20% of real photos may be flagged
m = keras.models.load_model("models/mobilenet_ft_lr1e-4_s2.keras")
yv = pd.read_csv("splits/val.csv")["label"].values
yt = pd.read_csv("splits/test.csv")["label"].values
pv = m.predict(make_ds("val"), verbose=0).ravel()
pt = m.predict(make_ds("test"), verbose=0).ravel()

fpr, tpr, thr = roc_curve(yv, pv)
ok = np.where(fpr <= MAX_FPR)[0]
t = float(thr[ok[np.argmax(tpr[ok])]])            # best recall subject to the false-alarm cap (validation only)

def rep(y, p):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return {"false_alarm_rate": round(fp / (fp + tn), 3), "ai_recall": round(tp / (tp + fn), 3),
            "precision": round(tp / max(tp + fp, 1), 3), "accuracy": round((tp + tn) / len(y), 3),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
rv, rt = rep(yv, pv), rep(yt, pt)
print("deployment threshold:", round(t, 3)); print("VAL :", rv); print("TEST:", rt)
json.dump({"threshold": t, "val": rv, "test": rt}, open("models/threshold.json", "w"), indent=1)
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metric("deploy_threshold", t)
    mlflow.log_metrics({f"deploy_val_{k}": v for k, v in rv.items() if isinstance(v, float)})
    mlflow.log_metrics({f"deploy_test_{k}": v for k, v in rt.items() if isinstance(v, float)})
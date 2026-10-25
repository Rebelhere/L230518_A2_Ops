import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from PIL import Image
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from src.data import make_ds

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--model_path", required=True)
p.add_argument("--run_id", required=True)
a = p.parse_args()
load_dotenv()

model = keras.models.load_model(a.model_path)
yv = pd.read_csv("splits/val.csv")["label"].values
te = pd.read_csv("splits/test.csv"); yt = te["label"].values
pv = model.predict(make_ds("val"), verbose=0).ravel()
pt = model.predict(make_ds("test"), verbose=0).ravel()

pr, rc, th = precision_recall_curve(yv, pv)
thr = float(th[int(np.argmax((2 * pr * rc / (pr + rc + 1e-9))[:-1]))])   # threshold from VAL only
yh = (pt >= thr).astype(int)
m = {"accuracy": accuracy_score(yt, yh), "precision": precision_score(yt, yh), "recall": recall_score(yt, yh),
     "f1": f1_score(yt, yh), "roc_auc": roc_auc_score(yt, pt)}
cm = confusion_matrix(yt, yh)
print(a.name, "threshold", round(thr, 3), {k: round(v, 4) for k, v in m.items()}); print(cm)

Path("reports/figs").mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): plt.text(j, i, v, ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["real", "ai"]); plt.yticks([0, 1], ["real", "ai"]); plt.xlabel("predicted"); plt.ylabel("true")
plt.title(f"{a.name}: confusion (test)"); fig.savefig(f"reports/figs/test_cm_{a.name}.png", dpi=120)
mlflow_cm = fig
fpr, tpr, _ = roc_curve(yt, pt)
fig2 = plt.figure(figsize=(4.5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{a.name}: ROC (test) AUC={m['roc_auc']:.3f}")
fig2.savefig(f"reports/figs/test_roc_{a.name}.png", dpi=120)

# 2.6: the 10 most confident mistakes
wrong = np.where(yh != yt)[0]
worst = wrong[np.argsort(-np.abs(pt[wrong] - yt[wrong]))][:10]
fig3, ax = plt.subplots(2, 5, figsize=(18, 8))
for a_, i in zip(ax.ravel(), worst):
    a_.imshow(Image.open(te.path[i]).convert("RGB")); a_.axis("off")
    a_.set_title(f"true {'AI' if yt[i] else 'REAL'} | p(AI)={pt[i]:.2f}\n{Path(te.path[i]).name[:28]}", fontsize=9)
plt.tight_layout(); fig3.savefig(f"reports/figs/misclassified_{a.name}.png", dpi=110)

with mlflow.start_run(run_id=a.run_id):
    mlflow.log_metrics({f"test_{k}": v for k, v in m.items()}); mlflow.log_metric("test_threshold", thr)
    mlflow.log_figure(mlflow_cm, "test_confusion_matrix.png"); mlflow.log_figure(fig2, "test_roc_curve.png")
    mlflow.log_figure(fig3, "misclassified_test.png")
pd.DataFrame([{"name": a.name, "threshold": round(thr, 3), **{k: round(v, 4) for k, v in m.items()},
               "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}]).to_csv(
    "reports/test_results.csv", mode="a", header=not Path("reports/test_results.csv").exists(), index=False)
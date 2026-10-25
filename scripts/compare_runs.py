import mlflow, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
runs = mlflow.search_runs(experiment_names=["VisionGuard"]).dropna(subset=["metrics.final_val_f1"])
cols = {"tags.mlflow.runName": "run", "params.model": "model", "params.lr": "head_lr", "params.ft_lr": "ft_lr",
        "params.augmentation": "aug", "params.ft_epochs": "ft_epochs",
        "metrics.final_val_accuracy": "val_acc", "metrics.final_val_precision": "val_prec",
        "metrics.final_val_recall": "val_rec", "metrics.final_val_f1": "val_f1",
        "metrics.final_val_roc_auc": "val_auc", "metrics.threshold": "thr"}
t = runs[list(cols)].rename(columns=cols)
num = ["val_acc", "val_prec", "val_rec", "val_f1", "val_auc", "thr"]
t[num] = t[num].astype(float).round(4)
t = t.sort_values("val_auc", ascending=False)
t.to_csv("reports/comparison.csv", index=False)
print(t.to_string(index=False))
ax = t.set_index("run")[["val_auc", "val_f1", "val_acc"]].plot.barh(figsize=(9, 5))
ax.invert_yaxis(); plt.xlabel("validation score"); plt.title("VisionGuard run comparison"); plt.tight_layout()
plt.savefig("reports/figs/comparison.png", dpi=120)
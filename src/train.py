import argparse, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from src.config import SEED, IMG_SIZE, BATCH
from src.data import make_ds
from src.models import build_cnn, build_mobilenet, unfreeze_last_block

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--model", choices=["cnn", "mobilenet"], default="mobilenet")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--ft_lr", type=float, default=1e-5)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--ft_epochs", type=int, default=10)     # 0 = no fine-tuning stage
p.add_argument("--no_aug", action="store_true")
p.add_argument("--logging", choices=["manual", "autolog"], default="manual")
p.add_argument("--test", action="store_true")            # evaluate on the test set
args = p.parse_args()

load_dotenv()
mlflow.set_experiment("VisionGuard")
keras.utils.set_random_seed(SEED)
Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)

def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"

def dvc_version():
    for line in Path("data.dvc").read_text().splitlines():
        if "md5:" in line: return line.split("md5:")[1].strip()
    return "unknown"

class EpochLogger(keras.callbacks.Callback):
    def __init__(self, offset=0): super().__init__(); self.offset = offset
    def on_epoch_end(self, epoch, logs=None):
        m = {k: float(v) for k, v in (logs or {}).items()}
        mlflow.log_metrics(m, step=self.offset + epoch)

def fit_stage(model, train, val, epochs, tag, offset):
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(f"models/{args.name}_{tag}.keras", monitor="val_loss", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-7)]
    if args.logging == "manual": cbs.append(EpochLogger(offset))
    return model.fit(train, validation_data=val, epochs=epochs, callbacks=cbs, verbose=2).history

def metrics_at(y, prob, t):
    yh = (prob >= t).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh), "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, prob)}

train_ds = make_ds("train", augment=not args.no_aug, shuffle=True)
val_ds, test_ds = make_ds("val"), make_ds("test")
yv = pd.read_csv("splits/val.csv")["label"].values
yt = pd.read_csv("splits/test.csv")["label"].values

if args.logging == "autolog":
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

with mlflow.start_run(run_name=args.name) as run:
    mlflow.set_tags({"git_commit": git_commit(), "dvc_data_md5": dvc_version(), "seed": str(SEED),
                     "model": args.model, "logging": args.logging})
    mlflow.log_params({**vars(args), "img_size": IMG_SIZE, "batch": BATCH, "augmentation": not args.no_aug})

    stages, hist = [], []
    if args.model == "cnn":
        model = build_cnn(args.dropout, args.lr)
        hist.append(fit_stage(model, train_ds, val_ds, args.epochs, "s1", 0)); stages.append("cnn")
    else:
        model = build_mobilenet(args.dropout, args.lr)
        hist.append(fit_stage(model, train_ds, val_ds, args.epochs, "s1", 0)); stages.append("frozen head")
        if args.ft_epochs > 0:
            unfreeze_last_block(model, args.ft_lr)
            hist.append(fit_stage(model, train_ds, val_ds, args.ft_epochs, "s2", len(hist[0]["loss"])))
            stages.append("fine-tune")

    # ---- 2.4: best epoch + curves ----
    full = {k: sum([h[k] for h in hist], []) for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    best = [(int(np.argmin(h["val_loss"])) + 1, float(min(h["val_loss"]))) for h in hist]
    for s, (e, v) in zip(stages, best): print(f"best epoch [{s}]: {e} (val_loss={v:.4f})")
    for i, (e, v) in enumerate(best): mlflow.log_metrics({f"best_epoch_stage{i+1}": e, f"best_val_loss_stage{i+1}": v})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    marks, off = [], 0
    for h, (e, _) in zip(hist, best): marks.append(off + e - 1); off += len(h["loss"])
    for a, (k, t) in zip(ax, [("loss", "Loss"), ("accuracy", "Accuracy")]):
        a.plot(full[k], label="train"); a.plot(full["val_" + k], label="val"); a.set_title(t); a.set_xlabel("epoch")
        for m in marks: a.axvline(m, color="red", ls="--", label="best epoch")
        if len(hist) > 1: a.axvline(len(hist[0]["loss"]) - 0.5, color="grey", ls=":", label="stage switch")
        h_, l_ = a.get_legend_handles_labels(); u = dict(zip(l_, h_)); a.legend(u.values(), u.keys())
    plt.tight_layout(); mlflow.log_figure(fig, "curves.png"); fig.savefig(f"reports/figs/curves_{args.name}.png", dpi=120); plt.close(fig)

    # ---- 2.5: threshold from VALIDATION data, then metrics ----
    pv = model.predict(val_ds, verbose=0).ravel()
    pr, rc, th = precision_recall_curve(yv, pv)
    f1s = 2 * pr * rc / (pr + rc + 1e-9)
    thr = float(th[int(np.argmax(f1s[:-1]))])
    mlflow.log_metric("threshold", thr)
    vm = metrics_at(yv, pv, thr)
    mlflow.log_metrics({f"final_val_{k}": v for k, v in vm.items()})
    row = {"name": args.name, "run_id": run.info.run_id, "model": args.model, "lr": args.lr, "ft_lr": args.ft_lr,
           "dropout": args.dropout, "augmentation": not args.no_aug, "ft_epochs": args.ft_epochs,
           "best_epochs": [b[0] for b in best], "threshold": round(thr, 3),
           **{f"val_{k}": round(v, 4) for k, v in vm.items()}}
    row.update({f"test_{k}": "" for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    y_eval, p_eval, split = (yv, pv, "val")
    if args.test:
        ptest = model.predict(test_ds, verbose=0).ravel()
        tm = metrics_at(yt, ptest, thr)
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        row.update({f"test_{k}": round(v, 4) for k, v in tm.items()})
        y_eval, p_eval, split = (yt, ptest, "test")
        print("TEST:", {k: round(v, 4) for k, v in tm.items()})
    print("VAL :", {k: round(v, 4) for k, v in vm.items()}, "threshold", round(thr, 3))

    # ---- 3.4 artifacts ----
    cm = confusion_matrix(y_eval, (p_eval >= thr).astype(int))
    fig = plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): plt.text(j, i, v, ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["real", "ai"]); plt.yticks([0, 1], ["real", "ai"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Confusion matrix ({split})")
    mlflow.log_figure(fig, f"confusion_matrix_{split}.png"); fig.savefig(f"reports/figs/cm_{args.name}.png", dpi=120); plt.close(fig)
    fpr, tpr, _ = roc_curve(y_eval, p_eval)
    fig = plt.figure(figsize=(4.5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({split}) AUC={roc_auc_score(y_eval, p_eval):.3f}")
    mlflow.log_figure(fig, f"roc_curve_{split}.png"); fig.savefig(f"reports/figs/roc_{args.name}.png", dpi=120); plt.close(fig)
    mlflow.log_artifacts("splits", artifact_path="splits")
    Path("reports/pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    mlflow.log_artifact("reports/pip_freeze.txt")
    mlflow.keras.log_model(model, artifact_path="model")

    out = Path("reports/runs_summary.csv")
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print("run_id:", run.info.run_id)
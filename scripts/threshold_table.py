import numpy as np, pandas as pd
from tensorflow import keras
from sklearn.metrics import roc_curve
from src.data import make_ds

m = keras.models.load_model("models/mobilenet_ft_lr1e-4_s2.keras")
y = pd.read_csv("splits/val.csv")["label"].values
p = m.predict(make_ds("val"), verbose=0).ravel()
rows = []
for t in (0.2, 0.3, 0.366, 0.4, 0.5, 0.6, 0.7):
    yh = (p >= t).astype(int)
    tp = ((yh == 1) & (y == 1)).sum(); fp = ((yh == 1) & (y == 0)).sum()
    fn = ((yh == 0) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    rows.append({"threshold": t, "real_called_fake_%": round(100 * fp / (fp + tn), 1),
                 "ai_caught_%": round(100 * tp / (tp + fn), 1), "precision_%": round(100 * tp / max(tp + fp, 1), 1)})
fpr, tpr, thr = roc_curve(y, p)
print(pd.DataFrame(rows).to_string(index=False))
print("Youden J threshold:", round(float(thr[np.argmax(tpr - fpr)]), 3))
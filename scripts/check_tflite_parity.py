import json
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from src.preprocess import decode_resize
from serving.lite import LitePredictor, decode_resize_np

# Keras + TF preprocessing (training/local serving) vs TFLite + NumPy preprocessing (Vercel), on every test image
thr = json.load(open("models/threshold.json"))["threshold"]
keras_model = keras.models.load_model("models/champion.keras")
lite = LitePredictor("models/champion.tflite", thr)
rows = []
for path, y in zip(*pd.read_csv("splits/test.csv")[["path", "label"]].values.T):
    raw = open(path, "rb").read()
    x_tf, x_np = decode_resize(raw).numpy(), decode_resize_np(raw)
    p_keras = float(keras_model(x_tf[None], training=False)[0, 0])
    rows.append({"path": path, "label": y, "p_keras": p_keras,
                 "p_lite_tfprep": lite.p_ai(x_tf),              # isolates the model conversion
                 "p_lite": lite.p_ai(x_np),                     # what Vercel computes
                 "max_pixel_diff": float(np.abs(x_tf - x_np).max())})
df = pd.DataFrame(rows)
df.to_csv("reports/tflite_parity.csv", index=False)
d_model, d_all = (df.p_lite_tfprep - df.p_keras).abs(), (df.p_lite - df.p_keras).abs()
agree = ((df.p_keras >= thr) == (df.p_lite >= thr)).sum()
print(f"test images: {len(df)}   threshold {thr:.3f}")
print(f"conversion only    |dp| max {d_model.max():.2e}  mean {d_model.mean():.2e}")
print(f"conversion + prep  |dp| max {d_all.max():.2e}  mean {d_all.mean():.2e}")
print(f"pixels |diff| max {df.max_pixel_diff.max():.3f}  median {df.max_pixel_diff.median():.3f} (0-255 scale)")
print(f"same decision: {agree}/{len(df)}")
print(df.sort_values("max_pixel_diff", ascending=False).head(3)[["path", "max_pixel_diff", "p_keras", "p_lite"]].to_string(index=False))

import json
import numpy as np, pandas as pd, tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from src.preprocess import decode_resize

model = keras.models.load_model("models/champion.keras")
thr = json.load(open("models/threshold.json"))["threshold"]
base = model.get_layer("mobilenetv2_1.00_224")
after = model.layers[model.layers.index(base) + 1:]            # GAP, dropout, dense
# preprocess_input is a graph op inside the saved model, not a listed layer, so it is re-applied here
prep = keras.applications.mobilenet_v2.preprocess_input

def run(layers, x):
    for l in layers: x = l(x, training=False) if isinstance(l, keras.layers.Dropout) else l(x)
    return x

def gradcam(img):
    x = prep(tf.expand_dims(img, 0))
    with tf.GradientTape() as tape:
        fmap = base(x, training=False)                         # (1, 7, 7, 1280) last conv block
        tape.watch(fmap)
        p = run(after, fmap)[0, 0]
    w = tf.reduce_mean(tape.gradient(p, fmap), axis=(1, 2))   # channel importance for p(AI)
    cam = tf.nn.relu(tf.reduce_sum(fmap * w[:, None, None, :], -1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-9)
    return float(p), tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]

df = pd.read_csv("splits/test.csv")
rows = []
for path, y in zip(df["path"], df["label"]):
    img = decode_resize(tf.io.read_file(path))
    p, cam = gradcam(img)
    assert abs(p - float(model(img[None], training=False)[0, 0])) < 1e-3, "grad-cam path differs from the model"
    rows.append((path, y, p, int(p >= thr), img.numpy() / 255.0, cam))
correct = [r for r in rows if r[1] == r[3]]
wrong = [r for r in rows if r[1] != r[3]]
by = lambda rs, y: [r for r in rs if r[1] == y]
fn, fp = by(wrong, 1), by(wrong, 0)                            # missed AI | real flagged as AI
pick = by(correct, 1)[:2] + by(correct, 0)[:2] + (fp[:2] + fn[:2] + fp[2:] + fn[2:])[:4]

fig, ax = plt.subplots(2, 8, figsize=(24, 6.5))
names = {0: "real", 1: "AI"}
for k, (path, y, p, yh, img, cam) in enumerate(pick):
    ax[0, k].imshow(img); ax[1, k].imshow(img); ax[1, k].imshow(cam, cmap="jet", alpha=0.45)
    ax[0, k].set_title(f"{'CORRECT' if y == yh else 'WRONG'}\ntrue {names[y]} / pred {names[yh]}\np(AI)={p:.2f}", fontsize=10,
                       color="green" if y == yh else "red")
    for a in ax[:, k]: a.axis("off")
plt.suptitle(f"Grad-CAM on last MobileNetV2 conv block (threshold {thr:.3f})")
plt.tight_layout(); fig.savefig("reports/figs/gradcam.png", dpi=110)
print("saved reports/figs/gradcam.png")
for path, y, p, yh, *_ in pick: print(f"{'ok ' if y == yh else 'BAD'} true={names[y]:4s} p={p:.3f} {path}")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from src.config import SEED
from src.data import make_ds
from src.models import build_cnn, build_mobilenet, unfreeze_last_block

keras.utils.set_random_seed(SEED)
Path("reports/figs").mkdir(parents=True, exist_ok=True)

# ---- 2.1: one augmented training batch ----
x, y = next(iter(make_ds("train", augment=True, shuffle=True)))
fig, ax = plt.subplots(2, 6, figsize=(16, 6))
for i, a in enumerate(ax.ravel()):
    a.imshow(np.clip(x[i].numpy() / 255.0, 0, 1)); a.axis("off")
    a.set_title("AI" if y[i] == 1 else "REAL")
plt.tight_layout(); plt.savefig("reports/figs/augmented_batch.png", dpi=120); plt.close()
print("batch:", x.shape, x.dtype, "labels:", y.shape)

# ---- 2.2: baseline CNN summary + table ----
cnn = build_cnn(); cnn.summary()
rows = [(l.name, l.__class__.__name__, str(l.output.shape), l.count_params()) for l in cnn.layers]
pd.DataFrame(rows, columns=["layer", "type", "output_shape", "params"]).to_csv("reports/cnn_layers.csv", index=False)
print(pd.DataFrame(rows, columns=["layer", "type", "output_shape", "params"]).to_string())

# ---- 2.3: transfer model trainable layers per stage ----
m = build_mobilenet()
base = m.get_layer("mobilenetv2_1.00_224")
print("\nSTAGE 1: base frozen. trainable weights:", [w.path for w in m.trainable_weights])
unfreeze_last_block(m)
tl = [l.name for l in base.layers if l.trainable]
print("STAGE 2 trainable base layers (%d):" % len(tl), tl)
print("trainable params:", sum(int(np.prod(w.shape)) for w in m.trainable_weights))
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

Path("reports/figs").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("reports/audit.csv")
df = df[(~df.corrupt) & (df["class"] != "unknown")]

# 1) sample grid: 8 per class
fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for r, cls in enumerate(["real", "ai"]):
    sample = df[df["class"] == cls].sample(8, random_state=1)
    for c, p in enumerate(sample.path):
        axes[r, c].imshow(Image.open(p).convert("RGB"))
        axes[r, c].axis("off")
    axes[r, 0].set_title(cls.upper(), loc="left", fontsize=14)
plt.tight_layout(); plt.savefig("reports/figs/sample_grid.png", dpi=120); plt.close()

# 2) histograms per class
for col in ["width", "height", "file_size"]:
    plt.figure(figsize=(7, 4))
    for cls, color in [("real", "tab:blue"), ("ai", "tab:orange")]:
        plt.hist(df[df["class"] == cls][col], bins=40, alpha=0.6, label=cls, color=color)
    plt.xlabel(col); plt.ylabel("count"); plt.legend(); plt.title(f"{col} per class")
    plt.tight_layout(); plt.savefig(f"reports/figs/hist_{col}.png", dpi=120); plt.close()
print("done")
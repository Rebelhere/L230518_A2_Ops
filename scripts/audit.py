import hashlib, sys
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATA = Path("data")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SEED = 42

def label_of(p: Path):
    parts = [x.lower() for x in p.relative_to(DATA).parts[:-1]]
    joined = " ".join(parts)
    if "real" in joined: return "real"
    if any(k in joined for k in ("ai", "fake", "generated")): return "ai"
    return "unknown"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
files = [p for p in DATA.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
print("files found:", len(files))
for i, p in enumerate(files):
    r = {"path": p.as_posix(), "class": label_of(p), "file_size": p.stat().st_size,
         "sha256": sha256(p), "corrupt": False, "width": None, "height": None,
         "mode": None, "format": None}
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:          # reopen: verify() invalidates the handle
            im.load()
            r.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception:
        r["corrupt"] = True
    rows.append(r)
    if i % 1000 == 0: print(i, "/", len(files))

df = pd.DataFrame(rows)
df["duplicate"] = df.duplicated("sha256", keep="first")
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/audit.csv", index=False)

print("\n=== counts per class ===")
print(df.groupby("class").size())
good = df[~df.corrupt]
print("\n=== min / median / max ===")
print(good.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T)
print("\ncorrupt files:", int(df.corrupt.sum()))
print("exact duplicates:", int(df.duplicate.sum()))
print("\nmodes:\n", good["mode"].value_counts())
print("\nformats:\n", good["format"].value_counts())
print("\nunknown-class files:", int((df["class"] == "unknown").sum()))

# ---- clean + stratified 70/15/15 split (seed fixed) ----
clean = df[(~df.corrupt) & (~df.duplicate) & (df["class"] != "unknown")].copy()
clean["label"] = (clean["class"] == "ai").astype(int)   # 1 = AI-generated
train, tmp = train_test_split(clean, test_size=0.30, stratify=clean["label"], random_state=SEED)
val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
Path("splits").mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label", "class", "sha256"]].to_csv(f"splits/{name}.csv", index=False)
    print(name, len(part), part["label"].value_counts().to_dict())
"""Load the champion Keras model once and classify image bytes.

    python -m serving.inference path/to/a.jpg path/to/b.png
"""
import json, os, sys, time
from pathlib import Path

MODEL_NAME = "L230518_A2_Ops_VisionGuard"

class Predictor:
    def __init__(self, model_path, threshold):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found at {path} (run scripts/export_champion.py "
                                    "or mount the models folder)")
        import tensorflow as tf                     # imported here so the web app can run in remote mode without TF
        from tensorflow import keras
        from src.config import IMG_SIZE
        from src.preprocess import decode_resize
        self._tf, self._prep = tf, decode_resize
        self.model = keras.models.load_model(path)  # once, at start-up
        self.threshold = float(threshold)
        meta = path.with_name("champion_meta.json")
        self.version = json.loads(meta.read_text()).get("version", "unknown") if meta.exists() else "unknown"
        self.name = MODEL_NAME
        self.model(self._tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))  # warm-up so the first user request is not slow

    def predict(self, raw):
        t0 = time.perf_counter()
        x = self._tf.expand_dims(self._prep(raw), 0)                 # identical to training (src/preprocess.py)
        p = float(self.model(x, training=False).numpy()[0][0])     # p(AI-generated)
        ai = p >= self.threshold
        return {"label": "AI-generated" if ai else "Real", "p_ai": round(p, 4),
                "confidence": round(p if ai else 1 - p, 4), "threshold": round(self.threshold, 4),
                "model_name": self.name, "model_version": str(self.version),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

if __name__ == "__main__":
    t0 = time.perf_counter()
    pr = Predictor(os.getenv("MODEL_PATH", "models/champion.keras"), os.getenv("THRESHOLD", "0.771"))
    print(f"model v{pr.version} loaded once in {time.perf_counter() - t0:.1f}s, threshold {pr.threshold:.3f}")
    for f in sys.argv[1:]:
        print(f, "->", pr.predict(Path(f).read_bytes()))

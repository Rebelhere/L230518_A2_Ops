"""TensorFlow-free predictor for the Vercel model API: TFLite model + LiteRT runtime + Pillow/NumPy preprocessing.

Preprocessing reproduces src/preprocess.py::decode_resize without TF; scripts/check_tflite_parity.py measures
how closely it (and the TFLite model) match the Keras path on the test set.
"""
import io, json, time
from pathlib import Path
import numpy as np
from PIL import Image

IMG_SIZE = 224
MODEL_NAME = "L230518_A2_Ops_VisionGuard"

def _axis(out_size, in_size):
    # tf.image.resize bilinear (TF2 default): half_pixel_centers=True, no antialiasing
    x = (np.arange(out_size) + 0.5) * (in_size / out_size) - 0.5
    lo_f = np.floor(x)
    lo = np.maximum(lo_f, 0).astype(int)
    hi = np.minimum(np.ceil(x), in_size - 1).astype(int)
    return lo, hi, (x - lo_f).astype(np.float32)

def decode_resize_np(raw, size=IMG_SIZE):
    """Bytes -> float32 (size, size, 3) in [0, 255], like tf.io.decode_image(channels=3) + tf.image.resize."""
    with Image.open(io.BytesIO(raw)) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float32)    # alpha dropped, L/P expanded to 3 channels
    h, w = img.shape[:2]
    y0, y1, fy = _axis(size, h)
    x0, x1, fx = _axis(size, w)
    top = img[y0][:, x0] + (img[y0][:, x1] - img[y0][:, x0]) * fx[None, :, None]
    bot = img[y1][:, x0] + (img[y1][:, x1] - img[y1][:, x0]) * fx[None, :, None]
    return top + (bot - top) * fy[:, None, None]

def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter      # Linux / Vercel
    except ImportError:
        import tensorflow as tf                                  # local fallback (LiteRT has no Windows wheel)
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(path))

class LitePredictor:
    def __init__(self, model_path, threshold):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found at {path} (run scripts/export_tflite.py)")
        self.it = _interpreter(path)                             # loaded once, at start-up
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.threshold = float(threshold)
        meta = path.with_name("champion_meta.json")
        self.version = json.loads(meta.read_text()).get("version", "unknown") if meta.exists() else "unknown"
        self.name = MODEL_NAME

    def p_ai(self, x):
        self.it.set_tensor(self.inp, x[None].astype(np.float32))
        self.it.invoke()
        return float(self.it.get_tensor(self.out)[0][0])

    def predict(self, raw):
        t0 = time.perf_counter()
        p = self.p_ai(decode_resize_np(raw))
        ai = p >= self.threshold
        return {"label": "AI-generated" if ai else "Real", "p_ai": round(p, 4),
                "confidence": round(p if ai else 1 - p, 4), "threshold": round(self.threshold, 4),
                "model_name": self.name, "model_version": str(self.version),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

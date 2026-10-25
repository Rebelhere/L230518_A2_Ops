"""Vercel entrypoint for the VisionGuard model API (TFLite + LiteRT, no TensorFlow).

Vercel's Flask preset finds `app` here and routes every request to it with the original path.
Run `python -m scripts.build_vercel` first: it copies serving/, api/main.py (as api_core/) and the model here.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "champion.tflite"))
os.environ.setdefault("THRESHOLD", "0.771")
os.environ.setdefault("MAX_UPLOAD_MB", "4")          # Vercel rejects request bodies over 4.5 MB

from api_core.main import create_api  # noqa: E402

app = create_api()                                   # model loaded once per function instance

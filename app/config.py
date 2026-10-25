import os

class Config:
    """Every setting comes from the environment (12-factor); defaults are only for local development."""
    def __init__(self):
        self.MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
        self.THRESHOLD = float(os.getenv("THRESHOLD", "0.771"))
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
        self.MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
        self.INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")          # local | remote
        self.MODEL_API_URL = os.getenv("MODEL_API_URL", "").rstrip("/")
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
        self.MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "30"))
        self.SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(16).hex()

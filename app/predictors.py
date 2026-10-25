import requests

class PredictionError(RuntimeError):
    """The model could not produce an answer (e.g. the cloud API is down)."""

class LocalPredictor:
    mode = "local"

    def __init__(self, cfg):
        from serving.inference import Predictor
        self._p = Predictor(cfg.MODEL_PATH, cfg.THRESHOLD)
        self.name, self.version = self._p.name, self._p.version

    def predict(self, raw, filename):
        return self._p.predict(raw)

    def health(self):
        return {"model_loaded": True, "model_version": str(self.version), "threshold": self._p.threshold}

class RemotePredictor:
    """Sends the image bytes to the model API hosted in the cloud (Task 7/8)."""
    mode = "remote"

    def __init__(self, cfg):
        if not cfg.MODEL_API_URL:
            raise ValueError("INFERENCE_MODE=remote needs MODEL_API_URL")
        self.url, self.timeout = cfg.MODEL_API_URL, cfg.MODEL_API_TIMEOUT
        self.http = requests.Session()
        self.http.headers["X-API-Key"] = cfg.MODEL_API_KEY

    def predict(self, raw, filename):
        try:
            r = self.http.post(f"{self.url}/predict", files={"image": (filename, raw)}, timeout=self.timeout)
        except requests.RequestException as e:
            raise PredictionError(f"model API unreachable: {e.__class__.__name__}") from e
        if r.status_code != 200:
            msg = r.json().get("error", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            raise PredictionError(f"model API returned {r.status_code}: {msg[:200]}")
        return r.json()

    def health(self):
        try:
            h = self.http.get(f"{self.url}/health", timeout=10).json()
            return {"model_api": self.url, "model_api_status": h.get("status"), "model_version": h.get("model_version")}
        except (requests.RequestException, ValueError) as e:
            return {"model_api": self.url, "model_api_status": f"unreachable ({e.__class__.__name__})"}

def make_predictor(cfg):
    return RemotePredictor(cfg) if cfg.INFERENCE_MODE == "remote" else LocalPredictor(cfg)

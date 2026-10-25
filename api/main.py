"""Model API deployed to the cloud (Task 7). JSON in/out, protected by an API key.

    POST /predict   header X-API-Key, multipart field "image"  -> {label, p_ai, confidence, ...}
    GET  /health    no auth                                     -> {status, model_version, threshold}
"""
import hmac, logging, os
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from serving.validation import InvalidImage, check_upload

def create_api(predictor=None, api_key=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
    key = api_key if api_key is not None else os.getenv("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set; refusing to start an unprotected model API")
    if predictor is None:
        path, thr = os.getenv("MODEL_PATH", "models/champion.keras"), os.getenv("THRESHOLD", "0.771")
        if path.endswith(".tflite"):                     # Vercel: TFLite + LiteRT, no TensorFlow
            from serving.lite import LitePredictor
            predictor = LitePredictor(path, thr)
        else:                                            # container (Docker / Azure): full Keras model
            from serving.inference import Predictor
            predictor = Predictor(path, thr)
    api.logger.info("model %s v%s loaded, threshold %.3f", predictor.name, predictor.version, predictor.threshold)

    @api.get("/")
    def index():
        return jsonify(service="VisionGuard model API (L230518_A2_Ops)", model_version=str(predictor.version),
                       endpoints={"GET /health": "status", "POST /predict": "multipart 'image' + header X-API-Key"})

    @api.get("/health")
    def health():
        return jsonify(status="ok", model_name=predictor.name, model_version=str(predictor.version),
                       threshold=predictor.threshold)

    @api.post("/predict")
    def predict():
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), key):
            return jsonify(error="missing or invalid X-API-Key"), 401
        f = request.files.get("image")
        if f is None:
            return jsonify(error="send the image as multipart form field 'image'"), 400
        raw = f.read()
        try:
            name, digest = check_upload(f.filename, raw)
        except InvalidImage as e:
            return jsonify(error=str(e)), 400
        out = predictor.predict(raw)
        api.logger.info("predict %s sha=%s -> %s p_ai=%.3f %.0fms",
                        name, digest[:12], out["label"], out["p_ai"], out["latency_ms"])
        return jsonify({**out, "filename": name, "sha256": digest})

    @api.errorhandler(HTTPException)
    def http_error(e):
        return jsonify(error=e.description if e.code != 413 else "file too large"), e.code

    return api

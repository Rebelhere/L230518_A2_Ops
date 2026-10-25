from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException
from app.predictors import PredictionError
from serving.validation import InvalidImage, check_upload

bp = Blueprint("vg", __name__)

def _ext(name): return current_app.extensions[f"vg_{name}"]

@bp.get("/")
def index():
    return render_template("index.html", max_mb=_ext("cfg").MAX_UPLOAD_MB)

@bp.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None or not f.filename:
        return _error(400, "No file selected", "Choose an image to upload first.")
    raw = f.read()                                         # bytes in memory only, never saved
    try:
        name, digest = check_upload(f.filename, raw)
    except InvalidImage as e:
        current_app.logger.info("rejected upload %r: %s", f.filename, e)
        return _error(400, "That file can't be checked", str(e))
    pred = _ext("predictor")
    try:
        out = pred.predict(raw, name)
    except PredictionError as e:
        current_app.logger.error("prediction failed: %s", e)
        return _error(503, "The model is unavailable", "The classifier could not be reached. Please try again shortly.")
    row = _ext("store").add(filename=name, sha256=digest, label=out["label"], p_ai=out["p_ai"],
                            confidence=out["confidence"], threshold=out["threshold"], model_name=out["model_name"],
                            model_version=str(out["model_version"]), mode=pred.mode, latency_ms=out["latency_ms"])
    current_app.logger.info("prediction #%s %s p_ai=%.3f (%s)", row.id, out["label"], out["p_ai"], pred.mode)
    return redirect(url_for("vg.result", pid=row.id), code=303)

@bp.get("/result/<int:pid>")
def result(pid):
    row = _ext("store").get(pid)
    if row is None:
        return _error(404, "Result not found", f"There is no prediction #{pid}.")
    return render_template("result.html", r=row)

@bp.get("/history")
def history():
    page = max(request.args.get("page", 1, type=int), 1)
    rows, total = _ext("store").page(page)
    return render_template("history.html", rows=rows, total=total, page=page, pages=max((total + 19) // 20, 1))

@bp.get("/health")
def health():
    body, code = {"status": "ok", "inference_mode": _ext("predictor").mode}, 200
    try:
        _ext("store").ping(); body["database"] = "ok"
    except Exception as e:
        body.update(database=f"error: {e.__class__.__name__}", status="degraded"); code = 503
    body.update(_ext("predictor").health())
    return jsonify(body), code

def _error(code, title, message):
    return render_template("error.html", code=code, title=title, message=message), code

@bp.app_errorhandler(HTTPException)
def http_error(e):
    if e.code == 413:
        mb = _ext("cfg").MAX_UPLOAD_MB
        return _error(413, "File too large", f"The upload limit is {mb:g} MB. Please choose a smaller image.")
    return _error(e.code, e.name, e.description)

@bp.app_errorhandler(Exception)
def server_error(e):
    current_app.logger.exception("unhandled error")
    return _error(500, "Something went wrong", "An unexpected error occurred. It has been logged.")

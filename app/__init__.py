import io, logging
from flask import Flask, Request
from app.config import Config
from app.db import Store
from app.predictors import make_predictor

class InMemoryRequest(Request):
    """Werkzeug spools uploads > 500 KB to a temp file; keep them in RAM instead (bounded by MAX_CONTENT_LENGTH)."""
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()

def create_app(cfg=None, predictor=None, store=None):
    cfg = cfg or Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config.update(SECRET_KEY=cfg.SECRET_KEY, MAX_CONTENT_LENGTH=int(cfg.MAX_UPLOAD_MB * 1024 * 1024))
    app.extensions["vg_cfg"] = cfg
    app.extensions["vg_store"] = store or Store(cfg.DATABASE_URL)
    app.logger.info("database ready (%s)", cfg.DATABASE_URL.split("@")[-1])    # never log the password
    app.extensions["vg_predictor"] = predictor or make_predictor(cfg)            # model loaded ONCE here
    app.logger.info("inference mode=%s", app.extensions["vg_predictor"].mode)
    from app.routes import bp
    app.register_blueprint(bp)
    return app

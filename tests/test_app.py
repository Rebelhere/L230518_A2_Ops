import io
import pytest
from PIL import Image
from app import create_app
from app.config import Config
from app.db import Store
from api.main import create_api

class FakePredictor:
    """Stands in for the Keras model so tests run in seconds without TensorFlow."""
    mode, name, version, threshold = "local", "L230518_A2_Ops_VisionGuard", "test", 0.771

    def predict(self, raw, filename=None):
        return {"label": "AI-generated", "p_ai": 0.9, "confidence": 0.9, "threshold": self.threshold,
                "model_name": self.name, "model_version": self.version, "latency_ms": 1.0}

    def health(self):
        return {"model_loaded": True, "model_version": self.version}

def png_bytes(size=(32, 32)):
    buf = io.BytesIO(); Image.new("RGB", size, (200, 30, 30)).save(buf, "PNG"); return buf.getvalue()

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    cfg = Config()
    app = create_app(cfg, predictor=FakePredictor(), store=Store(f"sqlite:///{tmp_path / 't.db'}"))
    return app.test_client()

def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")

def test_upload_is_logged_and_shown_in_history(client):
    r = upload(client, png_bytes(), "../../evil name.png")
    assert r.status_code == 303
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Likely AI-generated" in page and "evil_name.png" in page      # file name sanitised
    hist = client.get("/history").get_data(as_text=True)
    assert "1 request logged" in hist and "AI-generated" in hist

def test_wrong_file_type_rejected(client):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 400 and "Unsupported file type" in r.get_data(as_text=True)

def test_too_large_rejected(client):
    r = upload(client, b"\0" * (2 * 1024 * 1024), "big.jpg")
    assert r.status_code == 413 and "File too large" in r.get_data(as_text=True)

def test_fake_image_rejected(client):
    r = upload(client, b"\xff\xd8\xff\xe0 definitely not a jpeg", "photo.jpg")
    assert r.status_code == 400 and "could not be decoded" in r.get_data(as_text=True)

def test_missing_file(client):
    assert client.post("/predict", data={}).status_code == 400

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["database"] == "ok" and r.json["status"] == "ok"

def test_unknown_result_is_friendly_404(client):
    r = client.get("/result/999")
    assert r.status_code == 404 and "Back to upload" in r.get_data(as_text=True)

def test_api_requires_key():
    api = create_api(predictor=FakePredictor(), api_key="secret").test_client()
    files = {"image": (io.BytesIO(png_bytes()), "x.png")}
    assert api.post("/predict", data=files, content_type="multipart/form-data").status_code == 401
    r = api.post("/predict", data={"image": (io.BytesIO(png_bytes()), "x.png")},
                 headers={"X-API-Key": "secret"}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.json["label"] == "AI-generated"
    assert api.get("/health").json["status"] == "ok"
    assert "/predict" in str(api.get("/").json["endpoints"])

@pytest.mark.parametrize("mode,fmt", [("RGBA", "PNG"), ("L", "JPEG"), ("P", "PNG")])
def test_numpy_preprocessing_shape(mode, fmt):
    from serving.lite import decode_resize_np
    buf = io.BytesIO(); Image.new(mode, (301, 177)).save(buf, fmt)
    x = decode_resize_np(buf.getvalue())
    assert x.shape == (224, 224, 3) and x.dtype.name == "float32"

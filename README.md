# L230518_A2_Ops — VisionGuard

[![CI](https://github.com/Rebelhere/L230518_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/Rebelhere/L230518_A2_Ops/actions/workflows/ci.yml)

VisionGuard tells whether an uploaded image is **AI-generated or real**. A MobileNetV2 model fine-tuned with
TensorFlow/Keras is tracked with MLflow on DagsHub, served by Flask, logged to PostgreSQL, containerised with
Docker Compose, and its model API (TFLite) runs on Vercel.

- DagsHub (data, experiments, model registry): https://dagshub.com/Rebelhere/L230518_A2_Ops
- Registered model: `L230518_A2_Ops_VisionGuard`, alias `@champion`

## Layout
| Path | What |
|---|---|
| `scripts/` | audit, EDA, tuner, evaluation, registry, Grad-CAM |
| `src/` | config, `tf.data` pipeline, model builders, training with MLflow logging |
| `src/preprocess.py` | decode + resize, **shared by training and serving** |
| `serving/` | `Predictor` (loads the model once) and upload validation |
| `app/` | Flask UI: upload, result, history, health + SQLAlchemy database trail |
| `api/` | model API for the cloud (`POST /predict` with `X-API-Key`, `GET /health`) |
| `Dockerfile`, `docker-compose.yml` | web + Postgres stack; `Dockerfile.api` = cloud model image |

## Run it
```bash
cp .env.example .env                 # fill in passwords / keys
python -m scripts.export_champion    # fetch @champion from the registry into models/ (once)
docker compose up -d --build         # web on http://localhost:5000, Postgres internal only
docker compose ps                    # both services "healthy"
```
Use the cloud model instead of the local one: set `INFERENCE_MODE=remote`, `MODEL_API_URL`, `MODEL_API_KEY` in `.env`,
then `docker compose up -d`.

Without Docker: `pip install -r requirements-serve.txt && python wsgi.py web` (SQLite by default).

## Tests
```bash
pip install -r requirements-app.txt pytest ruff
ruff check app api serving tests wsgi.py && python -m pytest -q tests
```

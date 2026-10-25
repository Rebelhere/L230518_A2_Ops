# Runbook: steps that need your screenshots

Run everything from `C:\Users\It Zone\Desktop\Github\L230518_A2_Ops` in PowerShell with `(venv)` active, unless a step says otherwise.
Every screenshot must show the prompt (your username and project folder) and, ideally, the clock.
Save them as `report/img/<id>_<name>.png`; the names are given below.

---

## 6.1 Docker warm-up (ready-made `nginx` image)

| Step | What it shows | Screenshot |
|---|---|---|
| A | pull + list images | `6.1a_pull.png` |
| B | run + port map + volume mount | `6.1b_run.png` |
| C | page served from the mounted folder, at http://localhost:8081 in a browser | `6.1c_browser.png` |
| D | logs (two `GET /` lines after a browser refresh) | `6.1d_logs.png` |
| E | shell inside the container | `6.1e_shell.png` |
| F | stop / remove container + image | `6.1f_cleanup.png` |

```powershell
# A. pull + list images
docker pull nginx:alpine
docker images

# B. run + port map + volume mount
mkdir $HOME\nginx_toy; "<h1>L230518 Docker warm-up</h1>" | Out-File -Encoding ascii $HOME\nginx_toy\index.html
docker run -d --name toy-nginx -p 8081:80 -v "$HOME\nginx_toy:/usr/share/nginx/html:ro" nginx:alpine
docker ps

# D. logs (after refreshing the browser twice)
docker logs toy-nginx

# E. shell inside the container
docker exec -it toy-nginx sh
#   whoami; ls /usr/share/nginx/html; cat /etc/os-release; exit

# F. stop / remove container + image
docker stop toy-nginx; docker rm toy-nginx; docker ps -a
docker rmi nginx:alpine; docker images
```

---

## 6.2 / 6.3 Compose stack

```powershell
python -m scripts.export_champion            # only if models/champion.keras is missing
docker compose up -d --build
docker compose ps                            # SCREENSHOT 6.3a_ps_healthy.png  (both "healthy")
docker compose exec web whoami               # SCREENSHOT 6.2_whoami.png       -> app
docker compose exec web id
docker compose port db 5432                  # prints nothing / error: no published port
Test-NetConnection localhost -Port 5432      # SCREENSHOT 6.3b_db_closed.png   TcpTestSucceeded : False
docker compose exec db psql -U visionguard -d visionguard -c "select id, created_at, label, round(p_ai::numeric,3), model_version, mode from predictions order by id desc limit 10;"
                                             # SCREENSHOT 5_db_rows.png (after a few uploads)
```

Then open http://localhost:5000, upload a few images and screenshot the History page (I can also capture this one).

---

## 6.4 Image size: before vs after

```powershell
docker build -f Dockerfile.naive -t visionguard:naive .
docker images | Select-String "visionguard|l230518"          # SCREENSHOT 6.4a_sizes.png
docker history l230518_a2_ops-web --format "{{.Size}}`t{{.CreatedBy}}" | Select-Object -First 12   # 6.4b_history.png
docker history visionguard:naive --format "{{.Size}}`t{{.CreatedBy}}" | Select-Object -First 12
```

---

## 6.5 Operations and troubleshooting

```powershell
docker compose stop; docker compose ps
docker compose start; docker compose restart web
docker compose logs --tail 30 web
docker compose logs --tail 30 db
docker compose exec web sh     # ls /models; exit
docker compose exec db sh      # psql -U visionguard -d visionguard -c "\dt"; exit
```

**Break it on purpose: missing model file.**

```powershell
Rename-Item models\champion.keras champion.keras.bak
docker compose up -d --force-recreate web
docker compose logs --tail 40 web      # SCREENSHOT 6.5_failure.png -> FileNotFoundError: model file not found at /models/champion.keras
Rename-Item models\champion.keras.bak champion.keras
docker compose up -d --force-recreate web
docker compose ps                      # SCREENSHOT 6.5_fixed.png
```

**Optional second break: wrong database password.**
Change `POSTGRES_PASSWORD` in `.env`, then run `docker compose up -d --force-recreate web`.
The log shows `password authentication failed for user "visionguard"`.
Postgres keeps the old password because it lives inside the `pgdata` volume. Put the old value back to fix it.

---

## 6.6 GitHub Actions

**Toy repo.** The files are in `C:\Users\It Zone\Desktop\Github\ci_toy`. First create an empty **public** repo on GitHub named `L230518_A2_Ops_ci_toy`.

```powershell
cd ..\ci_toy
git init; git add .; git commit -m "toy CI: matrix 3.11/3.12"
git branch -M main
git remote add origin https://github.com/Rebelhere/L230518_A2_Ops_ci_toy.git
git push -u origin main                                   # green run

# make it red: break a test
(Get-Content test_calc.py) -replace '== 5', '== 6' | Set-Content test_calc.py
git commit -am "break test on purpose"; git push           # SCREENSHOT 6.6a_red.png (Actions tab)

# fix it
(Get-Content test_calc.py) -replace '== 6', '== 5' | Set-Content test_calc.py
git commit -am "fix test"; git push                        # SCREENSHOT 6.6b_green.png
cd ..\L230518_A2_Ops
```

**Project.** After pushing this repo, take these screenshots:
- the green run at https://github.com/Rebelhere/L230518_A2_Ops/actions → `6.6c_project_green.png`;
- the README with the badge → `6.6d_badge.png`.

---

## 7 Model API on Google Cloud Run

One-time setup:
1. Install the gcloud CLI: https://cloud.google.com/sdk/docs/install
2. In the console, create the project **`l230518-a2-ops`**. GCP IDs must be lowercase with hyphens. → `7a_project.png`
3. Link a billing account. The free tier covers this, and the limits below cap the cost.

```powershell
gcloud auth login
gcloud config set project l230518-a2-ops
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create l230518-a2-ops --repository-format=docker --location=us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev

# build the API image with the champion model baked in, and push it
docker build -f Dockerfile.api -t us-central1-docker.pkg.dev/l230518-a2-ops/l230518-a2-ops/api:v2 .
docker push us-central1-docker.pkg.dev/l230518-a2-ops/l230518-a2-ops/api:v2            # 7b_push.png

# test the same image locally first
docker run --rm -p 8080:8080 -e MODEL_API_KEY=localtest us-central1-docker.pkg.dev/l230518-a2-ops/l230518-a2-ops/api:v2

# deploy; the key comes from .env and is never typed into a file
$KEY = (Select-String '^MODEL_API_KEY=' .env).Line.Split('=',2)[1]
gcloud run deploy l230518-a2-ops-api `
  --image us-central1-docker.pkg.dev/l230518-a2-ops/l230518-a2-ops/api:v2 `
  --region us-central1 --memory 2Gi --cpu 1 --cpu-boost --timeout 120 `
  --min-instances 0 --max-instances 1 --allow-unauthenticated `
  --set-env-vars "MODEL_API_KEY=$KEY,THRESHOLD=0.771"                                   # 7c_deploy.png (shows Service URL)
```

`--allow-unauthenticated` only makes the URL reachable. Every `/predict` call still needs the `X-API-Key` header, and without it the API returns 401.

Screenshots to take:
- `7d_console.png`: the Cloud Run console page for the service, showing the URL, the revision and a green tick.
- `7e_logs.png`: the Logs tab after a few calls, showing the `predict ... p_ai=` lines.

**Postman.** Do this from your phone's hotspot, so the call comes from another network.
- `POST https://<service-url>/predict`
- Headers: `X-API-Key: <key>`
- Body: form-data, key `image` (type File)
- Screenshots: `7f_postman_ok.png` (200 + JSON), `7g_postman_401.png` (same call without the header), `7h_postman_400.png` (a .txt file).

---

## 8 Flask (localhost) calling the cloud API

In `.env`, set `INFERENCE_MODE=remote` and `MODEL_API_URL=https://<service-url>`. Then:

```powershell
docker compose up -d --force-recreate web
curl.exe http://localhost:5000/health        # model_api_status: ok   -> 8a_health_remote.png
```

Upload 5 or more different images: AI, real, a PNG, and one wrong file. Then take:
- `8b_result_*.png`: each result page;
- `8c_history.png`: the History page, with the "Served by: cloud API" column;
- `8d_cloudrun_logs.png`: the matching Cloud Run log lines.

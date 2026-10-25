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

## 7a Model API on Vercel (deployed pathway)

Full TensorFlow doesn't fit in a Vercel function (about 250 MB limit). Vercel therefore serves a **TFLite** copy of the champion model with the small **LiteRT** runtime. The API code is the same `api/main.py`.

**One-time setup:** sign up at https://vercel.com/signup with **Continue with GitHub**. No card is needed.

```powershell
npx vercel login                                 # opens the browser; approve
python -m scripts.export_champion                # models/champion.keras = registry @champion (if not done)
python -m scripts.export_tflite                  # -> models/champion.tflite
python -m scripts.check_tflite_parity            # SCREENSHOT 7a_parity.png  (146/146 same decision)
python -m scripts.build_vercel                   # copies code + model into deploy/vercel
cd deploy\vercel
npx vercel link --yes --project l230518-a2-ops-api
# API key from ..\..\.env into Vercel's encrypted env store (never into a file):
(Select-String '^MODEL_API_KEY=' ..\..\.env).Line.Split('=',2)[1] | npx vercel env add MODEL_API_KEY production
npx vercel deploy --prod                         # SCREENSHOT 7b_deploy.png -> prints https://l230518-a2-ops-api.vercel.app
cd ..\..
curl.exe https://l230518-a2-ops-api.vercel.app/health
```

Screenshots to take:
- `7c_dashboard.png`: the Vercel dashboard for the project, showing the production deployment, the domain and the "Ready" status.
- `7d_env.png`: Settings → Environment Variables, showing `MODEL_API_KEY` (encrypted, value hidden).
- `7e_logs.png`: the deployment's **Logs** tab after a few calls, showing the `predict ... p_ai=` lines.

**Postman.** Do this from your phone's hotspot, so the call comes from another network.
- `POST https://l230518-a2-ops-api.vercel.app/predict`
- Headers: `X-API-Key: <key>`
- Body: form-data, key `image` (type File)
- Screenshots: `7f_postman_ok.png` (200 + JSON), `7g_postman_401.png` (same call without the header), `7h_postman_400.png` (a .txt file).

---

## 7b Azure Container Apps (optional, later: full TensorFlow container)

Google Cloud Run was the first plan. Google needs a billing account, and it rejected my payment card, including a SadaPay virtual card (error `OR_BACR2_31`). Azure for Students needs only the university email and gives $100 of credit, with no card.

**One-time setup**

1. Activate https://azure.microsoft.com/free/students with your FAST NU email. → `7a_azure_students.png` (credit page)
2. Install the Azure CLI: https://aka.ms/installazurecliwindows. Then open a new PowerShell window.

**Sign in and create the resources**

```powershell
az login
az account show --query "{name:name, id:id}"          # "Azure for Students"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry

# resource group (names may contain _), in a region your student subscription allows
az group create -n L230518_A2_Ops -l eastus
# container registry (name: lowercase letters and digits only)
az acr create -g L230518_A2_Ops -n l230518a2ops --sku Basic --admin-enabled true
```

If `eastus` is refused with `RequestDisallowedByAzure`, student subscriptions only allow certain regions. The error message lists them. Use one of those (for example `centralindia` or `uaenorth`) everywhere below.

**Build the API image in Azure**

The image has the champion model built in. `az acr build` uploads only the build context: the code plus the 17 MB model, after `.dockerignore`. Azure's servers then download TensorFlow and build the image.

```powershell
python -m scripts.export_champion                      # models/champion.keras = registry @champion
az acr build -r l230518a2ops -t api:v2 -f Dockerfile.api .      # 7b_acr_build.png
```

If you get `TasksOperationsNotAllowed`, the subscription doesn't allow cloud builds. Build and push from your PC instead (slow):

```powershell
az acr login -n l230518a2ops
docker build -f Dockerfile.api -t l230518a2ops.azurecr.io/api:v2 .
docker push l230518a2ops.azurecr.io/api:v2
```

**Deploy**

The API key comes from `.env` and is stored as a Container Apps *secret*. It never appears in a file or in the image.

```powershell
$KEY = (Select-String '^MODEL_API_KEY=' .env).Line.Split('=',2)[1]
az containerapp env create -g L230518_A2_Ops -n l230518-a2-ops-env -l eastus
az containerapp create -g L230518_A2_Ops -n l230518-a2-ops-api `
  --environment l230518-a2-ops-env `
  --image l230518a2ops.azurecr.io/api:v2 --registry-server l230518a2ops.azurecr.io `
  --target-port 8080 --ingress external `
  --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 `
  --secrets api-key=$KEY `
  --env-vars MODEL_API_KEY=secretref:api-key THRESHOLD=0.771 `
  --query properties.configuration.ingress.fqdn          # 7c_deploy.png -> prints the API host name

curl.exe https://<fqdn>/health                           # {"status":"ok","model_version":"2",...}
```

Screenshots to take:
- `7d_portal.png`: the Azure portal page for the Container App, showing the Application Url, "Running" status and the replica settings.
- `7e_logs.png`: the portal's **Log stream**, or `az containerapp logs show -g L230518_A2_Ops -n l230518-a2-ops-api --follow`, showing the `predict ... p_ai=` lines after a few calls.

**Postman.** Do this from your phone's hotspot, so the call comes from another network.
- `POST https://<fqdn>/predict`
- Headers: `X-API-Key: <key>`
- Body: form-data, key `image` (type File)
- Screenshots: `7f_postman_ok.png` (200 + JSON), `7g_postman_401.png` (same call without the header), `7h_postman_400.png` (a .txt file).

**Saving credit when you're done grading:**

```powershell
az containerapp update -g L230518_A2_Ops -n l230518-a2-ops-api --max-replicas 0    # pause
az group delete -n L230518_A2_Ops                                                  # delete everything at the end
```

---

## 8 Flask (localhost) calling the cloud API

In `.env`, set `INFERENCE_MODE=remote` and `MODEL_API_URL=https://l230518-a2-ops-api.vercel.app` and `MAX_UPLOAD_MB=4`. Then:

```powershell
docker compose up -d --force-recreate web
curl.exe http://localhost:5000/health        # model_api_status: ok   -> 8a_health_remote.png
```

Upload 5 or more different images: AI, real, a PNG, and one wrong file. Then take:
- `8b_result_*.png`: each result page;
- `8c_history.png`: the History page, with the "Served by: cloud API" column;
- `8d_cloudrun_logs.png`: the matching Vercel log lines.

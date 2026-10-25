import shutil
from pathlib import Path

# assemble deploy/vercel: shared API + serving code and the TFLite champion (copies are git-ignored build output)
out = Path("deploy/vercel")
for d in ("serving", "api_core", "models"):
    shutil.rmtree(out / d, ignore_errors=True)
    (out / d).mkdir(parents=True)
for f in ("__init__.py", "validation.py", "lite.py"):
    shutil.copy(Path("serving") / f, out / "serving" / f)
(out / "api_core" / "__init__.py").touch()
shutil.copy("api/main.py", out / "api_core" / "main.py")
for f in ("champion.tflite", "champion_meta.json"):
    shutil.copy(Path("models") / f, out / "models" / f)
print("deploy/vercel ready:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))

"""Production entrypoints, served by waitress with Flask debug OFF.

    python wsgi.py web    # VisionGuard UI  (PORT, default 5000)
    python wsgi.py api    # model API       (PORT, default 8080)
"""
import os, sys
from dotenv import load_dotenv
from waitress import serve

if __name__ == "__main__":
    load_dotenv()                                   # no-op in Docker, where env comes from compose / Azure
    which = sys.argv[1] if len(sys.argv) > 1 else "web"
    if which == "api":
        from api.main import create_api
        app, port = create_api(), int(os.getenv("PORT", "8080"))
    else:
        from app import create_app
        app, port = create_app(), int(os.getenv("PORT", "5000"))
    app.debug = False
    serve(app, host=os.getenv("HOST", "0.0.0.0"), port=port, threads=int(os.getenv("THREADS", "4")))

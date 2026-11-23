"""Serve the app on this laptop with Waitress."""
import sys
from waitress import serve
from scripts.prepare_models import prepare

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    prepare()
    from app import app
    print("Open http://localhost:5000", flush=True)
    serve(app, host="127.0.0.1", port=5000, threads=1)

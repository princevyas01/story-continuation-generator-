"""Static file serving and SPA fallback router for compiled React frontend."""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from backend.app.config import settings

def setup_static_serving(app: FastAPI):
    """Mount compiled frontend static files and SPA catch-all handler."""
    dist_dir = settings.DIST_DIR
    assets_dir = os.path.join(dist_dir, "assets")

    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Do not catch API routes or docs
        if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return HTMLResponse(status_code=404, content="API route not found")

        index_file = os.path.join(dist_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)

        # Fallback if frontend has not been compiled yet
        return HTMLResponse(
            status_code=200,
            content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Story Continuation Generator - Backend Ready</title>
                <style>
                    body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                    .card { background: #1e293b; padding: 2.5rem; border-radius: 1rem; border: 1px solid #334155; max-width: 600px; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
                    h1 { color: #38bdf8; margin-top: 0; }
                    a { color: #818cf8; text-decoration: none; font-weight: bold; }
                    a:hover { text-decoration: underline; }
                    .badge { display: inline-block; background: #0284c7; color: white; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; margin-bottom: 1rem; }
                </style>
            </head>
            <body>
                <div class="card">
                    <span class="badge">FastAPI Backend Running</span>
                    <h1>Story Continuation Generator</h1>
                    <p>The neural LSTM backend is operational! Access API documentation at <a href="/docs">/docs</a> or <a href="/api/v1/health">/api/v1/health</a>.</p>
                </div>
            </body>
            </html>
            """
        )

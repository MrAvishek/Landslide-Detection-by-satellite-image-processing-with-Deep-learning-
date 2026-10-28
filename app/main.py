from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes.health import router as health_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Landslide Detection API",
    description="Satellite imagery analysis and landslide segmentation",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(health_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/v1")
def api_root():
    return {
        "message": "Landslide Detection API is running",
        "docs": "/docs",
    }
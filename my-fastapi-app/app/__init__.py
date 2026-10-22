import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# Get base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
template_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

# Create FastAPI app
app = FastAPI(
    title="MindMap Chat",
    description="A mind map based chat interface with AI",
    version="1.0.0"
)

# Mount static files with proper route name
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=template_dir)

# Import routes after app is created to avoid circular imports
from app import routes  # noqa: E402, F401  (registers the endpoints)

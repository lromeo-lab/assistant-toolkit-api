import logging
import sys
from fastapi import FastAPI
from app.api.v1.endpoints import ingestion, memory, assistant
from app.core.config import get_settings

# --- CORRECTED & ROBUST LOGGING SETUP ---
# Get the root logger
logger = logging.getLogger()

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Set the desired level
logger.setLevel(logging.INFO)

# Create a handler to stream logs to the console (stdout)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

# Add the handler to the root logger
logger.addHandler(handler)
# --- END OF LOGGING SETUP ---


# Load settings
settings = get_settings()

# Create the FastAPI app instance
app = FastAPI(
    title=settings.project_name,
    description="A toolkit API for managing document ingestion and interacting with a RAG assistant.",
    version="1.0.0"
)

# Include the API router from the ingestion endpoint file
# All routes defined in that router will be added to our app under the /api/v1 prefix
app.include_router(ingestion.router, prefix=settings.api_v1_str, tags=["Ingestion"])
app.include_router(memory.router, prefix=settings.api_v1_str, tags=["Chat Memory"])
app.include_router(assistant.router, prefix=settings.api_v1_str, tags=["Assistant"])

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": f"Welcome to the {settings.project_name}"}
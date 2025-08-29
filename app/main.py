import logging
import sys
from fastapi import FastAPI, Request
from app.api.v1.endpoints import (
    agent_management,
    file_management,
    thread_management,
    chat_management,
    assistant,
    worker_management)
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
    description="A toolkit API for managing document ingestion and interacting with a RAG Agent.",
    version="1.0.0"
)

# Include the API router from the ingestion endpoint file
# All routes defined in that router will be added to our app under the /api/v1 prefix
app.include_router(agent_management.router, prefix=settings.api_v1_str, tags=["Agent Management"])
app.include_router(thread_management.router, prefix=settings.api_v1_str, tags=["Thread Management"])
app.include_router(file_management.router, prefix=settings.api_v1_str, tags=["File Management"])
app.include_router(chat_management.router, prefix=settings.api_v1_str, tags=["Chat Management"])
app.include_router(worker_management.router, prefix=settings.api_v1_str, tags=["Worker Management"])
app.include_router(assistant.router, prefix=settings.api_v1_str, tags=["Agent Engine"])

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": f"Welcome to the {settings.project_name}"}

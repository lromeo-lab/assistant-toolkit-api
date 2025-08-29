import logging
from fastapi import APIRouter, Depends, BackgroundTasks, Body, HTTPException
import httpx

from app.core.config import Settings, get_settings
from app.services.chat_management_service import ChatManagementService

from app.api.v1.schemas import ChatIngestionRequest, MessageResponse
from app.api.v1.dependencies import get_chat_management_service

router = APIRouter()

# --- Background Task Helper ---
async def call_worker_endpoint(url: str, payload: dict):
    """Makes an async HTTP request to a worker endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            logging.info(f"Successfully triggered worker for thread '{payload.get('thread_id')}'.")
        except httpx.RequestError as e:
            logging.error(f"Failed to call worker endpoint at {url}: {e}")

# --- User-Facing Endpoint ---
@router.post("/schedule-ingest-chat", status_code=202, response_model=MessageResponse)
async def schedule_chat_ingestion(
    background_tasks: BackgroundTasks,
    chat: ChatIngestionRequest = Body(...),
    settings: Settings = Depends(get_settings)
):
    """
    USER-FACING: Quickly accepts a chat turn and schedules it for ingestion.
    """
    logging.info(f"Received request to schedule ingestion for thread '{chat.thread_id}'.")
    
    worker_url = f"{settings.internal_worker_url}{settings.api_v1_str}/worker/ingest-chat"
    
    background_tasks.add_task(
        call_worker_endpoint,
        url=worker_url,
        payload=chat.model_dump()
    )

    return MessageResponse(message="Chat turn received and scheduled for ingestion.")

@router.delete("/chats/{thread_id}", response_model=MessageResponse)
def delete_chat(
    thread_ids: list,
    service: ChatManagementService = Depends(get_chat_management_service)
):
    """Deletes all chat history for specific threads."""
    success = service.delete_chats(thread_ids=thread_ids)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete chats history.")
    return MessageResponse(message=f"Chats history for thread(s) '{thread_ids}' have been deleted.")

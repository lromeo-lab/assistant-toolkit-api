import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
#
from app.core.config import Settings, get_settings
from app.services.chat_management_service import ChatManagementService

# Create an API router
router = APIRouter()

# Instantiate the service
settings = get_settings()
chat_management_service = ChatManagementService(settings)

# --- Pydantic Models for Request Bodies ---
class Chat(BaseModel):
    user_query: str
    assistant_response: str
    thread_id: str
    turn_id: int


@router.post("/ingest-chat", status_code=202)
async def ingest_chat(
    background_tasks: BackgroundTasks,
    chat: Chat
):
    """
    Accepts a single conversational turn and schedules it for ingestion
    into the long-term memory vector database.
    """
    logging.info(f"Received request to ingest turn for thread '{chat.thread_id}'. Scheduling background task.")
    
    background_tasks.add_task(
        chat_management_service.ingest_chat,
        user_query=chat.user_query,
        assistant_response=chat.assistant_response,
        thread_id=chat.thread_id,
        turn_id=chat.turn_id
    )
    
    return {"message": "Chat turn received and scheduled for ingestion."}
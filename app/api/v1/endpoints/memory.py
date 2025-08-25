import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel
#
from app.core.config import Settings, get_settings
from app.services.memory_service import MemoryPipeline

# Create an API router
router = APIRouter()

# Instantiate the service
settings = get_settings()
memory_service = MemoryPipeline(settings)

# --- Pydantic Models for Request Bodies ---
class ChatTurn(BaseModel):
    db_name: str
    collection_name: str
    user_query: str
    assistant_response: str
    thread_id: str
    turn_id: Optional[int]  

class ChatHistory(BaseModel):
    db_name: str
    collection_name: str
    thread_id: str

@router.post("/ingest-turn", status_code=202)
async def ingest_chat_turn(
    background_tasks: BackgroundTasks,
    chat_turn: ChatTurn
):
    """
    Accepts a single conversational turn and schedules it for ingestion
    into the long-term memory vector database.
    """
    logging.info(f"Received request to ingest turn for thread '{chat_turn.thread_id}'. Scheduling background task.")
    
    background_tasks.add_task(
        memory_service.run,
        db_name=chat_turn.db_name,
        collection_name=chat_turn.collection_name,
        user_query=chat_turn.user_query,
        assistant_response=chat_turn.assistant_response,
        thread_id=chat_turn.thread_id,
        turn_id=chat_turn.turn_id
    )
    
    return {"message": "Chat turn received and scheduled for ingestion."}

@router.delete("/delete-history")
async def delete_chat_history(
    history: ChatHistory
):
    """
    Deletes an entire chat history from both short-term (Redis) and
    long-term (MongoDB) memory stores.
    """
    logging.info(f"Received request to delete chat history under thread '{history.thread_id}'")
    try:
        memory_service.delete_chat_history(
            db_name=history.db_name,
            collection_name=history.collection_name,
            thread_id=history.thread_id
        )
        return {"message": f"Deletion request for thread '{history.thread_id}' processed successfully."}
    except Exception as e:
        logging.exception("Failed to process delete-history request.")
        raise HTTPException(status_code=500, detail=str(e))

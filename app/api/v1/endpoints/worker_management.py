import logging
import shutil
from fastapi import APIRouter, Depends, Body

from app.services.chat_management_service import ChatManagementService
from app.services.file_management_service import FileManagementService

from app.api.v1.dependencies import get_chat_management_service, get_file_management_service
from app.api.v1.schemas import ChatIngestionRequest, FileIngestionRequest, MessageResponse

router = APIRouter()#include_in_schema=False

# --- Worker Endpoints ---
@router.post("/worker/ingest-chat", response_model=MessageResponse)
async def ingest_chat_worker(
    chat: ChatIngestionRequest = Body(...),
    service: ChatManagementService = Depends(get_chat_management_service)
):
    """
    WORKER ENDPOINT: Performs the actual heavy ingestion of a chat turn.
    """
    logging.info(f"Worker received job to ingest chat turn for thread '{chat.thread_id}'.")
    
    service.ingest_chat(
        user_query=chat.user_query,
        agent_response=chat.agent_response,
        thread_id=chat.thread_id,
        turn_id=chat.turn_id
    )
    
    return MessageResponse(message="Chat turn ingestion completed successfully.")


@router.post("/worker/ingest-files", response_model=MessageResponse)
async def ingest_files_worker(
    request: FileIngestionRequest = Body(...),
    service: FileManagementService = Depends(get_file_management_service)
):
    """
    WORKER ENDPOINT: Performs the actual heavy ingestion of files and cleans up resources.
    """
    logging.info(f"Worker received job to ingest {len(request.file_paths)} files.")
    try:
        service.ingest_files(
            file_paths=request.file_paths,
            owner_user_id=request.owner_user_id,
            agent_id=request.agent_id,
            thread_id=request.thread_id,
            user_ids=request.user_ids
        )
    finally:
        logging.info(f"Worker cleaning up temporary directory: {request.temp_dir}")
        shutil.rmtree(request.temp_dir, ignore_errors=True)
        
    return MessageResponse(message="File ingestion completed successfully.")
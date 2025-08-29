from typing import List
from fastapi import APIRouter, Depends, HTTPException, Body, Query

from app.services.thread_management_service import ThreadManagementService
from app.services.chat_management_service import ChatManagementService
from app.services.file_management_service import FileManagementService
from app.services.validation_management_service import ValidationManagementService

from app.api.v1.schemas import (
    ThreadCreateRequest,
    ThreadDeleteRequest,
    ThreadResponse,
    MessageResponse
)
from app.api.v1.dependencies import (
    get_thread_management_service,
    get_chat_management_service,
    get_file_management_service,
    get_validation_management_service
)

router = APIRouter()

# --- API Endpoints (Standardized) ---

@router.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(
    request: ThreadCreateRequest = Body(...),
    service: ThreadManagementService = Depends(get_thread_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """
    Creates a new thread for a user to chat with a specific agent.
    """
    validation_service.is_valid_agent(agent_id=request.agent_id)
    validation_service.has_access_to_agent(agent_id=request.agent_id, user_id=request.owner_user_id)
    validation_service.is_thread_duplicated(name=request.name)
    new_thread = service.create_thread(
        name=request.name,
        owner_user_id=request.owner_user_id,
        agent_id=request.agent_id
    )
    new_thread["thread_id"] = str(new_thread.pop("_id"))
    new_thread['created_at'] = new_thread['created_at'].isoformat()
    return ThreadResponse(**new_thread)

@router.get("/threads", response_model=List[ThreadResponse])
def list_threads_for_user(
    user_id: str = Query(..., description="List all threads for this user ID."),
    service: ThreadManagementService = Depends(get_thread_management_service)
):
    """
    Lists all threads owned by a specific user.
    """ 
    threads = service.list_threads_for_owner(owner_user_id=user_id)
    for thread in threads:
        thread["thread_id"] = str(thread.pop("_id"))
        thread['created_at'] = thread['created_at'].isoformat()
    return threads

@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    user_id: str = Query(..., description="The ID of the user making the request, for permission checking."),
    service: ThreadManagementService = Depends(get_thread_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Retrieves a single thread by its ID, if the user owns it."""
    validation_service.is_valid_thread(thread_id=thread_id)
    validation_service.is_owner_of_thread(thread_id=thread_id,owner_user_id=user_id) 
    thread = service.get_thread_by_id(thread_id=thread_id)
    thread["thread_id"] = str(thread.pop("_id"))
    thread['created_at'] = thread['created_at'].isoformat()
    return thread

@router.delete("/threads/{thread_id}", response_model=MessageResponse)
def delete_thread(
    thread_id: str,
    request: ThreadDeleteRequest = Body(...),
    service: ThreadManagementService = Depends(get_thread_management_service),
    chat_service: ChatManagementService = Depends(get_chat_management_service),
    file_service: FileManagementService = Depends(get_file_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """
    Deletes a thread and all of its associated files and chat history.
    """
    validation_service.is_valid_thread(thread_id=thread_id)
    validation_service.is_owner_of_thread(thread_id=thread_id,owner_user_id=request.owner_user_id)
    _ = file_service.delete_files_by_metadata({"metadata.thread_id":thread_id})
    _ = chat_service.delete_chats(thread_ids=[thread_id])
    _ = service.delete_thread_by_id(thread_id=thread_id, owner_user_id=request.owner_user_id)
    return MessageResponse(message=f"Thread '{thread_id}' and all associated chat history and files have been deleted.")

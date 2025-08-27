import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.thread_management_service import ThreadManagementService
from app.services.agent_management_service import AgentManagementService
from .agent_management import get_agent_management_service

router = APIRouter()

# --- Dependency Injection ---
_thread_management_service = None
def get_thread_management_service(settings: Settings = Depends(get_settings)) -> ThreadManagementService:
    global _thread_management_service
    if _thread_management_service is None:
        _thread_management_service = ThreadManagementService(settings)
    return _thread_management_service

# --- Pydantic Models (Standardized) ---

class ThreadCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="An optional human-friendly name for the thread.")
    owner_user_id: str = Field(..., description="The ID of the user creating the thread.")
    agent_id: str = Field(..., description="The ID of the agent this thread is associated with.")

class ThreadResponse(BaseModel):
    thread_id: str
    name: Optional[str]
    owner_user_id: str
    agent_id: str
    created_at: str

class DeleteThreadRequest(BaseModel):
    owner_user_id: str = Field(..., description="The ID of the user who owns the thread, for permission validation.")

class MessageResponse(BaseModel):
    """A standard response model for simple messages."""
    message: str

# --- API Endpoints (Standardized) ---

@router.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(
    request: ThreadCreateRequest = Body(...),
    service: ThreadManagementService = Depends(get_thread_management_service),
    agent_service: AgentManagementService = Depends(get_agent_management_service)
):
    """
    Creates a new thread for a user to chat with a specific agent.
    """
    # A user can only create a thread for an agent they have access to.
    if not agent_service.has_access_to_agent(agent_id=request.agent_id, user_id=request.owner_user_id):
        raise HTTPException(status_code=403, detail="Permission denied: You do not have access to this agent.")
    
    try:
        new_thread = service.create_thread(
            name=request.name, 
            owner_user_id=request.owner_user_id,
            agent_id=request.agent_id
        )
        new_thread['created_at'] = new_thread['created_at'].isoformat()
        return ThreadResponse(**new_thread)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        logging.exception("Failed to create thread.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

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
        thread['created_at'] = thread['created_at'].isoformat()
    return threads

@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    user_id: str = Query(..., description="The ID of the user making the request, for permission checking."),
    service: ThreadManagementService = Depends(get_thread_management_service)
):
    """Retrieves a single thread by its ID, if the user owns it."""
    if not service.is_owner_of_thread(thread_id=thread_id, owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Permission denied or thread not found.")
    
    thread = service.get_thread_by_id(thread_id=thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")
        
    thread['created_at'] = thread['created_at'].isoformat()
    return thread

@router.delete("/threads/{thread_id}", response_model=MessageResponse)
def delete_thread(
    thread_id: str,
    request: DeleteThreadRequest = Body(...),
    service: ThreadManagementService = Depends(get_thread_management_service)
):
    """
    Deletes a thread and all of its associated chat history.
    """
    if not service.is_owner_of_thread(thread_id=thread_id, owner_user_id=request.owner_user_id):
        raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to delete this thread.")
    
    success = service.delete_thread(thread_id=thread_id, owner_user_id=request.owner_user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found.")
        
    return MessageResponse(message=f"Thread '{thread_id}' and all associated chat history have been deleted.")

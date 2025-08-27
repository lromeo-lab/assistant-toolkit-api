import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, Body, Query
import tempfile
import os
import shutil
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.file_management_service import FileManagementService
from app.services.agent_management_service import AgentManagementService
from app.services.thread_management_service import ThreadManagementService
from .agent_management import get_agent_management_service
from .thread_management import get_thread_management_service

# Create an API router
router = APIRouter()

# --- Dependency Injection ---
_file_management_service = None
def get_file_management_service(settings: Settings = Depends(get_settings)) -> FileManagementService:
    global _file_management_service
    if _file_management_service is None:
        _file_management_service = FileManagementService(settings)
    return _file_management_service

# --- Background Task Helper ---
def run_ingestion_and_cleanup(service: FileManagementService, temp_dir: str, **kwargs):
    """Wrapper to run ingestion and then clean up the temp directory."""
    try:
        service.ingest_files(**kwargs)
    finally:
        logging.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- Pydantic Models (Standardized) ---
class FileResponse(BaseModel):
    file_id: str
    file_name: str
    user_ids: Optional[List[str]] = None

class FileListResponse(BaseModel):
    files: List[FileResponse]

class DeleteFileRequest(BaseModel):
    owner_user_id: str = Field(..., description="The ID of the user who owns the agent/thread, for permission validation.")

class MessageResponse(BaseModel):
    """A standard response model for simple messages."""
    message: str

class IngestionResponse(BaseModel):
    message: str
    agent_id: str
    filenames: List[str]
    applied_user_ids: List[str]
    excluded_user_ids: List[str]

# --- API Endpoints (Standardized) ---

@router.post("/files", status_code=202, response_model=IngestionResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    owner_user_id: str = Form(..., description="The ID of the user uploading the files, for permission validation."),
    files: List[UploadFile] = File(..., description="A list of files to be ingested."),
    agent_id: Optional[str] = Form(None, description="The unique ID of the agent to associate with these files."),
    thread_id: Optional[str] = Form(None, description="The unique ID of the thread to associate with these files."),
    user_ids: Optional[List[str]] = Form(None, description="A list of user IDs who have access to these files."),
    service: FileManagementService = Depends(get_file_management_service),
    agent_service: AgentManagementService = Depends(get_agent_management_service),
    thread_service: ThreadManagementService = Depends(get_thread_management_service)
):
    """
    Accepts files and ingests them, associating them with either an agent or a thread.
    """
    if user_ids:
        # Check if the list contains a single item that is a comma-separated string
        if len(user_ids) == 1 and ',' in user_ids[0]:
            user_ids = [uid.strip() for uid in user_ids[0].split(',')]
        else:
            user_ids = user_ids

    # Validate provided params
    if not agent_id and not thread_id:
        raise HTTPException(status_code=400, detail="You must provide either an 'agent_id' or a 'thread_id'.")
    if agent_id and thread_id:
        raise HTTPException(status_code=400, detail="Please provide either an 'agent_id' or a 'thread_id', not both.")
    
    final_file_user_ids = []
    excluded_user_ids = []
    
    # --- Conditional Logic for Permissions ---
    if agent_id:
        # AGENT PATH: Apply hierarchical permissions
        if not agent_service.is_owner_of_agent(agent_id=agent_id, owner_user_id=owner_user_id):
            raise HTTPException(status_code=403, detail="Permission denied: You do not own this agent.")
        
        agent = agent_service.get_agent_by_id(agent_id)
        agent_user_ids = agent.get("user_ids")

        permission_result = service.calculate_file_permissions(
            agent_user_ids=agent_user_ids,
            file_user_ids=user_ids,
            owner_user_id=owner_user_id
        )
        final_file_user_ids = permission_result["applied_user_ids"]
        excluded_user_ids = permission_result["excluded_user_ids"]

    elif thread_id:
        # THREAD PATH: Lock permissions to the thread owner
        if not thread_service.is_owner_of_thread(thread_id=thread_id, owner_user_id=owner_user_id):
            raise HTTPException(status_code=403, detail="Permission denied: You do not own this thread.")
        
        final_file_user_ids = [owner_user_id]
        if user_ids:
            logging.warning(f"User IDs provided for thread '{thread_id}' were ignored to enforce thread privacy.")
            excluded_user_ids = list(set(user_ids) - {owner_user_id})

    # Temp directory
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(file_path, "wb") as f: shutil.copyfileobj(file.file, f)
            file_paths.append(file_path)
        finally:
            file.file.close()

    # Run the process
    background_tasks.add_task(
        run_ingestion_and_cleanup,
        service=service,
        temp_dir=temp_dir,
        file_paths=file_paths,
        owner_user_id=owner_user_id,
        agent_id=agent_id,
        thread_id=thread_id,
        user_ids=final_file_user_ids
    )

    return IngestionResponse(
        message="Files received. Ingestion has started in the background.",
        agent_id=agent_id,
        filenames=[f.filename for f in files],
        applied_user_ids=final_file_user_ids,
        excluded_user_ids=excluded_user_ids
    )

@router.get("/agents/{agent_id}/files", response_model=FileListResponse)
def list_files_for_agent(
    agent_id: str,
    user_id: str = Query(..., description="The ID of the user making the request, for permission checking."),
    service: FileManagementService = Depends(get_file_management_service),
    agent_service: AgentManagementService = Depends(get_agent_management_service)
):
    """Lists all unique files for an agent that the user is permitted to see."""
    # 1. First, check if the user has any access to the agent at all.
    if not agent_service.has_access_to_agent(agent_id=agent_id, user_id=user_id):
        raise HTTPException(status_code=403, detail="Permission denied or agent not found.")
    
    # 2. Get the agent's details to find its owner.
    agent = agent_service.get_agent_by_id(agent_id=agent_id)
    if not agent:
         raise HTTPException(status_code=404, detail="Agent not found.")

    # 3. Call the service with the necessary info to get the correctly filtered file list.
    files_data = service.list_files_for_agent(
        agent_id=agent_id, 
        user_id=user_id, 
        agent_owner_id=agent["owner_user_id"]
    )
    return FileListResponse(files=files_data)

@router.get("/users/{user_id}/files", response_model=FileListResponse)
def list_files_for_user(
    user_id: str,
    by_owner: bool = Query(True, description="If true, lists files owned by the user. If false, lists files the user has access to."),
    service: FileManagementService = Depends(get_file_management_service)
):
    """Lists files for a user, either by ownership or by authorized access."""
    if by_owner:
        files_data = service.list_files_for_owner(owner_user_id=user_id)
    else:
        files_data = service.list_files_for_user(user_id=user_id)
    return FileListResponse(files=files_data)

@router.get("/threads/{thread_id}/files", response_model=FileListResponse)
def list_files_for_thread(
    thread_id: str,
    user_id: str = Query(..., description="The ID of the user making the request, for permission checking."),
    service: FileManagementService = Depends(get_file_management_service),
    thread_service: ThreadManagementService = Depends(get_thread_management_service)
):
    """Lists all unique files associated with a specific thread, if the user owns it."""
    if not thread_service.is_owner_of_thread(thread_id=thread_id, owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Permission denied or thread not found.")
        
    files_data = service.list_files_for_thread(thread_id=thread_id)
    return FileListResponse(files=files_data)

@router.delete("/files/{file_id}", response_model=MessageResponse)
def delete_file(
    file_id: str,
    request: DeleteFileRequest = Body(...),
    service: FileManagementService = Depends(get_file_management_service)
):
    """Deletes all nodes for a specific file_id after validating ownership."""
    if not service.is_owner_of_file(file_id=file_id, user_id=request.owner_user_id):
        raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to delete this file.")
    
    deleted_count = service.delete_file_by_id(file_id=file_id)
    
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="File not found.")

    return MessageResponse(message=f"File '{file_id}' and its {deleted_count} associated nodes have been deleted.")
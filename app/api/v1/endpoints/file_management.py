import logging
from typing import Optional, List, Tuple
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Body, Query
import tempfile
import os
import shutil
import httpx

from app.core.clients import get_worker_client

from app.services.agent_management_service import AgentManagementService
from app.services.thread_management_service import ThreadManagementService
from app.services.file_management_service import FileManagementService
from app.services.validation_management_service import ValidationManagementService

# --- Import schemas ---
from app.api.v1.schemas import (
    DeleteFileRequest,
    FileIngestionRequest,
    FileListResponse,
    FileIngestionResponse,
    MessageResponse
)

# --- Import injection dependencies ---
from app.api.v1.dependencies import (
    get_agent_management_service,
    get_thread_management_service,
    get_file_management_service,
    get_validation_management_service
)

# Create an API router
router = APIRouter()

# --- Private Helper Functions for the Upload Endpoint ---
async def call_worker_endpoint(
    url: str,
    payload: dict
):
    """Makes an async HTTP request to a worker endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            logging.info(f"Successfully triggered worker with payload: {payload}")
        except httpx.RequestError as e:
            logging.error(f"Failed to call worker endpoint at {url}: {e}")

def _parse_user_ids(
    user_ids_form: Optional[List[str]]
) -> Optional[List[str]]:
    """Parses user_ids from a form, handling the comma-separated string case."""
    if not user_ids_form:
        return None
    if len(user_ids_form) == 1 and ',' in user_ids_form[0]:
        return [uid.strip() for uid in user_ids_form[0].split(',')]
    return user_ids_form

def _save_files_to_temp_dir(
    files: List[UploadFile]
) -> Tuple[str, List[str]]:
    """Saves uploaded files to a temporary directory and returns the paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(file_path)
        finally:
            file.file.close()
    return temp_dir, file_paths

# --- Main User-Facing Endpoint ---

@router.post("/schedule-ingest-files", status_code=202, response_model=FileIngestionResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: FileIngestionRequest = Body(...),
    # --- Services and their dependencies ---
    worker_url = Depends(get_worker_client),
    agent_service: AgentManagementService = Depends(get_agent_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """
    Accepts files and schedules them for ingestion after validating permissions.
    """
    # 1. Parse and clean input data
    parsed_user_ids = _parse_user_ids(request.user_ids)

    # 2. Handle all permission logic
    validation_service.at_least_thread_or_agent(agent_id=request.agent_id, thread_id=request.thread_id)
    validation_service.not_both_thread_and_agent(agent_id=request.agent_id,thread_id=request.thread_id)
    
    if request.agent_id:
        validation_service.is_valid_agent(agent_id=request.agent_id)
        validation_service.is_owner_of_agent(agent_id=request.agent_id,owner_user_id=request.owner_user_id)
        
        agent = agent_service.get_agent_by_id(agent_id=request.agent_id)
        agent_user_ids = agent.get("user_ids", [])
        final_file_user_ids, excluded_user_ids=validation_service.adjust_file_on_agent_permissions(
            agent_user_ids=agent_user_ids,
            file_user_ids=parsed_user_ids
        )

    if request.thread_id:
        validation_service.is_valid_thread(thread_id=request.thread_id)
        validation_service.is_owner_of_thread(thread_id=request.thread_id,owner_user_id=request.owner_user_id)
        final_file_user_ids, excluded_user_ids=validation_service.adjust_file_on_thread_permissions(
            thread_owner_user_id=request.owner_user_id,
            file_user_ids=parsed_user_ids
        )

    # 3. Handle file I/O
    temp_dir, file_paths = _save_files_to_temp_dir(files)

    # 4. Schedule the background task
    payload = {
        "file_paths": file_paths,
        "temp_dir": temp_dir,
        "owner_user_id": request.owner_user_id,
        "agent_id": request.agent_id,
        "thread_id": request.thread_id,
        "user_ids": final_file_user_ids
    }

    # 5. Run the endpoint in a backgroung task
    background_tasks.add_task(
        call_worker_endpoint,
        url=f"{worker_url}/ingest-files",
        payload=payload
    )

    # 6. Return the response
    return FileIngestionResponse(
        message="Files received. Ingestion has started in the background.",
        agent_id=request.agent_id,
        filenames=[f.filename for f in files],
        applied_user_ids=final_file_user_ids,
        excluded_user_ids=excluded_user_ids
    )

@router.get("/agents/{agent_id}/files", response_model=FileListResponse)
def list_files_for_agent(
    agent_id: str,
    user_id: str = Query(...),
    service: FileManagementService = Depends(get_file_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Lists all unique files for an agent that the user is permitted to see."""
    validation_service.is_valid_agent(agent_id=agent_id)
    validation_service.has_access_to_agent(agent_id=agent_id,user_id=user_id)
    files_data = service.list_files_for_agent(agent_id=agent_id, user_id=user_id)
    return FileListResponse(files=files_data)

@router.get("/users/{user_id}/files", response_model=FileListResponse, response_model_exclude={"files": {"__all__": {"user_ids"}}})
def list_files_for_user(
    user_id: str,
    by_owner: bool = Query(True),
    service: FileManagementService = Depends(get_file_management_service)
):
    """Lists files for a user, either by ownership or by authorized access."""
    if by_owner:
        files_data = service.list_files_for_owner(owner_user_id=user_id)
    else:
        files_data = service.list_files_for_user(user_id=user_id)
    return FileListResponse(files=files_data)

@router.get("/threads/{thread_id}/files", response_model=FileListResponse, response_model_exclude={"files": {"__all__": {"user_ids"}}})
def list_files_for_thread(
    thread_id: str,
    user_id: str = Query(...),
    service: FileManagementService = Depends(get_file_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Lists all unique files associated with a specific thread, if the user owns it."""
    validation_service.is_valid_thread(thread_id=thread_id)
    validation_service.is_owner_of_thread(thread_id=thread_id,owner_user_id=user_id)
    files_data = service.list_files_for_thread(thread_id=thread_id)
    return FileListResponse(files=files_data)

@router.delete("/files/{file_id}", response_model=MessageResponse)
def delete_file(
    file_id: str,
    request: DeleteFileRequest = Body(...),
    service: FileManagementService = Depends(get_file_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Deletes all nodes for a specific file_id after validating ownership."""
    validation_service.is_valid_file(file_id=file_id)
    validation_service.is_owner_of_file(file_id=file_id,owner_user_id=request.owner_user_id)
    deleted_count = service.delete_file_by_id(file_id=file_id)
    return MessageResponse(message=f"File '{file_id}' and its {deleted_count} associated nodes have been deleted.")
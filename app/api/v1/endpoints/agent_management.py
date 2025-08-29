from typing import List
from fastapi import APIRouter, Depends, HTTPException, Body, Query

from app.services.agent_management_service import AgentManagementService
from app.services.thread_management_service import ThreadManagementService
from app.services.file_management_service import FileManagementService
from app.services.chat_management_service import ChatManagementService
from app.services.validation_management_service import ValidationManagementService

from app.api.v1.schemas import (
    AgentCreateRequest,
    AgentDeleteRequest,
    AgentResponse,
    MessageResponse
)
from app.api.v1.dependencies import (
    get_agent_management_service,
    get_thread_management_service,
    get_file_management_service,
    get_chat_management_service,
    get_validation_management_service
)

router = APIRouter()

# --- API Endpoints (Standardized) ---
@router.post("/agents", response_model=AgentResponse, status_code=201)
def create_agent(
    request: AgentCreateRequest = Body(...),
    service: AgentManagementService = Depends(get_agent_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Creates a new agent with llm model configurations."""
    validation_service.is_agent_duplicated(name=request.name)
    config_dict = request.config.model_dump(exclude_unset=True) if request.config else {}#?
    new_agent = service.create_agent(
        name=request.name, 
        owner_user_id=request.owner_user_id,
        config=config_dict,
        user_ids=request.user_ids
    )
    new_agent['created_at'] = new_agent['created_at'].isoformat()
    return AgentResponse(**new_agent)

@router.get("/agents", response_model=List[AgentResponse])
def list_agents_for_user(
    user_id: str = Query(...),
    by_owner: bool = Query(True),
    service: AgentManagementService = Depends(get_agent_management_service)
):
    """
    List the agents. You can list them by users with access or by owners.
    """ 
    if by_owner:
        agents_data = service.list_agents_for_owner(owner_user_id=user_id)
    else:
        agents_data = service.list_agents_for_user(user_id=user_id)
    for agent in agents_data:
        agent['agent_id'] = str(agent.pop('_id'))
        agent['created_at'] = agent['created_at'].isoformat()
    return agents_data

@router.get("/agents/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    user_id: str = Query(...),
    service: AgentManagementService = Depends(get_agent_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Retrieves a single agent by its ID, if the user has access."""
    validation_service.is_valid_agent(agent_id=agent_id)
    validation_service.has_access_to_agent(agent_id=agent_id,user_id=user_id)
    agent = service.get_agent_by_id(agent_id=agent_id)
    agent["agent_id"] = str(agent.pop("_id"))
    agent['created_at'] = agent['created_at'].isoformat()
    return AgentResponse(**agent)

@router.delete("/agents/{agent_id}", response_model=MessageResponse)
def delete_agent(
    agent_id: str,
    request: AgentDeleteRequest = Body(...),
    service: AgentManagementService = Depends(get_agent_management_service),
    thread_service: ThreadManagementService = Depends(get_thread_management_service),
    file_service: FileManagementService = Depends(get_file_management_service),
    chat_service: ChatManagementService = Depends(get_chat_management_service),
    validation_service: ValidationManagementService = Depends(get_validation_management_service)
):
    """Deletes an agent and all of its associated files."""
    validation_service.is_valid_agent(agent_id=agent_id)
    validation_service.is_owner_of_agent(agent_id=agent_id,owner_user_id=request.owner_user_id)
    thread_ids = [] # falta
    _ = file_service.delete_files_by_metadata({"metadata.agent_id":agent_id})
    _ = chat_service.delete_chats(thread_ids=thread_ids)
    _ = thread_service.delete_threads_by_metadata({"agent_id":agent_id})
    _ = service.delete_agent_by_id(agent_id=agent_id)
    return MessageResponse(message=f"Agent '{agent_id}' and all associated files have been deleted.")
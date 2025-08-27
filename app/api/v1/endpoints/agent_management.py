import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.agent_management_service import AgentManagementService

router = APIRouter()

# --- Dependency Injection (remains the same) ---
_agent_management_service = None
def get_agent_management_service(settings: Settings = Depends(get_settings)) -> AgentManagementService:
    global _agent_management_service
    if _agent_management_service is None:
        _agent_management_service = AgentManagementService(settings)
    return _agent_management_service

# --- Pydantic Models (Standardized) ---

class AgentConfig(BaseModel):
    """A model for agent-specific configurations. All fields are optional."""
    llm_model_name: str = Field("gpt-4.1-nano", description="e.g., 'gpt-4.1-nano'")
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    system_prompt: str = Field("You're a usefull assistant.", description="The persona/main instruction for the agent.")
    embedding_model_name: str = Field("text-embedding-3-small", description="e.g., 'text-embedding-3-small'")
    retriever_top_k: int = Field(20, description="Number of chunks returned by the retriever.")
    reranker_top_n: int = Field(3, gt=0, description="The limit of the Cohere re-rank output length.")
    chunk_size: int = Field(512, gt=0, description="Embedding chunk size")
    chunk_overlap: int = Field(50, ge=0, description="Embedding chunk overlap")
    ingestion_batch_size: int = Field(100, gt=0, description="Define file uploading mini-batches")

class AgentCreateRequest(BaseModel):
    name: str = Field(..., description="The human-friendly name for the agent.")
    owner_user_id: str = Field(..., description="The ID of the user creating the agent.")
    config: AgentConfig = Field(None, description="Agent-specific configurations.")
    user_ids: Optional[List[str]] = Field(None, description="The IDs of the users with access to the agent. If None, access will be granted for all users.")

class DeleteAgentRequest(BaseModel):
    owner_user_id: str = Field(..., description="The ID of the user who owns the agent, for permission validation.")

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    owner_user_id: str
    config: Dict
    user_ids: List[str]
    created_at: str

class MessageResponse(BaseModel):
    """A standard response model for simple messages."""
    message: str

# --- API Endpoints (Standardized) ---

@router.post("/agents", response_model=AgentResponse, status_code=201)
def create_agent(
    request: AgentCreateRequest = Body(...),
    service: AgentManagementService = Depends(get_agent_management_service)
):
    """Creates a new agent with llm model configurations."""
    try:
        config_dict = request.config.model_dump(exclude_unset=True) if request.config else {}
        new_agent = service.create_agent(
            name=request.name, 
            owner_user_id=request.owner_user_id,
            config=config_dict,
            user_ids=request.user_ids
        )
        new_agent['created_at'] = new_agent['created_at'].isoformat()
        return AgentResponse(**new_agent)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        logging.exception("Failed to create agent.")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get("/agents", response_model=List[AgentResponse])
def list_agents_for_user(
    user_id: str = Query(..., description="The user ID to filter agents by."),
    by_owner: bool = Query(True, description="If true, lists agents owned by the user. If false, lists agents the user has access to."),
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
        agent['created_at'] = agent['created_at'].isoformat()
    return agents_data

@router.get("/agents/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    user_id: str = Query(..., description="The ID of the user making the request, for permission checking."),
    service: AgentManagementService = Depends(get_agent_management_service)
):
    """Retrieves a single agent by its ID, if the user has access."""
    if not service.has_access_to_agent(agent_id=agent_id, user_id=user_id):
        raise HTTPException(status_code=403, detail="Permission denied or agent not found.")
    
    agent = service.get_agent_by_id(agent_id=agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
        
    agent['created_at'] = agent['created_at'].isoformat()
    return agent

@router.delete("/agents/{agent_id}", response_model=MessageResponse)
def delete_agent(
    agent_id: str,
    request: DeleteAgentRequest = Body(...),
    service: AgentManagementService = Depends(get_agent_management_service)
):
    """Deletes an agent and all of its associated files."""
    if not service.is_owner_of_agent(agent_id=agent_id, owner_user_id=request.owner_user_id):
        raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to delete this agent.")
    
    success = service.delete_agent(agent_id=agent_id, owner_user_id=request.owner_user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found.")
        
    return MessageResponse(message=f"Agent '{agent_id}' and all associated files have been deleted.")
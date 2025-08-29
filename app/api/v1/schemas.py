from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# --- General ---
class MessageResponse(BaseModel):
    message: str

# --- File Management Schemas ---
# ---- File DTOs
class FileBase(BaseModel):
    file_id: str
    file_name: str
    user_ids: List[str] = Field(default_factory=list)

class DeleteFileRequest(BaseModel):
    owner_user_id: str

class FileIngestionRequest(BaseModel):
    owner_user_id: str
    agent_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_ids: List[str] = Field(default_factory=list)

class FileListResponse(BaseModel):
    files: List[FileBase] = Field(default_factory=list)

class FileIngestionResponse(BaseModel):
    message: str
    agent_id: str
    filenames: List[str] = Field(default_factory=list)
    applied_user_ids: List[str] = Field(default_factory=list)
    excluded_user_ids: List[str] = Field(default_factory=list)

# --- Chat Management Schemas ---
class ChatIngestionRequest(BaseModel):
    user_query: str
    agent_response: str
    thread_id: str
    turn_id: int
# I'm missing someone?
# --- Agent Management Schemas ---
class AgentConfig(BaseModel):
    llm_model_name: str
    temperature: float
    system_prompt: str
    reranker_top_n: int

class AgentCreateRequest(BaseModel):
    name: str
    owner_user_id: str
    config: AgentConfig
    user_ids: Optional[List[str]]

class AgentDeleteRequest(BaseModel):
    owner_user_id: str

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    owner_user_id: str
    config: Dict
    user_ids: List[str]
    created_at: str

# --- Thread Management Schemas ---
class ThreadCreateRequest(BaseModel):
    name: Optional[str]
    owner_user_id: str
    agent_id: str

class ThreadDeleteRequest(BaseModel):
    owner_user_id: str

class ThreadResponse(BaseModel):
    thread_id: str
    name: Optional[str]
    owner_user_id: str
    agent_id: str
    created_at: str
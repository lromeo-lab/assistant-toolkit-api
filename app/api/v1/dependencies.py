from fastapi import Depends
from app.core.config import Settings, get_settings

# Import all services
from app.services.validation_management_service import ValidationManagementService
from app.services.agent_management_service import AgentManagementService
from app.services.file_management_service import FileManagementService
from app.services.thread_management_service import ThreadManagementService
from app.services.chat_management_service import ChatManagementService

# Import all shared clients
from app.core.clients import (
    get_mongo_client,
    get_redis_client,
    get_embed_model,
    get_text_splitter
)
import pymongo
import redis
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import TokenTextSplitter

# --- Singleton instances of services ---
_validation_management_service = None
_agent_management_service = None
_thread_management_service = None
_file_management_service = None
_chat_management_service = None


def get_validation_management_service(
    settings: Settings = Depends(get_settings),
    mongo_client: pymongo.MongoClient = Depends(get_mongo_client)
) -> ValidationManagementService:
    global _validation_management_service
    if _validation_management_service is None:
        _validation_management_service = ValidationManagementService(settings, mongo_client)
    return _validation_management_service

def get_agent_management_service(
    settings: Settings = Depends(get_settings),
    mongo_client: pymongo.MongoClient = Depends(get_mongo_client)
) -> AgentManagementService:
    global _agent_management_service
    if _agent_management_service is None:
        _agent_management_service = AgentManagementService(settings, mongo_client)
    return _agent_management_service

def get_thread_management_service(
    settings: Settings = Depends(get_settings),
    mongo_client: pymongo.MongoClient = Depends(get_mongo_client)
) -> ThreadManagementService:
    global _thread_management_service
    if _thread_management_service is None:
        _thread_management_service = ThreadManagementService(settings, mongo_client)
    return _thread_management_service

# --- Files Injection ---
_file_management_service = None
def get_file_management_service(
    settings: Settings = Depends(get_settings),
    mongo_client: pymongo.MongoClient = Depends(get_mongo_client),
    embed_model: BaseEmbedding = Depends(get_embed_model),
    text_splitter: TokenTextSplitter = Depends(get_text_splitter)
) -> FileManagementService:
    global _file_management_service
    if _file_management_service is None:
        _file_management_service = FileManagementService(
            settings=settings,
            mongo_client=mongo_client,
            embed_model=embed_model,
            text_splitter=text_splitter
        )
    return _file_management_service

# --- Chat Injection ---
def get_chat_management_service(
    settings: Settings = Depends(get_settings),
    mongo_client: pymongo.MongoClient = Depends(get_mongo_client),
    redis_client: redis.Redis = Depends(get_redis_client),
    embed_model: BaseEmbedding = Depends(get_embed_model),
    text_splitter: TokenTextSplitter = Depends(get_text_splitter)
) -> ChatManagementService:
    global _chat_management_service
    if _chat_management_service is None:
        _chat_management_service = ChatManagementService(
            settings=settings,
            mongo_client=mongo_client,
            redis_client=redis_client,
            embed_model=embed_model,
            text_splitter=text_splitter
        )
    return _chat_management_service
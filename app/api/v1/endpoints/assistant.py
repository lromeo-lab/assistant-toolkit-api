import logging
import certifi
import pymongo
import openai
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.storage.chat_store.redis import RedisChatStore

from app.core.config import Settings, get_settings
from app.services.assistant_service import RAGAssistantService

# --- API Router Setup ---
router = APIRouter()

# --- Pydantic Models for API Data Validation ---
class ChatRequest(BaseModel):
    agent_id: str = Field(..., description="The unique identifier for the agent.", example="agent-007")
    thread_id: str = Field(..., description="The unique identifier for the conversation thread.", example="thread-abc-123")
    query: str = Field(..., description="The user's question or message.", example="What is the status of project X?")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The assistant's response.", example="The status of project X is on track.")
    thread_id: str = Field(..., description="The conversation thread identifier.", example="thread-abc-123")
    agent_id: str = Field(..., description="The agent identifier.", example="agent-007")

# --- Dependency Injection for the Assistant Service ---
# Global variables to hold the singleton instances of our clients and service.
_mongo_client = None
_chat_store = None
_llm = None
_embed_model = None
_reranker = None
_assistant_service = None

def get_assistant_service(settings: Settings = Depends(get_settings)) -> RAGAssistantService:
    """
    Dependency function to create and return a singleton instance of the RAGAssistantService.
    This ensures that clients and models are initialized only once per application lifecycle.
    """
    global _mongo_client, _chat_store, _llm, _embed_model, _reranker, _assistant_service

    # This block ensures that all expensive objects are created only once.
    if _assistant_service is None:
        logging.info("Initializing shared clients and models for the assistant service...")
        
        openai.api_key = settings.llm.openai_api_key

        try:
            _mongo_client = pymongo.MongoClient(settings.database.mongo_uri, tlsCAFile=certifi.where())
            _mongo_client.admin.command('ping') 
            logging.info("MongoDB connection successful.")
        except pymongo.errors.ConnectionFailure as e:
            logging.error(f"MongoDB connection failed: {e}")
            raise HTTPException(status_code=503, detail="Could not connect to the database.")

        # Correctly initialize RedisChatStore
        _chat_store = RedisChatStore(redis_url=settings.database.redis_url, **{"ssl_ca_certs": certifi.where()})
        
        # Initialize LlamaIndex components using the nested settings
        _llm = OpenAI(
            model=settings.llm.model_name, 
            temperature=settings.llm.temperature, 
            api_key=settings.llm.openai_api_key
        )
        _embed_model = OpenAIEmbedding(
            model=settings.llm.embedding_model_name, 
            api_key=settings.llm.openai_api_key
        )
        _reranker = CohereRerank(
            api_key=settings.llm.cohere_api_key, 
            top_n=settings.llm.reranker_top_n
        )
        
        # Create the service instance with all its dependencies
        _assistant_service = RAGAssistantService(
            mongo_client=_mongo_client,
            chat_store=_chat_store,
            llm=_llm,
            embed_model=_embed_model,
            reranker=_reranker,
            settings=settings
        )
        logging.info("RAGAssistantService singleton instance created.")
        
    return _assistant_service

# --- API Endpoints ---
@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(
    request: ChatRequest = Body(...),
    service: RAGAssistantService = Depends(get_assistant_service)
) -> ChatResponse:
    """
    Main endpoint to interact with the RAG assistant.
    """
    try:
        logging.info(f"Received chat request for agent '{request.agent_id}' in thread '{request.thread_id}'")
        response_text = service.get_chat_response(
            agent_id=request.agent_id,
            thread_id=request.thread_id,
            query=request.query
        )
        return ChatResponse(
            response=response_text,
            thread_id=request.thread_id,
            agent_id=request.agent_id
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred in the chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
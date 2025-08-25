import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env from the project root (optional but fine)
load_dotenv()

# --- Nested Settings Models ---

class DataBaseSettings(BaseSettings):
    """Settings related to data stores like MongoDB and Redis."""
    mongo_uri: str
    redis_url: str
    db_name: str
    file_collection_name: str
    chat_history_collection_name: str
    atlas_vector_index_name: str
    atlas_search_index_name: str
    memory_token_limit: int
    file_retriever_top_k: int
    chat_retriever_top_k: int

    # No env_prefix needed; values come via nested path "DATABASE__..."
    model_config = SettingsConfigDict()

class LlmSettings(BaseSettings):
    """Settings for the LLMs, Embeddings, and Reranker models."""
    openai_api_key: str
    cohere_api_key: str
    model_name: str
    temperature: float
    reranker_top_n: int
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    ingestion_batch_size: int

    model_config = SettingsConfigDict()

class AssistantSettings(BaseSettings):
    """Tunable parameters for the RAG assistant's behavior."""
    chat_search_type: str
    file_search_type: str

    model_config = SettingsConfigDict()

# --- Main Application Settings ---

class Settings(BaseSettings):
    """Main application settings, composed of nested configuration models."""
    project_name: str = "Assistant Toolkit API"
    api_v1_str: str = "/api/v1"
    memory_management_root_endpoint: str

    database: DataBaseSettings
    llm: LlmSettings
    assistant: AssistantSettings

    # This is the important part: inside the class
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',   # enables DATABASE__MONGO_URI, LLM__MODEL_NAME, etc.
        #case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

@lru_cache()
def get_settings() -> "Settings":
    logging.info("Loading application settings...")
    return Settings()
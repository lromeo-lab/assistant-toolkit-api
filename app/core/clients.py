# In a new file: app/core/clients.py
import pymongo
import certifi
import redis
from fastapi import Depends
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from .config import Settings, get_settings

# --- Singleton instances of our clients ---
_mongo_client = None
_redis_client = None
_embed_model = None
_text_splitter = None
_worker_client = None

def get_mongo_client(settings: Settings = Depends(get_settings)) -> pymongo.MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = pymongo.MongoClient(settings.database.mongo_uri, tlsCAFile=certifi.where())
    return _mongo_client

def get_redis_client(settings: Settings = Depends(get_settings)) -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        redis_kwargs = {"ssl_ca_certs": certifi.where()}
        _redis_client = redis.from_url(settings.database.redis_url,**redis_kwargs)
    return _redis_client

def get_embed_model(settings: Settings = Depends(get_settings)) -> BaseEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = OpenAIEmbedding(
            model=settings.llm.embedding_model_name,
            api_key=settings.llm.openai_api_key
        )
    return _embed_model

def get_text_splitter(settings: Settings = Depends(get_settings)) -> TokenTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = TokenTextSplitter(
                chunk_size=settings.llm.chunk_size,
                chunk_overlap=settings.llm.chunk_overlap
            )
    return _text_splitter

def get_worker_client(settings: Settings = Depends(get_settings)):#Change the name to get_worker_url
    global _worker_client
    if _worker_client is None:
        _worker_client = f"{settings.internal_worker_url}{settings.api_v1_str}/worker/"
    return _worker_client
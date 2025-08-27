from typing import Optional
import logging
import pymongo
import certifi
from pymongo.operations import SearchIndexModel

# --- LlamaIndex and MongoDB Imports ---
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.storage.chat_store.redis import RedisChatStore

from app.core.config import Settings

class ChatManagementService:
    """
    A service class for processing chats and uploading them to a specified MongoDB Atlas collection,
    ensuring the necessary hybrid search indexes are created automatically.
    """
    def __init__(self, settings: Settings):
        """
        Initializes the pipeline with necessary configurations and clients.
        """
        self.settings = settings
        logging.info("Initializing MemoryPipeline service for MongoDB Atlas...")
        
        self.embed_model: BaseEmbedding = OpenAIEmbedding(
            model=self.settings.llm.embedding_model_name, 
            api_key=self.settings.llm.openai_api_key
        )
        
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.settings.llm.chunk_size,
            chunk_overlap=self.settings.llm.chunk_overlap
        )
        
        if not settings.database.mongo_uri:
            raise ValueError("MONGO_URI not found in .env file. Please add it.")
        self.mongo_client = pymongo.MongoClient(settings.database.mongo_uri, tlsCAFile=certifi.where())
        
        if not settings.database.redis_url:
            raise ValueError("REDIS_URL not found in .env file. Please add it.")
        self.redis_chat_store = RedisChatStore(redis_url=settings.database.redis_url, **{"ssl_ca_certs": certifi.where()})

        self.db_name = self.settings.database.db_name
        self.chat_collection_name = self.settings.database.chat_collection_name
        self.vector_index_name = self.settings.database.atlas_vector_index_name
        self.search_index_name = self.settings.database.atlas_search_index_name
        
        logging.info("ChatMemoryService initialized successfully.")

    def _ensure_chat_indexes(self):
        """
        Creates the necessary Vector Search and Full Text Search indexes if they don't exist.
        This should be called AFTER documents have been inserted into the collection.
        """
        collection = self.mongo_client[self.db_name][self.chat_collection_name]
        
        vector_search_model = SearchIndexModel(
            definition={
                "fields": [
                    {"type": "vector","path": "embedding","numDimensions": 1536,"similarity": "cosine"},
                    {"type": "filter", "path": "metadata.thread_id"},
                    {"type": "filter", "path": "metadata.turn_id"}
                ]
            },
            name=self.vector_index_name,
            type="vectorSearch",
        )
        
        full_text_model = SearchIndexModel(
            definition={"mappings": {"dynamic": True, "fields": {"text": {"type": "string"}}}},
            name=self.search_index_name,
            type="search",
        )
        
        # We reached the maximum number of indexes for free tier (3), 2 for docs and 1 for chat history (Text Search)
        # - IN PROD: we can activate this vector index to perform hybrid search over chats
        index_names = [self.search_index_name] # add self.vector_index_name in PROD
        models = [full_text_model] # add vector_search_model in PROD

        for index_name,model in zip(index_names,models):
            try:
                logging.info(f"Ensuring chat index '{index_name}' exists...")
                collection.create_search_index(model=model)
            except pymongo.errors.OperationFailure as e:
                if "already exists" in str(e).lower():
                    logging.warning(f"Index '{index_name}' already exists. Skipping creation.")
                else:
                    raise e

    def _load_and_prepare_chat(self, user_query: str, agent_response: str, thread_id: str, turn_id: int) -> Document:
        """
        Loads a chat interaction and enriches it with the provided metadata.
        """
        turn_text = f"User: {user_query}\nAgent: {agent_response}"
        
        doc = Document(
            text=turn_text,
            metadata={
                "thread_id": thread_id,
                "turn_id": turn_id
            }
        )
        return doc

    def ingest_chat(self, user_query: str, agent_response: str, thread_id: str, turn_id: int):
        """
        Runs the memory pipeline on a specific MongoDB database and collection.
        """

        logging.info(f"--- Starting batch ingestion for MongoDB collection '{self.db_name}.{self.chat_collection_name}' ---")
        
        try:

            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=self.mongo_client,
                db_name=self.db_name,
                collection_name=self.chat_collection_name,
                vector_index_name=self.vector_index_name,
                fulltext_index_name=self.search_index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            doc = self._load_and_prepare_chat(
                user_query,
                agent_response,
                thread_id,
                turn_id
            )

            if not doc:
                logging.warning("No chats were loaded. Aborting memory loading.")
                return

            nodes = self.text_splitter.get_nodes_from_documents([doc], show_progress=True)
            logging.info(f"Split chat into {len(nodes)} text chunks (nodes).")

            VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            
            # Now that the collection exists and has data, ensure the search indexes are created on it.
            self._ensure_chat_indexes()
            
            logging.info(f"--- Successfully processed and indexed all nodes for collection '{self.db_name}.{self.chat_collection_name}' ---")
        
        except Exception:
            logging.exception("An unexpected error occurred during the memory run.")

import logging
import pymongo
import redis

# --- LlamaIndex and MongoDB Imports ---
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
#from llama_index.storage.chat_store.redis import RedisChatStore

from app.core.config import Settings

class ChatManagementService:
    """
    A service class for managing chat history in both MongoDB (long-term)
    and Redis (short-term), using shared, injected clients.
    """
    def __init__(
        self, 
        settings: Settings,
        mongo_client: pymongo.MongoClient,
        redis_client: redis.Redis,
        embed_model: BaseEmbedding,
        text_splitter: TokenTextSplitter
    ):
        """
        Initializes the service with shared clients instead of creating new ones.
        """
        logging.info("Initializing ChatManagementService with injected clients...")
        
        # --- Use Injected, Shared Clients ---
        self.mongo_client = mongo_client
        self.redis_client = redis_client
        self.embed_model = embed_model
        self.text_splitter = text_splitter
        
        # --- Database and Collection Setup ---
        self.db_name = settings.database.db_name
        self.chat_collection_name = settings.database.chat_collection_name
        self.chat_collection = self.mongo_client[self.db_name][self.chat_collection_name]
        
        logging.info("ChatManagementService initialized successfully.")

    #def get_redis_chat_store(self) -> RedisChatStore:
    #    """
    #    Returns a LlamaIndex RedisChatStore configured with our shared, persistent client.
    #    """
    #    return RedisChatStore(redis_client=self.redis_client)

    def delete_chats(self, thread_ids: list) -> bool:
        """
        Deletes a chat history from both Redis (short-term) and MongoDB (long-term).
        """
        logging.info(f"Deleting all chat history for thread ids '{thread_ids}'...")
        try:
            redis_key = [f"chat_store/{thread_id}" for thread_id in thread_ids]
            deleted_redis_keys = self.redis_client.delete(*redis_key)
            logging.info(f"Deleted {deleted_redis_keys} key(s) from Redis.")
            query_filter = {"metadata.thread_id": {"$in": thread_ids}}
            mongo_result = self.chat_collection.delete_many(query_filter)
            logging.info(f"Deleted {mongo_result.deleted_count} instances from MongoDB.")
            return True
        except Exception as e:
            logging.exception(f"An error occurred while deleting chat history for threads '{thread_ids}': {e}")
            return False

    def ingest_chat(self, user_query: str, agent_response: str, thread_id: str, turn_id: int):
        """
        Ingests a single conversational turn into MongoDB for long-term retrieval.
        """
        logging.info(f"--- Ingesting chat turn {turn_id} for thread '{thread_id}' into MongoDB ---")
        try:
            text = f"User: {user_query}\nAgent: {agent_response}"
            metadata = {"thread_id": thread_id, "turn_id": turn_id}
            doc = Document(text=text, metadata=metadata)

            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=self.mongo_client,
                db_name=self.db_name,
                collection_name=self.chat_collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            nodes = self.text_splitter.get_nodes_from_documents([doc], show_progress=False)
            if not nodes:
                logging.warning("No nodes were produced from chat turn. Aborting ingestion.")
                return

            index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=self.embed_model)
            index.insert_nodes(nodes)
            
            logging.info(f"--- Successfully indexed chat turn {turn_id} ---")
        
        except Exception:
            logging.exception("An unexpected error occurred during the chat ingestion run.")
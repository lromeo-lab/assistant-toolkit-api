import logging
import pymongo
import certifi
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from llama_index.storage.chat_store.redis import RedisChatStore
from app.core.config import Settings

class ThreadManagementService:
    """
    A service class for managing threads, including creation, retrieval, deletion,
    and validation against a MongoDB collection.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mongo_client = pymongo.MongoClient(
            settings.database.mongo_uri,
            tlsCAFile=certifi.where()
        )
        self.redis_chat_store = RedisChatStore(
            redis_url=settings.database.redis_url,
            **{"ssl_ca_certs": certifi.where()}
        )
        self.db = self.mongo_client[settings.database.db_name]
        self.threads_collection = self.db[settings.database.thread_collection_name]
        self.chat_collection = self.db[settings.database.chat_collection_name]
        self.files_collection = self.db[settings.database.file_collection_name]
        logging.info("ThreadManagementService initialized.")

    def _generate_unique_id(self) -> str:
        """Generates a unique, prefixed ID for a new thread."""
        return f"thd_{uuid.uuid4().hex}"

    def create_thread(self, name: str, owner_user_id: str, agent_id: str) -> Dict:
        """
        Creates a new thread for a specific user, with optional configurations.
        """
        if self.threads_collection.find_one({"name": name, "owner_user_id": owner_user_id}):
            raise ValueError(f"Thread with name '{name}' already exists for this user.")

        new_thread = {
            "_id": self._generate_unique_id(),
            "name": name,
            "owner_user_id": owner_user_id,
            "agent_id": agent_id,
            "created_at": datetime.now(timezone.utc)
        }
        self.threads_collection.insert_one(new_thread)
        logging.info(f"Created new thread '{name}' with ID '{new_thread['_id']}' for user '{owner_user_id}'.")
        
        new_thread["thread_id"] = new_thread.pop("_id")
        return new_thread

    def delete_thread(self, thread_id: str, owner_user_id: str) -> bool:
        """
        Deletes an thread and all associated files.
        Validates that the user making the request is the owner of the thread.
        """
        thread = self.threads_collection.find_one({"_id": thread_id, "owner_user_id": owner_user_id})
        if not thread:
            # Return False if the thread doesn't exist or the user is not the owner
            logging.warning(f"Delete failed: Thread '{thread_id}' not found or user '{owner_user_id}' is not the owner.")
            return False

        # --- Delete associated files ---
        logging.info(f"Deleting all files associated with thread '{thread_id}'...")
        delete_result = self.files_collection.delete_many({"metadata.thread_id": thread_id})
        logging.info(f"Deleted {delete_result.deleted_count} associated files.")

        # --- Delete associated chats in long term memory ---
        logging.info(f"Deleting all chats in long term memory associated with thread '{thread_id}'...")
        delete_result = self.chat_collection.delete_many({"metadata.thread_id": thread_id})
        logging.info(f"Deleted long term memory from Mongo for thread '{thread_id}'.")

        # --- Delete associated chats in short term memory ---
        logging.info(f"Deleting all chats in short term memory associated with thread '{thread_id}'.")
        self.redis_chat_store.delete_messages(thread_id)
        logging.info(f"Deleted short term memory from Redis for thread '{thread_id}'.")

        # --- Now, delete the thread itself ---
        result = self.threads_collection.delete_one({"_id": thread_id})
        if result.deleted_count > 0:
            logging.info(f"Successfully deleted thread '{thread_id}'.")
            return True
        
        return False

    def get_thread_by_id(self, thread_id: str) -> Optional[Dict]:
        """Retrieves a single thread by its unique ID."""
        thread = self.threads_collection.find_one({"_id": thread_id})
        thread["thread_id"] = thread.pop("_id")
        return thread

    def list_threads_for_owner(self, owner_user_id: str) -> List[Dict]:
        """Lists all threads owned by a specific user."""
        threads = list(self.threads_collection.find({"owner_user_id": owner_user_id}))
        for thread in threads:
            thread["thread_id"] = thread.pop("_id")
        return threads
    
    def is_owner_of_thread(self, thread_id: str, owner_user_id: str) -> bool:
        """Checks if a user is the owner of the agent that a file belongs to."""
        thread = self.threads_collection.find_one({"_id": thread_id, "owner_user_id": owner_user_id})
        return thread is not None
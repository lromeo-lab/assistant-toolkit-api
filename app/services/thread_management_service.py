import logging
import pymongo

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import uuid

from app.core.config import Settings

class ThreadManagementService:
    """
    A service class for managing threads, including creation, retrieval, deletion,
    and validation against a MongoDB collection.
    """
    def __init__(
            self,
            settings: Settings,
            mongo_client: pymongo.MongoClient
        ):
        # --- Use Injected, Shared Clients ---
        self.mongo_client = mongo_client

        # --- Database and Collection Setup ---
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
        new_thread = {
            "_id": self._generate_unique_id(),
            "name": name,
            "owner_user_id": owner_user_id,
            "agent_id": agent_id,
            "created_at": datetime.now(timezone.utc)
        }
        self.threads_collection.insert_one(new_thread)
        logging.info(f"Created new thread '{name}' with ID '{new_thread['_id']}' for user '{owner_user_id}'.")
        return new_thread
    
    def get_thread_by_id(self, thread_id: str) -> Optional[Dict]:
        """Retrieves a single thread by its unique ID."""
        thread = self.threads_collection.find_one({"_id": thread_id})
        return thread

    def list_threads_for_owner(self, owner_user_id: str) -> List[Dict]:
        """Lists all threads owned by a specific user."""
        threads = list(self.threads_collection.find({"owner_user_id": owner_user_id}))
        return threads

    def delete_thread_by_id(self,thread_id: str,owner_user_id: str) -> bool:
        """
        Deletes an thread and all associated files.
        Validates that the user making the request is the owner of the thread.
        """
        try:
            _ = self.threads_collection.delete_one({"_id": thread_id})
            logging.info(f"Successfully deleted thread_id '{thread_id}'.")
            return True
        except Exception:
            logging.exception(f"An error occurred during thread deletion for thread_id '{thread_id}'.")
            return False
        
    def delete_threads_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Deletes all the threads instances based on a metadata filter.
        """
        if not metadata_filter:
            logging.warning("delete_threads_by_metadata called with an empty filter. Aborting to prevent accidental mass deletion.")
            return False
        logging.info(f"Attempting to delete threads with filter: {metadata_filter}")
        try:
            result = self.threads_collection.delete_many(metadata_filter)
            logging.info(f"Successfully deleted {result.deleted_count} thread instances.")
            return True
        except Exception as e:
            logging.exception(f"An error occurred during metadata deletion: {e}")
            return False
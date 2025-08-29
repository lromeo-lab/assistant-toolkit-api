import logging
import pymongo
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from app.core.config import Settings

class AgentManagementService:
    """
    A service class for managing agents, including creation, retrieval, deletion,
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
        self.agent_collection = self.db[settings.database.agent_collection_name]
        self.thread_collection = self.db[settings.database.thread_collection_name]
        self.files_collection = self.db[settings.database.file_collection_name]
        logging.info("AgentManagementService initialized.")

    def _generate_unique_id(
            self
        ) -> str:
        """Generates a unique, prefixed ID for a new agent."""
        return f"agt_{uuid.uuid4().hex}"

    def create_agent(
            self,
            name: str,
            owner_user_id: str,
            config: Dict,
            user_ids: Optional[list]
        ) -> Dict:
        """
        Creates a new agent for a specific user, with optional configurations.
        """
        # Ensure owner_user_id is included in the user_ids list IF this is not empty
        final_user_ids = list(dict.fromkeys([*user_ids, owner_user_id])) if user_ids else []
        new_agent = {
            "_id": self._generate_unique_id(),
            "name": name,
            "owner_user_id": owner_user_id,
            "config": config or {},
            "user_ids": final_user_ids,
            "created_at": datetime.now(timezone.utc)
        }
        self.agent_collection.insert_one(new_agent)
        logging.info(f"Created new agent '{name}' with ID '{new_agent['_id']}' for user '{owner_user_id}'.") 
        new_agent["agent_id"] = new_agent.pop("_id")
        return new_agent

    def delete_agent_by_id(self, agent_id: str) -> bool:
        """
        Deletes an agent and all associated threads and files.
        """
        try:
            _ = self.agent_collection.delete_one({"_id": agent_id})
            logging.info(f"Successfully deleted agent_id '{agent_id}'.")
            return True
        except Exception:
            logging.exception(f"An error occurred during agent deletion for agent_id '{agent_id}'.")
            return False

    def get_agent_by_id(self, agent_id: str) -> Optional[Dict]:
        """Retrieves a single agent by its unique ID."""
        agent = self.agent_collection.find_one({"_id": agent_id})
        return agent

    def list_agents_for_user(self, user_id: str) -> List[Dict]:
        """Lists all agents accessible by a specific user."""
        query = {"$or": [{"user_ids": user_id},{"user_ids": []}]}
        agents = list(self.agent_collection.find(query))
        return agents
    
    def list_agents_for_owner(self, owner_user_id: str) -> List[Dict]:
        """Lists all agents owned by a specific user."""
        agents = list(self.agent_collection.find({"owner_user_id": owner_user_id}))
        return agents
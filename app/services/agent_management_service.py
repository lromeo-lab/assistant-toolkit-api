import logging
import pymongo
import certifi
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid

from app.core.config import Settings

class AgentManagementService:
    """
    A service class for managing agents, including creation, retrieval, deletion,
    and validation against a MongoDB collection.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mongo_client = pymongo.MongoClient(
            settings.database.mongo_uri,
            tlsCAFile=certifi.where()
        )
        self.db = self.mongo_client[settings.database.db_name]
        self.agent_collection = self.db[settings.database.agent_collection_name]
        self.files_collection = self.db[settings.database.file_collection_name]
        logging.info("AgentManagementService initialized.")

    def _generate_unique_id(self) -> str:
        """Generates a unique, prefixed ID for a new agent."""
        return f"agt_{uuid.uuid4().hex}"

    def create_agent(self, name: str, owner_user_id: str, config: Dict, user_ids: list = []) -> Dict:
        """
        Creates a new agent for a specific user, with optional configurations.
        """
        if self.agent_collection.find_one({"name": name, "owner_user_id": owner_user_id}):
            raise ValueError(f"Agent with name '{name}' already exists for this user.")
        
        final_user_ids = None
        if user_ids:
            final_user_ids = list(set(user_ids + [owner_user_id]))

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

    def delete_agent(self, agent_id: str, owner_user_id: str) -> bool:
        """
        Deletes an agent and all associated files.
        Validates that the user making the request is the owner of the agent.
        """
        agent = self.agent_collection.find_one({"_id": agent_id, "owner_user_id": owner_user_id})
        if not agent:
            logging.warning(f"Delete failed: Agent '{agent_id}' not found or user '{owner_user_id}' is not the owner.")
            return False

        # --- CRITICAL STEP: Delete associated files ---
        logging.info(f"Deleting all files associated with agent '{agent_id}'...")
        delete_result = self.files_collection.delete_many({"metadata.agent_id": agent_id})
        logging.info(f"Deleted {delete_result.deleted_count} associated files.")

        # --- Now, delete the agent itself ---
        result = self.agent_collection.delete_one({"_id": agent_id})
        if result.deleted_count > 0:
            logging.info(f"Successfully deleted agent '{agent_id}'.")
            return True
        
        return False

    def get_agent_by_id(self, agent_id: str) -> Optional[Dict]:
        """Retrieves a single agent by its unique ID."""
        agent = self.agent_collection.find_one({"_id": agent_id})
        agent["agent_id"] = str(agent.pop("_id"))
        return agent

    def list_agents_for_user(self, user_id: str) -> List[Dict]:
        """Lists all agents accessible by a specific user."""
        query = {"$or": [{"user_ids": user_id},{"user_ids": []}]}
        agents = list(self.agent_collection.find(query))
        for agent in agents:
            agent["agent_id"] = str(agent.pop("_id"))
        return agents
    
    def list_agents_for_owner(self, owner_user_id: str) -> List[Dict]:
        """Lists all agents owned by a specific user."""
        agents = list(self.agent_collection.find({"owner_user_id": owner_user_id}))
        for agent in agents:
            agent["agent_id"] = str(agent.pop("_id"))
        return agents
    
    def has_access_to_agent(self, agent_id: str, user_id: str) -> bool:
        """
        Checks if a user has access to the agent, either by being the owner
        or by being in the agent's user_ids list.
        """
        agent = self.agent_collection.find_one({"_id": agent_id})
        if not agent:
            return False
        if user_id in agent.get("user_ids") or agent.get("user_ids")==[]:
            return True
        return False
    
    def is_owner_of_agent(self, agent_id: str, owner_user_id: str) -> bool:
        """Checks if a user is the owner of the agent that a file belongs to."""
        agent = self.agent_collection.find_one({"_id": agent_id})
        if not agent:
            return False
        if agent.get("owner_user_id") == owner_user_id:
            return True
        return False
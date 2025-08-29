import pymongo
from fastapi import HTTPException
from typing import Optional, List, Tuple

from app.core.config import Settings

class ValidationManagementService:
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
        self.file_collection = self.db[settings.database.file_collection_name]

    
    # --- Agent-based validation functions ---

    def is_valid_agent(self, agent_id: str):
        """
        Checks if an agent exist checking by id.
        """
        agent = self.agent_collection.find_one({"_id": agent_id})
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found.")
        else:
            pass

    def is_agent_duplicated(self, name:str):
        """
        Checks if an agent's name already exists.
        """
        if self.agent_collection.find_one({"name": name}):
            raise HTTPException(status_code=409, detail="Agent name already exists.")
        else:
            pass
    
    def is_owner_of_agent(self, agent_id: str, owner_user_id: str) -> bool:
        """
        Checks if a user is the owner of the agent.
        """
        agent = self.agent_collection.find_one({"_id": agent_id})
        if not agent.get("owner_user_id") == owner_user_id:
            raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to perform this task.")
        else:
            pass

    def has_access_to_agent(self, agent_id: str, user_id: str) -> bool:
        """
        Checks if a user has access to the agent.
        """
        agent = self.agent_collection.find_one({"_id": agent_id})
        if user_id not in agent.get("user_ids") and agent.get("user_ids")!=[]:
            raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to perform this task.")
        else:
            pass
    

    # --- Thread-based validation functions ---
    def is_valid_thread(self, thread_id: str):
        """
        Checks if a thread exist checking by id.
        """
        agent = self.thread_collection.find_one({"_id": thread_id})
        if not agent:
            raise HTTPException(status_code=404, detail="Thread not found.")
        else:
            pass

    def is_thread_duplicated(self, name:str):
        """
        Checks if an file's name already exists.
        """
        if self.thread_collection.find_one({"name": name}):
            raise HTTPException(status_code=409, detail="Thread name already.")
        else:
            pass
    
    def is_owner_of_thread(self, thread_id: str, owner_user_id: str) -> bool:
        """
        Checks if a user is the owner of the thread.
        """
        thread = self.thread_collection.find_one({"_id": thread_id})
        if not thread.get("owner_user_id") == owner_user_id:
            raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to perform this task.")
        else:
            pass

    
    # --- Thread-based validation functions ---
    def is_valid_file(self, file_id: str):
        """
        Checks if a file exist checking by id.
        """
        file = self.file_collection.find_one({"_id": file_id})
        if not file:
            raise HTTPException(status_code=404, detail="File not found.")
        else:
            pass

    def is_file_duplicated(self, name:str):
        """
        Checks if an file's name already exists.
        """
        if self.file_collection.find_one({"name": name}):
            raise HTTPException(status_code=409, detail="File name already exists.")
        else:
            pass
    
    def is_owner_of_file(self, file_id: str, owner_user_id: str) -> bool:
        """
        Checks if a user is the owner of the file.
        """
        file = self.file_collection.find_one({"_id": file_id})
        if not file.get("owner_user_id") == owner_user_id:
            raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to perform this task.")
        else:
            pass

    def has_access_to_file(self, file_id: str, user_id: str) -> bool:
        """
        Checks if a user has access to the file.
        """
        file = self.file_collection.find_one({"_id": file_id})
        if user_id not in file.get("user_ids") and file.get("user_ids")!=[]:
            raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to perform this task.")
        else:
            pass
    
    def adjust_file_on_agent_permissions(
            self,
            agent_user_ids: List[str],
            file_user_ids: Optional[List[str]]
        ) -> Tuple[list]:
        """
        Handles ON CASCADE permission adjustment logic for assigned users based on the agent permissions.
        """
        # --- Rule 1: File inherits permissions from the agent ---
        if file_user_ids is None or file_user_ids==[]:
            return (agent_user_ids, [])
        # --- Rule 2: File has specific permissions ---
        agent_permissions = set(agent_user_ids)
        file_request_permissions = set(file_user_ids)
        # If the agent is public (empty list), all requested users are applied.
        if not agent_permissions:
            applied_ids, excluded_ids = file_request_permissions, set()
        else:
            # If the agent is restricted, find the intersection.
            applied_ids = agent_permissions.intersection(file_request_permissions)
            excluded_ids = file_request_permissions.difference(agent_permissions)
        return (applied_ids, excluded_ids)
    
    def adjust_file_on_thread_permissions(
            self,
            thread_owner_user_id: str,
            file_user_ids: Optional[List[str]]
        ) -> Tuple[list]:
        """
        Handles ON CASCADE permission adjustment logic for assigned users based on the thread permissions.
        """
        applied_ids = [thread_owner_user_id]
        excluded_ids = list(set(file_user_ids or []) - {thread_owner_user_id})
        return applied_ids, excluded_ids
    
    # --- General validation functions ---
    def at_least_thread_or_agent(
            self,
            agent_id: Optional[str],
            thread_id: Optional[str]
        ) -> None:
        """
        Require either agent_id or thread_id.
        """
        if not agent_id and not thread_id:
            raise HTTPException(status_code=400, detail="You must provide either an 'agent_id' or a 'thread_id'.")
        else:
            pass

    def not_both_thread_and_agent(
            self,
            agent_id: Optional[str],
            thread_id: Optional[str]
        ) -> None:
        """
        Forbid sending both identifiers.
        """
        if agent_id and thread_id:
            raise HTTPException(status_code=400, detail="Please provide either an 'agent_id' or a 'thread_id', not both.")
        else:
            pass
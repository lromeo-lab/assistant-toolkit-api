import os
from typing import Optional, List, Dict, Any
import uuid
import logging
import pymongo
import certifi
from pymongo.operations import SearchIndexModel
from datetime import datetime, timezone

# --- LlamaIndex and MongoDB Imports ---
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
#from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from app.core.config import Settings

class FileManagementService:
    """
    A service class for processing documents and uploading them to a specified MongoDB Atlas collection,
    ensuring the necessary hybrid search indexes are created automatically.
    """
    def __init__(self, settings: Settings):
        """
        Initializes the pipeline with necessary configurations and clients.
        """
        self.settings = settings
        logging.info("Initializing IngestionPipeline service for MongoDB Atlas...")
        self.embed_model: BaseEmbedding = OpenAIEmbedding(
            model=self.settings.llm.embedding_model_name, 
            api_key=self.settings.llm.openai_api_key
        )
        if not settings.database.mongo_uri:
            raise ValueError("MONGO_URI not found in .env file. Please add it.")
        self.mongo_client = pymongo.MongoClient(settings.database.mongo_uri, tlsCAFile=certifi.where())
        
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.settings.llm.chunk_size,
            chunk_overlap=self.settings.llm.chunk_overlap
        )

        self.db_name = settings.database.db_name
        self.db = self.mongo_client[self.db_name]

        self.file_collection_name = self.settings.database.file_collection_name
        self.file_collection = self.db[self.file_collection_name]
        
        self.vector_index_name = self.settings.database.atlas_vector_index_name
        self.search_index_name = self.settings.database.atlas_search_index_name
        logging.info("IngestionPipeline service initialized successfully.")

    def _generate_file_id(self) -> str:
        """Generates a unique, prefixed ID for a new file."""
        return f"file_{uuid.uuid4().hex}"
    
    def _ensure_atlas_indexes(self):
        """
        Creates the necessary Vector Search and Full Text Search indexes if they don't exist.
        This should be called AFTER documents have been inserted into the collection.
        """
        
        vector_search_model = SearchIndexModel(
            definition={
                "fields": [
                    {"type": "vector","path": "embedding","numDimensions": 1536,"similarity": "cosine"},
                    {"type": "filter", "path": "metadata.agent_id"},
                    {"type": "filter", "path": "metadata.thread_id"},
                    {"type": "filter", "path": "metadata.user_ids"}
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
        
        index_names = [self.vector_index_name, self.search_index_name]
        models = [vector_search_model, full_text_model]

        for index_name,model in zip(index_names,models):
            try:
                logging.info(f"Ensuring file index '{index_name}' exists...")
                self.file_collection.create_search_index(model=model)
            except pymongo.errors.OperationFailure as e:
                if "already exists" in str(e).lower():
                    logging.warning(f"Index '{index_name}' already exists. Skipping creation.")
                else:
                    raise e

    def _load_and_prepare_docs(
            self,
            file_id_map: Dict[str, str],
            owner_user_id: str,
            agent_id: Optional[str],
            thread_id: Optional[str],
            user_ids: Optional[list]
        ) -> List[Document]:
        """
        Loads one or more documents and enriches them with the provided metadata.
        """
        all_docs = []
        # Loop through each file to load it and assign its unique metadata
        for file_id, file_path in file_id_map.items():
            try:
                reader = SimpleDirectoryReader(input_files=[file_path])
                docs = reader.load_data()
                logging.info(f"Loaded {len(docs)} document object(s) from '{os.path.basename(file_path)}'.")

                for doc in docs:
                    doc.metadata["file_id"] = file_id
                    doc.metadata["file_name"] = os.path.basename(file_path)
                    doc.metadata["owner_user_id"] = owner_user_id
                    doc.metadata["user_ids"] = user_ids
                    if agent_id:
                        doc.metadata["agent_id"] = agent_id
                    elif thread_id:
                        doc.metadata["thread_id"] = thread_id
                    doc.metadata["created_at"] = datetime.now(timezone.utc)
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Failed to load or prepare file {file_path}: {e}")
        
        logging.info(f"Successfully prepared a total of {len(all_docs)} document objects.")
        return all_docs

    def ingest_files(
            self,
            file_paths: List[str],
            owner_user_id: str,
            agent_id: Optional[str] = None,
            thread_id: Optional[str] = None,
            user_ids: Optional[list] = []
        ):
        """
        Creates and uploads a new vectorized file on a specific MongoDB database and collection.
        """
        if not agent_id and not thread_id:
            logging.error("INGESTION FAILED: You must provide either an 'agent_id' or a 'thread_id'.")
            return
        
        final_user_ids = []
        if user_ids:
            final_user_ids = list(set(user_ids + [owner_user_id]))

        logging.info(f"--- Starting batch ingestion for MongoDB collection '{self.db_name}.{self.file_collection_name}' ---")
        
        try:
            file_id_map = {self._generate_file_id(): path for path in file_paths}
            logging.info(f"Generated unique IDs for {len(file_id_map)} files.")

            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=self.mongo_client,
                db_name=self.db_name,
                collection_name=self.file_collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            documents = self._load_and_prepare_docs(file_id_map, owner_user_id, agent_id, thread_id, final_user_ids)
            if not documents:
                logging.warning("No documents were loaded. Aborting ingestion.")
                return

            nodes = self.text_splitter.get_nodes_from_documents(documents, show_progress=True)
            logging.info(f"Split documents into {len(nodes)} text chunks (nodes).")

            # Ingest the data in memory-safe batches.
            for i in range(0, len(nodes), self.settings.llm.ingestion_batch_size):
                batch_nodes = nodes[i:i + self.settings.llm.ingestion_batch_size]
                batch_num = (i // self.settings.llm.ingestion_batch_size) + 1
                total_batches = (len(nodes) + self.settings.llm.ingestion_batch_size - 1) // self.settings.llm.ingestion_batch_size
                
                logging.info(f"--- Processing Batch {batch_num}/{total_batches} ---")
                
                VectorStoreIndex(
                    nodes=batch_nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True
                )
            
            # Now that the collection exists and has data, ensure the search indexes are created on it.
            self._ensure_atlas_indexes()
            
            logging.info(f"--- Successfully processed and indexed all batches for collection '{self.db_name}.{self.file_collection_name}' ---")
        
        except Exception:
            logging.exception("An unexpected error occurred during the ingestion run.")
    

    def list_files_for_agent(self, agent_id: str, user_id: str, agent_owner_id: str) -> List[Dict]:
        """
        Lists all unique files for a given agent that are accessible by the user.
        - If the user is the agent owner, they see all files.
        - If the user is not the owner, they see only public files or files they have explicit access to.
        """
        match_clauses = [{"metadata.agent_id": agent_id}]

        if user_id != agent_owner_id:
            permission_filter = {
                "$or": [
                    {"metadata.user_ids": user_id},
                    {"metadata.user_ids": {"$exists": False}}
                ]
            }
            match_clauses.append(permission_filter)

        pipeline = [
            {"$match": {"$and": match_clauses}},
            {"$group": {
                "_id": "$metadata.file_id",
                "file_name": {"$first": "$metadata.file_name"},
                "user_ids": {"$first": "$metadata.user_ids"}
            }},
            {"$project": {"file_id": "$_id", "file_name": 1, "user_ids": 1, "_id": 0}}
        ]
        try:
            files = list(self.file_collection.aggregate(pipeline))
            logging.info(f"User '{user_id}' found {len(files)} accessible files for agent '{agent_id}'.")
            return files
        except Exception:
            logging.exception(f"Failed to list files for agent '{agent_id}'.")
            return []

    def list_files_for_user(self, user_id: str) -> List[Dict]:
        """
        Lists all unique files associated with a specific user, either as an owner 
        or as an authorized user, using an aggregation pipeline.
        """
        pipeline = [
            {"$match": {
                "$or": [
                    {"metadata.user_ids": user_id},
                    {"metadata.user_ids": {"$exists": False}}
                    ]
                }
            },
            {"$group": {
                "_id": "$metadata.file_id",
                "file_name": {"$first": "$metadata.file_name"}
            }},
            {"$project": {"file_id": "$_id","file_name": 1,"_id": 0}}
        ]
        try:
            files = list(self.file_collection.aggregate(pipeline))
            logging.info(f"Found {len(files)} unique files for user '{user_id}'.")
            return files
        except Exception:
            logging.exception(f"Failed to list files for user '{user_id}'.")
            return []
        
    def list_files_for_owner(self, owner_user_id: str) -> List[Dict]:
        """
        Lists all unique files associated with a specific user, either as an owner 
        or as an authorized user, using an aggregation pipeline.
        """
        pipeline = [
            {"$match": {"metadata.owner_user_id": owner_user_id}},
            {"$group": {
                "_id": "$metadata.file_id",
                "file_name": {"$first": "$metadata.file_name"}
            }},
            {"$project": {"file_id": "$_id","file_name": 1,"_id": 0}}
        ]
        try:
            files = list(self.file_collection.aggregate(pipeline))
            logging.info(f"Found {len(files)} unique files for user '{owner_user_id}'.")
            return files
        except Exception:
            logging.exception(f"Failed to list files for user '{owner_user_id}'.")
            return []

    def list_files_for_thread(self, thread_id: str) -> List[Dict]:
        """
        Lists all unique files associated with a specific thread using an aggregation pipeline.
        """
        pipeline = [
            {"$match": {"metadata.thread_id": thread_id}},
            {"$group": {
                "_id": "$metadata.file_id",
                "file_name": {"$first": "$metadata.file_name"}
            }},
            {"$project": {"file_id": "$_id","file_name": 1,"_id": 0}}
        ]
        try:
            files = list(self.file_collection.aggregate(pipeline))
            logging.info(f"Found {len(files)} unique files for thread '{thread_id}'.")
            return files
        except Exception:
            logging.exception(f"Failed to list files for thread '{thread_id}'.")
            return []

    def delete_file_by_id(self, file_id: str) -> int:
        """
        Deletes all nodes associated with a specific file_id for a given agent.
        Returns the number of deleted nodes.
        """
        if not file_id:
            raise ValueError("file_id must be provided.")

        logging.info(f"Attempting to delete all nodes for file_id '{file_id}'")
        try:
            result = self.file_collection.delete_many({"metadata.file_id": file_id})
            logging.info(f"Successfully deleted {result.deleted_count} nodes for file_id '{file_id}'.")
            return True
        except Exception:
            logging.exception(f"An error occurred during file deletion for file_id '{file_id}'.")
            return False
    
    def has_access_to_file(self, file_id: str, user_id: str) -> bool:
        """
        Checks if a user has access to the agent, either by being the owner
        or by being in the agent's user_ids list.
        """
        file = self.file_collection.find_one({"_id": file_id})
        if not file:
            return False
        if file.get("owner_user_id") == user_id:
            return True
        if user_id in file.get("user_ids", []):
            return True
        return False

    def is_owner_of_file(self, file_id: str, owner_user_id: str) -> bool:
        """Checks if a user is the owner of the agent that a file belongs to."""
        file = self.file_collection.find_one({"_id": file_id})
        if not file:
            return False
        if file.get("owner_user_id") == owner_user_id:
            return True
        return False
    
    def calculate_file_permissions(
            self,
            agent_user_ids: List[str],
            file_user_ids: Optional[List[str]],
            owner_user_id: str
        ) -> Dict[str, List[str]]:
        
        #Calculates the final user access list for a file based on hierarchical permissions.
        
        # --- Rule 1: File inherits permissions from the agent ---
        if file_user_ids is None or file_user_ids==[]:
            return {"applied_user_ids": agent_user_ids, "excluded_user_ids": []}

        # --- Rule 2: File has specific permissions ---
        agent_permissions = set(agent_user_ids)
        file_request_permissions = set(file_user_ids)

        # If the agent is public (empty list), all requested users are applied.
        if not agent_permissions:
            applied_ids = file_request_permissions
            excluded_ids = set()
        else:
            # If the agent is restricted, find the intersection.
            applied_ids = agent_permissions.intersection(file_request_permissions)
            excluded_ids = file_request_permissions.difference(agent_permissions)
        
        # Ensure the owner has access.
        applied_ids.add(owner_user_id)

        return {
            "applied_user_ids": list(applied_ids),
            "excluded_user_ids": list(excluded_ids)
        }
    
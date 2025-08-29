import os
from typing import Optional, List, Dict, Any
import uuid
import logging
import pymongo
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential

# --- LlamaIndex and MongoDB Imports ---
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from app.core.config import Settings

class FileManagementService:
    """
    A service class for processing documents and uploading them to a specified MongoDB Atlas collection,
    ensuring the necessary hybrid search indexes are created automatically.
    """
    def __init__(
            self,
            settings: Settings,
            mongo_client: pymongo.MongoClient,
            embed_model: BaseEmbedding,
            text_splitter: TokenTextSplitter
        ):
        logging.info("Initializing IngestionPipeline service for MongoDB Atlas...")
        self.embed_model = embed_model
        self.mongo_client = mongo_client
        self.text_splitter = text_splitter
        self.batch_size = settings.llm.ingestion_batch_size
        self.db_name = settings.database.db_name
        self.file_collection_name = settings.database.file_collection_name
        self.file_collection = self.mongo_client[self.db_name][self.file_collection_name]
        logging.info("IngestionPipeline service initialized successfully.")

    def _generate_file_id(self) -> str:
        return f"file_{uuid.uuid4().hex}"

    # --- REFACTORED: Now accepts the path_to_id_map directly ---
    def _load_and_prepare_docs(
            self,
            path_to_id_map: Dict[str, str],
            owner_user_id: str,
            agent_id: Optional[str],
            thread_id: Optional[str],
            user_ids: Optional[list]
        ) -> List[Document]:
        """
        Loads unique documents efficiently and enriches them with metadata.
        """
        all_unique_paths = list(path_to_id_map.keys())
        try:
            reader = SimpleDirectoryReader(input_files=all_unique_paths)
            docs = reader.load_data()
            logging.info(f"Loaded {len(docs)} document object(s) from {len(all_unique_paths)} unique file(s).")

            for doc in docs:
                file_path = doc.metadata.get("file_path")
                file_id = path_to_id_map.get(file_path)
                
                doc.metadata["file_id"] = file_id
                doc.metadata["file_name"] = os.path.basename(file_path)
                doc.metadata["owner_user_id"] = owner_user_id
                doc.metadata["user_ids"] = user_ids or []
                if agent_id:
                    doc.metadata["agent_id"] = agent_id
                elif thread_id:
                    doc.metadata["thread_id"] = thread_id
                doc.metadata["created_at"] = datetime.now(timezone.utc)
        except Exception as e:
            logging.exception(f"Failed to load or prepare files. Error: {e}")
            return []
            
        logging.info(f"Successfully prepared a total of {len(docs)} document objects.")
        return docs
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _insert_batch_with_retry(self, index: VectorStoreIndex, batch_nodes: List):
        index.insert_nodes(batch_nodes)

    def ingest_files(
            self,
            file_paths: List[str],
            owner_user_id: str,
            agent_id: Optional[str],
            thread_id: Optional[str],
            user_ids: Optional[list]
        ):
        """
        Deduplicates file paths, then creates and uploads vectorized files to MongoDB.
        """
        # Ensure owner_user_id is included in the user_ids list IF this is not empty
        final_user_ids = list(dict.fromkeys([*user_ids, owner_user_id])) if user_ids else []

        unique_file_paths = list(dict.fromkeys(file_paths))
        if len(unique_file_paths) < len(file_paths):
            logging.info(f"Removed {len(file_paths) - len(unique_file_paths)} duplicate file paths.")
        
        logging.info(f"--- Starting batch ingestion for {len(unique_file_paths)} unique file(s) ---")
        try:
            path_to_id_map = {path: self._generate_file_id() for path in unique_file_paths}

            docs = self._load_and_prepare_docs(path_to_id_map, owner_user_id, agent_id, thread_id, final_user_ids)
            if not docs:
                logging.warning("No documents were prepared. Aborting ingestion.")
                return

            nodes = self.text_splitter.get_nodes_from_documents(docs, show_progress=False)
            if not nodes:
                logging.warning("No nodes were produced from documents. Aborting ingestion.")
                return
            
            logging.info(f"Documents split into {len(nodes)} text chunks (nodes).")

            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=self.mongo_client,
                db_name=self.db_name,
                collection_name=self.file_collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=self.embed_model)

            total_batches = (len(nodes) + self.batch_size - 1) // self.batch_size
            for i in range(0, len(nodes), self.batch_size):
                batch_nodes = nodes[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                
                logging.info(f"--- Processing Batch {batch_num}/{total_batches} ---")
                self._insert_batch_with_retry(index, batch_nodes)
            
            logging.info(f"--- Successfully processed and indexed all batches ---")
        
        except Exception:
            logging.exception("An unexpected error occurred during the ingestion run.")
    

    def list_files_for_agent(self, agent_id: str, user_id: str) -> List[Dict]:
        """
        Lists all unique files for a given agent that are accessible by the user.
        - If the user is the agent owner, they see all files.
        - If the user is not the owner, they see only public files or files they have explicit access to.
        """
        match_clauses = [
            {"metadata.agent_id": agent_id},
            {"$or": [{"metadata.user_ids": user_id},{"metadata.user_ids": {"$exists": False}}]}
        ]

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
        match_clause = {
            "$or": [{"metadata.user_ids": user_id},{"metadata.user_ids": {"$exists": False}}]
        }
        pipeline = [
            {"$match": match_clause},
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
        """
        logging.info(f"Attempting to delete all nodes for file_id '{file_id}'")
        try:
            result = self.file_collection.delete_many({"metadata.file_id": file_id})
            logging.info(f"Successfully deleted {result.deleted_count} nodes for file_id '{file_id}'.")
            return result.deleted_count
        except Exception:
            logging.exception(f"An error occurred during file deletion for file_id '{file_id}'.")
            return False
    
    def delete_files_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Deletes all nodes (document chunks) from the file collection based on a metadata filter.
        """
        if not metadata_filter:
            logging.warning("delete_files_by_metadata called with an empty filter. Aborting to prevent accidental mass deletion.")
            return False
        logging.info(f"Attempting to delete file nodes with filter: {metadata_filter}")
        try:
            result = self.file_collection.delete_many(metadata_filter)
            logging.info(f"Successfully deleted {result.deleted_count} file nodes.")
            return True
        except Exception as e:
            logging.exception(f"An error occurred during metadata deletion: {e}")
            return False
        
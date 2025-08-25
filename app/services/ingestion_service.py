import os
from typing import Optional, List, Dict, Any
import uuid
import logging
import pymongo
import certifi
from pymongo.operations import SearchIndexModel

# --- LlamaIndex and MongoDB Imports ---
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
#from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from app.core.config import Settings

class IngestionPipeline:
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
        # --- CHANGE IS HERE: Use the more reliable TokenTextSplitter ---
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.settings.llm.chunk_size,
            chunk_overlap=self.settings.llm.chunk_overlap
        )
        logging.info("IngestionPipeline service initialized successfully.")

    def _ensure_atlas_indexes(self, db_name: str, collection_name: str, vector_index_name: str, search_index_name: str):
        """
        Creates the necessary Vector Search and Atlas Search indexes if they don't exist.
        This should be called AFTER documents have been inserted into the collection.
        """
        collection = self.mongo_client[db_name][collection_name]
        
        vector_search_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1536,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "metadata.agent_id"},
                    {"type": "filter", "path": "metadata.thread_id"},
                    {"type": "filter", "path": "metadata.file_name"},
                ]
            },
            name=vector_index_name,
            type="vectorSearch",
        )
        
        full_text_model = SearchIndexModel(
            definition={"mappings": {"dynamic": True, "fields": {"text": {"type": "string"}}}},
            name=search_index_name,
            type="search",
        )
        
        # --- BUG FIX IS HERE ---
        # We now handle each index creation explicitly to use the correct name in the log,
        # avoiding the '.name' attribute error.
        try:
            logging.info(f"Ensuring Atlas index '{vector_index_name}' exists...")
            collection.create_search_index(model=vector_search_model)
        except pymongo.errors.OperationFailure as e:
            if "already exists" in str(e).lower():
                logging.warning(f"Index '{vector_index_name}' already exists. Skipping creation.")
            else:
                raise e
        
        try:
            logging.info(f"Ensuring Atlas index '{search_index_name}' exists...")
            collection.create_search_index(model=full_text_model)
        except pymongo.errors.OperationFailure as e:
            if "already exists" in str(e).lower():
                logging.warning(f"Index '{search_index_name}' already exists. Skipping creation.")
            else:
                raise e

    def _load_and_prepare_docs(
        self, file_paths: List[str], agent_id: Optional[str], thread_id: Optional[str]
    ) -> List[Document]:
        """
        Loads one or more documents and enriches them with the provided metadata.
        """
        reader = SimpleDirectoryReader(input_files=file_paths)
        docs = reader.load_data()

        for doc in docs:
            doc.metadata["file_name"] = os.path.basename(doc.metadata.get("file_path", ""))
            if agent_id:
                doc.metadata["agent_id"] = agent_id
            elif thread_id:
                doc.metadata["thread_id"] = thread_id
        
        logging.info(f"Successfully loaded {len(file_paths)} document(s).")
        return docs

    def run(self, db_name: str, collection_name: str, file_paths: List[str], agent_id: Optional[str] = None, thread_id: Optional[str] = None):
        """
        Runs the ingestion pipeline on a specific MongoDB database and collection.
        """
        if not agent_id and not thread_id:
            logging.error("INGESTION FAILED: You must provide either an 'agent_id' or a 'thread_id'.")
            return

        logging.info(f"--- Starting batch ingestion for MongoDB collection '{db_name}.{collection_name}' ---")
        
        try:
            vector_index_name = "vector_index"
            search_index_name = "search_index"

            vector_store = MongoDBAtlasVectorSearch(
                mongodb_client=self.mongo_client,
                db_name=db_name,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                search_index_name=search_index_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            documents = self._load_and_prepare_docs(file_paths, agent_id, thread_id)
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
            self._ensure_atlas_indexes(db_name, collection_name, vector_index_name, search_index_name)
            
            logging.info(f"--- Successfully processed and indexed all batches for collection '{db_name}.{collection_name}' ---")
        
        except Exception:
            logging.exception("An unexpected error occurred during the ingestion run.")
    
    def delete_by_metadata(self, db_name: str, collection_name: str, metadata_filter: Dict[str, Any]):
        """
        Deletes documents from a specific collection based on a metadata filter.
        """
        logging.info(f"Attempting to delete documents from '{db_name}.{collection_name}' with filter: {metadata_filter}")
        try:
            collection = self.mongo_client[db_name][collection_name]
            query_filter = {f"metadata.{key}": value for key, value in metadata_filter.items()}
            result = collection.delete_many(query_filter)
            logging.info(f"Successfully deleted {result.deleted_count} documents.")
        except Exception:
            logging.exception(f"An error occurred during metadata deletion from collection '{db_name}.{collection_name}'.")

    def delete_collection(self, db_name: str, collection_name: str):
        """
        Deletes an entire MongoDB collection. This is a destructive operation.
        """
        logging.warning(f"--- DESTRUCTIVE OPERATION: DELETING ENTIRE COLLECTION: {db_name}.{collection_name} ---")
        try:
            self.mongo_client[db_name].drop_collection(collection_name)
            logging.info(f"Successfully sent request to delete collection '{collection_name}'.")
        except Exception:
            logging.exception(f"An error occurred during deletion of collection '{collection_name}'.")
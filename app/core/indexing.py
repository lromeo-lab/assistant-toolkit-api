import pymongo
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
import logging
from typing import List, Dict

def create_atlas_indexes(
    collection: Collection,
    vector_index_name: str = None,
    search_index_name: str = None,
    vector_fields: List[Dict] = None,
    search_fields: Dict = None
):
    """
    Creates the necessary Vector Search and Full Text Search indexes if they don't exist.
    """
    existing_indexes = {index['name'] for index in collection.list_search_indexes()}

    # --- 1. Create Vector Search Index ---
    if vector_index_name and vector_index_name not in existing_indexes:
        vector_definition = {
            "fields": [
                {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
                *(vector_fields or [])
            ]
        }
        vector_search_model = SearchIndexModel(
            definition=vector_definition,
            name=vector_index_name,
            type="vectorSearch",
        )
        try:
            logging.info(f"Creating vector index '{vector_index_name}'...")
            collection.create_search_index(model=vector_search_model)
            logging.info(f"Successfully created index '{vector_index_name}'.")
        except pymongo.errors.OperationFailure as e:
            logging.error(f"Failed to create index '{vector_index_name}': {e}")
            raise
    elif vector_index_name:
        logging.info(f"Index '{vector_index_name}' already exists. Skipping.")

    # --- 2. Create Full-Text Search Index ---
    if search_index_name and search_index_name not in existing_indexes:
        search_definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string"},
                    "metadata": {
                        "type": "document",
                        "fields": search_fields or {}
                    }
                }
            }
        }
        full_text_model = SearchIndexModel(
            definition=search_definition,
            name=search_index_name,
            type="search",
        )
        try:
            logging.info(f"Creating search index '{search_index_name}'...")
            collection.create_search_index(model=full_text_model)
            logging.info(f"Successfully created index '{search_index_name}'.")
        except pymongo.errors.OperationFailure as e:
            logging.error(f"Failed to create index '{search_index_name}': {e}")
            raise
    elif search_index_name:
        logging.info(f"Index '{search_index_name}' already exists. Skipping.")
                    
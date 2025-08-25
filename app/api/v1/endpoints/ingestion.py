import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, Body
import tempfile
import os

from app.core.config import Settings, get_settings
from app.services.ingestion_service import IngestionPipeline

# Create an API router
router = APIRouter()

# Instantiate the pipeline service once at the module level for efficiency
settings = get_settings()
ingestion_pipeline = IngestionPipeline(settings)

@router.post("/ingest", status_code=202)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="A list of files to be ingested."),
    db_name: str = Form(..., description="The name of the target MongoDB database."),
    collection_name: str = Form(..., description="The name of the target MongoDB collection within the database."),
    agent_id: Optional[str] = Form(None, description="The agent ID to associate with these files."),
    thread_id: Optional[str] = Form(None, description="The thread ID to associate with these files if no agent_id is provided.")
):
    """
    Accepts a batch of files and ingests them into a specified MongoDB Atlas collection.
    """
    if not agent_id and not thread_id:
        raise HTTPException(
            status_code=422, 
            detail="Validation Error: You must provide either an 'agent_id' or a 'thread_id'."
        )

    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(file_path)

    logging.info(f"Received {len(files)} files. Scheduling for background ingestion.")

    # Update the background task to call the service with the correct parameters
    background_tasks.add_task(
        ingestion_pipeline.run,
        db_name=db_name,
        collection_name=collection_name,
        file_paths=file_paths,
        agent_id=agent_id,
        thread_id=thread_id
    )

    return {
        "message": "Files received. Ingestion has started in the background.",
        "database": db_name,
        "collection": collection_name,
        "filenames": [file.filename for file in files]
    }

@router.delete("/delete-vectors")
async def delete_vectors_by_metadata(
    db_name: str = Body(..., description="The name of the target MongoDB database."),
    collection_name: str = Body(..., description="The name of the target MongoDB collection."),
    metadata_filter: Dict[str, Any] = Body(..., description="A dictionary specifying the metadata to filter by. Example: {'file_name': 'document_to_delete.txt'}")    
):
    """
    Deletes documents from a specific collection based on a metadata filter.
    """
    logging.info(f"Received request to delete vectors from '{db_name}.{collection_name}' with filter: {metadata_filter}")
    try:
        ingestion_pipeline.delete_by_metadata(
            db_name=db_name,
            collection_name=collection_name,
            metadata_filter=metadata_filter
        )
        return {"message": "Delete request sent successfully.", "database": db_name, "collection": collection_name, "filter": metadata_filter}
    except Exception as e:
        logging.exception("Failed to process delete-by-metadata request.")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-collection")
async def delete_collection(
    db_name: str = Body(..., embed=True, description="The name of the MongoDB database."),
    collection_name: str = Body(..., embed=True, description="The name of the MongoDB collection to delete.")
):
    """
    Deletes an entire MongoDB collection. This is a destructive operation.
    """
    logging.warning(f"Received request to delete entire collection: '{db_name}.{collection_name}'")
    try:
        ingestion_pipeline.delete_collection(db_name=db_name, collection_name=collection_name)
        return {"message": f"Request to delete collection '{db_name}.{collection_name}' sent successfully."}
    except Exception as e:
        logging.exception("Failed to process delete-collection request.")
        raise HTTPException(status_code=500, detail=str(e))
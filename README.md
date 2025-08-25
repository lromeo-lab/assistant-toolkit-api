# Assistant Toolkit API 🚀

This is a FastAPI-based API for a RAG (Retrieval-Augmented Generation) assistant. It uses LlamaIndex to connect a language model with documents and chat history stored in MongoDB.

This project was created for a YouTube tutorial demonstrating how to build and deploy a modern RAG API.

## Features
- **Dynamic Retrieval:** Uses a custom retriever for hybrid search (vector + keyword) on MongoDB.
- **Chat Memory:** Integrates with Redis for short-term conversation memory.
- **Modular Design:** Built with a clean service layer and dependency injection for scalability.

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  Create a `.env` file and fill in your API keys, database URIs and the rest of the environment variables based on the `.env.example` file.
4.  Run the server:
    ```bash
    uvicorn app.main:app --reload
    ```
import logging
import certifi
import pymongo
import requests
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.retrievers import BaseRetriever, RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import PydanticMultiSelector
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from app.core.config import Settings

# --- Custom Retriever Class ---
class MongoCustomRetriever(BaseRetriever):
    """
    A custom LlamaIndex retriever that performs a manual hybrid search against
    MongoDB Atlas by running vector and keyword queries concurrently.
    """
    def __init__(
        self, 
        mongo_client: pymongo.MongoClient, 
        embed_model: OpenAIEmbedding, 
        settings: Settings,
        collection_name: str,
        search_type: str,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None, 
        top_k: int = 10
    ):
        self.mongo_client = mongo_client
        self.collection = self.mongo_client[settings.database.db_name][collection_name]
        self.embed_model = embed_model
        self.settings = settings
        self.search_type = search_type
        self.agent_id = agent_id
        self.thread_id = thread_id
        self.top_k = top_k
        super().__init__()

    def _run_query(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper to execute a MongoDB aggregation pipeline."""
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logging.error(f"Error executing MongoDB pipeline: {e}")
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """The core retrieval logic called by LlamaIndex."""
        query_text = query_bundle.query_str
        logging.info(f"MongoCustomRetriever executing retrieve for query: '{query_text}'")
        query_embedding = self.embed_model.get_query_embedding(query_text)

        # --- DYNAMIC FILTER CONSTRUCTION ---
        # Build the filter for the $vectorSearch pipeline using MQL syntax.
        vector_filter_clauses = []
        if self.agent_id:
            vector_filter_clauses.append({"metadata.agent_id": self.agent_id})
        if self.thread_id:
            vector_filter_clauses.append({"metadata.thread_id": self.thread_id})
        
        vector_filter = {}
        if len(vector_filter_clauses) > 1:
            vector_filter = {"$and": vector_filter_clauses}
        elif vector_filter_clauses:
            vector_filter = vector_filter_clauses[0]

        # Build the filter for the $search pipeline using Atlas Search syntax.
        keyword_filter_clauses = []
        if self.agent_id:
            keyword_filter_clauses.append({"text": {"query": self.agent_id, "path": "metadata.agent_id"}})
        if self.thread_id:
            keyword_filter_clauses.append({"text": {"query": self.thread_id, "path": "metadata.thread_id"}})

        # --- DYNAMIC PIPELINE DEFINITION ---
        vector_pipeline_stage = {
            "$vectorSearch": {
                "index": self.settings.database.atlas_vector_index_name, 
                "path": "embedding",
                "queryVector": query_embedding, 
                "numCandidates": 150, 
                "limit": self.top_k
            }
        }
        if vector_filter:
            vector_pipeline_stage["$vectorSearch"]["filter"] = vector_filter

        vector_pipeline = [
            vector_pipeline_stage,
            {"$project": {"_id": 1, "text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]

        keyword_pipeline_stage = {
            "$search": {
                "index": self.settings.database.atlas_search_index_name,
                "compound": {
                    "must": [{"text": {"query": query_text, "path": "text"}}]
                }
            }
        }
        if keyword_filter_clauses:
            keyword_pipeline_stage["$search"]["compound"]["filter"] = keyword_filter_clauses

        keyword_pipeline = [
            keyword_pipeline_stage,
            {"$project": {"_id": 1, "text": 1, "metadata": 1, "score": {"$meta": "searchScore"}}},
            {"$limit": self.top_k}
        ]

        
        final_results_list = []
        if self.search_type == "Hybrid":
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_vector = executor.submit(self._run_query, vector_pipeline)
                future_keyword = executor.submit(self._run_query, keyword_pipeline)
                vector_results = future_vector.result()
                keyword_results = future_keyword.result()
            
            final_docs_dict = {}
            for doc in keyword_results + vector_results:
                final_docs_dict[str(doc["_id"])] = doc
            final_results_list = list(final_docs_dict.values())
        elif self.search_type == "Vector":
            final_results_list = self._run_query(vector_pipeline)
        elif self.search_type == "Keyword":
            final_results_list = self._run_query(keyword_pipeline)

        final_nodes = []
        for doc in final_results_list:
            node = TextNode(text=doc.get("text", ""), metadata=doc.get("metadata", {}), id_=str(doc["_id"]))
            score = doc.get("score", 0.0)
            final_nodes.append(NodeWithScore(node=node, score=score))
            
        logging.info(f"Retriever found {len(final_nodes)} documents via '{self.search_type}' search.")
        return final_nodes

# --- Main Assistant Service Class ---
class RAGAssistantService:
    """Orchestrates the RAG pipeline using injected dependencies."""
    def __init__(
        self,
        mongo_client: pymongo.MongoClient,
        chat_store: RedisChatStore,
        llm: OpenAI,
        embed_model: OpenAIEmbedding,
        reranker: CohereRerank,
        settings: Settings
    ):
        self.mongo_client = mongo_client
        self.chat_store = chat_store
        self.llm = llm
        self.embed_model = embed_model
        self.reranker = reranker
        self.settings = settings
        logging.info("RAGAssistantService initialized with shared components.")

    def _create_retriever(
        self, 
        collection_name: str, 
        search_type: str, 
        top_k: int,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> MongoCustomRetriever:
        """Factory method to create a configured MongoCustomRetriever."""
        return MongoCustomRetriever(
            mongo_client=self.mongo_client,
            embed_model=self.embed_model,
            settings=self.settings,
            collection_name=collection_name,
            search_type=search_type,
            agent_id=agent_id,
            thread_id=thread_id,
            top_k=top_k
        )

    def _ingest_turn_to_memory(self, thread_id: str, turn_id: int, query: str, response: str):
        """Calls the memory management API to ingest a conversational turn."""
        payload = {
            "db_name": self.settings.database.db_name,
            "collection_name": self.settings.database.chat_history_collection_name,
            "thread_id": thread_id, "turn_id": turn_id,
            "user_query": query, "assistant_response": response
        }
        try:
            endpoint = f"{self.settings.memory_management_root_endpoint}/api/v1/ingest-turn"
            res = requests.post(endpoint, json=payload)
            res.raise_for_status()
            logging.info(f"Successfully sent turn {turn_id} for thread '{thread_id}' to memory API.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send turn to memory API: {e}")

    def get_chat_response(self, agent_id: str, thread_id: str, query: str) -> str:
        """Builds the chat engine, gets a response, and handles memory ingestion."""
        logging.info(f"--- Chatting with Agent '{agent_id}' on Thread {thread_id} ---")
        logging.info(f"Original user query: '{query}'")

        # 1. Create specialized retriever tools
        file_retriever = self._create_retriever(
            agent_id=agent_id,
            thread_id=None,#in the future I can add an thread-based filter just including it here
            collection_name=self.settings.database.file_collection_name,
            search_type=self.settings.assistant.file_search_type,
            top_k=self.settings.database.file_retriever_top_k
        ) 
        file_tool = RetrieverTool.from_defaults(
            retriever=file_retriever,
            description="Use this for questions about specific documentation the user provided."
        )
        
        chat_retriever = self._create_retriever(
            agent_id=None,#in the future I can add an agent-based filter just including it here
            thread_id=thread_id,
            collection_name=self.settings.database.chat_history_collection_name,
            search_type=self.settings.assistant.chat_search_type,
            top_k=self.settings.database.chat_retriever_top_k
        )
        chat_tool = RetrieverTool.from_defaults(
            retriever=chat_retriever,
            description="Use this for questions about past conversations not in the recent chat history."
        )

        # 2. Configure short-term memory
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.settings.database.memory_token_limit,
            chat_store=self.chat_store,
            chat_store_key=thread_id
        )
        
        # 3. Build the router and query engine
        router_retriever = RouterRetriever(
            selector=PydanticMultiSelector.from_defaults(llm=self.llm),
            retriever_tools=[file_tool, chat_tool],
        )
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=router_retriever,
            node_postprocessors=[self.reranker]
        )

        # 4. Build the final chat engine
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=router_retriever,
            query_engine=query_engine,
            memory=memory,
            llm=self.llm,
            system_prompt=(
                f"You are an expert assistant for agent '{agent_id}'. "
                "Answer questions based only on the provided context. "
                "If the context does not contain the answer, state that you don't know."
            ),
            verbose=True
        )

        # --- CORRECTED OBSERVABILITY SECTION ---
        # We now get the condensed question directly from the engine's internal method.
        chat_history = memory.get()
        condensed_question = chat_engine._condense_question(chat_history, query)
        logging.info(f"Condensed question: '{condensed_question}'")

        selected_tools_result = router_retriever._selector.select(
            [t.metadata for t in [file_tool, chat_tool]],
            QueryBundle(condensed_question)
        )
        for i, res in enumerate(selected_tools_result.selections):
            logging.info(f"Retriever selection {i+1}: Tool index '{res.index}' with reason: {res.reason}")
        
        retrieved_nodes = router_retriever.retrieve(condensed_question)
        logging.info(f"Total documents retrieved before reranking: {len(retrieved_nodes)}")
        if not retrieved_nodes:
            logging.warning("Retrieval step returned zero documents. The final response may be empty.")

        reranked_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle=QueryBundle(condensed_question))
        logging.info(f"Total documents remaining after reranking: {len(reranked_nodes)}")
        # --- END OF OBSERVABILITY SECTION ---

        # 5. Get the final response
        response = chat_engine.chat(query)
        response_text = str(response)

        logging.info(f"Final response generated by LLM: '{response_text}'")
        logging.info(f"--- Finished Chat Turn for Agent '{agent_id}' ---")

        # 6. Ingest to long-term memory
        redis_client = self.chat_store._redis_client
        turn_id = redis_client.llen(thread_id) // 2 + 1
        
        self._ingest_turn_to_memory(
            thread_id=thread_id,
            turn_id=turn_id,
            query=query,
            response=response_text
        )

        return response_text
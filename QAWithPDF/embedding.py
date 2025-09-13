from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.exception import customexception
from logger import logging

import sys
import os
from pathlib import Path

# Default embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
# Default local cache directory for index persistence
CACHE_DIR = os.getenv("CACHE_DIR", "./cache/vector_index")


def download_gemini_embedding(
    model,
    documents,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 20,
    cache_dir: str = CACHE_DIR,
    use_cache: bool = True,
):
    """
    Build or load a query engine using Gemini embeddings + provided LLM model.
    Supports caching with LlamaIndex persistent storage.

    Args:
        model: LLM model instance to use for queries.
        documents: List of Document objects to index (ignored if cache is used).
        embedding_model_name: Gemini embedding model (default: gemini-embedding-001).
        chunk_size: Chunk size for document splitting.
        chunk_overlap: Overlap between chunks.
        cache_dir: Directory to store cached index.
        use_cache: Whether to load cached index if available.

    Returns:
        query_engine: A query engine configured with Gemini embeddings + LLM.
    """
    try:
        logging.info("Initializing Gemini embedding model: %s", embedding_model_name)

        # Configure Gemini embedding
        gemini_embed_model = GeminiEmbedding(model_name=embedding_model_name)

        # Update global settings
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

        cache_path = Path(cache_dir)

        # Helper to check cache completeness
        def is_cache_valid(path: Path) -> bool:
            required_files = ["docstore.json", "index_store.json", "vector_store.json"]
            return all((path / f).exists() for f in required_files)

        # Load from cache if valid
        if use_cache and cache_path.exists() and is_cache_valid(cache_path):
            logging.info("Loading cached VectorStoreIndex from %s", cache_dir)
            storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
            index = load_index_from_storage(storage_context)
        else:
            logging.info("Building new VectorStoreIndex from %d documents", len(documents))
            index = VectorStoreIndex.from_documents(documents)

            if use_cache:
                cache_path.mkdir(parents=True, exist_ok=True)
                index.storage_context.persist(persist_dir=cache_dir)
                logging.info("Index cached at %s", cache_dir)

        query_engine = index.as_query_engine()
        logging.info("Gemini RAG query engine initialized successfully")

        return query_engine

    except Exception as e:
        logging.error("Failed to initialize Gemini RAG engine: %s", str(e))
        raise customexception(e, sys)
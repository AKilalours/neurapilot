"""ChromaDB vector store wrapper for NeuraPilot.

One Chroma collection per course, namespaced under the base collection prefix.
MMR (Maximal Marginal Relevance) retrieval reduces redundancy in retrieved chunks.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from neurapilot.config import Settings


def collection_name(settings: Settings, course_id: str) -> str:
    """Generate a valid Chroma collection name for a course."""
    safe = "".join(c for c in course_id.lower() if c.isalnum() or c in ("-", "_")).strip("-_")
    name = f"{settings.base_collection}__{safe or 'default'}"
    # Chroma collection names must be 3-63 chars
    return name[:63]


def get_vector_store(
    settings: Settings,
    embeddings: Embeddings,
    course_id: str,
) -> Chroma:
    """Return a Chroma vector store for the given course."""
    return Chroma(
        collection_name=collection_name(settings, course_id),
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


def get_retriever(
    settings: Settings,
    embeddings: Embeddings,
    course_id: str,
) -> VectorStoreRetriever:
    """Return an MMR retriever that balances relevance with diversity.

    MMR prevents fetching near-duplicate chunks from the same document section,
    which improves context quality for multi-chunk answers.
    """
    store = get_vector_store(settings, embeddings, course_id)
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.candidate_k,
            "fetch_k": settings.candidate_k * 2,
            "lambda_mult": settings.mmr_lambda,
        },
    )


def delete_course_collection(
    settings: Settings,
    embeddings: Embeddings,
    course_id: str,
) -> None:
    """Remove all vectors for a course (used when re-ingesting)."""
    store = get_vector_store(settings, embeddings, course_id)
    try:
        store.delete_collection()
    except Exception:
        pass  # Collection may not exist yet

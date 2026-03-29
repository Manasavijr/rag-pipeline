"""
VectorStore — manages FAISS index with HuggingFace embeddings.
Handles add, search, persist, and load operations.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store with sentence-transformer embeddings.
    Fully local — no external API calls for embeddings.
    """

    def __init__(self):
        self._index: Optional[FAISS] = None
        self._doc_count: int = 0
        self._chunk_count: int = 0
        self._index_path = Path(settings.FAISS_INDEX_PATH)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded")

    def load_existing(self) -> bool:
        """Load index from disk if it exists."""
        index_file = str(self._index_path)
        if Path(f"{index_file}.faiss").exists():
            try:
                self._index = FAISS.load_local(
                    index_file,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._chunk_count = self._index.index.ntotal
                logger.info(f"Loaded existing FAISS index ({self._chunk_count} vectors)")
                return True
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
        return False

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the FAISS index."""
        if not documents:
            raise ValueError("No documents to add")

        logger.info(f"Adding {len(documents)} chunks to FAISS index...")

        if self._index is None:
            self._index = FAISS.from_documents(documents, self.embeddings)
        else:
            self._index.add_documents(documents)

        self._chunk_count = self._index.index.ntotal
        self._doc_count += 1
        self._save()
        logger.info(f"Index now has {self._chunk_count} total vectors")

    def search(
        self, query: str, top_k: int = None, score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Semantic search returning (document, score) tuples.
        Score is cosine similarity (higher = more relevant).
        """
        if self._index is None:
            raise RuntimeError("No documents indexed yet. Upload documents first.")

        k = top_k or settings.FAISS_TOP_K
        threshold = score_threshold or settings.MIN_RELEVANCE_SCORE

        results = self._index.similarity_search_with_score(query, k=k)

        # Filter by relevance threshold and convert distance to similarity
        filtered = []
        for doc, score in results:
            # FAISS returns L2 distance — convert to 0-1 similarity
            similarity = 1 / (1 + score)
            if similarity >= threshold:
                filtered.append((doc, round(similarity, 4)))

        logger.info(f"Search '{query[:50]}...' → {len(filtered)}/{len(results)} results above threshold")
        return filtered

    def _save(self) -> None:
        """Persist index to disk."""
        if self._index:
            self._index.save_local(str(self._index_path))
            logger.info(f"Index saved to {self._index_path}")

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._chunk_count = 0
        self._doc_count = 0
        import shutil
        if self._index_path.parent.exists():
            for f in self._index_path.parent.glob("faiss_index*"):
                f.unlink()
        logger.info("Index cleared")

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def doc_count(self) -> int:
        return self._doc_count

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from empathy_engine.config.settings import get_settings
from empathy_engine.utils.errors import VectorStoreError
from empathy_engine.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class StoredItem:
    id: int
    session_id: Optional[str]
    text: str
    timestamp: str
    meta: Dict[str, Any]


class FaissVectorStore:
    """FAISS-backed vector store with JSON metadata."""

    def __init__(self) -> None:
        settings = get_settings()
        self._index_path = settings.faiss_index_path
        self._metadata_path = settings.faiss_metadata_path
        self._dim = settings.embedding_dimension
        self._use_ip = settings.faiss_use_inner_product

        self._index = self._load_index()
        self._metadata: List[StoredItem] = self._load_metadata()

    def _load_index(self) -> faiss.Index:
        if os.path.exists(self._index_path):
            try:
                logger.info("Loading FAISS index from %s", self._index_path)
                index = faiss.read_index(self._index_path)
                if index.d != self._dim:
                    raise VectorStoreError(
                        f"FAISS index dimension {index.d} != expected {self._dim}"
                    )
                return index
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load FAISS index: %s. Recreating.", exc)

        metric = faiss.METRIC_INNER_PRODUCT if self._use_ip else faiss.METRIC_L2
        if metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(self._dim)
        else:
            index = faiss.IndexFlatL2(self._dim)
        return index

    def _load_metadata(self) -> List[StoredItem]:
        if not os.path.exists(self._metadata_path):
            return []
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load FAISS metadata: %s. Resetting.", exc)
            return []

        items: List[StoredItem] = []
        for entry in raw:
            items.append(
                StoredItem(
                    id=int(entry["id"]),
                    session_id=entry.get("session_id"),
                    text=entry["text"],
                    timestamp=entry["timestamp"],
                    meta=entry.get("meta", {}),
                )
            )
        return items

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        data = [
            {
                "id": item.id,
                "session_id": item.session_id,
                "text": item.text,
                "timestamp": item.timestamp,
                "meta": item.meta,
            }
            for item in self._metadata
        ]
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @property
    def is_ready(self) -> bool:
        return self._index is not None

    def add_item(
        self,
        embedding: np.ndarray,
        text: str,
        session_id: Optional[str],
        meta: Dict[str, Any],
    ) -> int:
        if embedding.shape[0] != self._dim:
            raise VectorStoreError(
                f"Embedding dimension {embedding.shape[0]} != expected {self._dim}"
            )
        vec = embedding.astype("float32")
        if self._use_ip:
            # Normalize for cosine similarity.
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

        next_id = len(self._metadata)
        self._index.add(vec.reshape(1, -1))

        item = StoredItem(
            id=next_id,
            session_id=session_id,
            text=text,
            timestamp=datetime.utcnow().isoformat(),
            meta=meta,
        )
        self._metadata.append(item)
        self._persist()
        return next_id

    def search(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[StoredItem, float]]:
        if self._index.ntotal == 0:
            return []
        vec = embedding.astype("float32").reshape(1, -1)
        if self._use_ip:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        distances, indices = self._index.search(vec, k)
        results: List[Tuple[StoredItem, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            results.append((self._metadata[idx], float(dist)))
        return results

    def get_recent_for_session(
        self,
        session_id: str,
        limit: int = 5,
    ) -> List[StoredItem]:
        items = [m for m in self._metadata if m.session_id == session_id]
        items.sort(key=lambda i: i.timestamp, reverse=True)
        return items[:limit]


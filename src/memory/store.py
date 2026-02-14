import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import json
from pathlib import Path
from loguru import logger
from ..agents.base import MemoryItem

# Try to import FAISS with proper error handling
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False


class SimpleVectorStore:
    """Simple numpy-based vector store fallback"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        logger.info("SimpleVectorStore initialized")

    def add(self, vectors: np.ndarray):
        """Add vectors to store"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        for v in vectors:
            self.vectors.append(v)

    def search(self, query_vectors: np.ndarray, k: int):
        """Search for similar vectors"""
        import numpy.linalg as LA

        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)

        results = []
        for query_vec in query_vectors:
            similarities = []
            for i, stored_vec in enumerate(self.vectors):
                # Cosine similarity
                similarity = np.dot(query_vec, stored_vec) / (
                    LA.norm(query_vec) * LA.norm(stored_vec) + 1e-8
                )
                similarities.append((similarity, i))

            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Get top k results
            distances = []
            indices = []
            for similarity, idx in similarities[:k]:
                distances.append(1.0 - similarity)  # Convert to distance
                indices.append(idx)

            results.append((np.array(distances), np.array(indices)))

        return results


class VectorStore:
    """FAISS-based vector store for agent memory with fallback"""

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.use_fallback = False

        if FAISS_AVAILABLE:
            try:
                # Initialize FAISS index
                if index_type == "flat":
                    self.index = faiss.IndexFlatL2(dimension)
                elif index_type == "ivf":
                    quantizer = faiss.IndexFlatL2(dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                else:
                    raise ValueError(f"Unsupported index type: {index_type}")
                logger.info(f"FAISS VectorStore initialized with {index_type} index")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
                self.use_fallback = True
                self.index = SimpleVectorStore(dimension)
        else:
            # Fallback to simple numpy-based storage
            self.use_fallback = True
            self.index = SimpleVectorStore(dimension)
            logger.warning("Using simple numpy vector storage (FAISS not available)")

        # Storage for memory items
        self.memory_items: List[MemoryItem] = []
        self.id_to_index: Dict[str, int] = {}

        # Simple embedding function (in production, use proper embeddings)
        self.embedding_model = None

    async def add(self, memory_item: MemoryItem):
        """Add a memory item to the vector store"""
        # Generate embedding (simple TF-IDF style for now)
        embedding = await self._get_embedding(memory_item.content)

        # Add vectors store
        vectors = np.array([embedding], dtype=np.float32)
        if self.use_fallback or not FAISS_AVAILABLE:
            self.index.add(vectors)
        else:
            try:
                self.index.add(vectors)
            except Exception as e:
                logger.warning(f"FAISS add failed: {e}")
                # Fallback to simple storage
                if not self.use_fallback:
                    self.index = SimpleVectorStore(self.dimension)
                    self.use_fallback = True
                self.index.add(vectors)

        # Store the memory item
        index = len(self.memory_items)
        self.memory_items.append(memory_item)
        self.id_to_index[memory_item.id] = index

        logger.debug(f"Added memory item {memory_item.id} to vector store")

    async def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search for similar memory items"""
        if len(self.memory_items) == 0:
            return []

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        query_vectors = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, len(self.memory_items))

        try:
            if self.use_fallback or not FAISS_AVAILABLE:
                # Use fallback search
                search_results = self.index.search(query_vectors, k)
                distances, indices = search_results[0]  # Get first result
            else:
                # Use FAISS search
                if self.index.ntotal > 0:
                    # Standard FAISS search
                    distances, indices = self.index.search(query_vectors, k)
                else:
                    return []

            # Return memory items
            results = []
            for i, idx in enumerate(indices):
                if idx >= 0 and idx < len(self.memory_items):
                    results.append(self.memory_items[idx])
            return results

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            # Fallback to simple text search
            query_lower = query.lower()
            results = []
            for item in self.memory_items:
                if query_lower in item.content.lower():
                    results.append(item)
            return results[:top_k]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Simple embedding function (replace with proper model in production)"""
        # For now, use a simple hash-based embedding
        # In production, use sentence transformers or similar

        # Create a simple character-based embedding
        embedding = np.zeros(self.dimension, dtype=np.float32)

        # Simple character frequency embedding
        for i, char in enumerate(text.lower()):
            if i < self.dimension:
                embedding[i] = ord(char) / 255.0

        # Add some variation
        embedding = embedding + np.random.normal(0, 0.01, self.dimension)

        return embedding

    def save(self, path: Path):
        """Save the vector store to disk"""
        path.mkdir(parents=True, exist_ok=True)

        if FAISS_AVAILABLE and not self.use_fallback and self.index is not None:
            # Save FAISS index
            try:
                faiss.write_index(self.index, str(path / "faiss.index"))
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

        # Save fallback vectors if needed
        if self.use_fallback and hasattr(self.index, "vectors"):
            with open(path / "vectors.json", "w") as f:
                json.dump([v.tolist() for v in self.index.vectors], f)

        # Save memory items
        with open(path / "memory_items.json", "w") as f:
            memory_data = [asdict(item) for item in self.memory_items]
            json.dump(memory_data, f)

        logger.info(f"VectorStore saved to {path}")

    def load(self, path: Path):
        """Load the vector store from disk"""
        if not path.exists():
            logger.warning(f"VectorStore path {path} does not exist")
            return

        if FAISS_AVAILABLE and not self.use_fallback:
            # Try to load FAISS index
            faiss_path = path / "faiss.index"
            if faiss_path.exists():
                try:
                    self.index = faiss.read_index(str(faiss_path))
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
                    # Switch to fallback
                    self.use_fallback = True
                    self.index = SimpleVectorStore(self.dimension)
            else:
                self.use_fallback = True
                self.index = SimpleVectorStore(self.dimension)

        # Load fallback vectors if needed
        if self.use_fallback:
            vectors_path = path / "vectors.json"
            if vectors_path.exists():
                try:
                    with open(vectors_path, "r") as f:
                        vectors_data = json.load(f)
                        self.index.vectors = [np.array(v) for v in vectors_data]
                except Exception as e:
                    logger.warning(f"Failed to load vectors: {e}")

        # Load memory items
        items_path = path / "memory_items.json"
        if items_path.exists():
            with open(items_path, "r") as f:
                memory_data = json.load(f)
                self.memory_items = [MemoryItem(**data) for data in memory_data]

            # Rebuild index mapping
            for i, item in enumerate(self.memory_items):
                self.id_to_index[item.id] = i

        logger.info(f"VectorStore loaded from {path}")


class KeyValueStore:
    """Simple key-value store for structured memory"""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        logger.info("KeyValueStore initialized")

    async def set(self, key: str, value: Any):
        """Set a key-value pair"""
        self.store[key] = value
        logger.debug(f"Set key {key}")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key"""
        return self.store.get(key, default)

    async def delete(self, key: str):
        """Delete a key"""
        if key in self.store:
            del self.store[key]
            logger.debug(f"Deleted key {key}")

    async def keys(self) -> List[str]:
        """Get all keys"""
        return list(self.store.keys())

    async def clear(self):
        """Clear all data"""
        self.store.clear()
        logger.info("KeyValueStore cleared")


class MemoryStore:
    """Combined memory store with vector and key-value capabilities"""

    def __init__(self, vector_dimension: int = 384, use_vectors: bool = True):
        self.use_vectors = use_vectors

        if use_vectors:
            self.vector_store = VectorStore(vector_dimension)
        else:
            self.vector_store = None

        self.kv_store = KeyValueStore()

        logger.info("MemoryStore initialized")

    async def add(self, memory_item: MemoryItem):
        """Add a memory item"""
        if self.use_vectors and self.vector_store:
            await self.vector_store.add(memory_item)

        # Also add structured data to kv store
        await self.kv_store.set(f"memory:{memory_item.id}", memory_item)

    async def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search memory items"""
        if self.use_vectors and self.vector_store:
            return await self.vector_store.search(query, top_k)
        else:
            # Simple text search fallback
            all_keys = await self.kv_store.keys()
            results = []

            for key in all_keys:
                if key.startswith("memory:"):
                    item = await self.kv_store.get(key)
                    if query.lower() in item.content.lower():
                        results.append(item)

            results.sort(key=lambda x: x.importance, reverse=True)
            return results[:top_k]

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get specific memory item by ID"""
        return await self.kv_store.get(f"memory:{memory_id}")

    async def set_kv(self, key: str, value: Any):
        """Set key-value data"""
        await self.kv_store.set(key, value)

    async def get_kv(self, key: str, default: Any = None) -> Any:
        """Get key-value data"""
        return await self.kv_store.get(key, default)

    def save(self, path: Path):
        """Save memory store to disk"""
        path.mkdir(parents=True, exist_ok=True)

        if self.use_vectors and self.vector_store:
            self.vector_store.save(path)

        # Save kv store
        with open(path / "kv_store.json", "w") as f:
            json.dump(self.kv_store.store, f)

        logger.info(f"MemoryStore saved to {path}")

    def load(self, path: Path):
        """Load memory store from disk"""
        if not path.exists():
            logger.warning(f"MemoryStore path {path} does not exist")
            return

        if self.use_vectors and self.vector_store:
            self.vector_store.load(path)

        # Load kv store
        kv_path = path / "kv_store.json"
        if kv_path.exists():
            with open(kv_path, "r") as f:
                self.kv_store.store = json.load(f)

        logger.info(f"MemoryStore loaded from {path}")

"""
Semantic Similarity Cache
=========================

Approximate nearest neighbor search for cache hits beyond exact token matching.

Uses embeddings-based similarity to find cached sequences similar to input,
enabling cache reuse for semantically equivalent prompts.

Techniques:
- Embedding-based representation
- Approximate nearest neighbor search (HNSW)
- Similarity threshold matching
- Semantic cache statistics

Sprint 2.2 Days 3-4 - Advanced Caching
Created: 2025-12-27
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CachedSemanticSequence:
    """Cached sequence with semantic information."""
    sequence_id: str
    tokens: torch.Tensor  # Token IDs
    embedding: torch.Tensor  # Sequence embedding
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # Cached KV
    similarity_hits: int = 0
    exact_hits: int = 0
    timestamp: float = 0.0
    embedding_dim: int = 768


class EmbeddingModel(nn.Module):
    """Simple embedding model for sequences."""
    
    def __init__(self, vocab_size: int = 32000, embedding_dim: int = 768):
        """Initialize embedding model."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed token sequence.
        
        Args:
            tokens: Token IDs [seq_len]
        
        Returns:
            Sequence embedding [embedding_dim]
        """
        seq_len = tokens.shape[0]
        positions = torch.arange(seq_len, device=tokens.device)
        
        # Embeddings
        token_embeds = self.token_embedding(tokens)  # [seq_len, dim]
        pos_embeds = self.position_embedding(positions)  # [seq_len, dim]
        
        # Combine and pool
        embeds = token_embeds + pos_embeds
        embeds = self.norm(embeds)
        
        # Mean pooling
        sequence_embedding = embeds.mean(dim=0)
        
        return sequence_embedding / (torch.norm(sequence_embedding) + 1e-8)


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search.
    
    Simplified implementation for semantic cache matching.
    """
    
    def __init__(self, embedding_dim: int, max_neighbors: int = 10):
        """
        Initialize HNSW index.
        
        Args:
            embedding_dim: Embedding dimension
            max_neighbors: Max neighbors per node
        """
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self.query_count = 0
        self.search_cost = 0.0
    
    def add(self, sequence_id: str, embedding: torch.Tensor):
        """
        Add embedding to index.
        
        Args:
            sequence_id: Sequence identifier
            embedding: Normalized embedding [embedding_dim]
        """
        self.embeddings[sequence_id] = embedding.clone().detach()
        self.neighbors[sequence_id] = []
        
        # Connect to nearest neighbors
        for existing_id, existing_emb in self.embeddings.items():
            if existing_id == sequence_id:
                continue
            
            similarity = torch.dot(embedding, existing_emb).item()
            
            # Add bidirectional connections
            if len(self.neighbors[sequence_id]) < self.max_neighbors:
                self.neighbors[sequence_id].append((existing_id, similarity))
            
            if len(self.neighbors[existing_id]) < self.max_neighbors:
                self.neighbors[existing_id].append((sequence_id, similarity))
        
        # Sort by similarity (highest first)
        for seq_id in self.neighbors:
            self.neighbors[seq_id].sort(key=lambda x: x[1], reverse=True)
            if len(self.neighbors[seq_id]) > self.max_neighbors:
                self.neighbors[seq_id] = self.neighbors[seq_id][:self.max_neighbors]
    
    def search(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
        ef: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            ef: Search width parameter
        
        Returns:
            List of (sequence_id, similarity) tuples
        """
        self.query_count += 1
        
        if not self.embeddings:
            return []
        
        # Start from random entry point
        import random
        current_id = random.choice(list(self.embeddings.keys()))
        
        # Greedy search
        candidates = [(torch.dot(query_embedding, self.embeddings[current_id]).item(), current_id)]
        visited = {current_id}
        
        for _ in range(ef):
            worst_dist = max(candidates, key=lambda x: x[0])[0]
            
            # Explore neighbors
            for neighbor_id, _ in self.neighbors.get(current_id, []):
                if neighbor_id in visited:
                    continue
                
                visited.add(neighbor_id)
                similarity = torch.dot(query_embedding, self.embeddings[neighbor_id]).item()
                
                if similarity > worst_dist:
                    candidates.append((similarity, neighbor_id))
                    candidates.sort(reverse=True)
                    if len(candidates) > k:
                        candidates = candidates[:k]
        
        # Return top k
        return [(seq_id, sim) for sim, seq_id in sorted(candidates, reverse=True)[:k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        avg_neighbors = np.mean([len(n) for n in self.neighbors.values()])
        
        return {
            "num_embeddings": len(self.embeddings),
            "avg_neighbors": avg_neighbors,
            "query_count": self.query_count,
            "total_embeddings_bytes": sum(e.numel() * 4 for e in self.embeddings.values())
        }


class SemanticCache:
    """
    Semantic similarity-based cache for KV sequences.
    
    Enables cache reuse for semantically equivalent prompts.
    """
    
    def __init__(
        self,
        embedding_model: Optional[nn.Module] = None,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            embedding_model: Model to generate embeddings
            embedding_dim: Embedding dimension
            similarity_threshold: Minimum similarity for cache hit
            max_cache_size: Maximum cached sequences
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Use provided model or create default
        if embedding_model is None:
            self.embedding_model = EmbeddingModel(embedding_dim=embedding_dim)
        else:
            self.embedding_model = embedding_model
        
        # Index for semantic search
        self.index = HNSWIndex(embedding_dim, max_neighbors=10)
        
        # Cache storage
        self.cache: Dict[str, CachedSemanticSequence] = {}
        
        # Statistics
        self.total_queries = 0
        self.semantic_hits = 0
        self.exact_hits = 0
        
        logger.info(f"SemanticCache initialized with threshold={similarity_threshold}")
    
    def get_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for token sequence.
        
        Args:
            tokens: Token IDs
        
        Returns:
            Sequence embedding
        """
        with torch.no_grad():
            embedding = self.embedding_model(tokens)
        return embedding
    
    def add_sequence(
        self,
        sequence_id: str,
        tokens: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Add sequence to semantic cache.
        
        Args:
            sequence_id: Unique sequence ID
            tokens: Token IDs
            kv_cache: Optional cached KV tensors
        """
        # Check capacity
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # Get embedding
        embedding = self.get_embedding(tokens)
        
        # Create cached entry
        import time
        cached = CachedSemanticSequence(
            sequence_id=sequence_id,
            tokens=tokens.clone(),
            embedding=embedding,
            kv_cache=kv_cache,
            timestamp=time.time(),
            embedding_dim=self.embedding_dim
        )
        
        # Store and index
        self.cache[sequence_id] = cached
        self.index.add(sequence_id, embedding)
        
        logger.debug(f"Added {sequence_id} to semantic cache")
    
    def find_similar(
        self,
        tokens: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[str, float, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Find semantically similar cached sequences.
        
        Args:
            tokens: Query token sequence
            k: Number of results
        
        Returns:
            List of (sequence_id, similarity, kv_cache)
        """
        self.total_queries += 1
        
        # Get query embedding
        query_embedding = self.get_embedding(tokens)
        
        # Search index
        results = self.index.search(query_embedding, k=k)
        
        # Filter by threshold and collect KV cache
        similar = []
        for seq_id, similarity in results:
            if similarity >= self.similarity_threshold:
                cached = self.cache.get(seq_id)
                if cached:
                    kv = (cached.kv_cache[0], cached.kv_cache[1]) if cached.kv_cache else None
                    similar.append((seq_id, similarity, kv))
                    
                    # Update hit count
                    cached.similarity_hits += 1
                    self.semantic_hits += 1
        
        return similar
    
    def _evict_oldest(self):
        """Evict oldest cached sequence."""
        if not self.cache:
            return
        
        oldest_id = min(self.cache.keys(), 
                       key=lambda x: self.cache[x].timestamp)
        del self.cache[oldest_id]
        logger.debug(f"Evicted {oldest_id} from semantic cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.semantic_hits / max(1, self.total_queries)
        
        return {
            "cache_size": len(self.cache),
            "total_queries": self.total_queries,
            "semantic_hits": self.semantic_hits,
            "hit_rate": hit_rate,
            "index_stats": self.index.get_statistics(),
            "cache_memory_mb": sum(
                (s.tokens.numel() * 4 + s.embedding.numel() * 4) / 1e6
                for s in self.cache.values()
            )
        }


class HybridSemanticCache:
    """
    Hybrid cache combining exact matching (fast) and semantic similarity (comprehensive).
    """
    
    def __init__(
        self,
        embedding_model: Optional[nn.Module] = None,
        semantic_threshold: float = 0.85,
        max_cache_size: int = 1000
    ):
        """Initialize hybrid cache."""
        self.exact_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.semantic_cache = SemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=semantic_threshold,
            max_cache_size=max_cache_size
        )
        self.exact_hits = 0
        self.semantic_hits = 0
    
    def find_cached(
        self,
        tokens: torch.Tensor,
        use_exact: bool = True,
        use_semantic: bool = True
    ) -> Optional[Tuple[str, torch.Tensor, torch.Tensor, float]]:
        """
        Find cached result (exact first, then semantic).
        
        Args:
            tokens: Query tokens
            use_exact: Try exact match
            use_semantic: Try semantic match
        
        Returns:
            (sequence_id, k, v, similarity) or None
        """
        # Try exact match first
        if use_exact:
            tokens_str = str(tokens.tolist())
            if tokens_str in self.exact_cache:
                k, v = self.exact_cache[tokens_str]
                self.exact_hits += 1
                return (tokens_str, k, v, 1.0)
        
        # Try semantic match
        if use_semantic:
            similar = self.semantic_cache.find_similar(tokens, k=1)
            if similar:
                seq_id, similarity, kv = similar[0]
                if kv:
                    self.semantic_hits += 1
                    return (seq_id, kv[0], kv[1], similarity)
        
        return None
    
    def cache_result(
        self,
        tokens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_exact: bool = True,
        use_semantic: bool = True
    ):
        """Cache generated KV."""
        seq_id = f"seq_{len(self.exact_cache) + len(self.semantic_cache.cache)}"
        
        if use_exact:
            tokens_str = str(tokens.tolist())
            self.exact_cache[tokens_str] = (k.clone(), v.clone())
        
        if use_semantic:
            self.semantic_cache.add_sequence(seq_id, tokens, (k, v))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hybrid cache statistics."""
        total_hits = self.exact_hits + self.semantic_hits
        total_requests = total_hits + self.semantic_cache.total_queries
        
        return {
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": len(self.semantic_cache.cache),
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "total_hit_rate": self.exact_hits / max(1, total_requests) if total_hits > 0 else 0,
            "semantic_hit_rate": self.semantic_cache.get_statistics()["hit_rate"]
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Semantic Cache...")
    
    # Create cache
    cache = SemanticCache(similarity_threshold=0.8)
    
    # Add sequences
    seq1 = torch.tensor([1, 2, 3, 4, 5])
    seq2 = torch.tensor([1, 2, 3, 4, 5])  # Exact duplicate
    seq3 = torch.tensor([1, 2, 3, 6, 7])  # Similar but different
    
    cache.add_sequence("seq_1", seq1)
    cache.add_sequence("seq_2", seq2)
    cache.add_sequence("seq_3", seq3)
    
    print("Cached 3 sequences")
    
    # Query similar
    query = torch.tensor([1, 2, 3, 4, 5])
    results = cache.find_similar(query)
    
    print(f"\nFound {len(results)} similar sequences:")
    for seq_id, sim, _ in results:
        print(f"  {seq_id}: similarity={sim:.3f}")
    
    # Stats
    stats = cache.get_statistics()
    print(f"\nCache stats:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")
    
    print("\nSemantic cache test passed!")

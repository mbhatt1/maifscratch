"""
Optimized semantic embedding implementation for high-performance search.
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

@dataclass
class SemanticEmbedding:
    """Represents a semantic embedding with metadata."""
    vector: List[float]
    source_hash: str
    model_name: str
    timestamp: float
    metadata: Optional[Dict] = None

class OptimizedSemanticEmbedder:
    """High-performance semantic embedder with batch processing and ANN search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = True):
        self.model_name = model_name
        self.embeddings: List[SemanticEmbedding] = []
        self.embedding_matrix = None
        self.faiss_index = None
        self.use_gpu = use_gpu
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Enable GPU if available and requested
                device = 'cuda' if use_gpu and self._cuda_available() else 'cpu'
                self.model = SentenceTransformer(model_name, device=device)
                print(f"Loaded model {model_name} on {device}")
            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")
                self.model = None
        else:
            print("Warning: sentence-transformers not available")
            self.model = None
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[SemanticEmbedding]:
        """Generate embeddings for multiple texts using efficient batch processing."""
        if not self.model:
            return self._fallback_embeddings(texts)
        
        embeddings = []
        
        # Process in batches for optimal performance
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Batch encode - this is much faster than individual encoding
            batch_vectors = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Pre-normalize for faster cosine similarity
            )
            
            # Create SemanticEmbedding objects
            for j, (text, vector) in enumerate(zip(batch_texts, batch_vectors)):
                source_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                embedding = SemanticEmbedding(
                    vector=vector.tolist(),
                    source_hash=source_hash,
                    model_name=self.model_name,
                    timestamp=time.time()
                )
                embeddings.append(embedding)
        
        return embeddings
    
    def build_search_index(self, embeddings: List[SemanticEmbedding]) -> None:
        """Build FAISS index for fast approximate nearest neighbor search."""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available. Using brute-force search.")
            self.embedding_matrix = np.array([emb.vector for emb in embeddings])
            return
        
        # Convert embeddings to matrix
        embedding_vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
        dimension = embedding_vectors.shape[1]
        
        # Create FAISS index - using HNSW for best performance
        if len(embeddings) > 10000:
            # Use HNSW for large datasets
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
            index.hnsw.efConstruction = 200  # Higher = better quality, slower build
            index.hnsw.efSearch = 50  # Higher = better recall, slower search
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        
        # Add vectors to index
        index.add(embedding_vectors)
        
        self.faiss_index = index
        self.embedding_matrix = embedding_vectors
        self.embeddings = embeddings
        
        print(f"Built FAISS index with {len(embeddings)} vectors")
    
    def search_similar(self, query_embedding: SemanticEmbedding, k: int = 10) -> List[Tuple[int, float]]:
        """Fast similarity search using FAISS index."""
        query_vector = np.array([query_embedding.vector], dtype=np.float32)
        
        if self.faiss_index is not None:
            # Use FAISS for fast search
            similarities, indices = self.faiss_index.search(query_vector, k)
            return [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities[0])]
        
        elif self.embedding_matrix is not None:
            # Fallback to optimized numpy computation
            similarities = np.dot(self.embedding_matrix, query_vector.T).flatten()
            top_indices = np.argsort(similarities)[-k:][::-1]  # Top k in descending order
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        else:
            raise ValueError("No search index built. Call build_search_index() first.")
    
    def embed_text_single(self, text: str) -> SemanticEmbedding:
        """Generate embedding for a single text (for queries)."""
        if not self.model:
            return self._fallback_embedding(text)
        
        vector = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        source_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return SemanticEmbedding(
            vector=vector.tolist(),
            source_hash=source_hash,
            model_name=self.model_name,
            timestamp=time.time()
        )
    
    def _fallback_embeddings(self, texts: List[str]) -> List[SemanticEmbedding]:
        """Fallback to random embeddings when model is not available."""
        embeddings = []
        for text in texts:
            # Generate deterministic "embedding" based on text hash
            text_hash = hashlib.sha256(text.encode()).digest()
            vector = np.frombuffer(text_hash, dtype=np.uint8)[:384].astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            source_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            embedding = SemanticEmbedding(
                vector=vector.tolist(),
                source_hash=source_hash,
                model_name="fallback",
                timestamp=time.time()
            )
            embeddings.append(embedding)
        
        return embeddings
    
    def _fallback_embedding(self, text: str) -> SemanticEmbedding:
        """Fallback to random embedding when model is not available."""
        return self._fallback_embeddings([text])[0]

# Backward compatibility wrapper
class SemanticEmbedder(OptimizedSemanticEmbedder):
    """Backward compatible interface with performance optimizations."""
    
    def embed_texts(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[SemanticEmbedding]:
        """Generate embeddings for multiple texts (optimized batch version)."""
        return self.embed_texts_batch(texts, batch_size=64)
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[SemanticEmbedding]:
        """Ensure the method is available in the wrapper."""
        return super().embed_texts_batch(texts, batch_size)
    
    def embed_text(self, text: str, metadata: Optional[Dict] = None) -> SemanticEmbedding:
        """Generate embedding for single text."""
        return self.embed_text_single(text)
    
    def compute_similarity(self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding) -> float:
        """Compute cosine similarity between two embeddings."""
        v1 = np.array(embedding1.vector)
        v2 = np.array(embedding2.vector)
        
        # For normalized vectors, dot product = cosine similarity
        return float(np.dot(v1, v2))
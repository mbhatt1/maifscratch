"""
Semantic embedding and knowledge graph functionality for MAIF.
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

@dataclass
class SemanticEmbedding:
    """Represents a semantic embedding with metadata."""
    vector: List[float]
    source_hash: str
    model_name: str
    timestamp: float
    metadata: Optional[Dict] = None

@dataclass
class KnowledgeTriple:
    """Represents a knowledge graph triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None

class SemanticEmbedder:
    """Generates and manages semantic embeddings for multimodal content."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings: List[SemanticEmbedding] = []
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")
                self.model = None
        else:
            print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
            self.model = None
    
    def embed_text(self, text: str, metadata: Optional[Dict] = None) -> SemanticEmbedding:
        """Generate embedding for text content."""
        if not self.model:
            # Fallback: simple hash-based pseudo-embedding
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            vector = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 768*2), 2)]
            vector = vector[:384]  # Truncate to reasonable size
        else:
            vector = self.model.encode(text).tolist()
        
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        
        embedding = SemanticEmbedding(
            vector=vector,
            source_hash=source_hash,
            model_name=self.model_name,
            timestamp=time.time(),
            metadata=metadata
        )
        
        self.embeddings.append(embedding)
        return embedding
    
    def embed_texts(self, texts: List[str], metadata_list: Optional[List[Dict]] = None) -> List[SemanticEmbedding]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        metadata_list = metadata_list or [None] * len(texts)
        
        for text, metadata in zip(texts, metadata_list):
            embedding = self.embed_text(text, metadata)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding) -> float:
        """Compute cosine similarity between two embeddings."""
        v1 = np.array(embedding1.vector)
        v2 = np.array(embedding2.vector)
        
        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search_similar(self, query_embedding: SemanticEmbedding, top_k: int = 5) -> List[Tuple[SemanticEmbedding, float]]:
        """Find most similar embeddings to a query."""
        similarities = []
        
        for embedding in self.embeddings:
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append((embedding, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_embeddings_data(self) -> List[Dict]:
        """Get embeddings in serializable format."""
        return [
            {
                "vector": emb.vector,
                "source_hash": emb.source_hash,
                "model_name": emb.model_name,
                "timestamp": emb.timestamp,
                "metadata": emb.metadata or {}
            }
            for emb in self.embeddings
        ]

class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from multimodal content."""
    
    def __init__(self):
        self.triples: List[KnowledgeTriple] = []
        self.entities: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
    
    def add_triple(self, subject: str, predicate: str, obj: str, 
                   confidence: float = 1.0, source: Optional[str] = None) -> KnowledgeTriple:
        """Add a knowledge triple to the graph."""
        triple = KnowledgeTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source
        )
        
        self.triples.append(triple)
        
        # Update entity and relation tracking
        self._update_entity(subject)
        self._update_entity(obj)
        self._update_relation(predicate)
        
        return triple
    
    def _update_entity(self, entity: str):
        """Update entity metadata."""
        if entity not in self.entities:
            self.entities[entity] = {
                "mentions": 0,
                "relations": set()
            }
        self.entities[entity]["mentions"] += 1
    
    def _update_relation(self, relation: str):
        """Update relation metadata."""
        if relation not in self.relations:
            self.relations[relation] = {
                "frequency": 0,
                "subjects": set(),
                "objects": set()
            }
        self.relations[relation]["frequency"] += 1
    
    def extract_entities_from_text(self, text: str, source: Optional[str] = None) -> List[str]:
        """Extract entities from text (simple implementation)."""
        # This is a very basic implementation
        # In practice, you'd use NLP libraries like spaCy or NLTK
        import re
        
        # Simple pattern for capitalized words (potential entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Add entities to graph with basic relations
        for i, entity in enumerate(entities):
            if i > 0:
                self.add_triple(entities[i-1], "mentions_with", entity, source=source)
        
        return entities
    
    def find_related_entities(self, entity: str, max_depth: int = 2) -> List[Tuple[str, str, int]]:
        """Find entities related to a given entity."""
        related = []
        visited = set()
        
        def _traverse(current_entity: str, depth: int, path: str):
            if depth > max_depth or current_entity in visited:
                return
            
            visited.add(current_entity)
            
            for triple in self.triples:
                if triple.subject == current_entity:
                    new_path = f"{path} -> {triple.predicate} -> {triple.object}"
                    related.append((triple.object, new_path, depth))
                    _traverse(triple.object, depth + 1, new_path)
                elif triple.object == current_entity:
                    new_path = f"{path} <- {triple.predicate} <- {triple.subject}"
                    related.append((triple.subject, new_path, depth))
                    _traverse(triple.subject, depth + 1, new_path)
        
        _traverse(entity, 0, entity)
        return related
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            "total_triples": len(self.triples),
            "unique_entities": len(self.entities),
            "unique_relations": len(self.relations),
            "avg_confidence": sum(t.confidence for t in self.triples) / len(self.triples) if self.triples else 0,
            "top_entities": sorted(
                [(entity, data["mentions"]) for entity, data in self.entities.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
            "top_relations": sorted(
                [(relation, data["frequency"]) for relation, data in self.relations.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    def export_to_json(self) -> Dict:
        """Export knowledge graph to JSON format."""
        return {
            "triples": [
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "confidence": t.confidence,
                    "source": t.source
                }
                for t in self.triples
            ],
            "entities": {
                entity: {
                    "mentions": data["mentions"],
                    "relations": list(data["relations"])
                }
                for entity, data in self.entities.items()
            },
            "relations": {
                relation: {
                    "frequency": data["frequency"],
                    "subjects": list(data["subjects"]),
                    "objects": list(data["objects"])
                }
                for relation, data in self.relations.items()
            },
            "statistics": self.get_graph_statistics()
        }
    
    def import_from_json(self, data: Dict):
        """Import knowledge graph from JSON format."""
        self.triples = []
        self.entities = {}
        self.relations = {}
        
        for triple_data in data.get("triples", []):
            self.add_triple(
                triple_data["subject"],
                triple_data["predicate"],
                triple_data["object"],
                triple_data.get("confidence", 1.0),
                triple_data.get("source")
            )

class CrossModalAttention:
    """Implements Adaptive Cross-Modal Attention Mechanism (ACAM)."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.attention_weights: Dict[Tuple[str, str], float] = {}
    
    def compute_coherence_score(self, embedding1: List[float], embedding2: List[float], 
                               trust_score: float = 1.0) -> float:
        """Compute coherence score between two embeddings."""
        # Cosine similarity
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm1 * norm2)
        
        # Combine with trust score
        coherence = cosine_sim * trust_score
        return max(0.0, min(1.0, coherence))  # Clamp to [0, 1]
    
    def compute_attention_weights(self, embeddings: Dict[str, List[float]], 
                                 trust_scores: Optional[Dict[str, float]] = None) -> Dict[Tuple[str, str], float]:
        """Compute attention weights between all modality pairs."""
        trust_scores = trust_scores or {}
        modalities = list(embeddings.keys())
        
        attention_weights = {}
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    trust_score = min(
                        trust_scores.get(mod1, 1.0),
                        trust_scores.get(mod2, 1.0)
                    )
                    
                    coherence = self.compute_coherence_score(
                        embeddings[mod1],
                        embeddings[mod2],
                        trust_score
                    )
                    
                    # Apply softmax-like normalization
                    attention_weights[(mod1, mod2)] = coherence
        
        # Normalize weights
        total_weight = sum(attention_weights.values())
        if total_weight > 0:
            for key in attention_weights:
                attention_weights[key] /= total_weight
        
        self.attention_weights = attention_weights
        return attention_weights
    
    def get_attended_representation(self, embeddings: Dict[str, List[float]], 
                                   query_modality: str) -> List[float]:
        """Get attention-weighted representation for a query modality."""
        if query_modality not in embeddings:
            return []
        
        attended = np.array(embeddings[query_modality])
        
        for (mod1, mod2), weight in self.attention_weights.items():
            if mod1 == query_modality and mod2 in embeddings:
                attended += weight * np.array(embeddings[mod2])
        
        return attended.tolist()

class HierarchicalSemanticCompression:
    """Implements Hierarchical Semantic Compression (HSC) algorithm."""
    
    def __init__(self, compression_levels: int = 3):
        self.compression_levels = compression_levels
        self.compression_ratios = {}
        
    def compress_embeddings(self, embeddings: List[List[float]],
                           semantic_clusters: Optional[List[int]] = None) -> Dict[str, Any]:
        """Compress embeddings using hierarchical semantic approach."""
        if not embeddings:
            return {"compressed_data": [], "metadata": {}}
        
        embeddings_array = np.array(embeddings)
        
        # Level 1: Dimensionality reduction using PCA-like approach
        level1_compressed = self._apply_dimensionality_reduction(embeddings_array)
        
        # Level 2: Semantic clustering and centroid compression
        level2_compressed, clusters = self._apply_semantic_clustering(level1_compressed, semantic_clusters)
        
        # Level 3: Quantization and encoding
        level3_compressed = self._apply_quantization(level2_compressed)
        
        compression_metadata = {
            "original_shape": embeddings_array.shape,
            "compression_levels": self.compression_levels,
            "clusters": clusters,
            "compression_ratio": len(embeddings_array.flatten()) / len(level3_compressed.flatten())
        }
        
        return {
            "compressed_data": level3_compressed.tolist(),
            "metadata": compression_metadata
        }
    
    def _apply_dimensionality_reduction(self, embeddings: np.ndarray, target_dim: Optional[int] = None) -> np.ndarray:
        """Apply dimensionality reduction while preserving semantic relationships."""
        if target_dim is None:
            target_dim = min(embeddings.shape[1] // 2, 192)  # Reduce by half, max 192
        
        # Simple SVD-based reduction (in practice, would use more sophisticated methods)
        U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)
        reduced = U[:, :target_dim] @ np.diag(s[:target_dim])
        
        return reduced
    
    def _apply_semantic_clustering(self, embeddings: np.ndarray,
                                  predefined_clusters: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
        """Apply semantic clustering to group similar embeddings."""
        if predefined_clusters:
            clusters = predefined_clusters
        else:
            # Simple k-means-like clustering
            n_clusters = min(max(1, len(embeddings) // 10), min(20, len(embeddings)))
            if n_clusters >= len(embeddings):
                n_clusters = max(1, len(embeddings) // 2)
            clusters = self._simple_kmeans(embeddings, n_clusters)
        
        # Compress using cluster centroids
        unique_clusters = list(set(clusters))
        centroids = []
        
        for cluster_id in unique_clusters:
            cluster_embeddings = embeddings[np.array(clusters) == cluster_id]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)
        
        return np.array(centroids), clusters
    
    def _simple_kmeans(self, embeddings: np.ndarray, k: int) -> List[int]:
        """Simple k-means clustering implementation."""
        n_samples, n_features = embeddings.shape
        
        # Ensure k doesn't exceed n_samples
        k = min(k, n_samples)
        
        # Initialize centroids randomly
        if k == n_samples:
            # Each point is its own cluster
            return list(range(n_samples))
        
        centroids = embeddings[np.random.choice(n_samples, k, replace=False)]
        clusters = [0] * n_samples
        
        for _ in range(10):  # Max 10 iterations
            # Assign points to nearest centroid
            for i, embedding in enumerate(embeddings):
                distances = [np.linalg.norm(embedding - centroid) for centroid in centroids]
                clusters[i] = np.argmin(distances)
            
            # Update centroids
            for j in range(k):
                cluster_points = embeddings[np.array(clusters) == j]
                if len(cluster_points) > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
        
        return clusters
    
    def _apply_quantization(self, embeddings: np.ndarray, bits: int = 8) -> np.ndarray:
        """Apply quantization to reduce storage requirements."""
        # Normalize to [0, 1] range
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)
        normalized = (embeddings - min_val) / (max_val - min_val + 1e-8)
        
        # Quantize to specified bits
        quantized = np.round(normalized * (2**bits - 1)).astype(np.uint8)
        
        return quantized
    
    def decompress_embeddings(self, compressed_data: Dict[str, Any]) -> List[List[float]]:
        """Decompress embeddings from HSC format."""
        compressed_array = np.array(compressed_data["compressed_data"])
        metadata = compressed_data["metadata"]
        
        # Reverse quantization
        dequantized = compressed_array.astype(np.float32) / 255.0
        
        # Expand from centroids (simplified reconstruction)
        # In practice, would use more sophisticated reconstruction
        expanded = np.repeat(dequantized,
                           metadata["original_shape"][0] // len(dequantized) + 1,
                           axis=0)[:metadata["original_shape"][0]]
        
        return expanded.tolist()

class CryptographicSemanticBinding:
    """Implements Cryptographic Semantic Binding (CSB) algorithm."""
    
    def __init__(self):
        self.bindings = {}
        
    def create_semantic_commitment(self, embedding: List[float], source_data: str,
                                 salt: Optional[str] = None) -> Dict[str, str]:
        """Create cryptographic commitment binding embedding to source data."""
        import hashlib
        import secrets
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Create commitment using hash-based scheme
        embedding_bytes = np.array(embedding).tobytes()
        source_bytes = source_data.encode('utf-8')
        
        # Commitment = H(embedding || source_data || salt)
        commitment_input = embedding_bytes + source_bytes + salt.encode('utf-8')
        commitment = hashlib.sha256(commitment_input).hexdigest()
        
        # Create verification hash for embedding authenticity
        embedding_hash = hashlib.sha256(embedding_bytes + salt.encode('utf-8')).hexdigest()
        
        # Create source data hash
        source_hash = hashlib.sha256(source_bytes + salt.encode('utf-8')).hexdigest()
        
        binding = {
            "commitment": commitment,
            "embedding_hash": embedding_hash,
            "source_hash": source_hash,
            "salt": salt,
            "timestamp": str(int(time.time()))
        }
        
        self.bindings[commitment] = binding
        return binding
    
    def verify_semantic_binding(self, embedding: List[float], source_data: str,
                               binding: Dict[str, str]) -> bool:
        """Verify that embedding corresponds to source data using CSB."""
        import hashlib
        
        try:
            salt = binding["salt"]
            
            # Recreate commitment
            embedding_bytes = np.array(embedding).tobytes()
            source_bytes = source_data.encode('utf-8')
            commitment_input = embedding_bytes + source_bytes + salt.encode('utf-8')
            computed_commitment = hashlib.sha256(commitment_input).hexdigest()
            
            # Verify commitment matches
            if computed_commitment != binding["commitment"]:
                return False
            
            # Verify embedding hash
            computed_embedding_hash = hashlib.sha256(embedding_bytes + salt.encode('utf-8')).hexdigest()
            if computed_embedding_hash != binding["embedding_hash"]:
                return False
            
            # Verify source hash
            computed_source_hash = hashlib.sha256(source_bytes + salt.encode('utf-8')).hexdigest()
            if computed_source_hash != binding["source_hash"]:
                return False
            
            return True
            
        except Exception:
            return False
    
    def create_zero_knowledge_proof(self, embedding: List[float],
                                   binding: Dict[str, str]) -> Dict[str, str]:
        """Create basic zero-knowledge proof for embedding authenticity (simplified)."""
        import hashlib
        import secrets
        
        # Simplified ZK proof - in practice would use more sophisticated schemes
        nonce = secrets.token_hex(16)
        
        # Create proof that we know the embedding without revealing it
        embedding_bytes = np.array(embedding).tobytes()
        salt = binding["salt"]
        
        # Proof = H(embedding || nonce || salt)
        proof_input = embedding_bytes + nonce.encode('utf-8') + salt.encode('utf-8')
        proof = hashlib.sha256(proof_input).hexdigest()
        
        return {
            "proof": proof,
            "nonce": nonce,
            "commitment": binding["commitment"],
            "timestamp": str(int(time.time()))
        }
    
    def verify_zero_knowledge_proof(self, proof_data: Dict[str, str],
                                   binding: Dict[str, str]) -> bool:
        """Verify zero-knowledge proof (simplified implementation)."""
        # This is a simplified verification - real ZK proofs are much more complex
        return proof_data.get("commitment") == binding.get("commitment")

class DeepSemanticUnderstanding:
    """Enhanced cross-modal AI with deep semantic understanding across all modalities."""
    
    def __init__(self):
        self.modality_processors = {}
        self.cross_modal_attention = CrossModalAttention()
        self.semantic_memory = {}
        
    def register_modality_processor(self, modality: str, processor_func):
        """Register a processor function for a specific modality."""
        self.modality_processors[modality] = processor_func
    
    def process_multimodal_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs from multiple modalities with deep semantic understanding."""
        processed_embeddings = {}
        semantic_features = {}
        
        # Process each modality
        for modality, data in inputs.items():
            if modality in self.modality_processors:
                embedding = self.modality_processors[modality](data)
                processed_embeddings[modality] = embedding
                
                # Extract semantic features
                semantic_features[modality] = self._extract_semantic_features(data, modality)
        
        # Apply cross-modal attention
        attention_weights = self.cross_modal_attention.compute_attention_weights(processed_embeddings)
        
        # Create unified semantic representation
        unified_representation = self._create_unified_representation(
            processed_embeddings, semantic_features, attention_weights
        )
        
        return {
            "embeddings": processed_embeddings,
            "semantic_features": semantic_features,
            "attention_weights": attention_weights,
            "unified_representation": unified_representation
        }
    
    def _extract_semantic_features(self, data: Any, modality: str) -> Dict[str, Any]:
        """Extract semantic features specific to each modality."""
        features = {"modality": modality}
        
        if modality == "text":
            # Extract text-specific semantic features
            features.update({
                "length": len(str(data)),
                "word_count": len(str(data).split()),
                "sentiment": self._simple_sentiment_analysis(str(data)),
                "entities": self._extract_simple_entities(str(data))
            })
        elif modality == "image":
            # Extract image-specific semantic features (placeholder)
            features.update({
                "type": "image",
                "estimated_complexity": "medium"  # Would use actual image analysis
            })
        elif modality == "audio":
            # Extract audio-specific semantic features (placeholder)
            features.update({
                "type": "audio",
                "estimated_duration": "unknown"  # Would use actual audio analysis
            })
        
        return features
    
    def _simple_sentiment_analysis(self, text: str) -> str:
        """Simple sentiment analysis (placeholder for more sophisticated analysis)."""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Simple entity extraction (placeholder for NLP libraries)."""
        import re
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))  # Remove duplicates
    
    def _create_unified_representation(self, embeddings: Dict[str, List[float]],
                                     semantic_features: Dict[str, Dict],
                                     attention_weights: Dict[Tuple[str, str], float]) -> List[float]:
        """Create unified semantic representation across all modalities."""
        if not embeddings:
            return []
        
        # Start with the first modality's embedding
        modalities = list(embeddings.keys())
        unified = np.array(embeddings[modalities[0]])
        
        # Apply attention-weighted combination
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j and (mod1, mod2) in attention_weights:
                    weight = attention_weights[(mod1, mod2)]
                    unified += weight * np.array(embeddings[mod2])
        
        # Normalize the result
        unified = unified / np.linalg.norm(unified)
        
        return unified.tolist()
    
    def semantic_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic reasoning across modalities."""
        # Process query
        query_embedding = self.modality_processors.get("text", lambda x: [0.0] * 384)(query)
        
        # Find relevant context
        relevant_context = self._find_relevant_context(query_embedding, context)
        
        # Generate reasoning result
        reasoning_result = {
            "query": query,
            "relevant_context": relevant_context,
            "confidence": self._calculate_reasoning_confidence(query_embedding, relevant_context),
            "explanation": self._generate_explanation(query, relevant_context)
        }
        
        return reasoning_result
    
    def _find_relevant_context(self, query_embedding: List[float],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Find context most relevant to the query."""
        # Simplified relevance calculation
        relevant = {}
        
        if "embeddings" in context:
            for modality, embedding in context["embeddings"].items():
                try:
                    # Ensure embeddings are numpy arrays and same length
                    query_vec = np.array(query_embedding)
                    embed_vec = np.array(embedding)
                    
                    # Pad shorter vector to match longer one
                    if len(query_vec) != len(embed_vec):
                        max_len = max(len(query_vec), len(embed_vec))
                        if len(query_vec) < max_len:
                            query_vec = np.pad(query_vec, (0, max_len - len(query_vec)))
                        if len(embed_vec) < max_len:
                            embed_vec = np.pad(embed_vec, (0, max_len - len(embed_vec)))
                    
                    # Calculate similarity
                    query_norm = np.linalg.norm(query_vec)
                    embed_norm = np.linalg.norm(embed_vec)
                    
                    if query_norm > 0 and embed_norm > 0:
                        similarity = np.dot(query_vec, embed_vec) / (query_norm * embed_norm)
                        # Lower threshold for relevance to get some results
                        if similarity > 0.1:
                            relevant[modality] = {
                                "embedding": embedding,
                                "similarity": float(similarity)
                            }
                except Exception as e:
                    # Skip problematic embeddings
                    continue
        
        return relevant
    
    def _calculate_reasoning_confidence(self, query_embedding: List[float],
                                      relevant_context: Dict[str, Any]) -> float:
        """Calculate confidence in reasoning result."""
        if not relevant_context:
            return 0.0
        
        # Average similarity as confidence measure
        similarities = [ctx["similarity"] for ctx in relevant_context.values()]
        return sum(similarities) / len(similarities)
    
    def _generate_explanation(self, query: str, relevant_context: Dict[str, Any]) -> str:
        """Generate human-readable explanation of reasoning."""
        if not relevant_context:
            return f"No relevant context found for query: {query}"
        
        modalities = list(relevant_context.keys())
        explanation = f"Found relevant information in {len(modalities)} modalities: {', '.join(modalities)}. "
        
        for modality, ctx in relevant_context.items():
            explanation += f"{modality} shows {ctx['similarity']:.2f} similarity. "
        
        return explanation

# Import time for timestamps
import time
#!/usr/bin/env python3
"""Quick test to verify semantic fixes."""

from maif.semantic import (
    SemanticEmbedder, HierarchicalSemanticCompression, 
    CryptographicSemanticBinding, DeepSemanticUnderstanding
)

def test_semantic_embedder():
    """Test semantic embedder."""
    embedder = SemanticEmbedder(model_name="test-model")
    print(f"✓ SemanticEmbedder initialized: {embedder.model_name}")
    
    # Test embedding
    embedding = embedder.embed_text("test text", metadata={"id": 1})
    print(f"✓ Embedding vector length: {len(embedding.vector)}")
    assert embedder.model_name == "test-model"
    assert len(embedding.vector) > 0

def test_hierarchical_compression():
    """Test hierarchical compression."""
    hsc = HierarchicalSemanticCompression()
    
    # Test with target_compression_ratio parameter
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    result = hsc.compress_embeddings(embeddings, target_compression_ratio=2.0)
    print(f"✓ Compression result keys: {list(result.keys())}")
    
    # Test _apply_semantic_clustering with num_clusters
    labels = hsc._apply_semantic_clustering(embeddings, num_clusters=2)
    print(f"✓ Clustering labels: {labels}")
    assert "compressed_embeddings" in result
    assert len(labels) == len(embeddings)

def test_cryptographic_binding():
    """Test cryptographic semantic binding."""
    csb = CryptographicSemanticBinding()
    
    # Test create_semantic_commitment
    embedding = [0.1, 0.2, 0.3]
    commitment = csb.create_semantic_commitment(embedding, "test data")
    print(f"✓ Commitment keys: {list(commitment.keys())}")
    
    # Test zero-knowledge proof
    proof = csb.create_zero_knowledge_proof(embedding, "secret")
    print(f"✓ Proof keys: {list(proof.keys())}")
    assert "commitment_hash" in commitment
    assert "response" in proof

def test_deep_semantic_understanding():
    """Test deep semantic understanding."""
    dsu = DeepSemanticUnderstanding()
    
    # Check attributes
    print(f"✓ Has embedder: {hasattr(dsu, 'embedder')}")
    print(f"✓ Has kg_builder: {hasattr(dsu, 'kg_builder')}")
    print(f"✓ Has attention: {hasattr(dsu, 'attention')}")
    
    # Test process_multimodal_input
    inputs = {"text": "test text", "metadata": {"test": True}}
    result = dsu.process_multimodal_input(inputs)
    print(f"✓ Process result keys: {list(result.keys())}")
    
    # Test semantic reasoning
    query = "test query"
    context = {"text_data": ["test context"]}
    reasoning = dsu.semantic_reasoning(query, context)
    print(f"✓ Reasoning keys: {list(reasoning.keys())}")
    
    assert hasattr(dsu, 'embedder')
    assert hasattr(dsu, 'kg_builder')
    assert "understanding_score" in result

if __name__ == "__main__":
    try:
        test_semantic_embedder()
        test_hierarchical_compression()
        test_cryptographic_binding()
        test_deep_semantic_understanding()
        print("\n🎉 All semantic tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
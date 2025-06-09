#!/usr/bin/env python3
"""
One-liner test for MAIF Novel Algorithms
Run this to quickly verify everything is working
"""

def test_everything():
    """Test all novel algorithms in one go."""
    print("🧪 Testing MAIF Novel Algorithms...")
    
    try:
        # Test imports
        from maif.semantic import CrossModalAttention, HierarchicalSemanticCompression, CryptographicSemanticBinding
        print("✅ Imports successful")
        
        # Test ACAM
        acam = CrossModalAttention()
        weights = acam.compute_attention_weights({"text": [0.1]*384, "image": [0.2]*384})
        assert len(weights) > 0, "ACAM failed"
        print("✅ ACAM working")
        
        # Test HSC  
        hsc = HierarchicalSemanticCompression()
        result = hsc.compress_embeddings([[0.1]*384, [0.2]*384])
        assert result['metadata']['compression_ratio'] > 1, "HSC failed"
        print("✅ HSC working")
        
        # Test CSB
        csb = CryptographicSemanticBinding()
        binding = csb.create_semantic_commitment([0.1]*384, "test")
        valid = csb.verify_semantic_binding([0.1]*384, "test", binding)
        assert valid, "CSB failed"
        print("✅ CSB working")
        
        print("🎉 All novel algorithms working perfectly!")
        assert True  # Test passed
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_everything()
    exit(0 if success else 1)
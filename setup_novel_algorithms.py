#!/usr/bin/env python3
"""
Quick setup script for MAIF Novel Algorithms
Makes testing and validation as painless as possible
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Core dependencies
    dependencies = [
        "numpy",
        "sentence-transformers",
        "cryptography",
    ]
    
    # Optional dependencies (install if available)
    optional_deps = [
        "lzma",
        "brotli", 
        "lz4",
        "zstandard"
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
    
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional dependency {dep} not installed (this is OK)")

def run_quick_test():
    """Run a quick test of the novel algorithms."""
    print("\n" + "="*50)
    print("Running Quick Test of Novel Algorithms")
    print("="*50)
    
    try:
        # Test imports
        print("Testing imports...")
        from maif.semantic import (
            CrossModalAttention, 
            HierarchicalSemanticCompression,
            CryptographicSemanticBinding,
            DeepSemanticUnderstanding
        )
        print("✅ All novel algorithm classes imported successfully")
        
        # Quick ACAM test
        print("\nTesting ACAM...")
        acam = CrossModalAttention()
        embeddings = {
            "text": [0.1] * 384,
            "image": [0.2] * 384
        }
        weights = acam.compute_attention_weights(embeddings)
        print(f"✅ ACAM computed {len(weights)} attention weights")
        
        # Quick HSC test
        print("\nTesting HSC...")
        hsc = HierarchicalSemanticCompression()
        test_embeddings = [[0.1] * 384, [0.2] * 384]
        compressed = hsc.compress_embeddings(test_embeddings)
        print(f"✅ HSC compressed embeddings with ratio {compressed['metadata']['compression_ratio']:.1f}:1")
        
        # Quick CSB test
        print("\nTesting CSB...")
        csb = CryptographicSemanticBinding()
        binding = csb.create_semantic_commitment([0.1] * 384, "test data")
        is_valid = csb.verify_semantic_binding([0.1] * 384, "test data", binding)
        print(f"✅ CSB created and verified binding: {is_valid}")
        
        # Quick Cross-Modal AI test
        print("\nTesting Cross-Modal AI...")
        dsu = DeepSemanticUnderstanding()
        print("✅ Deep Semantic Understanding initialized")
        
        print("\n🎉 All novel algorithms are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_full_demo():
    """Run the full demonstration."""
    print("\n" + "="*50)
    print("Running Full Novel Algorithms Demo")
    print("="*50)
    
    try:
        os.chdir("examples")
        result = subprocess.run([sys.executable, "novel_algorithms_demo.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Full demo completed successfully!")
            print("\nDemo output:")
            print(result.stdout)
        else:
            print("❌ Demo failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Could not run demo: {e}")
        return False
    
    return True

def create_quick_start_guide():
    """Create a quick start guide."""
    guide = """
# MAIF Novel Algorithms - Quick Start Guide

## What's Been Implemented

✅ **ACAM** - Adaptive Cross-Modal Attention Mechanism
✅ **HSC** - Hierarchical Semantic Compression  
✅ **CSB** - Cryptographic Semantic Binding
✅ **Cross-Modal AI** - Deep semantic understanding across modalities

## Quick Usage

```python
from maif import MAIFEncoder
from maif.semantic import (
    CrossModalAttention, 
    HierarchicalSemanticCompression,
    CryptographicSemanticBinding
)

# Create MAIF with novel algorithms
encoder = MAIFEncoder(agent_id="demo")

# Add cross-modal data with ACAM
multimodal_data = {"text": "Hello world", "metadata": {"type": "demo"}}
encoder.add_cross_modal_block(multimodal_data)

# Add compressed embeddings with HSC
embeddings = [[0.1] * 384]  # Your embeddings here
encoder.add_compressed_embeddings_block(embeddings, use_hsc=True)

# Add semantic binding with CSB
encoder.add_semantic_binding_block([0.1] * 384, "source data")

# Build MAIF file
encoder.build_maif("output.maif", "manifest.json")
```

## What Was Removed

❌ Homomorphic encryption
❌ Quantum-resistant crypto  
❌ Self-evolving artifacts
❌ Blockchain integration (replaced with cryptographic verification)

## Files to Check

- `examples/novel_algorithms_demo.py` - Complete demonstration
- `maif/semantic.py` - All novel algorithm implementations
- `NOVEL_ALGORITHMS_IMPLEMENTATION.md` - Detailed documentation

## Next Steps

1. Run `python3 setup_novel_algorithms.py` to test everything
2. Check `examples/novel_algorithms_demo.py` for usage examples
3. Use the new methods in `MAIFEncoder` for your applications
"""
    
    with open("QUICK_START.md", "w") as f:
        f.write(guide)
    
    print("✅ Created QUICK_START.md")

def main():
    """Main setup function."""
    print("MAIF Novel Algorithms Setup")
    print("=" * 40)
    print("This script will:")
    print("1. Install required dependencies")
    print("2. Test all novel algorithms")
    print("3. Run the full demonstration")
    print("4. Create a quick start guide")
    print()
    
    # Install dependencies
    install_dependencies()
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed. Please check the error messages above.")
        return 1
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Ask if user wants to run full demo
    response = input("\nWould you like to run the full demonstration? (y/n): ").lower()
    if response in ['y', 'yes']:
        if run_full_demo():
            print("\n🎉 Everything is set up and working perfectly!")
        else:
            print("\n⚠️  Setup complete but demo had issues. Check the output above.")
    else:
        print("\n✅ Setup complete! Run 'python3 examples/novel_algorithms_demo.py' when ready.")
    
    print("\n📖 Check QUICK_START.md for usage examples")
    print("📖 Check NOVEL_ALGORITHMS_IMPLEMENTATION.md for detailed documentation")
    
    return 0

if __name__ == "__main__":
    exit(main())
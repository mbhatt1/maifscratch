#!/usr/bin/env python3
"""Quick test to verify our fixes."""

import os
import tempfile
from maif.core import MAIFEncoder
from maif.validation import MAIFValidator
from maif.semantic import DeepSemanticUnderstanding

def test_validation_fix():
    """Test that validation now works correctly."""
    print("Testing validation fix...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Test content", metadata={"test": True})
        
        maif_path = os.path.join(temp_dir, "test.maif")
        manifest_path = os.path.join(temp_dir, "test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Validate it
        validator = MAIFValidator()
        result = validator.validate_file(maif_path, manifest_path)
        
        print(f"  Validation result: is_valid={result.is_valid}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        
        if result.errors:
            for error in result.errors:
                print(f"    Error: {error}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"    Warning: {warning}")
        
        return result.is_valid or len(result.errors) == 0

def test_semantic_fix():
    """Test that semantic understanding works."""
    print("Testing semantic fix...")
    
    try:
        dsu = DeepSemanticUnderstanding()
        
        # Test process_multimodal_input
        inputs = {"text": "test text", "metadata": {"test": True}}
        result = dsu.process_multimodal_input(inputs)
        
        print(f"  Result keys: {list(result.keys())}")
        has_understanding_score = "understanding_score" in result
        print(f"  Has understanding_score: {has_understanding_score}")
        
        return has_understanding_score
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("Running quick fix verification tests...\n")
    
    validation_ok = test_validation_fix()
    print(f"Validation fix: {'✓ PASS' if validation_ok else '✗ FAIL'}\n")
    
    semantic_ok = test_semantic_fix()
    print(f"Semantic fix: {'✓ PASS' if semantic_ok else '✗ FAIL'}\n")
    
    if validation_ok and semantic_ok:
        print("🎉 All fixes appear to be working!")
    else:
        print("❌ Some fixes still need work")
#!/usr/bin/env python3
"""
Test script to verify MAIF SDK imports work correctly.
"""

import sys
import traceback

def test_imports():
    """Test all SDK imports."""
    
    print("Testing MAIF SDK imports...")
    print("-" * 50)
    
    # Test 1: Import the main SDK module
    print("\n1. Testing main SDK import...")
    try:
        import maif_sdk
        print("✓ Successfully imported maif_sdk")
        print(f"  - Version: {maif_sdk.__version__}")
        print(f"  - AWS Available: {maif_sdk.AWS_AVAILABLE}")
        print(f"  - FUSE Available: {maif_sdk.FUSE_AVAILABLE}")
        print(f"  - gRPC Available: {maif_sdk.GRPC_AVAILABLE}")
    except ImportError as e:
        print(f"✗ Failed to import maif_sdk: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Import core SDK components
    print("\n2. Testing core component imports...")
    try:
        from maif_sdk import MAIFClient, Artifact, ContentItem
        print("✓ Successfully imported MAIFClient, Artifact, ContentItem")
    except ImportError as e:
        print(f"✗ Failed to import core components: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Import types
    print("\n3. Testing type imports...")
    try:
        from maif_sdk import (
            ContentType, SecurityLevel, CompressionLevel,
            ContentMetadata, SecurityOptions, ProcessingOptions
        )
        print("✓ Successfully imported all types")
        print(f"  - ContentType values: {[ct.value for ct in ContentType]}")
        print(f"  - SecurityLevel values: {[sl.value for sl in SecurityLevel]}")
    except ImportError as e:
        print(f"✗ Failed to import types: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Import convenience functions
    print("\n4. Testing convenience function imports...")
    try:
        from maif_sdk import quick_write, quick_read
        print("✓ Successfully imported quick_write, quick_read")
    except ImportError as e:
        print(f"✗ Failed to import convenience functions: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Import factory functions
    print("\n5. Testing factory function imports...")
    try:
        from maif_sdk import create_client, create_artifact
        print("✓ Successfully imported create_client, create_artifact")
    except ImportError as e:
        print(f"✗ Failed to import factory functions: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Test AWS imports (if available)
    print("\n6. Testing AWS component imports...")
    if maif_sdk.AWS_AVAILABLE:
        try:
            from maif_sdk import AWSConfig, create_aws_backends, create_aws_client
            print("✓ Successfully imported AWS components")
            # Test AWSConfig creation
            config = AWSConfig()
            print(f"  - Default AWS region: {config.region}")
        except ImportError as e:
            print(f"✗ Failed to import AWS components: {e}")
            traceback.print_exc()
            return False
    else:
        print("⚠ AWS components not available (boto3 not installed)")
    
    # Test 7: Try to create a client (without MAIF core)
    print("\n7. Testing client creation...")
    try:
        # This should fail gracefully if MAIF core is not installed
        client = MAIFClient(agent_id="test_agent", use_aws=False)
        print("✓ Successfully created MAIFClient")
    except ImportError as e:
        print(f"⚠ Expected import error (MAIF core not installed): {e}")
    except Exception as e:
        print(f"✗ Unexpected error creating client: {e}")
        traceback.print_exc()
        return False
    
    # Test 8: Test artifact creation
    print("\n8. Testing artifact creation...")
    try:
        artifact = Artifact(name="test_artifact")
        print("✓ Successfully created Artifact")
        print(f"  - Name: {artifact.name}")
        print(f"  - Security Level: {artifact.security_level.value}")
        print(f"  - Compression Level: {artifact.compression_level.value}")
    except Exception as e:
        print(f"✗ Failed to create artifact: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 50)
    print("✓ All import tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
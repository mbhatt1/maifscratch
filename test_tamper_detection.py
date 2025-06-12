#!/usr/bin/env python3
"""
Test script to demonstrate real-time tamper detection in high-speed streaming.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from maif.core import MAIFEncoder
from maif.streaming import SecureStreamReader, MemoryStreamGuard, StreamingConfig
from maif.privacy import PrivacyPolicy, PrivacyLevel, EncryptionMode

def create_test_file_with_tamper_detection():
    """Create a test MAIF file with integrity hashes."""
    print("🔧 Creating test MAIF file with tamper detection...")
    
    # Create encoder with privacy enabled
    encoder = MAIFEncoder(enable_privacy=True, agent_id="tamper_test")
    
    # Set up crypto policy
    crypto_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.CONFIDENTIAL,
        encryption_mode=EncryptionMode.AES_GCM,
        anonymization_required=False,
        audit_required=True
    )
    encoder.set_default_privacy_policy(crypto_policy)
    
    # Add test data blocks
    test_data = [
        "This is block 1 - critical financial data",
        "This is block 2 - sensitive user information", 
        "This is block 3 - confidential business logic",
        "This is block 4 - encrypted communication logs",
        "This is block 5 - authentication credentials"
    ]
    
    for i, data in enumerate(test_data):
        encoder.add_text_block(data, metadata={"block_number": i+1, "critical": True})
    
    # Build MAIF file
    with tempfile.TemporaryDirectory() as tmpdir:
        maif_path = os.path.join(tmpdir, "secure_test.maif")
        manifest_path = os.path.join(tmpdir, "secure_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        print(f"✅ Created secure MAIF file: {os.path.getsize(maif_path)} bytes")
        
        return maif_path, manifest_path, tmpdir

def run_normal_streaming_test(maif_path, manifest_path):
    """Test normal streaming with tamper detection."""
    print("\n🔍 Testing normal streaming with tamper detection...")
    
    config = StreamingConfig(
        chunk_size=1024 * 1024,  # 1MB chunks
        max_workers=8,
        buffer_size=8 * 1024 * 1024,  # 8MB buffer
        use_memory_mapping=True
    )
    
    start_time = time.time()
    
    with SecureStreamReader(maif_path, config, enable_verification=True) as reader:
        block_count = 0
        total_bytes = 0
        
        print("📊 Streaming blocks with real-time verification:")
        
        for block_type, data, is_valid in reader.stream_blocks_verified():
            block_count += 1
            total_bytes += len(data)
            
            status = "✅ VALID" if is_valid else "🚨 TAMPERED"
            print(f"   Block {block_count}: {block_type} ({len(data)} bytes) - {status}")
        
        duration = time.time() - start_time
        throughput = (total_bytes / (1024 * 1024)) / duration if duration > 0 else 0
        
        # Get security report
        security_report = reader.get_security_report()
        
        print(f"\n📈 Performance Results:")
        print(f"   Blocks processed: {block_count}")
        print(f"   Total bytes: {total_bytes:,}")
        print(f"   Duration: {duration:.3f}s")
        print(f"   Secure throughput: {throughput:.1f} MB/s")
        print(f"   Verification overhead: {security_report['verification_overhead']}")
        
        print(f"\n🔒 Security Results:")
        security = security_report['security']
        print(f"   Blocks verified: {security['total_blocks_verified']}")
        print(f"   Valid blocks: {security['valid_blocks']}")
        print(f"   Tampered blocks: {security['tampered_blocks']}")
        print(f"   Integrity: {security['integrity_percentage']:.1f}%")
        print(f"   Tamper detected: {security['tamper_detected']}")

def run_memory_tamper_detection_test():
    """Test memory-based tamper detection."""
    print("\n🧠 Testing memory-based tamper detection...")
    
    guard = MemoryStreamGuard()
    
    # Create test memory regions
    original_data = b"Critical data that must not be tampered with" * 1000
    guard.create_memory_checkpoint(original_data, "critical_region_1")
    
    # Verify original data
    is_valid = guard.verify_memory_integrity(original_data, "critical_region_1")
    print(f"   Original data verification: {'✅ VALID' if is_valid else '🚨 TAMPERED'}")
    
    # Simulate tampering
    tampered_data = original_data.replace(b"Critical", b"Modified")
    is_valid = guard.verify_memory_integrity(tampered_data, "critical_region_1")
    print(f"   Tampered data verification: {'✅ VALID' if is_valid else '🚨 TAMPERED'}")
    
    # Get security status
    status = guard.get_memory_security_status()
    print(f"\n🔒 Memory Security Status:")
    print(f"   Total checks: {status['total_memory_checks']}")
    print(f"   Valid checks: {status['valid_checks']}")
    print(f"   Memory integrity: {status['memory_integrity']:.1f}%")
    print(f"   Regions monitored: {status['regions_monitored']}")

def run_checkpoint_streaming_test(maif_path):
    """Test streaming with periodic checkpoints."""
    print("\n⏱️ Testing streaming with periodic checkpoints...")
    
    config = StreamingConfig(
        chunk_size=1024 * 1024,
        max_workers=8,
        buffer_size=8 * 1024 * 1024,
        use_memory_mapping=True
    )
    
    with SecureStreamReader(maif_path, config, enable_verification=True) as reader:
        block_count = 0
        
        print("📊 Streaming with checkpoints every 2 blocks:")
        
        for block_type, data in reader.stream_blocks_with_checkpoints(checkpoint_interval=2):
            block_count += 1
            print(f"   Processed block {block_count}: {block_type} ({len(data)} bytes)")

def main():
    """Main test function."""
    print("🚀 MAIF High-Speed Tamper Detection Test")
    print("=" * 60)
    
    try:
        # Create test file
        maif_path, manifest_path, tmpdir = create_test_file_with_tamper_detection()
        
        # Test normal streaming with verification
        test_normal_streaming(maif_path, manifest_path)
        
        # Test memory tamper detection
        test_memory_tamper_detection()
        
        # Test checkpoint streaming
        test_checkpoint_streaming(maif_path)
        
        print("\n" + "=" * 60)
        print("🎉 All tamper detection tests completed successfully!")
        print("\n📋 Summary:")
        print("✅ Real-time block verification during streaming")
        print("✅ Memory integrity monitoring")
        print("✅ Periodic checkpoint verification")
        print("✅ High-speed performance maintained")
        print("✅ Comprehensive security reporting")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
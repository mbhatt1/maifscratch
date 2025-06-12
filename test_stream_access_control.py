#!/usr/bin/env python3
"""
Stream-Level Access Control Demo
===============================

Demonstrates granular access control enforcement during MAIF streaming operations.
Shows per-block permissions, time-based access, rate limiting, and content filtering.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add the parent directory to the path so we can import maif modules
sys.path.insert(0, str(Path(__file__).parent))

from maif.stream_access_control import (
    StreamAccessController, StreamAccessRule, AccessLevel, AccessDecision,
    SecureStreamReader, StreamAccessPolicies
)
from maif.streaming import MAIFStreamWriter, StreamingConfig
from maif.core import MAIFEncoder


def create_test_maif_file(filepath: str) -> None:
    """Create a test MAIF file with various block types."""
    print(f"📁 Creating test MAIF file: {filepath}")
    
    # Create encoder
    encoder = MAIFEncoder()
    
    # Add different types of blocks
    encoder.add_text_block("public_data", "This is public information that everyone can read.")
    encoder.add_text_block("sensitive_data", "This contains sensitive information - restricted access.")
    encoder.add_text_block("admin_data", "Administrative data - admin access only.")
    
    # Add some binary blocks
    encoder.add_binary_block("image_data", b"FAKE_IMAGE_DATA" * 1000, {"type": "image", "format": "png"})
    encoder.add_binary_block("video_data", b"FAKE_VIDEO_DATA" * 5000, {"type": "video", "format": "mp4"})
    
    # Build the MAIF file
    manifest_path = filepath.replace('.maif', '_manifest.json')
    encoder.build_maif(filepath, manifest_path)
    
    print(f"✅ Created MAIF file with {len(encoder.blocks)} blocks")


def demo_basic_access_control():
    """Demonstrate basic access control functionality."""
    print("\n" + "="*60)
    print("🔐 DEMO 1: Basic Access Control")
    print("="*60)
    
    # Create access controller
    controller = StreamAccessController()
    
    # Create test file
    test_file = "test_access_control.maif"
    create_test_maif_file(test_file)
    
    try:
        # Add basic read rule for user1
        read_rule = StreamAccessRule(
            rule_id="user1_read",
            user_id="user1",
            resource_pattern=".*test_access_control.*",
            access_level=AccessLevel.READ,
            description="Basic read access for user1"
        )
        controller.add_rule(read_rule)
        
        # Test successful access
        print("\n📖 Testing authorized access...")
        with SecureStreamReader(test_file, "user1", controller) as reader:
            block_count = 0
            for block_type, block_data in reader.stream_blocks_secure():
                block_count += 1
                print(f"   ✅ Read block {block_count}: {block_type} ({len(block_data)} bytes)")
            
            stats = reader.get_session_stats()
            print(f"   📊 Session stats: {stats['blocks_read']} blocks, {stats['bytes_read']} bytes")
        
        # Test unauthorized access
        print("\n🚫 Testing unauthorized access...")
        try:
            with SecureStreamReader(test_file, "user2", controller) as reader:
                for block_type, block_data in reader.stream_blocks_secure():
                    print(f"   ❌ This should not happen: {block_type}")
        except PermissionError as e:
            print(f"   ✅ Access correctly denied: {e}")
    
    finally:
        # Cleanup
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def demo_time_based_access():
    """Demonstrate time-based access control."""
    print("\n" + "="*60)
    print("⏰ DEMO 2: Time-Based Access Control")
    print("="*60)
    
    controller = StreamAccessController()
    test_file = "test_time_access.maif"
    create_test_maif_file(test_file)
    
    try:
        # Create a rule that expires in 3 seconds
        time_rule = StreamAccessPolicies.create_time_limited_rule(
            user_id="temp_user",
            resource_pattern=".*test_time_access.*",
            duration_seconds=3,
            access_level=AccessLevel.READ
        )
        controller.add_rule(time_rule)
        
        print(f"🕐 Created time-limited rule (expires in 3 seconds)")
        print(f"   Rule ID: {time_rule.rule_id}")
        print(f"   Valid until: {time.ctime(time_rule.valid_until)}")
        
        # Test access immediately (should work)
        print("\n📖 Testing immediate access...")
        with SecureStreamReader(test_file, "temp_user", controller) as reader:
            block_count = 0
            for block_type, block_data in reader.stream_blocks_secure():
                block_count += 1
                if block_count <= 2:  # Only read first 2 blocks
                    print(f"   ✅ Read block {block_count}: {block_type}")
                else:
                    break
        
        # Wait for expiration
        print("\n⏳ Waiting for access to expire...")
        time.sleep(4)
        
        # Test access after expiration (should fail)
        print("\n🚫 Testing expired access...")
        try:
            with SecureStreamReader(test_file, "temp_user", controller) as reader:
                for block_type, block_data in reader.stream_blocks_secure():
                    print(f"   ❌ This should not happen: {block_type}")
        except PermissionError as e:
            print(f"   ✅ Access correctly expired: {e}")
    
    finally:
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def demo_rate_limiting():
    """Demonstrate rate limiting."""
    print("\n" + "="*60)
    print("🚦 DEMO 3: Rate Limiting")
    print("="*60)
    
    controller = StreamAccessController()
    test_file = "test_rate_limit.maif"
    create_test_maif_file(test_file)
    
    try:
        # Create a rate-limited rule (1 MB/s max)
        rate_rule = StreamAccessPolicies.create_rate_limited_rule(
            user_id="limited_user",
            resource_pattern=".*test_rate_limit.*",
            max_mbps=1,  # 1 MB/s limit
            access_level=AccessLevel.READ
        )
        controller.add_rule(rate_rule)
        
        print(f"🚦 Created rate-limited rule (1 MB/s max)")
        print(f"   Rule ID: {rate_rule.rule_id}")
        
        # Test normal access (should work initially)
        print("\n📖 Testing rate-limited access...")
        with SecureStreamReader(test_file, "limited_user", controller) as reader:
            start_time = time.time()
            block_count = 0
            total_bytes = 0
            
            try:
                for block_type, block_data in reader.stream_blocks_secure():
                    block_count += 1
                    total_bytes += len(block_data)
                    elapsed = time.time() - start_time
                    rate_mbps = (total_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                    
                    print(f"   📊 Block {block_count}: {len(block_data)} bytes, Rate: {rate_mbps:.2f} MB/s")
                    
                    # Small delay to avoid hitting rate limit immediately
                    time.sleep(0.1)
                    
            except PermissionError as e:
                print(f"   🚦 Rate limit hit: {e}")
                elapsed = time.time() - start_time
                rate_mbps = (total_bytes / (1024 * 1024)) / max(elapsed, 0.001)
                print(f"   📊 Final rate: {rate_mbps:.2f} MB/s after {elapsed:.2f}s")
    
    finally:
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def demo_content_filtering():
    """Demonstrate content-based access control."""
    print("\n" + "="*60)
    print("🎯 DEMO 4: Content-Based Access Control")
    print("="*60)
    
    controller = StreamAccessController()
    test_file = "test_content_filter.maif"
    create_test_maif_file(test_file)
    
    try:
        # Create a content-filtered rule (only allow text blocks)
        content_rule = StreamAccessPolicies.create_content_filtered_rule(
            user_id="text_only_user",
            resource_pattern=".*test_content_filter.*",
            allowed_types={"text"},
            access_level=AccessLevel.READ
        )
        controller.add_rule(content_rule)
        
        print(f"🎯 Created content-filtered rule (text blocks only)")
        print(f"   Rule ID: {content_rule.rule_id}")
        print(f"   Allowed types: {content_rule.allowed_block_types}")
        
        # Test filtered access
        print("\n📖 Testing content-filtered access...")
        with SecureStreamReader(test_file, "text_only_user", controller) as reader:
            block_count = 0
            allowed_count = 0
            
            try:
                for block_type, block_data in reader.stream_blocks_secure():
                    block_count += 1
                    allowed_count += 1
                    print(f"   ✅ Allowed block {block_count}: {block_type} ({len(block_data)} bytes)")
                    
            except PermissionError as e:
                print(f"   🚫 Content blocked: {e}")
                print(f"   📊 Processed {allowed_count} allowed blocks before hitting restriction")
    
    finally:
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def demo_custom_validation():
    """Demonstrate custom validation functions."""
    print("\n" + "="*60)
    print("🔧 DEMO 5: Custom Validation")
    print("="*60)
    
    def sensitive_data_validator(user_id: str, block_type: str, block_data: bytes) -> bool:
        """Custom validator that blocks sensitive content."""
        data_str = block_data.decode('utf-8', errors='ignore').lower()
        sensitive_keywords = ['sensitive', 'admin', 'password', 'secret']
        
        has_sensitive = any(keyword in data_str for keyword in sensitive_keywords)
        if has_sensitive:
            print(f"   🔍 Custom validator: Blocked sensitive content for {user_id}")
            return False
        return True
    
    controller = StreamAccessController()
    test_file = "test_custom_validation.maif"
    create_test_maif_file(test_file)
    
    try:
        # Create rule with custom validator
        custom_rule = StreamAccessRule(
            rule_id="custom_validation_rule",
            user_id="validated_user",
            resource_pattern=".*test_custom_validation.*",
            access_level=AccessLevel.READ,
            custom_validator=sensitive_data_validator,
            description="Custom validation to block sensitive content"
        )
        controller.add_rule(custom_rule)
        
        print(f"🔧 Created custom validation rule")
        print(f"   Rule ID: {custom_rule.rule_id}")
        print(f"   Validator: Blocks content with sensitive keywords")
        
        # Test custom validation
        print("\n📖 Testing custom validation...")
        with SecureStreamReader(test_file, "validated_user", controller) as reader:
            block_count = 0
            allowed_count = 0
            
            try:
                for block_type, block_data in reader.stream_blocks_secure():
                    block_count += 1
                    allowed_count += 1
                    data_preview = block_data.decode('utf-8', errors='ignore')[:50]
                    print(f"   ✅ Allowed block {block_count}: {block_type}")
                    print(f"      Preview: {data_preview}...")
                    
            except PermissionError as e:
                print(f"   🚫 Custom validation failed: {e}")
                print(f"   📊 Processed {allowed_count} blocks before validation failure")
    
    finally:
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def demo_audit_logging():
    """Demonstrate audit logging capabilities."""
    print("\n" + "="*60)
    print("📋 DEMO 6: Audit Logging")
    print("="*60)
    
    controller = StreamAccessController()
    test_file = "test_audit.maif"
    create_test_maif_file(test_file)
    
    try:
        # Add multiple rules
        rules = [
            StreamAccessRule("audit_user1", "user1", ".*test_audit.*", AccessLevel.READ),
            StreamAccessRule("audit_user2", "user2", ".*test_audit.*", AccessLevel.WRITE),
        ]
        
        for rule in rules:
            controller.add_rule(rule)
        
        # Perform various operations
        print("🔄 Performing various operations to generate audit log...")
        
        # Successful access
        with SecureStreamReader(test_file, "user1", controller) as reader:
            for i, (block_type, block_data) in enumerate(reader.stream_blocks_secure()):
                if i >= 2:  # Only read first 2 blocks
                    break
        
        # Failed access
        try:
            with SecureStreamReader(test_file, "unauthorized_user", controller) as reader:
                for block_type, block_data in reader.stream_blocks_secure():
                    pass
        except PermissionError:
            pass  # Expected
        
        # Show audit log
        print("\n📋 Audit Log:")
        audit_entries = controller.get_audit_log(limit=10)
        for i, entry in enumerate(audit_entries, 1):
            timestamp = time.ctime(entry['timestamp'])
            event_type = entry['event_type']
            details = entry['details']
            print(f"   {i}. [{timestamp}] {event_type}")
            for key, value in details.items():
                print(f"      {key}: {value}")
            print()
    
    finally:
        for f in [test_file, test_file.replace('.maif', '_manifest.json')]:
            if os.path.exists(f):
                os.remove(f)


def main():
    """Run all stream access control demos."""
    print("🔐 MAIF Stream-Level Access Control Demo")
    print("=" * 60)
    print("This demo shows how access control rules can be enforced")
    print("at the stream level during MAIF operations, providing:")
    print("• Per-block access control")
    print("• Time-based access expiration")
    print("• Rate limiting")
    print("• Content-based filtering")
    print("• Custom validation")
    print("• Comprehensive audit logging")
    
    try:
        demo_basic_access_control()
        demo_time_based_access()
        demo_rate_limiting()
        demo_content_filtering()
        demo_custom_validation()
        demo_audit_logging()
        
        print("\n" + "="*60)
        print("🎉 All stream access control demos completed successfully!")
        print("="*60)
        print("\n💡 Key Benefits of Stream-Level Access Control:")
        print("   • Granular per-block permissions")
        print("   • Dynamic access rules during streaming")
        print("   • Real-time rate limiting and throttling")
        print("   • Content-aware access decisions")
        print("   • Comprehensive audit trail")
        print("   • Zero performance impact on allowed operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
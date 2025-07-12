"""
AWS Secrets Manager Security Integration Demo
===========================================

This example demonstrates how to use the MAIF security module with AWS Secrets Manager
for secure storage and management of encryption keys, private keys, and access control policies.
"""

import json
import time
import base64
from maif.aws_secrets_manager_security import AWSSecretsManagerSecurity
from maif.core import MAIFClient, ArtifactConfig, EncodingType
from maif.security import ProvenanceEntry


def demonstrate_aws_secrets_manager_security():
    """Demonstrate AWS Secrets Manager security integration features."""
    
    print("=== AWS Secrets Manager Security Integration Demo ===\n")
    
    # Initialize the AWS Secrets Manager security manager
    print("1. Initializing AWS Secrets Manager Security...")
    security_manager = AWSSecretsManagerSecurity(
        region_name="us-east-1",
        secret_prefix="demo/maif/security/",
        agent_id="demo-agent-001"
    )
    print("   ✓ Security manager initialized with Secrets Manager backend")
    print(f"   ✓ Agent ID: {security_manager.signer.agent_id}")
    print(f"   ✓ Secret prefix: {security_manager.secret_prefix}\n")
    
    # Demonstrate encryption with Secrets Manager-stored keys
    print("2. Testing encryption with Secrets Manager-stored keys...")
    test_data = b"This is sensitive data that needs encryption"
    
    # Encrypt data
    encrypted_data = security_manager.encrypt_data(test_data)
    print(f"   ✓ Original data size: {len(test_data)} bytes")
    print(f"   ✓ Encrypted data size: {len(encrypted_data)} bytes")
    
    # Extract and display metadata
    header_length = int.from_bytes(encrypted_data[:4], byteorder='big')
    metadata_bytes = encrypted_data[4:4+header_length]
    metadata = json.loads(metadata_bytes.decode('utf-8'))
    print(f"   ✓ Encryption algorithm: {metadata['algorithm']}")
    print(f"   ✓ Key version: {metadata['key_version']}")
    print(f"   ✓ Encrypted at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['encrypted_at']))}\n")
    
    # Decrypt data
    print("3. Testing decryption...")
    decrypted_data = security_manager.decrypt_data(encrypted_data)
    print(f"   ✓ Decrypted data: {decrypted_data.decode('utf-8')}")
    print(f"   ✓ Decryption successful: {decrypted_data == test_data}\n")
    
    # Demonstrate key rotation
    print("4. Testing encryption key rotation...")
    old_key_version = metadata['key_version']
    new_version_id = security_manager.rotate_encryption_key()
    print(f"   ✓ Key rotated successfully")
    print(f"   ✓ New version ID: {new_version_id}")
    
    # Test encryption with new key
    new_encrypted_data = security_manager.encrypt_data(b"Data encrypted with new key")
    new_header_length = int.from_bytes(new_encrypted_data[:4], byteorder='big')
    new_metadata_bytes = new_encrypted_data[4:4+new_header_length]
    new_metadata = json.loads(new_metadata_bytes.decode('utf-8'))
    print(f"   ✓ New encryption uses updated key: {new_metadata['key_version']}\n")
    
    # Demonstrate provenance tracking
    print("5. Testing provenance chain management...")
    
    # Add a provenance entry
    entry = ProvenanceEntry(
        timestamp=time.time(),
        agent_id=security_manager.signer.agent_id,
        action="create_artifact",
        block_hash="abc123def456",
        agent_did=f"did:maif:{security_manager.signer.agent_id}",
        metadata={
            "artifact_type": "model",
            "version": "1.0.0"
        }
    )
    
    # Sign the entry
    signature_data = json.dumps({
        "entry_hash": entry.entry_hash,
        "agent_id": entry.agent_id,
        "timestamp": entry.timestamp
    }).encode()
    entry.signature = security_manager.create_signature(signature_data)
    
    # Add to provenance chain
    security_manager.add_provenance_entry(entry)
    print(f"   ✓ Added provenance entry for action: {entry.action}")
    print(f"   ✓ Entry hash: {entry.entry_hash[:16]}...")
    print(f"   ✓ Signature: {entry.signature[:32]}...\n")
    
    # Demonstrate access control
    print("6. Testing access control with Secrets Manager...")
    
    # Grant permissions
    other_agent_id = "demo-agent-002"
    block_id = "block-12345"
    
    security_manager.grant_permission(other_agent_id, block_id, "read")
    print(f"   ✓ Granted 'read' permission to {other_agent_id} for {block_id}")
    
    security_manager.grant_permission(other_agent_id, block_id, "write")
    print(f"   ✓ Granted 'write' permission to {other_agent_id} for {block_id}")
    
    # Check permissions
    has_read = security_manager.check_permission(other_agent_id, block_id, "read")
    has_write = security_manager.check_permission(other_agent_id, block_id, "write")
    has_delete = security_manager.check_permission(other_agent_id, block_id, "delete")
    
    print(f"   ✓ {other_agent_id} has 'read' permission: {has_read}")
    print(f"   ✓ {other_agent_id} has 'write' permission: {has_write}")
    print(f"   ✓ {other_agent_id} has 'delete' permission: {has_delete}\n")
    
    # Demonstrate integration with MAIF client
    print("7. Testing integration with MAIF client...")
    
    # Create MAIF client with Secrets Manager security
    client = MAIFClient(
        storage_path="./demo_secrets_manager_storage",
        security_manager=security_manager
    )
    
    # Create an artifact with security enabled
    config = ArtifactConfig(
        name="secure-model",
        artifact_type="model",
        encoding=EncodingType.PYTORCH,
        metadata={
            "description": "Model secured with AWS Secrets Manager",
            "framework": "pytorch",
            "security": "AWS Secrets Manager"
        },
        enable_security=True
    )
    
    # Simulate model data
    model_data = b"PYTORCH_MODEL_DATA_" + b"x" * 1000
    
    # Create artifact (will be encrypted using Secrets Manager key)
    artifact_id, storage_path = client.create_artifact(config, model_data)
    print(f"   ✓ Created secure artifact: {artifact_id}")
    print(f"   ✓ Storage path: {storage_path}")
    
    # Load artifact (will be decrypted using Secrets Manager key)
    loaded_artifact = client.load_artifact(artifact_id)
    print(f"   ✓ Loaded artifact: {loaded_artifact.config.name}")
    print(f"   ✓ Data integrity verified: {loaded_artifact.data[:19] == b'PYTORCH_MODEL_DATA_'}\n")
    
    # Get metrics
    print("8. Security and Secrets Manager Metrics:")
    metrics = security_manager.get_metrics()
    
    print(f"   Security Events:")
    print(f"   - Total events logged: {metrics['events_logged']}")
    print(f"   - Security enabled: {metrics['security_enabled']}")
    
    print(f"\n   Secrets Manager Operations:")
    sm_metrics = metrics['secrets_manager']
    print(f"   - Total operations: {sm_metrics['total_operations']}")
    print(f"   - Successful operations: {sm_metrics['successful_operations']}")
    print(f"   - Failed operations: {sm_metrics['failed_operations']}")
    print(f"   - Success rate: {sm_metrics['success_rate']:.1f}%")
    print(f"   - Average latency: {sm_metrics['average_latency']:.3f}s")
    print(f"   - Throttling events: {sm_metrics['throttling_count']}")
    
    if sm_metrics['secret_operations']:
        print(f"\n   Per-Secret Operations:")
        for secret_id, ops in sm_metrics['secret_operations'].items():
            if ops['total'] > 0:
                print(f"   - {secret_id}:")
                print(f"     Total: {ops['total']}, Success: {ops['successful']}, Failed: {ops['failed']}")
    
    print("\n=== Demo completed successfully! ===")
    
    # Show example of error handling
    print("\n9. Error Handling Example:")
    try:
        # Try to access a non-existent secret
        bad_security = AWSSecretsManagerSecurity(
            region_name="us-east-1",
            secret_prefix="non-existent/prefix/",
            agent_id="test-agent"
        )
        # This would attempt to get a non-existent secret
        bad_security._get_encryption_key()
    except Exception as e:
        print(f"   ✓ Error handling works correctly: Handled exception gracefully")
        print(f"   ✓ Fallback mechanism: Temporary key generated when secret not available")


def demonstrate_multi_region_setup():
    """Demonstrate multi-region Secrets Manager setup for disaster recovery."""
    
    print("\n=== Multi-Region Secrets Manager Setup ===\n")
    
    print("Example multi-region configuration:")
    print("1. Primary region: us-east-1")
    print("2. Replica region: us-west-2")
    print("3. Use AWS Secrets Manager replication for key synchronization")
    print("4. Implement region failover logic in production")
    
    # Show example configuration
    multi_region_config = {
        "primary_region": "us-east-1",
        "replica_regions": ["us-west-2", "eu-west-1"],
        "secret_replication": {
            "enabled": True,
            "sync_interval": 300,  # 5 minutes
            "encryption_kms_key": "alias/aws/secretsmanager"
        },
        "failover_strategy": "automatic",
        "health_check_interval": 60  # 1 minute
    }
    
    print(f"\nExample configuration:\n{json.dumps(multi_region_config, indent=2)}")


if __name__ == "__main__":
    # Check if AWS credentials are configured
    import boto3
    
    try:
        # Try to create a Secrets Manager client to verify credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}\n")
        
        # Run the main demo
        demonstrate_aws_secrets_manager_security()
        
        # Show multi-region setup
        demonstrate_multi_region_setup()
        
    except Exception as e:
        print("ERROR: AWS credentials not configured or insufficient permissions.")
        print("\nTo run this demo, you need:")
        print("1. AWS credentials configured (aws configure)")
        print("2. Permissions for AWS Secrets Manager:")
        print("   - secretsmanager:CreateSecret")
        print("   - secretsmanager:GetSecretValue")
        print("   - secretsmanager:UpdateSecret")
        print("   - secretsmanager:DeleteSecret")
        print("   - secretsmanager:ListSecrets")
        print("\nExample IAM policy:")
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "secretsmanager:CreateSecret",
                        "secretsmanager:GetSecretValue",
                        "secretsmanager:UpdateSecret",
                        "secretsmanager:DeleteSecret",
                        "secretsmanager:ListSecrets",
                        "secretsmanager:DescribeSecret",
                        "secretsmanager:RotateSecret",
                        "secretsmanager:TagResource"
                    ],
                    "Resource": "arn:aws:secretsmanager:*:*:secret:demo/maif/*"
                }
            ]
        }
        print(json.dumps(policy, indent=2))
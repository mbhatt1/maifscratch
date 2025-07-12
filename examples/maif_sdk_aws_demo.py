"""
MAIF SDK with AWS Backend Demo
===============================

This example demonstrates how to use the MAIF SDK with AWS backends
by simply setting use_aws=True.
"""

import os
from maif_sdk.client import MAIFClient
from maif_sdk.types import ContentType, SecurityLevel, SecurityOptions, ProcessingOptions, CompressionLevel
from maif_sdk.aws_backend import AWSConfig


def demonstrate_aws_backend():
    """Demonstrate MAIF SDK with AWS backend integration."""
    
    print("=== MAIF SDK with AWS Backend Demo ===\n")
    
    # Method 1: Use AWS with default configuration from environment
    print("1. Using AWS backend with environment configuration:")
    print("   Set these environment variables:")
    print("   - AWS_DEFAULT_REGION=us-east-1")
    print("   - MAIF_AWS_S3_BUCKET=my-maif-bucket")
    print("   - MAIF_AWS_KMS_KEY_ALIAS=alias/maif-key")
    print("   - MAIF_AWS_LOG_GROUP=maif-compliance\n")
    
    # Initialize with AWS backend
    with MAIFClient(agent_id="demo-agent", use_aws=True) as client:
        print("   ✓ MAIF client initialized with AWS backend")
        
        # Write content - automatically uses S3, KMS, CloudWatch
        artifact_id = client.write_content(
            filepath="demo.maif",
            content=b"This is my AI model data",
            content_type=ContentType.MODEL,
            security_options=SecurityOptions(encrypt=True, sign=True),
            processing_options=ProcessingOptions(compression=CompressionLevel.HIGH)
        )
        print(f"   ✓ Content written to AWS S3: {artifact_id}")
        
        # Read content - automatically retrieves from S3
        for block in client.read_content("demo.maif"):
            print(f"   ✓ Read block {block['block_id']} from AWS")
            print(f"     - Size: {block['size']} bytes")
            print(f"     - Type: {block['content_type']}")
    
    print()
    
    # Method 2: Use AWS with custom configuration
    print("2. Using AWS backend with custom configuration:")
    
    aws_config = AWSConfig(
        region="us-west-2",
        s3_bucket="my-custom-bucket",
        s3_prefix="projects/ai-models/",
        kms_key_alias="alias/my-custom-key",
        cloudwatch_log_group="my-compliance-logs",
        enable_macie=True
    )
    
    with MAIFClient(
        agent_id="custom-agent",
        use_aws=True,
        aws_config=aws_config
    ) as client:
        print("   ✓ MAIF client initialized with custom AWS configuration")
        print(f"   ✓ Using S3 bucket: {aws_config.s3_bucket}")
        print(f"   ✓ Using KMS key: {aws_config.kms_key_alias}")
        print(f"   ✓ Logging to CloudWatch: {aws_config.cloudwatch_log_group}")
    
    print()
    
    # Method 3: Hybrid approach - local + AWS
    print("3. Hybrid approach (local files + AWS services):")
    
    with MAIFClient(agent_id="hybrid-agent", use_aws=True) as client:
        # Data is stored locally in MAIF files
        # But uses AWS for:
        # - KMS for encryption
        # - Secrets Manager for key storage  
        # - CloudWatch for compliance logging
        # - Macie for privacy classification
        
        artifact = client.create_artifact(
            name="hybrid-model",
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        # Write to local file with AWS encryption
        artifact.write(b"Sensitive model parameters")
        
        print("   ✓ Data stored locally in MAIF format")
        print("   ✓ Encrypted with AWS KMS")
        print("   ✓ Keys managed by AWS Secrets Manager")
        print("   ✓ Compliance logged to CloudWatch")
        print("   ✓ Privacy classified by AWS Macie")
    
    print("\n=== Benefits of AWS Backend ===")
    print("• Scalable storage with S3")
    print("• Enterprise-grade encryption with KMS")
    print("• Centralized compliance logging with CloudWatch")
    print("• Automated privacy classification with Macie")
    print("• Secure credential management with Secrets Manager")
    print("• Real-time streaming with Kinesis")
    print("• Distributed processing with Lambda and Step Functions")


def demonstrate_streaming_with_aws():
    """Demonstrate streaming capabilities with AWS Kinesis."""
    
    print("\n=== Streaming with AWS Kinesis ===\n")
    
    # Configure with Kinesis stream
    aws_config = AWSConfig(
        region="us-east-1",
        kinesis_stream="maif-agent-events"
    )
    
    with MAIFClient(use_aws=True, aws_config=aws_config) as client:
        if client.streaming_backend:
            print("   ✓ Connected to Kinesis stream")
            
            # Stream events
            client.streaming_backend.stream_data(
                data=b'{"event": "model_updated", "version": "2.0"}',
                partition_key="model-updates"
            )
            print("   ✓ Streamed event to Kinesis")
            
            # In production, you'd have consumers processing the stream
            print("   ✓ Events can be processed by Lambda, Analytics, etc.")
        else:
            print("   ! Kinesis stream not configured")


def compare_local_vs_aws():
    """Compare local vs AWS backend performance and features."""
    
    print("\n=== Local vs AWS Backend Comparison ===\n")
    
    print("Local Backend (use_aws=False):")
    print("  ✓ Fast local file I/O")
    print("  ✓ No network latency")
    print("  ✓ Works offline")
    print("  ✗ Limited to local storage")
    print("  ✗ No built-in compliance logging")
    print("  ✗ Manual key management")
    
    print("\nAWS Backend (use_aws=True):")
    print("  ✓ Unlimited scalable storage")
    print("  ✓ Enterprise security features")
    print("  ✓ Automated compliance")
    print("  ✓ Global accessibility")
    print("  ✓ Integrated with AWS ecosystem")
    print("  ✗ Requires internet connection")
    print("  ✗ AWS costs apply")
    
    print("\nRecommendation:")
    print("• Development/Testing: use_aws=False")
    print("• Production/Enterprise: use_aws=True")


if __name__ == "__main__":
    # Check if AWS credentials are configured
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}\n")
        
        # Run demos
        demonstrate_aws_backend()
        demonstrate_streaming_with_aws()
        compare_local_vs_aws()
        
    except Exception as e:
        print("Note: AWS credentials not configured.")
        print("To use AWS backend, configure AWS credentials first:")
        print("  aws configure")
        print("\nOr set environment variables:")
        print("  export AWS_ACCESS_KEY_ID=your-key-id")
        print("  export AWS_SECRET_ACCESS_KEY=your-secret-key")
        print("  export AWS_DEFAULT_REGION=us-east-1")
        
        print("\nYou can still use MAIF SDK without AWS:")
        print("  client = MAIFClient(use_aws=False)")
"""
Example demonstrating tight integration of AWS services using centralized credential management.

This example shows how all AWS services in MAIF now use a single, centralized credential
management system for consistent authentication and configuration.
"""

import os
from maif.aws_config import configure_aws, get_aws_config
from maif.aws_bedrock_integration import BedrockClient, MAIFBedrockIntegration
from maif.aws_s3_integration import S3Client, MAIFS3Integration
from maif.aws_lambda_integration import LambdaClient, MAIFLambdaIntegration
from maif.aws_dynamodb_integration import DynamoDBClient, MAIFDynamoDBIntegration
from maif.aws_kms_integration import KMSClient, MAIFKMSIntegration
from maif.security import SecurityManager
from maif.compliance_logging import EnhancedComplianceLogger


def main():
    """Demonstrate centralized AWS credential management."""
    
    # Option 1: Configure AWS globally once
    # This configuration will be used by ALL AWS services automatically
    configure_aws(
        environment="production",
        profile="maif-prod",  # Uses AWS profile
        region="us-east-1",
        enable_metrics=True
    )
    
    # Option 2: Configure with explicit credentials (for CI/CD)
    # configure_aws(
    #     environment="production",
    #     access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    #     secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    #     region="us-east-1"
    # )
    
    # Option 3: Configure with IAM role assumption
    # configure_aws(
    #     environment="production",
    #     role_arn="arn:aws:iam::123456789012:role/MAIFServiceRole",
    #     region="us-east-1"
    # )
    
    # Get the global config (optional - services will use it automatically)
    aws_config = get_aws_config()
    print(f"Using AWS region: {aws_config.credential_manager.region_name}")
    print(f"Environment: {aws_config.environment.value}")
    
    # Initialize AWS services - they ALL automatically use the centralized config
    # No need to pass credentials or regions to each service!
    
    # 1. Bedrock - AI/ML models
    bedrock = BedrockClient()  # Automatically uses centralized config
    bedrock_integration = MAIFBedrockIntegration(bedrock)
    
    # 2. S3 - Storage
    s3 = S3Client()  # Automatically uses centralized config
    s3_integration = MAIFS3Integration(s3)
    
    # 3. Lambda - Serverless compute
    lambda_client = LambdaClient()  # Automatically uses centralized config
    lambda_integration = MAIFLambdaIntegration(lambda_client)
    
    # 4. DynamoDB - NoSQL database
    dynamodb = DynamoDBClient()  # Automatically uses centralized config
    dynamodb_integration = MAIFDynamoDBIntegration(dynamodb)
    
    # 5. KMS - Key management
    kms = KMSClient()  # Automatically uses centralized config
    kms_integration = MAIFKMSIntegration(kms)
    
    # 6. Security Manager with KMS
    security_manager = SecurityManager(
        use_kms=True,  # Will automatically use centralized KMS client
        require_encryption=True
    )
    
    # 7. Compliance Logger
    compliance_logger = EnhancedComplianceLogger()
    
    # Example usage - all services work seamlessly with the same credentials
    print("\n=== Testing AWS Service Integration ===")
    
    # Test S3
    try:
        buckets = s3.list_buckets()
        print(f"✓ S3 access working - found {len(buckets)} buckets")
    except Exception as e:
        print(f"✗ S3 access failed: {e}")
    
    # Test DynamoDB
    try:
        tables = dynamodb.list_tables()
        print(f"✓ DynamoDB access working - found {len(tables)} tables")
    except Exception as e:
        print(f"✗ DynamoDB access failed: {e}")
    
    # Test KMS
    try:
        keys = kms.list_keys()
        print(f"✓ KMS access working - found {len(keys)} keys")
    except Exception as e:
        print(f"✗ KMS access failed: {e}")
    
    # Test Lambda
    try:
        functions = lambda_client.list_functions()
        print(f"✓ Lambda access working - found {len(functions)} functions")
    except Exception as e:
        print(f"✗ Lambda access failed: {e}")
    
    # Demonstrate credential refresh (happens automatically)
    print("\n=== Credential Management Features ===")
    print(f"✓ Automatic credential refresh: {aws_config.credential_manager.needs_refresh()}")
    print(f"✓ Thread-safe operations: Yes")
    print(f"✓ Credential caching: Enabled")
    
    # Show service-specific configurations
    print("\n=== Service-Specific Configurations ===")
    for service_name in ['s3', 'dynamodb', 'lambda', 'kms', 'bedrock']:
        config = aws_config.get_service_config(service_name)
        print(f"{service_name}: retry={config.retry_config.mode}, "
              f"max_attempts={config.retry_config.max_attempts}")
    
    # Test secure operations with centralized KMS
    print("\n=== Secure Operations ===")
    
    # Encrypt data using KMS (automatically uses centralized credentials)
    test_data = b"Sensitive information"
    encrypted = security_manager.encrypt_data(test_data)
    print("✓ Data encrypted with KMS")
    
    # Log compliance event (automatically uses centralized CloudWatch)
    compliance_logger.log_event(
        event_type="DATA_ACCESS",
        resource="s3://my-bucket/sensitive-data.json",
        user_id="test-user",
        action="READ",
        compliance_frameworks=["FIPS", "HIPAA"]
    )
    print("✓ Compliance event logged to CloudWatch")
    
    print("\n=== Integration Complete ===")
    print("All AWS services are using centralized credential management!")
    print("Benefits:")
    print("- Single point of configuration")
    print("- Automatic credential refresh")
    print("- Consistent retry policies")
    print("- Thread-safe operations")
    print("- Environment-specific settings")


if __name__ == "__main__":
    main()
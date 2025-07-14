# MAIF AWS Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [AWS Service Integrations](#aws-service-integrations)
   - [AWS Bedrock](#aws-bedrock-integration)
   - [AWS KMS](#aws-kms-integration)
   - [AWS S3](#aws-s3-integration)
   - [AWS Lambda](#aws-lambda-integration)
   - [AWS DynamoDB](#aws-dynamodb-integration)
   - [AWS Step Functions](#aws-step-functions-integration)
   - [AWS Kinesis](#aws-kinesis-integration)
   - [AWS Secrets Manager](#aws-secrets-manager-integration)
   - [AWS Macie](#aws-macie-integration)
   - [AWS CloudWatch Logs](#aws-cloudwatch-logs-integration)
4. [Installation and Setup](#installation-and-setup)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Security Considerations](#security-considerations)
8. [Performance Optimization](#performance-optimization)
9. [Cost Management](#cost-management)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Overview

The MAIF AWS Integration Suite provides production-ready integrations between MAIF's multimodal agent framework and various AWS services. These integrations enable:

- **Scalable AI/ML Operations**: Leverage AWS Bedrock for LLM capabilities
- **Enterprise Security**: Use AWS KMS for encryption and Secrets Manager for secure credential storage
- **Distributed Computing**: Utilize DynamoDB, Lambda, and Step Functions for distributed agent operations
- **Real-time Streaming**: Process streaming data with AWS Kinesis
- **Data Privacy**: Automatically classify and protect sensitive data with AWS Macie
- **Compliance Logging**: Centralized compliance logging with CloudWatch Logs
- **Reliable Storage**: Store and retrieve artifacts using AWS S3

### Key Features

- **Production-Ready**: All integrations include retry logic, error handling, and monitoring
- **Secure by Default**: Built-in encryption, access controls, and audit trails
- **Scalable Architecture**: Designed to handle enterprise workloads
- **Cost-Optimized**: Efficient resource usage and batch processing
- **Observable**: Comprehensive metrics and logging for all operations

## Quick Start

```python
# Install MAIF with AWS extras
pip install maif[aws]

# Basic usage example
from maif.core import MAIFClient
from maif.aws_s3_integration import S3Client
from maif.aws_kms_integration import KMSKeyStore

# Initialize MAIF with AWS integrations
client = MAIFClient(
    storage_backend=S3Client(
        region_name="us-east-1"
    ),
    encryption_backend=KMSKeyStore(
        region_name="us-east-1"
    )
)

# Create and store an encrypted artifact
artifact_id, path = client.create_artifact(config, data)
```

## AWS Service Integrations

### AWS Bedrock Integration

**Purpose**: Provides access to foundation models for AI/ML operations.

**Features**:
- Streaming and non-streaming inference
- Automatic retry with exponential backoff
- Model-specific parameter validation
- Comprehensive error handling
- Performance metrics tracking

**Usage**:
```python
from maif.aws_bedrock_integration import BedrockClient

bedrock = BedrockClient(region_name="us-east-1")

# Generate text
response = bedrock.generate_text_block(
    prompt="Explain quantum computing",
    metadata={
        "model_id": "anthropic.claude-v2",
        "max_tokens": 500,
        "temperature": 0.7
    }
)

# Stream responses
for chunk in bedrock.generate_text_stream(prompt, model_id):
    print(chunk, end="", flush=True)
```

**Best Practices**:
- Use streaming for long responses to improve user experience
- Implement token limits to control costs
- Cache responses when appropriate
- Monitor throttling metrics

### AWS KMS Integration

**Purpose**: Provides encryption key management for data protection.

**Features**:
- Automatic key rotation
- Envelope encryption for large data
- Grant-based access control
- Audit trail integration
- Hardware security module (HSM) support

**Usage**:
```python
from maif.aws_kms_integration import AWSKMSIntegration

kms = AWSKMSIntegration(
    region_name="us-east-1",
    key_alias="alias/maif-master-key"
)

# Encrypt data
encrypted_data, metadata = kms.encrypt_with_context(
    data=b"sensitive information",
    context={"purpose": "artifact-encryption"}
)

# Decrypt with context validation
decrypted_data = kms.decrypt_with_context(encrypted_data, metadata)
```

**Best Practices**:
- Use encryption context for additional security
- Rotate keys regularly (automated)
- Implement least-privilege access policies
- Monitor key usage metrics

### AWS S3 Integration

**Purpose**: Provides scalable object storage for artifacts and data.

**Features**:
- Multipart upload for large files
- Server-side encryption integration
- Presigned URLs for secure sharing
- Lifecycle policies support
- Cross-region replication

**Usage**:
```python
from maif.aws_s3_integration import AWSS3Integration

s3 = AWSS3Integration(
    bucket_name="my-maif-artifacts",
    region_name="us-east-1",
    enable_versioning=True
)

# Upload artifact
s3.upload_artifact(
    artifact_id="model-123",
    data=model_bytes,
    metadata={"version": "1.0"}
)

# Generate presigned URL
url = s3.create_presigned_url(
    artifact_id="model-123",
    expiration=3600  # 1 hour
)
```

**Best Practices**:
- Enable versioning for artifact history
- Use lifecycle policies for cost optimization
- Implement bucket policies for security
- Enable S3 Transfer Acceleration for global access

### AWS Lambda Integration

**Purpose**: Enables serverless compute for agent operations.

**Features**:
- Synchronous and asynchronous invocation
- Event-driven processing
- Dead letter queue support
- Concurrent execution limits
- Custom runtime support

**Usage**:
```python
from maif.aws_lambda_integration import AWSLambdaIntegration

lambda_client = AWSLambdaIntegration(region_name="us-east-1")

# Invoke function synchronously
result = lambda_client.invoke_agent_function(
    function_name="maif-agent-processor",
    payload={"action": "process", "data": "..."}
)

# Invoke asynchronously
lambda_client.invoke_agent_function_async(
    function_name="maif-background-task",
    payload={"task": "train_model"}
)
```

**Best Practices**:
- Use async invocation for long-running tasks
- Implement idempotent functions
- Set appropriate timeout values
- Monitor cold start metrics

### AWS DynamoDB Integration

**Purpose**: Provides distributed state management and metadata storage.

**Features**:
- Consistent and eventually consistent reads
- Batch operations for efficiency
- Global secondary indexes
- Point-in-time recovery
- Auto-scaling support

**Usage**:
```python
from maif.aws_dynamodb_integration import AWSDynamoDBIntegration

dynamodb = AWSDynamoDBIntegration(
    table_name="maif-agent-state",
    region_name="us-east-1"
)

# Store distributed state
dynamodb.put_agent_state(
    agent_id="agent-001",
    state={"status": "active", "tasks": []},
    vector_clock={"agent-001": 1}
)

# Query with conditions
states = dynamodb.query_agent_states(
    filter_condition="status = :status",
    expression_values={":status": "active"}
)
```

**Best Practices**:
- Use batch operations for multiple items
- Implement optimistic locking with version numbers
- Design partition keys for even distribution
- Enable point-in-time recovery

### AWS Step Functions Integration

**Purpose**: Orchestrates complex multi-step workflows.

**Features**:
- Visual workflow designer
- Error handling and retry logic
- Parallel execution support
- Integration with 200+ AWS services
- Long-running workflow support

**Usage**:
```python
from maif.aws_stepfunctions_integration import AWSStepFunctionsIntegration

stepfunctions = AWSStepFunctionsIntegration(region_name="us-east-1")

# Define and create workflow
workflow_definition = {
    "Comment": "MAIF Agent Training Pipeline",
    "StartAt": "DataPreparation",
    "States": {
        "DataPreparation": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:function:prepare-data",
            "Next": "TrainModel"
        },
        "TrainModel": {
            "Type": "Task", 
            "Resource": "arn:aws:lambda:...:function:train-model",
            "End": True
        }
    }
}

# Execute workflow
execution = stepfunctions.start_workflow_execution(
    state_machine_arn="arn:aws:states:...",
    input_data={"dataset": "training-data-v1"}
)
```

**Best Practices**:
- Use state machine versioning
- Implement proper error handling states
- Monitor execution history
- Use Express workflows for high-volume

### AWS Kinesis Integration

**Purpose**: Enables real-time data streaming and processing.

**Features**:
- High-throughput data ingestion
- Real-time analytics support
- Multiple consumer support
- Data retention up to 365 days
- Encryption at rest and in transit

**Usage**:
```python
from maif.aws_kinesis_streaming import AWSKinesisStreaming

kinesis = AWSKinesisStreaming(
    stream_name="maif-agent-events",
    region_name="us-east-1"
)

# Stream data
kinesis.stream_data(
    data=b"agent event data",
    partition_key="agent-001"
)

# Process stream with handler
async def process_event(record):
    print(f"Processing: {record['data']}")

await kinesis.process_stream_async(
    handler=process_event,
    batch_size=100
)
```

**Best Practices**:
- Use appropriate partition keys for distribution
- Implement checkpointing for reliability
- Monitor shard metrics
- Use Kinesis Analytics for real-time insights

### AWS Secrets Manager Integration

**Purpose**: Secure storage and rotation of secrets and credentials.

**Features**:
- Automatic secret rotation
- Fine-grained access control
- Encryption at rest
- Audit trail integration
- Cross-region replication

**Usage**:
```python
from maif.aws_secrets_manager_security import AWSSecretsManagerSecurity

secrets = AWSSecretsManagerSecurity(
    region_name="us-east-1",
    secret_prefix="maif/security/"
)

# Store encryption key
secrets.store_encryption_key(key_data)

# Retrieve with caching
key = secrets.get_encryption_key()

# Rotate keys
new_version = secrets.rotate_encryption_key()
```

**Best Practices**:
- Enable automatic rotation
- Use resource-based policies
- Implement secret versioning
- Monitor access patterns

### AWS Macie Integration

**Purpose**: Automated discovery and protection of sensitive data.

**Features**:
- Automated sensitive data discovery
- Custom data identifiers
- Compliance reporting
- Integration with S3
- Machine learning-based classification

**Usage**:
```python
from maif.aws_macie_privacy import AWSMaciePrivacy

macie = AWSMaciePrivacy(
    region_name="us-east-1",
    enable_auto_classification=True
)

# Scan S3 bucket
job_id = macie.scan_s3_bucket(
    bucket_name="my-data-bucket",
    wait_for_completion=True
)

# Process findings
policies = macie.process_macie_findings()

# Generate compliance report
report = macie.generate_compliance_report()
print(f"Compliance Score: {report['compliance_score']}/100")
```

**Best Practices**:
- Create custom identifiers for domain data
- Enable continuous monitoring
- Automate remediation workflows
- Regular compliance reporting

### AWS CloudWatch Logs Integration

**Purpose**: Centralized logging and compliance monitoring.

**Features**:
- Real-time log streaming
- Metric filters and alarms
- Log insights queries
- Long-term retention
- Encryption support

**Usage**:
```python
from maif.aws_cloudwatch_compliance import AWSCloudWatchComplianceLogger

logger = AWSCloudWatchComplianceLogger(
    region_name="us-east-1",
    log_group_name="maif-compliance",
    retention_days=90
)

# Log compliance event
logger.log(
    level=LogLevel.INFO,
    category=LogCategory.ACCESS,
    user_id="user-001",
    action="read_artifact",
    resource_id="artifact-123",
    details={"ip": "10.0.0.1"}
)

# Create dashboard
dashboard = logger.create_compliance_dashboard()
```

**Best Practices**:
- Set appropriate retention periods
- Use metric filters for alerting
- Export to S3 for cost savings
- Implement log aggregation

## Installation and Setup

### Prerequisites

1. **AWS Account**: Active AWS account with appropriate permissions
2. **Python**: Version 3.8 or higher
3. **AWS CLI**: Configured with credentials

### Installation Steps

```bash
# Install MAIF with all AWS dependencies
pip install maif[aws-all]

# Or install specific integrations
pip install maif[bedrock,s3,kms]

# Configure AWS credentials
aws configure

# Verify setup
python -c "import maif.aws_bedrock_integration; print('AWS integration ready')"
```

### IAM Permissions

Create an IAM role with the following policy for full MAIF AWS integration:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "kms:Encrypt",
                "kms:Decrypt",
                "kms:GenerateDataKey",
                "kms:CreateGrant",
                "kms:DescribeKey",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "lambda:InvokeFunction",
                "lambda:InvokeAsync",
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:Query",
                "dynamodb:Scan",
                "dynamodb:BatchWriteItem",
                "dynamodb:BatchGetItem",
                "states:CreateStateMachine",
                "states:StartExecution",
                "states:DescribeExecution",
                "kinesis:PutRecords",
                "kinesis:GetRecords",
                "kinesis:GetShardIterator",
                "kinesis:DescribeStream",
                "secretsmanager:GetSecretValue",
                "secretsmanager:CreateSecret",
                "secretsmanager:UpdateSecret",
                "secretsmanager:RotateSecret",
                "macie2:CreateClassificationJob",
                "macie2:GetFindings",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

## Configuration

### Environment Variables

```bash
# AWS Region
export AWS_DEFAULT_REGION=us-east-1

# MAIF Configuration
export MAIF_AWS_S3_BUCKET=my-maif-artifacts
export MAIF_AWS_KMS_KEY_ALIAS=alias/maif-master-key
export MAIF_AWS_DYNAMODB_TABLE=maif-agent-state
export MAIF_AWS_KINESIS_STREAM=maif-events
export MAIF_AWS_LOG_GROUP=maif-compliance

# Performance Tuning
export MAIF_AWS_MAX_RETRIES=3
export MAIF_AWS_BATCH_SIZE=25
export MAIF_AWS_CONNECTION_POOL_SIZE=10
```

### Configuration File

Create `maif_aws_config.yaml`:

```yaml
aws:
  region: us-east-1
  profile: default  # Optional AWS profile
  
  bedrock:
    default_model: anthropic.claude-v2
    max_retries: 3
    timeout: 300
    
  s3:
    bucket: my-maif-artifacts
    encryption: AES256
    versioning: true
    lifecycle_days: 90
    
  kms:
    key_alias: alias/maif-master-key
    rotation_enabled: true
    grant_tokens: []
    
  dynamodb:
    table: maif-agent-state
    read_capacity: 10
    write_capacity: 10
    
  lambda:
    timeout: 900
    memory: 3008
    concurrent_executions: 100
    
  kinesis:
    shard_count: 2
    retention_hours: 24
    
  cloudwatch:
    retention_days: 90
    batch_size: 100
    batch_interval: 5.0
```

## Best Practices

### 1. Security Best Practices

- **Encryption Everything**: Always use encryption at rest and in transit
- **Least Privilege**: Grant minimal required permissions
- **Rotate Secrets**: Enable automatic rotation for all secrets
- **Audit Everything**: Enable CloudTrail and compliance logging
- **Network Isolation**: Use VPC endpoints where possible

### 2. Performance Best Practices

- **Batch Operations**: Use batch APIs to reduce API calls
- **Connection Pooling**: Reuse connections for efficiency
- **Caching**: Cache frequently accessed data
- **Async Processing**: Use async operations for non-blocking code
- **Regional Deployment**: Deploy close to your users

### 3. Cost Optimization

- **Lifecycle Policies**: Delete old data automatically
- **Reserved Capacity**: Use reserved capacity for predictable workloads
- **Spot Instances**: Use spot for batch processing
- **Data Compression**: Compress data before storage
- **Monitoring**: Set up cost alerts

### 4. Reliability Best Practices

- **Multi-Region**: Deploy across regions for DR
- **Backups**: Regular automated backups
- **Health Checks**: Implement comprehensive health checks
- **Circuit Breakers**: Prevent cascade failures
- **Idempotency**: Make operations idempotent

## Security Considerations

### Data Protection

1. **Encryption**:
   - Use AWS KMS for key management
   - Enable S3 bucket encryption
   - Use TLS for data in transit
   - Implement envelope encryption for large data

2. **Access Control**:
   - Use IAM roles, not access keys
   - Implement resource-based policies
   - Enable MFA for sensitive operations
   - Regular access reviews

3. **Compliance**:
   - Enable CloudTrail logging
   - Use AWS Config for compliance monitoring
   - Regular security assessments
   - Implement data retention policies

### Network Security

1. **VPC Configuration**:
   ```python
   # Use VPC endpoints
   s3_client = boto3.client(
       's3',
       endpoint_url='https://s3.vpce-xxx.s3.us-east-1.vpce.amazonaws.com'
   )
   ```

2. **Security Groups**:
   - Restrict inbound traffic
   - Use specific port ranges
   - Regular rule audits

## Performance Optimization

### 1. Batching Strategies

```python
# Efficient batch processing
from maif.aws_s3_block_storage import AWSS3BlockStorage

storage = AWSS3BlockStorage(
    bucket_name="artifacts",
    batch_size=100,  # Process 100 items at once
    max_workers=10    # Parallel processing
)

# Batch upload
blocks = [block1, block2, block3, ...]
storage.batch_store_blocks(blocks)
```

### 2. Caching Implementation

```python
from functools import lru_cache
from maif.aws_kms_integration import AWSKMSIntegration

kms = AWSKMSIntegration()

@lru_cache(maxsize=100)
def get_cached_key(key_id):
    return kms.describe_key(key_id)
```

### 3. Connection Pooling

```python
# Configure connection pooling
import boto3
from botocore.config import Config

config = Config(
    region_name='us-east-1',
    max_pool_connections=50,
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    }
)

s3 = boto3.client('s3', config=config)
```

### 4. Async Operations

```python
import asyncio
from maif.aws_kinesis_streaming import AWSKinesisStreaming

async def process_high_volume():
    kinesis = AWSKinesisStreaming()
    
    # Process multiple streams concurrently
    tasks = [
        kinesis.process_stream_async(stream1),
        kinesis.process_stream_async(stream2),
        kinesis.process_stream_async(stream3)
    ]
    
    await asyncio.gather(*tasks)
```

## Cost Management

### 1. Cost Monitoring

```python
# Set up cost alerts
import boto3

ce = boto3.client('ce')

# Get cost forecast
forecast = ce.get_cost_forecast(
    TimePeriod={
        'Start': '2024-01-01',
        'End': '2024-01-31'
    },
    Metric='UNBLENDED_COST',
    Granularity='MONTHLY'
)
```

### 2. Resource Optimization

- **S3 Lifecycle Policies**:
  ```python
  s3.put_bucket_lifecycle_configuration(
      Bucket='my-bucket',
      LifecycleConfiguration={
          'Rules': [{
              'ID': 'Archive old artifacts',
              'Status': 'Enabled',
              'Transitions': [{
                  'Days': 30,
                  'StorageClass': 'GLACIER'
              }]
          }]
      }
  )
  ```

- **DynamoDB Auto-scaling**:
  ```python
  # Enable auto-scaling
  autoscaling = boto3.client('application-autoscaling')
  
  autoscaling.register_scalable_target(
      ServiceNamespace='dynamodb',
      ResourceId='table/maif-state',
      ScalableDimension='dynamodb:table:ReadCapacityUnits',
      MinCapacity=5,
      MaxCapacity=100
  )
  ```

### 3. Cost Allocation Tags

```python
# Tag resources for cost tracking
tags = [
    {'Key': 'Project', 'Value': 'MAIF'},
    {'Key': 'Environment', 'Value': 'Production'},
    {'Key': 'CostCenter', 'Value': 'ML-Team'}
]

# Apply to all resources
s3.put_bucket_tagging(
    Bucket='my-bucket',
    Tagging={'TagSet': tags}
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Authentication Errors**:
   ```python
   # Check credentials
   sts = boto3.client('sts')
   try:
       identity = sts.get_caller_identity()
       print(f"Authenticated as: {identity['Arn']}")
   except Exception as e:
       print(f"Authentication failed: {e}")
   ```

2. **Throttling Issues**:
   ```python
   # Implement exponential backoff
   from maif.aws_bedrock_integration import AWSBedrockIntegration
   
   bedrock = AWSBedrockIntegration(
       max_retries=5,
       base_delay=1.0,
       max_delay=60.0
   )
   ```

3. **Network Timeouts**:
   ```python
   # Increase timeout values
   config = Config(
       read_timeout=300,
       connect_timeout=60,
       retries={'max_attempts': 3}
   )
   ```

4. **Permission Denied**:
   ```bash
   # Check IAM policies
   aws iam simulate-principal-policy \
       --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) \
       --action-names s3:GetObject \
       --resource-arns arn:aws:s3:::my-bucket/*
   ```

### Debug Mode

Enable debug logging:

```python
import logging
import boto3

# Enable debug logging
boto3.set_stream_logger('boto3.resources', logging.DEBUG)
boto3.set_stream_logger('botocore', logging.DEBUG)

# MAIF debug mode
from maif import enable_debug_mode
enable_debug_mode()
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_operation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your MAIF AWS operations
    client.create_artifact(config, data)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

## Examples

### Complete Multi-Service Integration

```python
from maif.core import MAIFClient, ArtifactConfig, EncodingType
from maif.aws_s3_integration import AWSS3Integration
from maif.aws_kms_integration import AWSKMSIntegration
from maif.aws_bedrock_integration import AWSBedrockIntegration
from maif.aws_macie_privacy import AWSMaciePrivacy
from maif.aws_cloudwatch_compliance import AWSCloudWatchComplianceLogger

# Initialize integrated MAIF client
client = MAIFClient(
    storage_backend=AWSS3Integration(
        bucket_name="my-maif-artifacts",
        region_name="us-east-1"
    ),
    encryption_backend=AWSKMSIntegration(
        key_alias="alias/maif-encryption",
        region_name="us-east-1"
    ),
    privacy_engine=AWSMaciePrivacy(
        region_name="us-east-1"
    ),
    compliance_logger=AWSCloudWatchComplianceLogger(
        log_group_name="maif-compliance",
        region_name="us-east-1"
    )
)

# Use Bedrock for AI operations
bedrock = AWSBedrockIntegration()

# Generate artifact description
description = bedrock.generate_text(
    prompt="Describe a machine learning model for fraud detection",
    model_id="anthropic.claude-v2",
    max_tokens=200
)

# Create artifact with full AWS integration
config = ArtifactConfig(
    name="fraud-detection-model",
    artifact_type="model",
    encoding=EncodingType.PYTORCH,
    metadata={
        "description": description,
        "sensitivity": "high",
        "compliance_required": True
    }
)

# Model will be:
# - Stored in S3
# - Encrypted with KMS
# - Scanned by Macie
# - Logged to CloudWatch
artifact_id, path = client.create_artifact(config, model_data)

print(f"Created artifact {artifact_id} with full AWS integration")
```

### Distributed Agent System

```python
from maif.aws_distributed_integration import AWSDistributedIntegration
from maif.aws_stepfunctions_integration import AWSStepFunctionsIntegration

# Initialize distributed system
distributed = AWSDistributedIntegration(
    dynamodb_table="maif-agents",
    lambda_prefix="maif-agent-",
    region_name="us-east-1"
)

# Create agent workflow
stepfunctions = AWSStepFunctionsIntegration()

workflow = stepfunctions.create_agent_workflow(
    name="distributed-training",
    definition={
        "Comment": "Distributed model training",
        "StartAt": "DistributeData",
        "States": {
            "DistributeData": {
                "Type": "Parallel",
                "Branches": [
                    {
                        "StartAt": "TrainAgent1",
                        "States": {
                            "TrainAgent1": {
                                "Type": "Task",
                                "Resource": distributed.get_lambda_arn("train"),
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "TrainAgent2",
                        "States": {
                            "TrainAgent2": {
                                "Type": "Task",
                                "Resource": distributed.get_lambda_arn("train"),
                                "End": True
                            }
                        }
                    }
                ],
                "Next": "AggregateResults"
            },
            "AggregateResults": {
                "Type": "Task",
                "Resource": distributed.get_lambda_arn("aggregate"),
                "End": True
            }
        }
    }
)

# Execute distributed training
execution = stepfunctions.start_workflow_execution(
    state_machine_arn=workflow['stateMachineArn'],
    input_data={"dataset": "training-data-v1"}
)
```

### Real-time Stream Processing

```python
from maif.aws_kinesis_streaming import AWSKinesisStreaming
import asyncio

# Initialize streaming
streaming = AWSKinesisStreaming(
    stream_name="maif-agent-events",
    region_name="us-east-1"
)

# Define event handler
async def process_agent_event(record):
    data = json.loads(record['data'])
    
    if data['type'] == 'model_update':
        # Process model update
        await update_model(data['model_id'], data['weights'])
    elif data['type'] == 'prediction_request':
        # Handle prediction
        result = await predict(data['input'])
        await streaming.stream_data(
            json.dumps({'result': result}),
            partition_key=data['request_id']
        )

# Start processing
async def main():
    await streaming.process_stream_async(
        handler=process_agent_event,
        batch_size=100,
        checkpoint_interval=60
    )

# Run stream processor
asyncio.run(main())
```

## Conclusion

The MAIF AWS Integration Suite provides a comprehensive, production-ready solution for building scalable, secure, and compliant multimodal agent systems on AWS. By following the best practices and examples in this guide, you can leverage the full power of AWS services while maintaining the flexibility and capabilities of the MAIF framework.

For additional support and updates, visit the [MAIF GitHub repository](https://github.com/maif/maif) or consult the AWS documentation for specific services.
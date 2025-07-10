# AWS Integration with MAIF Agentic Framework

This guide explains how to use the decorator-based AWS integration with the MAIF agentic framework.

## Overview

The MAIF agentic framework provides simple decorator-based APIs for integrating with AWS services. These decorators make it easy to:

1. Create agents with MAIF that leverage AWS services
2. Enhance agent systems with AWS capabilities
3. Add AWS service access to specific methods

## Available Decorators

### Agent Creation Decorators

| Decorator | Description |
|-----------|-------------|
| `@maif_agent(**config)` | Create a MAIF agent from any class |
| `@aws_agent(**config)` | Create a fully AWS-integrated MAIF agent |

### AWS Service Decorators

| Decorator | Description |
|-----------|-------------|
| `@aws_bedrock()` | Add AWS Bedrock capabilities to a method |
| `@aws_kms()` | Add AWS KMS capabilities to a method |
| `@aws_s3()` | Add AWS S3 capabilities to a method |
| `@aws_lambda()` | Add AWS Lambda capabilities to a method |
| `@aws_dynamodb()` | Add AWS DynamoDB capabilities to a method |

### Agent System Enhancement Decorators

| Decorator | Description |
|-----------|-------------|
| `@enhance_perception_with_bedrock()` | Enhance agent perception with AWS Bedrock |
| `@enhance_reasoning_with_bedrock()` | Enhance agent reasoning with AWS Bedrock |
| `@enhance_execution_with_aws()` | Enhance agent execution with AWS services |

## Usage Examples

### Simple AWS-Enhanced Agent

```python
from maif.aws_decorators import maif_agent, aws_bedrock, aws_s3

@maif_agent(workspace="./agent_workspace")
class SimpleAgent:
    def __init__(self, name="SimpleAgent"):
        self.name = name
    
    @aws_bedrock()
    def generate_response(self, prompt, bedrock=None):
        # bedrock is automatically injected
        return bedrock.generate_text_block(prompt)
    
    @aws_s3()
    def store_result(self, text, bucket_name, s3_client=None):
        # s3_client is automatically injected
        key = f"results/{int(time.time())}.txt"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=text.encode('utf-8')
        )
        return f"s3://{bucket_name}/{key}"
    
    async def process(self):
        # Your agent logic here
        response = self.generate_response("What's the weather today?")
        s3_path = self.store_result(response["text"], "my-bucket")
```

### Fully AWS-Integrated Agent

```python
from maif.aws_decorators import aws_agent

@aws_agent(workspace="./agent_workspace")
class FullAWSAgent:
    async def process(self):
        # Uses AWS-enhanced systems automatically
        perception = await self.perceive("Analyze this data", "text")
        reasoning = await self.reason([perception])
        plan = await self.plan("create summary", [reasoning])
        result = await self.execute(plan)
        
        # All operations use AWS services under the hood
```

### Custom AWS-Enhanced Agent

```python
from maif.agentic_framework import MAIFAgent
from maif.aws_decorators import enhance_perception_with_bedrock, enhance_reasoning_with_bedrock

@enhance_perception_with_bedrock()
@enhance_reasoning_with_bedrock()
class CustomAWSAgent(MAIFAgent):
    async def run(self):
        while self.state != AgentState.TERMINATED:
            try:
                # Use AWS-enhanced perception and reasoning
                perception = await self.perceive("Input data", "text")
                reasoning = await self.reason([perception])
                
                # Use standard planning and execution
                plan = await self.plan("analyze data", [reasoning])
                result = await self.execute(plan)
                
                await asyncio.sleep(5.0)
            except Exception as e:
                print(f"Error: {e}")
```

## AWS Service Integration Details

### AWS Bedrock Integration

The `@aws_bedrock()` decorator provides access to AWS Bedrock for:

- Text generation with Claude, Titan, and other models
- Embedding generation for semantic search
- Image generation with Stable Diffusion
- Image analysis and understanding

Example:

```python
@aws_bedrock()
def analyze_image(self, image_data, bedrock=None):
    # Generate description of image
    description = bedrock.analyze_image(image_data)
    
    # Generate embeddings for the description
    embedding = bedrock.embed_text(description)
    
    return {
        "description": description,
        "embedding": embedding
    }
```

### AWS KMS Integration

The `@aws_kms()` decorator provides access to AWS KMS for:

- Secure key management
- Digital signatures
- Verification operations

Example:

```python
@aws_kms()
def sign_data(self, data, key_store=None, verifier=None):
    # List available keys
    keys = key_store.list_kms_keys()
    key_id = keys[0]['KeyId']
    
    # Sign data
    signature_metadata = sign_block_data_with_kms(
        verifier, data.encode(), key_id
    )
    
    return signature_metadata
```

### AWS S3 Integration

The `@aws_s3()` decorator provides access to AWS S3 for:

- Storing artifacts
- Retrieving artifacts
- Managing S3 buckets

Example:

```python
@aws_s3()
def store_artifact(self, artifact, bucket_name, s3_client=None):
    # Save artifact to file
    filepath = artifact.save("temp.maif")
    
    # Upload to S3
    s3_client.upload_file(
        filepath, 
        bucket_name, 
        f"artifacts/{artifact.name}.maif"
    )
    
    return f"s3://{bucket_name}/artifacts/{artifact.name}.maif"
```

## Enhanced Agent Systems

### AWS-Enhanced Perception System

The `AWSEnhancedPerceptionSystem` uses AWS Bedrock to:

- Generate embeddings for text
- Analyze images
- Process audio
- Handle multimodal inputs

### AWS-Enhanced Reasoning System

The `AWSEnhancedReasoningSystem` uses AWS Bedrock to:

- Analyze context
- Generate insights
- Create reasoning artifacts

### AWS-Enhanced Execution System

The `AWSExecutionSystem` provides executors for:

- Invoking Lambda functions
- Storing data in S3
- Querying DynamoDB
- Generating images with Bedrock

## Configuration

All decorators accept AWS configuration parameters:

```python
@aws_bedrock(
    region_name="us-west-2",
    profile_name="my-profile"
)
def my_method(self, bedrock=None):
    # Use bedrock with custom configuration
```

## Complete Example

See the [AWS Agent Demo](../examples/aws_agent_demo.py) for a complete example of using the AWS decorators with the MAIF agentic framework.

## AWS Credentials

The decorators use the standard AWS credential chain:

1. Environment variables
2. Shared credential file (~/.aws/credentials)
3. AWS IAM role for EC2/ECS

Make sure your AWS credentials are properly configured before using these decorators.
"""
AWS Agent with Backends Demo
============================

Demonstrates how to use MAIF agent decorators with AWS backends enabled.
When use_aws=True is set, agents automatically use:
- S3 for artifact storage
- KMS for encryption
- Secrets Manager for key management
- CloudWatch for compliance logging
- Macie for privacy classification
"""

import asyncio
import os
from pathlib import Path
from maif.aws_decorators import maif_agent, aws_agent
from maif_sdk.aws_backend import AWSConfig
from maif_sdk.types import SecurityLevel


# Example 1: Basic MAIF agent with AWS backends
@maif_agent(
    workspace="./agent_data/basic",
    use_aws=True  # Enable AWS backends
)
class BasicAWSAgent:
    """Basic agent that uses AWS backends for storage."""
    
    async def process(self, data: str):
        # Access the MAIF client (automatically configured with AWS backends)
        client = self.maif_client
        
        # Create content - stored in S3, encrypted with KMS
        artifact = await client.create_artifact("analysis", SecurityLevel.L4_REGULATED)
        artifact.add_text(data, title="Input Data")
        artifact.add_analysis({
            "processed": True,
            "timestamp": str(self.created),
            "agent_id": self.agent_id
        })
        
        # Save artifact - goes to S3
        artifact_id = await client.write_content(artifact)
        print(f"Saved artifact {artifact_id} to S3")
        
        return artifact_id


# Example 2: AWS agent with all enhancements
@aws_agent(
    workspace="./agent_data/enhanced",
    region_name="us-east-1"
    # use_aws=True is default for aws_agent
)
class EnhancedAWSAgent:
    """Enhanced agent with full AWS integration."""
    
    async def analyze_document(self, document_path: str):
        # This agent has:
        # - AWS backends for storage (S3, KMS, etc.)
        # - Bedrock-enhanced perception
        # - Bedrock-enhanced reasoning
        # - AWS execution capabilities
        # - Step Functions integration
        
        # Read document
        with open(document_path, 'r') as f:
            content = f.read()
        
        # Perceive - uses Bedrock for embeddings and analysis
        perception = await self.perceive({
            "text": content,
            "type": "document"
        })
        
        # Reason - uses Bedrock for advanced reasoning
        decision = await self.reason(perception)
        
        # Execute - can trigger Lambda functions, Step Functions, etc.
        result = await self.execute(decision)
        
        # Store results in S3
        client = self.maif_client
        artifact = await client.create_artifact("document_analysis", SecurityLevel.L5_CRITICAL)
        
        artifact.add_text(content, title="Original Document")
        artifact.add_analysis({
            "perception": perception,
            "decision": decision,
            "result": result
        })
        
        artifact_id = await client.write_content(artifact)
        print(f"Analysis complete. Results stored in S3: {artifact_id}")
        
        return artifact_id


# Example 3: Agent with custom AWS configuration
custom_aws_config = AWSConfig(
    region_name="us-west-2",
    s3_bucket="my-custom-maif-bucket",
    kms_key_id="alias/my-custom-key",
    use_encryption=True,
    use_compliance_logging=True
)

@maif_agent(
    workspace="./agent_data/custom",
    use_aws=True,
    aws_config=custom_aws_config
)
class CustomConfigAgent:
    """Agent with custom AWS configuration."""
    
    async def process_sensitive_data(self, data: dict):
        client = self.maif_client
        
        # Create highly secure artifact
        artifact = await client.create_artifact(
            "sensitive_data",
            SecurityLevel.L5_CRITICAL
        )
        
        # Add encrypted data
        artifact.add_data(data, title="Sensitive Information")
        
        # Enable additional security features
        artifact.metadata.update({
            "compliance_required": True,
            "encryption_algorithm": "AES-256-GCM",
            "access_control": "strict"
        })
        
        # Save to custom S3 bucket with custom KMS key
        artifact_id = await client.write_content(artifact)
        print(f"Sensitive data stored securely: {artifact_id}")
        
        # Compliance logging happens automatically
        return artifact_id


# Example 4: Multi-agent system with AWS backends
@aws_agent(workspace="./agent_data/collector")
class DataCollectorAgent:
    """Collects data and stores in S3."""
    
    async def collect_data(self, sources: list):
        client = self.maif_client
        collected_artifacts = []
        
        for source in sources:
            artifact = await client.create_artifact(
                f"data_from_{source}",
                SecurityLevel.L3_CONFIDENTIAL
            )
            
            # Simulate data collection
            artifact.add_data({
                "source": source,
                "timestamp": str(self.created),
                "data": f"Sample data from {source}"
            })
            
            artifact_id = await client.write_content(artifact)
            collected_artifacts.append(artifact_id)
            print(f"Collected data from {source}: {artifact_id}")
        
        return collected_artifacts


@aws_agent(workspace="./agent_data/processor")
class DataProcessorAgent:
    """Processes data from S3."""
    
    async def process_artifacts(self, artifact_ids: list):
        client = self.maif_client
        results = []
        
        for artifact_id in artifact_ids:
            # Read from S3
            artifact = await client.read_content(artifact_id)
            
            # Process data
            data = artifact.get_data()
            processed_result = {
                "original_id": artifact_id,
                "processed": True,
                "summary": f"Processed {len(data)} data items"
            }
            
            # Store processed results
            result_artifact = await client.create_artifact(
                f"processed_{artifact_id}",
                SecurityLevel.L3_CONFIDENTIAL
            )
            result_artifact.add_analysis(processed_result)
            
            result_id = await client.write_content(result_artifact)
            results.append(result_id)
            print(f"Processed {artifact_id} -> {result_id}")
        
        return results


async def main():
    """Demonstrate various AWS-enabled agents."""
    
    print("=== AWS Agent Backends Demo ===\n")
    
    # Set AWS credentials (in production, use IAM roles or env vars)
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Example 1: Basic agent with AWS backends
    print("1. Basic AWS Agent:")
    basic_agent = BasicAWSAgent()
    await basic_agent.initialize()
    
    artifact_id = await basic_agent.process("Hello from AWS-enabled agent!")
    print(f"   Created artifact: {artifact_id}\n")
    
    # Example 2: Enhanced AWS agent
    print("2. Enhanced AWS Agent:")
    enhanced_agent = EnhancedAWSAgent()
    await enhanced_agent.initialize()
    
    # Create test document
    test_doc = Path("./test_document.txt")
    test_doc.write_text("This is a test document for AWS agent processing.")
    
    doc_artifact_id = await enhanced_agent.analyze_document(str(test_doc))
    print(f"   Document analysis: {doc_artifact_id}\n")
    
    # Example 3: Custom configuration
    print("3. Custom Config Agent:")
    custom_agent = CustomConfigAgent()
    await custom_agent.initialize()
    
    sensitive_artifact_id = await custom_agent.process_sensitive_data({
        "user_id": "12345",
        "ssn": "XXX-XX-XXXX",
        "credit_score": 750
    })
    print(f"   Sensitive data: {sensitive_artifact_id}\n")
    
    # Example 4: Multi-agent workflow
    print("4. Multi-Agent Workflow:")
    collector = DataCollectorAgent()
    processor = DataProcessorAgent()
    
    await collector.initialize()
    await processor.initialize()
    
    # Collect data
    collected = await collector.collect_data(["API", "Database", "FileSystem"])
    
    # Process collected data
    processed = await processor.process_artifacts(collected)
    
    print(f"   Workflow complete: {len(collected)} collected, {len(processed)} processed\n")
    
    # Cleanup
    test_doc.unlink(missing_ok=True)
    
    print("Demo complete! All artifacts stored in AWS S3 with KMS encryption.")
    print("\nBenefits of AWS backends:")
    print("- Scalable S3 storage")
    print("- KMS encryption for security")
    print("- Secrets Manager for key management")
    print("- CloudWatch for compliance logging")
    print("- Macie for privacy classification")
    print("- Seamless integration with existing MAIF code")


if __name__ == "__main__":
    asyncio.run(main())
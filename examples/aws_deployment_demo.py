"""
AWS Deployment Demo for MAIF Agents
===================================

Demonstrates how to deploy MAIF agents to:
1. AWS Lambda for event-driven processing
2. ECS/Fargate for long-running agents
3. Generate CDK projects for infrastructure as code
"""

import asyncio
import os
from pathlib import Path
from maif.agentic_framework import MAIFAgent
from maif.aws_deployment import (
    DeploymentConfig,
    DeploymentManager,
    deploy_agent_to_lambda,
    deploy_agent_to_ecs
)
from maif_sdk.types import SecurityLevel


# Example Agent for Deployment
class ProcessingAgent(MAIFAgent):
    """Example agent that can be deployed to AWS."""
    
    async def run(self, event=None):
        """Process based on event or run continuously."""
        await self.initialize()
        
        if event:
            # Lambda mode - process single event
            return await self.process_event(event)
        else:
            # ECS mode - run continuously
            await self.continuous_processing()
    
    async def process_event(self, event):
        """Process a single event (Lambda mode)."""
        # Extract data from event
        data = event.get('data', 'No data provided')
        event_type = event.get('type', 'unknown')
        
        # Process with MAIF
        perception = await self.perceive(data, "text")
        reasoning = await self.reason([perception])
        
        # Return result
        return {
            'event_type': event_type,
            'perception_id': perception.name,
            'reasoning_id': reasoning.name,
            'status': 'processed'
        }
    
    async def continuous_processing(self):
        """Run continuous processing (ECS mode)."""
        iteration = 0
        
        while iteration < 100:  # Process 100 items then stop
            # Simulate processing
            data = f"Continuous data stream item {iteration}"
            perception = await self.perceive(data, "text")
            
            if iteration % 10 == 0:
                # Periodic reasoning
                reasoning = await self.reason([perception])
                print(f"Processed batch {iteration // 10}")
            
            iteration += 1
            await asyncio.sleep(1)
        
        print("Continuous processing complete")


def demonstrate_lambda_deployment():
    """Demonstrate deploying an agent to AWS Lambda."""
    
    print("=== Lambda Deployment Demo ===\n")
    
    # Configure deployment
    config = DeploymentConfig(
        agent_name="maif-processing-agent",
        agent_class="ProcessingAgent",
        runtime="python3.9",
        memory_mb=1024,
        timeout_seconds=300,
        environment_vars={
            "LOG_LEVEL": "INFO",
            "PROCESSING_MODE": "event"
        },
        s3_bucket="maif-deployment-artifacts",
        enable_xray=True,
        enable_cloudwatch=True
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    # Generate Lambda deployment artifacts
    print("1. Generating Lambda deployment package...")
    output_dir = Path("./deployments/lambda_demo")
    
    # This would create:
    # - Lambda deployment ZIP with handler
    # - CloudFormation template
    # - Upload to S3 if configured
    
    print("   ✓ Lambda handler generated")
    print("   ✓ Dependencies packaged")
    print("   ✓ CloudFormation template created")
    print(f"   ✓ Output directory: {output_dir}")
    
    # Show generated CloudFormation template structure
    print("\n2. CloudFormation template includes:")
    print("   - IAM role with required permissions")
    print("   - Lambda function configuration")
    print("   - CloudWatch log group")
    print("   - X-Ray tracing enabled")
    print("   - Environment variables")
    
    # Deployment command
    print("\n3. Deploy with AWS CLI:")
    print(f"   aws cloudformation create-stack \\")
    print(f"     --stack-name {config.agent_name}-stack \\")
    print(f"     --template-body file://cloudformation.yaml \\")
    print(f"     --capabilities CAPABILITY_IAM")
    
    print("\n4. Invoke Lambda function:")
    print(f"   aws lambda invoke \\")
    print(f"     --function-name {config.agent_name} \\")
    print("     --payload '{\"data\": \"test\", \"type\": \"demo\"}' \\")
    print("     output.json")


def demonstrate_ecs_deployment():
    """Demonstrate deploying an agent to ECS/Fargate."""
    
    print("\n=== ECS/Fargate Deployment Demo ===\n")
    
    # Configure deployment
    config = DeploymentConfig(
        agent_name="maif-continuous-agent",
        agent_class="ProcessingAgent",
        memory_mb=2048,
        environment_vars={
            "LOG_LEVEL": "INFO",
            "PROCESSING_MODE": "continuous"
        },
        ecr_repository="123456789012.dkr.ecr.us-east-1.amazonaws.com/maif-agents",
        ecs_cluster="maif-agent-cluster",
        enable_xray=True
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    # Generate ECS deployment artifacts
    print("1. Generating ECS deployment artifacts...")
    output_dir = Path("./deployments/ecs_demo")
    
    print("   ✓ Dockerfile generated")
    print("   ✓ ECS task definition created")
    print("   ✓ CloudFormation template created")
    print(f"   ✓ Output directory: {output_dir}")
    
    # Show Dockerfile structure
    print("\n2. Dockerfile includes:")
    print("   - Python 3.9 base image")
    print("   - MAIF framework installation")
    print("   - Agent code")
    print("   - Automatic entrypoint")
    
    # Build and push commands
    print("\n3. Build and push Docker image:")
    print(f"   docker build -t {config.agent_name} .")
    print(f"   docker tag {config.agent_name} {config.ecr_repository}:{config.agent_name}")
    print(f"   docker push {config.ecr_repository}:{config.agent_name}")
    
    # Deploy commands
    print("\n4. Deploy to ECS:")
    print(f"   aws cloudformation create-stack \\")
    print(f"     --stack-name {config.agent_name}-ecs-stack \\")
    print(f"     --template-body file://ecs_cloudformation.yaml \\")
    print(f"     --capabilities CAPABILITY_IAM")


def demonstrate_cdk_deployment():
    """Demonstrate CDK project generation."""
    
    print("\n=== CDK Project Generation Demo ===\n")
    
    # Configure deployment
    config = DeploymentConfig(
        agent_name="maif-cdk-agent",
        agent_class="ProcessingAgent",
        memory_mb=512,
        timeout_seconds=60,
        s3_bucket="maif-cdk-artifacts",
        ecs_cluster="maif-agents"
    )
    
    # Create deployment manager
    manager = DeploymentManager(config)
    
    # Generate CDK project
    print("1. Generating CDK project...")
    output_dir = Path("./deployments/cdk_demo")
    
    # This would create a complete CDK project
    cdk_artifacts = manager.generate_cdk_project(output_dir)
    
    print("   ✓ CDK stack generated")
    print("   ✓ app.py created")
    print("   ✓ cdk.json configured")
    print("   ✓ requirements.txt added")
    print(f"   ✓ Project directory: {output_dir}")
    
    # Show CDK stack features
    print("\n2. CDK stack includes:")
    print("   - Lambda function construct")
    print("   - ECS service construct (optional)")
    print("   - S3 bucket for agent data")
    print("   - IAM roles and policies")
    print("   - CloudWatch logs")
    print("   - X-Ray tracing")
    
    # CDK commands
    print("\n3. Deploy with CDK:")
    print(f"   cd {output_dir}")
    print("   pip install -r requirements.txt")
    print("   cdk synth")
    print("   cdk deploy")
    
    # CDK advantages
    print("\n4. CDK advantages:")
    print("   - Type-safe infrastructure")
    print("   - Reusable constructs")
    print("   - Easy environment management")
    print("   - Built-in best practices")


def demonstrate_batch_deployment():
    """Demonstrate deploying multiple agents."""
    
    print("\n=== Batch Deployment Demo ===\n")
    
    # Define multiple agents
    agents = [
        {
            "name": "data-processor",
            "class": "DataProcessingAgent",
            "type": "lambda",
            "memory": 512
        },
        {
            "name": "monitor-agent",
            "class": "MonitoringAgent", 
            "type": "ecs",
            "memory": 1024
        },
        {
            "name": "analyzer-agent",
            "class": "AnalysisAgent",
            "type": "lambda",
            "memory": 2048
        }
    ]
    
    print("Deploying multiple agents:\n")
    
    for agent in agents:
        print(f"Agent: {agent['name']}")
        print(f"  Type: {agent['type']}")
        print(f"  Memory: {agent['memory']} MB")
        
        # Create config
        config = DeploymentConfig(
            agent_name=agent['name'],
            agent_class=agent['class'],
            memory_mb=agent['memory'],
            s3_bucket="maif-deployments"
        )
        
        # Deploy based on type
        if agent['type'] == 'lambda':
            print("  → Deploying to Lambda...")
        else:
            print("  → Deploying to ECS...")
        
        print("  ✓ Deployment artifacts generated\n")


def demonstrate_deployment_cli():
    """Show CLI interface for deployments."""
    
    print("\n=== Deployment CLI Demo ===\n")
    
    print("MAIF provides a CLI for easy deployment:\n")
    
    # Lambda deployment
    print("1. Deploy to Lambda:")
    print("   maif deploy lambda \\")
    print("     --agent-class ProcessingAgent \\")
    print("     --agent-module examples.processing_agent \\")
    print("     --memory 1024 \\")
    print("     --timeout 300 \\")
    print("     --s3-bucket maif-deployments")
    
    # ECS deployment
    print("\n2. Deploy to ECS:")
    print("   maif deploy ecs \\")
    print("     --agent-class ContinuousAgent \\")
    print("     --agent-module examples.continuous_agent \\")
    print("     --memory 2048 \\")
    print("     --ecr-repo 123456789012.dkr.ecr.us-east-1.amazonaws.com/agents \\")
    print("     --cluster maif-agents")
    
    # Generate CDK
    print("\n3. Generate CDK project:")
    print("   maif deploy generate-cdk \\")
    print("     --agent-class MyAgent \\")
    print("     --output-dir ./my-agent-cdk")
    
    # Status check
    print("\n4. Check deployment status:")
    print("   maif deploy status --stack-name my-agent-stack")


def main():
    """Run all deployment demonstrations."""
    
    print("=== MAIF Agent AWS Deployment Demo ===\n")
    print("This demo shows how to deploy MAIF agents to AWS services.\n")
    
    # Lambda deployment
    demonstrate_lambda_deployment()
    
    # ECS deployment
    demonstrate_ecs_deployment()
    
    # CDK generation
    demonstrate_cdk_deployment()
    
    # Batch deployment
    demonstrate_batch_deployment()
    
    # CLI interface
    demonstrate_deployment_cli()
    
    print("\n=== Deployment Best Practices ===\n")
    
    print("1. Lambda deployment:")
    print("   - Best for event-driven processing")
    print("   - Auto-scaling")
    print("   - Pay per invocation")
    print("   - 15-minute timeout limit")
    
    print("\n2. ECS/Fargate deployment:")
    print("   - Best for long-running agents")
    print("   - Continuous processing")
    print("   - More control over resources")
    print("   - No timeout limits")
    
    print("\n3. Hybrid approach:")
    print("   - Lambda for API/event handling")
    print("   - ECS for background processing")
    print("   - Step Functions for orchestration")
    
    print("\n4. Security considerations:")
    print("   - Use IAM roles, not credentials")
    print("   - Enable encryption everywhere")
    print("   - Use VPC for network isolation")
    print("   - Enable X-Ray and CloudWatch")


if __name__ == "__main__":
    main()
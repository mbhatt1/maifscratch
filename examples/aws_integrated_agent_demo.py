"""
AWS Integrated Agent Demo
=========================

Demonstrates agents that automatically:
1. Spin up AWS systems on start (X-Ray, Bedrock, CloudWatch, S3, etc.)
2. Use AWS services during operation
3. Dump complete state to MAIF on shutdown
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from maif.agentic_framework import MAIFAgent, AgentState
from maif_sdk.aws_backend import AWSConfig
from maif_sdk.types import SecurityLevel


class DataAnalysisAgent(MAIFAgent):
    """Example agent that analyzes data with automatic AWS integration."""
    
    async def run(self):
        """Main agent loop."""
        await self.initialize()
        
        # Process some sample data
        for i in range(3):
            # Perceive input
            perception = await self.perceive(
                f"Sample data item {i+1}: Analyze market trends for Q{i+1} 2024",
                "text"
            )
            
            # Apply reasoning
            reasoning = await self.reason([perception])
            
            # Create plan
            plan = await self.plan("analyze and summarize", [reasoning])
            
            # Execute plan
            result = await self.execute(plan)
            
            # Learn from experience
            await self.learn([perception, reasoning, result])
            
            print(f"Processed item {i+1}")
            await asyncio.sleep(1)  # Simulate work
        
        print("Agent processing complete")


class MonitoringAgent(MAIFAgent):
    """Agent that monitors system metrics with AWS integration."""
    
    async def run(self):
        """Monitor system continuously."""
        await self.initialize()
        
        monitoring_data = []
        
        for i in range(5):
            # Simulate monitoring
            metric_data = {
                "timestamp": i,
                "cpu_usage": 45 + i * 5,
                "memory_usage": 60 + i * 3,
                "active_connections": 100 + i * 10
            }
            
            # Process as perception
            perception = await self.perceive(
                str(metric_data),
                "text"
            )
            
            monitoring_data.append(perception)
            
            # Analyze if threshold exceeded
            if metric_data["cpu_usage"] > 50:
                reasoning = await self.reason(monitoring_data[-3:])
                print(f"Alert: High CPU usage detected - {metric_data['cpu_usage']}%")
            
            await asyncio.sleep(0.5)
        
        print("Monitoring complete")


async def demonstrate_aws_integrated_agents():
    """Demonstrate agents with automatic AWS integration."""
    
    print("=== AWS Integrated Agent Demo ===\n")
    
    # Set AWS credentials (in production, use IAM roles)
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Configure AWS settings
    aws_config = AWSConfig(
        region_name="us-east-1",
        s3_bucket=os.environ.get('MAIF_S3_BUCKET', 'maif-agent-data'),
        use_encryption=True,
        use_compliance_logging=True
    )
    
    # Example 1: Agent with AWS enabled
    print("1. Creating Data Analysis Agent with AWS Integration...")
    aws_agent = DataAnalysisAgent(
        agent_id="data_analyst_001",
        workspace_path="./agent_workspace/aws",
        config={
            "xray_sampling_rate": 1.0,  # 100% for demo
            "analysis_mode": "comprehensive"
        },
        use_aws=True,
        aws_config=aws_config
    )
    
    print("   ✓ AWS systems initialized:")
    print("     - X-Ray tracing enabled")
    print("     - Bedrock for embeddings")
    print("     - S3 for artifact storage")
    print("     - CloudWatch for compliance logging")
    print()
    
    # Example 2: Agent without AWS (for comparison)
    print("2. Creating Local Agent (no AWS)...")
    local_agent = MonitoringAgent(
        agent_id="monitor_001",
        workspace_path="./agent_workspace/local",
        config={"monitoring_interval": 0.5},
        use_aws=False
    )
    
    print("   ✓ Local systems initialized")
    print()
    
    # Run agents
    print("3. Running agents...")
    
    # Create tasks
    aws_task = asyncio.create_task(aws_agent.run())
    local_task = asyncio.create_task(local_agent.run())
    
    # Wait for completion
    await aws_task
    await local_task
    
    print("\n4. Shutting down agents and dumping state...")
    
    # Shutdown AWS agent
    print("\n   AWS Agent shutdown:")
    aws_dump_path = aws_agent.shutdown()
    print(f"   ✓ State dumped to: {aws_dump_path}")
    print("   ✓ Final state includes:")
    print("     - Complete agent configuration")
    print("     - All perceptions and reasoning artifacts")
    print("     - AWS service usage metrics")
    print("     - X-Ray trace data")
    print("     - CloudWatch compliance logs")
    
    # Shutdown local agent
    print("\n   Local Agent shutdown:")
    local_dump_path = local_agent.shutdown()
    print(f"   ✓ State dumped to: {local_dump_path}")
    
    # Compare storage
    print("\n5. Storage comparison:")
    print(f"   AWS Agent artifacts: Stored in S3 bucket '{aws_config.s3_bucket}'")
    print(f"   Local Agent artifacts: Stored in '{local_agent.workspace_path}'")
    
    print("\n6. Benefits of AWS Integration:")
    print("   - Automatic tracing of all operations with X-Ray")
    print("   - Scalable S3 storage for artifacts")
    print("   - KMS encryption for sensitive data")
    print("   - CloudWatch logs for compliance")
    print("   - Bedrock for advanced AI capabilities")
    print("   - Complete state preservation on shutdown")
    
    print("\nDemo complete! Check:")
    print("- AWS X-Ray console for trace data")
    print("- S3 bucket for stored artifacts")
    print("- CloudWatch logs for compliance events")
    print(f"- Local MAIF dumps at: {aws_dump_path} and {local_dump_path}")


class GracefulShutdownAgent(MAIFAgent):
    """Agent that demonstrates graceful shutdown on signals."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[Signal {signum} received] Initiating graceful shutdown...")
        self.running = False
    
    async def run(self):
        """Run until shutdown signal."""
        await self.initialize()
        
        print("Agent running. Press Ctrl+C to trigger graceful shutdown...")
        
        iteration = 0
        while self.running:
            # Simulate work
            perception = await self.perceive(
                f"Continuous data stream item {iteration}",
                "text"
            )
            
            if iteration % 5 == 0:
                print(f"Processed {iteration} items...")
            
            iteration += 1
            await asyncio.sleep(0.2)
        
        print(f"Agent stopped after processing {iteration} items")


async def demonstrate_graceful_shutdown():
    """Demonstrate graceful shutdown with state dump."""
    
    print("\n=== Graceful Shutdown Demo ===\n")
    
    # Create agent with AWS
    agent = GracefulShutdownAgent(
        agent_id="graceful_001",
        workspace_path="./agent_workspace/graceful",
        config={"process_rate": 5},
        use_aws=True
    )
    
    print("Starting agent with automatic state preservation...")
    print("AWS services spinning up...")
    
    try:
        # Run agent
        await agent.run()
    except KeyboardInterrupt:
        pass
    
    # Shutdown will dump complete state
    print("\nPerforming shutdown sequence...")
    dump_path = agent.shutdown()
    
    print(f"\n✓ Agent state successfully preserved at: {dump_path}")
    print("✓ State includes all processed data and AWS metrics")
    print("✓ Agent can be resumed from this state dump")


if __name__ == "__main__":
    # Run basic demo
    asyncio.run(demonstrate_aws_integrated_agents())
    
    # Uncomment to run graceful shutdown demo
    # asyncio.run(demonstrate_graceful_shutdown())
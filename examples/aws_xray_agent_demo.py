"""
AWS X-Ray Agent Demo
====================

Demonstrates how to use AWS X-Ray for distributed tracing of MAIF agents.
X-Ray provides insights into agent operations, performance metrics, and error tracking.
"""

import asyncio
import os
import time
from pathlib import Path
from maif.aws_decorators import (
    maif_agent, aws_agent, aws_xray, trace_aws_call,
    xray_trace, xray_subsegment
)
from maif_sdk.types import SecurityLevel
from maif.aws_xray_integration import xray  # Direct X-Ray recorder access


# Example 1: Basic agent with X-Ray tracing
@maif_agent(
    workspace="./agent_data/xray_basic",
    use_aws=True,
    enable_xray=True,
    xray_service_name="BasicXRayAgent"
)
class BasicXRayAgent:
    """Basic agent with automatic X-Ray tracing of all operations."""
    
    async def process_data(self, data: dict):
        # All agent operations are automatically traced
        
        # Add custom annotations
        if hasattr(self, '_xray_integration'):
            self._xray_integration.add_annotation('data_size', len(str(data)))
            self._xray_integration.add_metadata('input_data', data)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Create artifact
        client = self.maif_client
        artifact = await client.create_artifact("analysis", SecurityLevel.L3_CONFIDENTIAL)
        artifact.add_analysis({
            "processed": True,
            "timestamp": time.time(),
            "result": "Data processed successfully"
        })
        
        # Save to S3 (automatically traced)
        artifact_id = await client.write_content(artifact)
        
        return artifact_id


# Example 2: AWS agent with full X-Ray integration
@aws_agent(
    workspace="./agent_data/xray_enhanced",
    enable_xray=True,
    xray_service_name="EnhancedXRayAgent"
)
class EnhancedXRayAgent:
    """Enhanced agent with comprehensive X-Ray tracing."""
    
    @xray_subsegment("data_validation")
    async def validate_data(self, data: dict):
        """Validate input data - traced as subsegment."""
        required_fields = ['type', 'content', 'priority']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Add validation metadata
        xray.current_subsegment().put_metadata('validated_fields', required_fields)
        
        return True
    
    @trace_aws_call("bedrock")
    async def analyze_with_bedrock(self, content: str):
        """Analyze content using Bedrock - AWS call traced."""
        # Bedrock analysis happens here
        # The decorator automatically traces this as an AWS service call
        
        # Simulate Bedrock call
        await asyncio.sleep(0.2)
        
        return {
            "sentiment": "positive",
            "entities": ["MAIF", "AWS", "X-Ray"],
            "summary": "Content analyzed successfully"
        }
    
    @xray_trace("complex_workflow")
    async def run_complex_workflow(self, data: dict):
        """Run a complex multi-step workflow with detailed tracing."""
        
        # Step 1: Validate data
        await self.validate_data(data)
        
        # Step 2: Perceive input
        perception = await self.perceive(data['content'], "text")
        
        # Step 3: Analyze with Bedrock
        analysis = await self.analyze_with_bedrock(data['content'])
        
        # Step 4: Reason about results
        reasoning = await self.reason([perception])
        
        # Step 5: Execute plan
        result = await self.execute({
            "action": "store_results",
            "data": {
                "perception": perception,
                "analysis": analysis,
                "reasoning": reasoning
            }
        })
        
        return result


# Example 3: Custom X-Ray decorated class
@aws_xray(service_name="DataProcessor", sampling_rate=1.0)  # 100% sampling for demo
class DataProcessorAgent:
    """All methods in this class are automatically traced."""
    
    def __init__(self):
        self.processed_count = 0
    
    async def preprocess(self, raw_data: str):
        """Preprocessing step - automatically traced."""
        # Add custom metadata
        if hasattr(self, '_xray_integration'):
            self._xray_integration.add_metadata('raw_data_length', len(raw_data))
        
        # Simulate preprocessing
        await asyncio.sleep(0.05)
        
        return raw_data.strip().lower()
    
    async def transform(self, data: str):
        """Transform step - automatically traced."""
        # Simulate transformation
        await asyncio.sleep(0.05)
        
        transformed = {
            "original": data,
            "transformed": data.upper(),
            "word_count": len(data.split())
        }
        
        return transformed
    
    async def save_results(self, results: dict):
        """Save step - automatically traced."""
        self.processed_count += 1
        
        # Add metrics
        if hasattr(self, '_xray_integration'):
            self._xray_integration.add_annotation('processed_count', self.processed_count)
        
        # Simulate save
        await asyncio.sleep(0.05)
        
        return f"saved_{self.processed_count}"


# Example 4: Multi-agent system with distributed tracing
@aws_agent(workspace="./agent_data/collector", xray_service_name="CollectorAgent")
class CollectorAgent:
    """Collects data from various sources."""
    
    @xray_trace("collect_from_source")
    async def collect_from_source(self, source: str):
        # Trace header for distributed tracing
        trace_header = None
        if hasattr(self, '_xray_integration'):
            trace_header = self._xray_integration.get_trace_header()
        
        # Simulate data collection
        await asyncio.sleep(0.1)
        
        data = {
            "source": source,
            "data": f"Sample data from {source}",
            "trace_id": trace_header
        }
        
        return data


@aws_agent(workspace="./agent_data/analyzer", xray_service_name="AnalyzerAgent")
class AnalyzerAgent:
    """Analyzes collected data."""
    
    async def analyze_batch(self, data_batch: list):
        # Continue distributed trace
        for data in data_batch:
            if 'trace_id' in data:
                # This would continue the trace from the collector
                xray.begin_segment(name="analyze_item", trace_id=data['trace_id'])
                
                try:
                    # Analyze data
                    result = await self.analyze_item(data)
                    xray.current_segment().put_annotation('status', 'success')
                    
                except Exception as e:
                    xray.current_segment().add_exception(e)
                    raise
                    
                finally:
                    xray.end_segment()


async def demonstrate_xray_features():
    """Demonstrate various X-Ray features with MAIF agents."""
    
    print("=== AWS X-Ray Agent Demo ===\n")
    
    # Set AWS credentials (in production, use IAM roles)
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    # Example 1: Basic X-Ray tracing
    print("1. Basic X-Ray Agent:")
    basic_agent = BasicXRayAgent()
    await basic_agent.initialize()
    
    result1 = await basic_agent.process_data({
        "user": "demo_user",
        "action": "analyze",
        "data": "Sample data for X-Ray tracing"
    })
    print(f"   Result: {result1}")
    print("   ✓ Check X-Ray console for trace\n")
    
    # Example 2: Enhanced tracing with subsegments
    print("2. Enhanced X-Ray Agent:")
    enhanced_agent = EnhancedXRayAgent()
    await enhanced_agent.initialize()
    
    try:
        result2 = await enhanced_agent.run_complex_workflow({
            "type": "document",
            "content": "This is a test document for X-Ray tracing demonstration.",
            "priority": "high"
        })
        print(f"   Workflow completed: {result2}")
    except Exception as e:
        print(f"   Error (traced in X-Ray): {e}")
    print("   ✓ Check X-Ray service map for detailed traces\n")
    
    # Example 3: Custom decorated class
    print("3. Custom X-Ray Decorated Class:")
    processor = DataProcessorAgent()
    
    # Process multiple items to show aggregation
    for i in range(3):
        raw = f"sample data item {i+1}"
        preprocessed = await processor.preprocess(raw)
        transformed = await processor.transform(preprocessed)
        saved = await processor.save_results(transformed)
        print(f"   Processed item {i+1}: {saved}")
    
    print("   ✓ Check X-Ray for aggregated metrics\n")
    
    # Example 4: Distributed tracing
    print("4. Distributed Tracing:")
    collector = CollectorAgent()
    analyzer = AnalyzerAgent()
    
    await collector.initialize()
    await analyzer.initialize()
    
    # Collect data from multiple sources
    sources = ["API", "Database", "FileSystem"]
    collected_data = []
    
    for source in sources:
        data = await collector.collect_from_source(source)
        collected_data.append(data)
        print(f"   Collected from {source}")
    
    # Analyze collected data (continues the trace)
    await analyzer.analyze_batch(collected_data)
    print("   ✓ Check X-Ray for distributed trace across agents\n")
    
    print("Demo complete! Check AWS X-Ray console for:")
    print("- Service map showing agent interactions")
    print("- Trace timelines with detailed segments")
    print("- Performance metrics and annotations")
    print("- Error tracking and exceptions")
    print("\nX-Ray provides valuable insights for:")
    print("- Performance optimization")
    print("- Error debugging")
    print("- Understanding agent workflows")
    print("- Monitoring production systems")


if __name__ == "__main__":
    # Note: Ensure X-Ray daemon is running locally or use AWS environment
    # docker run -p 2000:2000/udp amazon/aws-xray-daemon -o
    
    asyncio.run(demonstrate_xray_features())
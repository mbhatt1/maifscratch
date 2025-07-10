#!/usr/bin/env python3
"""
Bedrock Agent Swarm Demo
========================

Demonstrates the use of multiple AWS Bedrock models in a swarm of agents that share the same MAIF storage.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAIF components
from maif.bedrock_swarm import BedrockAgentSwarm
from maif.aws_bedrock_integration import BedrockModelProvider

# Example tasks for the agent swarm
EXAMPLE_TASKS = [
    {
        "task_id": "task_1",
        "type": "all",
        "data": "Analyze the benefits of using multiple AI models in a swarm configuration.",
        "input_type": "text",
        "goal": "provide comprehensive analysis",
        "aggregation": "all"
    },
    {
        "task_id": "task_2",
        "type": "provider",
        "provider": "anthropic",
        "data": "Compare the Claude model family with other large language models.",
        "input_type": "text",
        "goal": "compare models",
        "aggregation": "all"
    },
    {
        "task_id": "task_3",
        "type": "all",
        "data": "What are the key considerations when implementing a multi-agent system?",
        "input_type": "text",
        "goal": "identify key considerations",
        "aggregation": "vote"
    },
    {
        "task_id": "task_4",
        "type": "all",
        "data": "What are the ethical implications of using AI in healthcare?",
        "input_type": "text",
        "goal": "analyze ethical implications",
        "aggregation": "weighted_vote",
        "provider_weights": {
            "anthropic": 1.0,  # Higher weight for Claude models on ethical questions
            "amazon": 0.8,
            "ai21": 0.7,
            "cohere": 0.7
        }
    },
    {
        "task_id": "task_5",
        "type": "all",
        "data": "Provide recommendations for implementing a secure cloud architecture.",
        "input_type": "text",
        "goal": "provide security recommendations",
        "aggregation": "ensemble"
    },
    {
        "task_id": "task_6",
        "type": "all",
        "data": "What are the most promising applications of AI in the next 5 years?",
        "input_type": "text",
        "goal": "predict AI applications",
        "aggregation": "semantic_merge"
    }
]

async def demonstrate_bedrock_swarm():
    """Demonstrate the Bedrock agent swarm."""
    print("\n=== Bedrock Agent Swarm Demo ===")
    
    # Create workspace
    workspace = Path("./demo_workspace/bedrock_swarm")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create agent swarm
    swarm = BedrockAgentSwarm(str(workspace))
    
    # Add agents with different models
    swarm.add_agent_with_model(
        "claude_agent",
        BedrockModelProvider.ANTHROPIC,
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "us-east-1"
    )
    
    swarm.add_agent_with_model(
        "titan_agent",
        BedrockModelProvider.AMAZON,
        "amazon.titan-text-express-v1",
        "us-east-1"
    )
    
    swarm.add_agent_with_model(
        "jurassic_agent",
        BedrockModelProvider.AI21,
        "ai21.j2-ultra-v1",
        "us-east-1"
    )
    
    swarm.add_agent_with_model(
        "command_agent",
        BedrockModelProvider.COHERE,
        "cohere.command-text-v14",
        "us-east-1"
    )
    
    # Start swarm in background
    swarm_task = asyncio.create_task(swarm.run())
    
    try:
        # Submit tasks
        for task in EXAMPLE_TASKS:
            task_id = await swarm.submit_task(task)
            logger.info(f"Submitted task {task_id}")
        
        # Wait for results
        for task in EXAMPLE_TASKS:
            task_id = task["task_id"]
            logger.info(f"Waiting for result of task {task_id}")
            
            result = await swarm.get_result(task_id, timeout=120.0)
            if result:
                print(f"\nResults for task {task_id}:")
                print(f"Task: {task['data']}")
                
                # Display results based on aggregation method
                if task["aggregation"] == "vote":
                    print(f"Consensus: {result.get('consensus', 'No consensus')}")
                    print(f"Vote count: {result.get('vote_count', 0)}/{result.get('total_votes', 0)}")
                
                elif task["aggregation"] == "weighted_vote":
                    print(f"Weighted Consensus: {result.get('consensus', 'No consensus')}")
                    print(f"Weighted score: {result.get('weighted_score', 0)}")
                    print("\nContributing agents:")
                    for agent_id, weight in result.get("contributing_agents", []):
                        print(f"- {agent_id}: weight {weight:.2f}")
                
                elif task["aggregation"] == "ensemble":
                    print("Ensemble Result:")
                    print(result.get("ensemble_result", "No ensemble result"))
                
                elif task["aggregation"] == "semantic_merge":
                    print("Semantically Merged Result:")
                    print(result.get("merged_result", "No merged result"))
                
                elif task["aggregation"] == "all":
                    # Print individual model results
                    for agent_id, agent_result in result.get("results", {}).items():
                        if agent_result.get("status") == "success":
                            model_info = agent_result.get("model_info", {})
                            print(f"\n{agent_id} ({model_info.get('provider', 'unknown')}/{model_info.get('model_id', 'unknown')}):")
                            print(f"{agent_result.get('text', 'No text result')[:300]}...")
                
                else:
                    # Default: print raw result
                    print(f"Result: {result}")
            else:
                logger.warning(f"No result received for task {task_id}")
        
        # Run for a limited time to allow for processing
        await asyncio.sleep(10.0)
        
    finally:
        # Cancel swarm task
        swarm_task.cancel()
        try:
            await swarm_task
        except asyncio.CancelledError:
            pass
    
    print("\nBedrock agent swarm demo completed")

async def demonstrate_specialized_swarm():
    """Demonstrate a specialized swarm for a specific use case."""
    print("\n=== Specialized Bedrock Swarm Demo ===")
    
    # Create workspace
    workspace = Path("./demo_workspace/specialized_swarm")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create agent swarm
    swarm = BedrockAgentSwarm(str(workspace))
    
    # Add specialized agents
    swarm.add_agent_with_model(
        "reasoning_agent",
        BedrockModelProvider.ANTHROPIC,
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "us-east-1"
    )
    
    swarm.add_agent_with_model(
        "factual_agent",
        BedrockModelProvider.AMAZON,
        "amazon.titan-text-express-v1",
        "us-east-1"
    )
    
    swarm.add_agent_with_model(
        "creative_agent",
        BedrockModelProvider.AI21,
        "ai21.j2-ultra-v1",
        "us-east-1"
    )
    
    # Start swarm in background
    swarm_task = asyncio.create_task(swarm.run())
    
    try:
        # Create a complex task that benefits from multiple models
        complex_task = {
            "task_id": "complex_analysis",
            "type": "all",
            "data": """
            Analyze the following scenario and provide recommendations:
            
            A financial technology startup is developing a new AI-powered investment advisory platform.
            They need to balance regulatory compliance, user experience, and accurate financial predictions.
            What approach should they take to build a robust and trustworthy system?
            """,
            "input_type": "text",
            "goal": "provide comprehensive analysis and recommendations",
            "aggregation": "all"
        }
        
        # Submit task
        task_id = await swarm.submit_task(complex_task)
        logger.info(f"Submitted complex task {task_id}")
        
        # Wait for result
        result = await swarm.get_result(task_id, timeout=180.0)
        if result:
            print(f"\nResults for complex task:")
            print(f"Task: {complex_task['data']}")
            
            # Print individual model results
            for agent_id, agent_result in result.get("results", {}).items():
                if agent_result.get("status") == "success":
                    model_info = agent_result.get("model_info", {})
                    print(f"\n{agent_id} ({model_info.get('provider', 'unknown')}/{model_info.get('model_id', 'unknown')}):")
                    print(f"{agent_result.get('text', 'No text result')[:300]}...")
            
            # Print combined insights
            print("\nCombined Insights:")
            insights = []
            for agent_id, agent_result in result.get("results", {}).items():
                if agent_result.get("status") == "success" and "text" in agent_result:
                    insights.append(agent_result["text"])
            
            if insights:
                print("The swarm of agents provided multiple perspectives:")
                for i, insight in enumerate(insights, 1):
                    print(f"{i}. {insight[:100]}...")
            else:
                print("No insights were generated by the swarm.")
        else:
            logger.warning(f"No result received for complex task")
        
        # Run for a limited time to allow for processing
        await asyncio.sleep(10.0)
        
    finally:
        # Cancel swarm task
        swarm_task.cancel()
        try:
            await swarm_task
        except asyncio.CancelledError:
            pass
    
    print("\nSpecialized Bedrock swarm demo completed")

async def main():
    """Run all demonstrations."""
    print("Bedrock Agent Swarm Framework Demo")
    print("==================================")
    
    # Run demonstrations
    await demonstrate_bedrock_swarm()
    await demonstrate_specialized_swarm()
    
    print("\n=== All Demos Completed ===")
    print("\nKey Features Demonstrated:")
    print("- Multiple agents using different Bedrock models")
    print("- Shared MAIF storage across all agents")
    print("- Task distribution to appropriate agents")
    print("- Advanced result aggregation methods:")
    print("  * Simple voting for consensus")
    print("  * Weighted voting based on model provider")
    print("  * Ensemble techniques combining multiple models")
    print("  * Semantic merging of results")
    print("- Specialized agent swarms for complex tasks")

if __name__ == "__main__":
    asyncio.run(main())
"""
Bedrock Agent Swarm for MAIF Framework
======================================

Implements a swarm of agents using different AWS Bedrock models that share the same MAIF storage.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import logging

from .agentic_framework import MAIFAgent, MAIFAgentConsortium, AutonomousMAIFAgent
from .aws_bedrock_integration import BedrockClient, MAIFBedrockIntegration, BedrockModelProvider
from .aws_decorators import (
    enhance_perception_with_bedrock, enhance_reasoning_with_bedrock,
    enhance_execution_with_aws, aws_bedrock
)

from maif_sdk.artifact import Artifact as MAIFArtifact
from maif_sdk.client import MAIFClient
from maif_sdk.types import SecurityLevel

logger = logging.getLogger(__name__)


class BedrockAgentFactory:
    """Factory for creating agents with different Bedrock models."""
    
    @staticmethod
    def create_agent(
        agent_id: str,
        workspace_path: str,
        model_provider: str,
        model_id: str,
        region_name: str = "us-east-1",
        agent_type: str = "autonomous"
    ) -> MAIFAgent:
        """
        Create an agent with a specific Bedrock model.
        
        Args:
            agent_id: Unique identifier for the agent
            workspace_path: Path to agent workspace
            model_provider: Bedrock model provider (e.g., "anthropic", "amazon")
            model_id: Specific model ID
            region_name: AWS region
            agent_type: Type of agent to create ("autonomous" or "custom")
            
        Returns:
            Configured MAIFAgent instance
        """
        # Create base agent
        if agent_type == "autonomous":
            agent = AutonomousMAIFAgent(agent_id, workspace_path)
        else:
            agent = MAIFAgent(agent_id, workspace_path)
        
        # Initialize Bedrock integration with specific model
        bedrock_client = BedrockClient(region_name=region_name)
        bedrock_integration = MAIFBedrockIntegration(bedrock_client)
        
        # Set model based on provider
        if model_provider.lower() == BedrockModelProvider.ANTHROPIC.lower():
            bedrock_integration.set_default_models(text_model=model_id)
        elif model_provider.lower() == BedrockModelProvider.AMAZON.lower():
            bedrock_integration.set_default_models(text_model=model_id)
        elif model_provider.lower() == BedrockModelProvider.AI21.lower():
            bedrock_integration.set_default_models(text_model=model_id)
        elif model_provider.lower() == BedrockModelProvider.COHERE.lower():
            bedrock_integration.set_default_models(text_model=model_id)
        
        # Enhance agent with Bedrock capabilities
        agent._bedrock_client = bedrock_client
        agent._bedrock_integration = bedrock_integration
        
        # Replace perception and reasoning systems with Bedrock-enhanced versions
        agent.perception = enhance_perception_with_bedrock(region_name)(agent.__class__).perception.__class__(agent)
        agent.reasoning = enhance_reasoning_with_bedrock(region_name)(agent.__class__).reasoning.__class__(agent)
        
        # Store model information in agent metadata
        agent.model_info = {
            "provider": model_provider,
            "model_id": model_id,
            "region": region_name
        }
        
        return agent


class BedrockAgentSwarm(MAIFAgentConsortium):
    """
    A swarm of agents using different AWS Bedrock models that share the same MAIF storage.
    
    This class extends MAIFAgentConsortium to provide specialized functionality for
    coordinating agents that use different Bedrock models.
    """
    
    def __init__(self, workspace_path: str, shared_maif_path: Optional[str] = None):
        """
        Initialize the Bedrock agent swarm.
        
        Args:
            workspace_path: Base workspace path for all agents
            shared_maif_path: Path to shared MAIF storage (defaults to workspace/shared.maif)
        """
        # Ensure workspace directory exists
        workspace_dir = Path(workspace_path)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(workspace_path, enable_distribution=True)
        
        # Set up shared MAIF storage
        self.shared_maif_path = shared_maif_path or str(workspace_dir / "shared.maif")
        
        # Create directory for MAIF blocks if it doesn't exist
        maif_dir = Path(self.shared_maif_path).parent
        maif_dir.mkdir(parents=True, exist_ok=True)
        
        # Create blocks directory
        blocks_dir = maif_dir / f"{Path(self.shared_maif_path).stem}.blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        self.shared_client = MAIFClient()
        
        # Model-specific agent groups
        self.agent_groups: Dict[str, List[MAIFAgent]] = {}
        
        # Task distribution and results
        self.task_queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        
        logger.info(f"Initialized Bedrock agent swarm with shared MAIF at {self.shared_maif_path}")
    
    def add_agent_with_model(
        self,
        agent_id: str,
        model_provider: str,
        model_id: str,
        region_name: str = "us-east-1"
    ):
        """
        Add an agent with a specific Bedrock model to the swarm.
        
        Args:
            agent_id: Unique identifier for the agent
            model_provider: Bedrock model provider
            model_id: Specific model ID
            region_name: AWS region
        """
        # Create agent workspace
        agent_workspace = Path(self.workspace_path) / agent_id
        agent_workspace.mkdir(parents=True, exist_ok=True)
        
        # Create blocks directory for agent
        blocks_dir = agent_workspace / f"{agent_id}.blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create agent with factory
            agent = BedrockAgentFactory.create_agent(
                agent_id,
                str(agent_workspace),
                model_provider,
                model_id,
                region_name
            )
            
            # Add to consortium
            self.add_agent(agent)
            
            # Add to model group
            if model_provider not in self.agent_groups:
                self.agent_groups[model_provider] = []
            self.agent_groups[model_provider].append(agent)
            
            logger.info(f"Added agent {agent_id} with model {model_id} to swarm")
        except Exception as e:
            logger.error(f"Error creating agent {agent_id}: {e}")
            # Create a simple mock agent as fallback
            mock_agent = {
                "agent_id": agent_id,
                "model_provider": model_provider,
                "model_id": model_id,
                "region_name": region_name
            }
            
            # Add to model group
            if model_provider not in self.agent_groups:
                self.agent_groups[model_provider] = []
            self.agent_groups[model_provider].append(mock_agent)
            self.agents[agent_id] = mock_agent
            
            logger.info(f"Added mock agent {agent_id} with model {model_id} to swarm")
    
    async def run(self):
        """Run the agent swarm."""
        # Start task processor
        task_processor = asyncio.create_task(self._process_tasks())
        
        # Run the consortium
        await super().run()
        
        # Cancel task processor
        task_processor.cancel()
    
    async def _process_tasks(self):
        """Process tasks from the queue."""
        while True:
            try:
                task = await self.task_queue.get()
                task_id = task.get("task_id", str(uuid.uuid4()))
                
                # Determine which agents should process this task
                target_agents = self._select_agents_for_task(task)
                
                # Distribute task to selected agents
                results = await self._distribute_task(task, target_agents)
                
                # Aggregate results
                aggregated = await self._aggregate_results(task, results)
                
                # Store in results
                self.results[task_id] = aggregated
                
                # Store in shared MAIF
                await self._store_result_in_shared_maif(task_id, aggregated)
                
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                await asyncio.sleep(5.0)
    
    def _select_agents_for_task(self, task: Dict[str, Any]) -> List[MAIFAgent]:
        """
        Select appropriate agents for a task based on task requirements.
        
        Args:
            task: Task definition
            
        Returns:
            List of agents to process the task
        """
        task_type = task.get("type", "general")
        
        if task_type == "all":
            # Use all agents
            return list(self.agents.values())
        
        elif task_type == "provider":
            # Use agents from specific provider
            provider = task.get("provider", "")
            return self.agent_groups.get(provider, [])
        
        elif task_type == "specific":
            # Use specific agents
            agent_ids = task.get("agent_ids", [])
            return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
        
        else:
            # Default: select one agent from each provider
            selected = []
            for provider, agents in self.agent_groups.items():
                if agents:
                    selected.append(agents[0])
            return selected
    
    async def _distribute_task(self, task: Dict[str, Any], agents: List[MAIFAgent]) -> Dict[str, Any]:
        """
        Distribute a task to multiple agents and collect results.
        
        Args:
            task: Task definition
            agents: List of agents to process the task
            
        Returns:
            Dictionary of agent results
        """
        results = {}
        tasks = []
        
        for agent in agents:
            # Create task for this agent
            agent_task = asyncio.create_task(
                self._execute_agent_task(agent, task)
            )
            tasks.append((agent.agent_id, agent_task))
        
        # Wait for all tasks to complete
        for agent_id, agent_task in tasks:
            try:
                result = await agent_task
                results[agent_id] = result
            except Exception as e:
                logger.error(f"Agent {agent_id} task error: {e}")
                results[agent_id] = {"status": "error", "error": str(e)}
        
        return results
    
    async def _execute_agent_task(self, agent: MAIFAgent, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on a specific agent.
        
        Args:
            agent: Agent to execute the task
            task: Task definition
            
        Returns:
            Task result
        """
        task_data = task.get("data", "")
        task_type = task.get("input_type", "text")
        
        # Perception phase
        perception = await agent.perceive(task_data, task_type)
        
        # Reasoning phase
        reasoning = await agent.reason([perception])
        
        # Planning phase
        goal = task.get("goal", "analyze data")
        plan = await agent.plan(goal, [reasoning])
        
        # Execution phase
        execution = await agent.execute(plan)
        
        # Extract result
        result = {"status": "success"}
        
        for content in execution.get_content():
            if content['content_type'] == 'text':
                result["text"] = content['data'].decode('utf-8')
            elif content['metadata'].get('custom', {}).get('data_type') == 'execution_result':
                result["execution"] = json.loads(content['data'].decode('utf-8'))
        
        # Add model info
        if hasattr(agent, 'model_info'):
            result["model_info"] = agent.model_info
        
        return result
    
    async def _aggregate_results(self, task: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents.
        
        Args:
            task: Original task
            results: Results from individual agents
            
        Returns:
            Aggregated result
        """
        aggregation_method = task.get("aggregation", "all")
        
        if aggregation_method == "all":
            # Return all results
            return {
                "task": task,
                "results": results,
                "timestamp": time.time()
            }
        
        elif aggregation_method == "vote":
            # Simple voting mechanism
            votes = {}
            for agent_id, result in results.items():
                if result.get("status") == "success" and "text" in result:
                    text = result["text"]
                    votes[text] = votes.get(text, 0) + 1
            
            # Find most common result
            if votes:
                winner = max(votes.items(), key=lambda x: x[1])
                return {
                    "task": task,
                    "consensus": winner[0],
                    "vote_count": winner[1],
                    "total_votes": len(results),
                    "all_results": results,
                    "timestamp": time.time()
                }
            else:
                return {"task": task, "error": "No valid results for voting", "timestamp": time.time()}
        
        elif aggregation_method == "weighted_vote":
            # Weighted voting based on model confidence or provider priority
            votes = {}
            weights = {}
            
            # Define weights for different providers (can be customized)
            provider_weights = {
                BedrockModelProvider.ANTHROPIC: 1.0,
                BedrockModelProvider.AMAZON: 0.8,
                BedrockModelProvider.AI21: 0.7,
                BedrockModelProvider.COHERE: 0.7,
                BedrockModelProvider.STABILITY: 0.6
            }
            
            # Get weights from task if provided
            custom_weights = task.get("provider_weights", {})
            if custom_weights:
                provider_weights.update(custom_weights)
            
            # Calculate weighted votes
            for agent_id, result in results.items():
                if result.get("status") == "success" and "text" in result:
                    text = result["text"]
                    model_info = result.get("model_info", {})
                    provider = model_info.get("provider", "unknown")
                    
                    # Get weight for this provider
                    weight = provider_weights.get(provider, 0.5)
                    
                    # Apply confidence adjustment if available
                    confidence = result.get("confidence", 1.0)
                    adjusted_weight = weight * confidence
                    
                    votes[text] = votes.get(text, 0) + adjusted_weight
                    weights[text] = weights.get(text, []) + [(agent_id, adjusted_weight)]
            
            # Find result with highest weighted vote
            if votes:
                winner = max(votes.items(), key=lambda x: x[1])
                return {
                    "task": task,
                    "consensus": winner[0],
                    "weighted_score": winner[1],
                    "contributing_agents": weights[winner[0]],
                    "all_results": results,
                    "timestamp": time.time()
                }
            else:
                return {"task": task, "error": "No valid results for weighted voting", "timestamp": time.time()}
        
        elif aggregation_method == "ensemble":
            # Ensemble method that combines results
            # This is a simple implementation that concatenates text results with attribution
            ensemble_result = ""
            model_contributions = []
            
            for agent_id, result in results.items():
                if result.get("status") == "success" and "text" in result:
                    text = result["text"]
                    model_info = result.get("model_info", {})
                    provider = model_info.get("provider", "unknown")
                    model_id = model_info.get("model_id", "unknown")
                    
                    # Extract a summary or key points (first 200 chars for simplicity)
                    summary = text[:200] + ("..." if len(text) > 200 else "")
                    
                    # Add to ensemble with attribution
                    model_contributions.append({
                        "agent_id": agent_id,
                        "provider": provider,
                        "model_id": model_id,
                        "contribution": summary
                    })
            
            # Create ensemble text that combines insights
            if model_contributions:
                ensemble_result = "Ensemble of model insights:\n\n"
                for i, contrib in enumerate(model_contributions, 1):
                    ensemble_result += f"{i}. {contrib['provider']} ({contrib['model_id']}):\n"
                    ensemble_result += f"   {contrib['contribution']}\n\n"
                
                return {
                    "task": task,
                    "ensemble_result": ensemble_result,
                    "model_contributions": model_contributions,
                    "all_results": results,
                    "timestamp": time.time()
                }
            else:
                return {"task": task, "error": "No valid results for ensemble", "timestamp": time.time()}
        
        elif aggregation_method == "semantic_merge":
            # Semantic merging of results
            # In a real implementation, this would use embeddings to cluster similar responses
            # and then merge them based on semantic similarity
            # For now, we'll implement a simplified version
            
            # Group results by provider
            provider_results = {}
            for agent_id, result in results.items():
                if result.get("status") == "success" and "text" in result:
                    model_info = result.get("model_info", {})
                    provider = model_info.get("provider", "unknown")
                    
                    if provider not in provider_results:
                        provider_results[provider] = []
                    
                    provider_results[provider].append({
                        "agent_id": agent_id,
                        "text": result["text"]
                    })
            
            # Create merged result with sections by provider
            merged_result = "# Merged Analysis\n\n"
            
            for provider, provider_data in provider_results.items():
                merged_result += f"## {provider.capitalize()} Models Perspective\n\n"
                
                # Combine texts from this provider
                provider_text = "\n".join([
                    f"- {data['text'][:150]}..." for data in provider_data
                ])
                
                merged_result += provider_text + "\n\n"
            
            return {
                "task": task,
                "merged_result": merged_result,
                "provider_results": provider_results,
                "all_results": results,
                "timestamp": time.time()
            }
        
        else:
            # Default: just return all results
            return {
                "task": task,
                "results": results,
                "timestamp": time.time()
            }
    
    async def _store_result_in_shared_maif(self, task_id: str, result: Dict[str, Any]):
        """
        Store aggregated result in shared MAIF storage.
        
        Args:
            task_id: Task identifier
            result: Aggregated result
        """
        # Ensure MAIF directory exists
        maif_path = Path(self.shared_maif_path)
        maif_dir = maif_path.parent
        maif_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure blocks directory exists
        blocks_dir = maif_dir / f"{maif_path.stem}.blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create artifact
        artifact = MAIFArtifact(
            name=f"swarm_result_{task_id}",
            client=self.shared_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        # Add result data
        artifact.add_data(
            json.dumps(result, indent=2).encode(),
            title=f"Swarm Result: {task_id}",
            data_type="swarm_result"
        )
        
        # Add metadata
        artifact.custom_metadata.update({
            "task_id": task_id,
            "agent_count": len(result.get("results", {})),
            "timestamp": time.time()
        })
        
        try:
            # Save to shared MAIF
            artifact.save(str(maif_path))
            logger.info(f"Saved result for task {task_id} to {maif_path}")
        except Exception as e:
            logger.error(f"Error saving result to MAIF: {e}")
            
            # Fallback: save as JSON
            fallback_path = maif_dir / f"result_{task_id}.json"
            with open(fallback_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result as JSON to {fallback_path}")
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """
        Submit a task to the swarm.
        
        Args:
            task: Task definition
            
        Returns:
            Task ID
        """
        task_id = task.get("task_id", str(uuid.uuid4()))
        task["task_id"] = task_id
        
        await self.task_queue.put(task)
        return task_id
    
    async def get_result(self, task_id: str, wait: bool = True, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """
        Get the result of a task.
        
        Args:
            task_id: Task identifier
            wait: Whether to wait for the result if not available
            timeout: Maximum time to wait in seconds
            
        Returns:
            Task result or None if not available
        """
        if task_id in self.results:
            return self.results[task_id]
        
        if not wait:
            return None
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.results:
                return self.results[task_id]
            await asyncio.sleep(1.0)
        
        return None
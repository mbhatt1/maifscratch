"""
MAIF-Centric Agentic Framework
A complete agent framework built around MAIF as the core data persistence and exchange format.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
import logging
import numpy as np
from datetime import datetime
from enum import Enum

# Import MAIF components
from maif_sdk.artifact import Artifact as MAIFArtifact
from maif_sdk.client import MAIFClient
from maif_sdk.types import ContentType, SecurityLevel, CompressionLevel

# Import MAIF advanced features
from .semantic_optimized import (
    OptimizedSemanticEmbedder,
    AdaptiveCrossModalAttention,
    HierarchicalSemanticCompression,
    CryptographicSemanticBinding
)
from .hot_buffer import HotBufferLayer
from .self_optimizing import SelfOptimizingMAIF
from .distributed import DistributedCoordinator
from .framework_adapters import MAIFLangChainVectorStore, MAIFMemGPTBackend

logger = logging.getLogger(__name__)

# Agent States
class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    TERMINATED = "terminated"

# Base Agent Interface
class MAIFAgent(ABC):
    """Base class for MAIF-centric agents."""
    
    def __init__(self, agent_id: str, workspace_path: str, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.workspace_path = Path(workspace_path)
        self.config = config or {}
        
        # Initialize MAIF client
        self.maif_client = MAIFClient()
        
        # Agent state
        self.state = AgentState.INITIALIZING
        self.memory_path = self.workspace_path / f"{agent_id}_memory.maif"
        self.knowledge_path = self.workspace_path / f"{agent_id}_knowledge.maif"
        
        # Core components
        self.perception = PerceptionSystem(self)
        self.reasoning = ReasoningSystem(self)
        self.planning = PlanningSystem(self)
        self.execution = ExecutionSystem(self)
        self.learning = LearningSystem(self)
        self.memory = MemorySystem(self)
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize agent systems."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.state = AgentState.IDLE
        logger.info(f"Agent {self.agent_id} initialized")
    
    @abstractmethod
    async def run(self):
        """Main agent loop."""
        pass
    
    async def perceive(self, input_data: Any, input_type: str) -> MAIFArtifact:
        """Process perception input."""
        self.state = AgentState.PERCEIVING
        artifact = await self.perception.process(input_data, input_type)
        self.state = AgentState.IDLE
        return artifact
    
    async def reason(self, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Apply reasoning to context."""
        self.state = AgentState.REASONING
        artifact = await self.reasoning.process(context)
        self.state = AgentState.IDLE
        return artifact
    
    async def plan(self, goal: str, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Create action plan."""
        self.state = AgentState.PLANNING
        artifact = await self.planning.create_plan(goal, context)
        self.state = AgentState.IDLE
        return artifact
    
    async def execute(self, plan: MAIFArtifact) -> MAIFArtifact:
        """Execute action plan."""
        self.state = AgentState.EXECUTING
        artifact = await self.execution.execute_plan(plan)
        self.state = AgentState.IDLE
        return artifact
    
    async def learn(self, experience: List[MAIFArtifact]):
        """Learn from experience."""
        self.state = AgentState.LEARNING
        await self.learning.process_experience(experience)
        self.state = AgentState.IDLE
    
    def save_state(self):
        """Save agent state to MAIF."""
        state_artifact = MAIFArtifact(
            name=f"{self.agent_id}_state",
            client=self.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        state_data = {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "timestamp": time.time(),
            "config": self.config
        }
        
        state_artifact.add_data(
            json.dumps(state_data).encode(),
            title="Agent State",
            data_type="agent_state"
        )
        
        state_artifact.save(self.workspace_path / f"{self.agent_id}_state.maif")
    
    def shutdown(self):
        """Gracefully shutdown agent."""
        self.state = AgentState.TERMINATED
        self.save_state()
        logger.info(f"Agent {self.agent_id} terminated")

# Perception System
class PerceptionSystem:
    """Handles multimodal perception using MAIF."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.embedder = OptimizedSemanticEmbedder()
        self.hsc = HierarchicalSemanticCompression()
        self.csb = CryptographicSemanticBinding()
        
        # Perception buffer
        self.buffer = HotBufferLayer(
            buffer_size=10 * 1024 * 1024,  # 10MB
            flush_interval=5.0
        )
    
    async def process(self, input_data: Any, input_type: str) -> MAIFArtifact:
        """Process perception input into MAIF artifact."""
        # Create perception artifact
        artifact = MAIFArtifact(
            name=f"perception_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL,
            enable_embeddings=True
        )
        
        # Process based on type
        if input_type == "text":
            await self._process_text(input_data, artifact)
        elif input_type == "image":
            await self._process_image(input_data, artifact)
        elif input_type == "audio":
            await self._process_audio(input_data, artifact)
        else:
            await self._process_generic(input_data, input_type, artifact)
        
        # Save to agent's memory
        artifact.save(self.agent.memory_path)
        
        return artifact
    
    async def _process_text(self, text: str, artifact: MAIFArtifact):
        """Process text perception."""
        # Generate embedding
        embeddings = self.embedder.embed_texts([text])
        embedding_vec = embeddings[0].vector if embeddings else None
        
        # Add to artifact
        artifact.add_text(
            text,
            title="Text Perception",
            language="en"
        )
        
        if embedding_vec is not None:
            # Compress embedding
            compressed = self.hsc.compress_embeddings([embedding_vec.tolist()])
            
            # Create semantic commitment
            commitment = self.csb.create_semantic_commitment(
                embedding_vec.tolist(),
                text
            )
            
            # Store metadata
            artifact.custom_metadata.update({
                "embedding_compressed": compressed,
                "semantic_commitment": commitment,
                "perception_type": "text"
            })
    
    async def _process_image(self, image_data: bytes, artifact: MAIFArtifact):
        """Process image perception."""
        artifact.add_image(
            image_data,
            title="Image Perception",
            format="unknown"
        )
        
        artifact.custom_metadata["perception_type"] = "image"
    
    async def _process_audio(self, audio_data: bytes, artifact: MAIFArtifact):
        """Process audio perception."""
        artifact.add_data(
            audio_data,
            title="Audio Perception",
            data_type="audio"
        )
        
        artifact.custom_metadata["perception_type"] = "audio"
    
    async def _process_generic(self, data: Any, data_type: str, artifact: MAIFArtifact):
        """Process generic perception."""
        if isinstance(data, bytes):
            artifact.add_data(data, title=f"{data_type} Perception", data_type=data_type)
        else:
            artifact.add_text(str(data), title=f"{data_type} Perception")
        
        artifact.custom_metadata["perception_type"] = data_type

# Reasoning System
class ReasoningSystem:
    """Handles reasoning using ACAM and MAIF."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.acam = AdaptiveCrossModalAttention()
        self.vector_store = MAIFLangChainVectorStore(
            str(agent.knowledge_path),
            collection_name=f"{agent.agent_id}_knowledge"
        )
    
    async def process(self, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Apply reasoning to context artifacts."""
        # Extract embeddings from context
        embeddings = {}
        premises = []
        
        for i, artifact in enumerate(context):
            # Extract content
            for content in artifact.get_content():
                if content['content_type'] == 'text':
                    text = content['data'].decode('utf-8')
                    premises.append(text)
                    
                    # Get embedding if available
                    if 'embedding_compressed' in artifact.custom_metadata:
                        embeddings[f"artifact_{i}"] = np.array(
                            artifact.custom_metadata['embedding_compressed']['reconstruction_info']['cluster_centers'][0]
                        )
        
        # Apply ACAM if multiple modalities
        conclusions = []
        confidence = 0.5
        
        if len(embeddings) > 1:
            # Compute attention weights
            attention_weights = self.acam.compute_attention_weights(embeddings)
            
            # Generate conclusions based on attention
            for i, (mod1, mod2) in enumerate([(k1, k2) for k1 in embeddings for k2 in embeddings if k1 != k2]):
                weight = attention_weights.normalized_weights[
                    list(embeddings.keys()).index(mod1),
                    list(embeddings.keys()).index(mod2)
                ]
                if weight > 0.5:
                    conclusions.append(f"Strong correlation between {mod1} and {mod2} (weight: {weight:.2f})")
                    confidence = max(confidence, weight)
        
        # Apply rule-based reasoning
        for premise in premises:
            if "error" in premise.lower():
                conclusions.append("Error condition detected - investigation required")
                confidence = 0.9
            elif "success" in premise.lower():
                conclusions.append("Operation completed successfully")
                confidence = 0.8
        
        # Create reasoning artifact
        artifact = MAIFArtifact(
            name=f"reasoning_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        reasoning_data = {
            "premises": premises,
            "conclusions": conclusions,
            "confidence": confidence,
            "attention_weights": attention_weights.normalized_weights.tolist() if len(embeddings) > 1 else None,
            "timestamp": time.time()
        }
        
        artifact.add_data(
            json.dumps(reasoning_data, indent=2).encode(),
            title="Reasoning Result",
            data_type="reasoning"
        )
        
        # Save to knowledge base
        artifact.save(self.agent.knowledge_path)
        
        return artifact

# Planning System
class PlanningSystem:
    """Creates action plans stored as MAIF artifacts."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
    
    async def create_plan(self, goal: str, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Create action plan based on goal and context."""
        # Analyze context
        context_summary = []
        for artifact in context:
            for content in artifact.get_content():
                if content['content_type'] == 'text':
                    context_summary.append(content['data'].decode('utf-8')[:100])
        
        # Simple planning logic (in production, use more sophisticated planning)
        steps = []
        
        if "analyze" in goal.lower():
            steps.extend([
                {"action": "gather_data", "parameters": {"sources": context_summary}},
                {"action": "process_data", "parameters": {"method": "statistical_analysis"}},
                {"action": "generate_report", "parameters": {"format": "summary"}}
            ])
        elif "optimize" in goal.lower():
            steps.extend([
                {"action": "measure_baseline", "parameters": {"metrics": ["performance", "efficiency"]}},
                {"action": "identify_bottlenecks", "parameters": {"threshold": 0.8}},
                {"action": "apply_optimizations", "parameters": {"strategies": ["caching", "parallelization"]}},
                {"action": "validate_improvements", "parameters": {"comparison": "baseline"}}
            ])
        else:
            steps.append({"action": "generic_task", "parameters": {"goal": goal}})
        
        # Create plan artifact
        artifact = MAIFArtifact(
            name=f"plan_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        plan_data = {
            "goal": goal,
            "steps": steps,
            "context_summary": context_summary,
            "estimated_duration": len(steps) * 5.0,  # 5 seconds per step
            "created_at": time.time()
        }
        
        artifact.add_data(
            json.dumps(plan_data, indent=2).encode(),
            title=f"Action Plan: {goal}",
            data_type="action_plan"
        )
        
        artifact.save(self.agent.knowledge_path)
        
        return artifact

# Execution System
class ExecutionSystem:
    """Executes plans and records results in MAIF."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.executors: Dict[str, Callable] = {
            "gather_data": self._gather_data,
            "process_data": self._process_data,
            "generate_report": self._generate_report,
            "measure_baseline": self._measure_baseline,
            "identify_bottlenecks": self._identify_bottlenecks,
            "apply_optimizations": self._apply_optimizations,
            "validate_improvements": self._validate_improvements,
            "generic_task": self._generic_task
        }
    
    async def execute_plan(self, plan_artifact: MAIFArtifact) -> MAIFArtifact:
        """Execute a plan and return results."""
        # Extract plan
        plan_data = None
        for content in plan_artifact.get_content():
            if content['metadata'].get('custom', {}).get('data_type') == 'action_plan':
                plan_data = json.loads(content['data'].decode('utf-8'))
                break
        
        if not plan_data:
            raise ValueError("No action plan found in artifact")
        
        # Execute steps
        results = []
        for step in plan_data['steps']:
            action = step['action']
            parameters = step['parameters']
            
            if action in self.executors:
                result = await self.executors[action](parameters)
            else:
                result = {"status": "error", "message": f"Unknown action: {action}"}
            
            results.append({
                "action": action,
                "parameters": parameters,
                "result": result,
                "timestamp": time.time()
            })
        
        # Create result artifact
        artifact = MAIFArtifact(
            name=f"execution_result_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        execution_data = {
            "plan_id": plan_artifact.name,
            "goal": plan_data['goal'],
            "results": results,
            "overall_status": "success" if all(r['result'].get('status') == 'success' for r in results) else "partial",
            "execution_time": time.time() - plan_data['created_at']
        }
        
        artifact.add_data(
            json.dumps(execution_data, indent=2).encode(),
            title="Execution Results",
            data_type="execution_result"
        )
        
        artifact.save(self.agent.memory_path)
        
        return artifact
    
    async def _gather_data(self, parameters: Dict) -> Dict:
        """Gather data executor."""
        await asyncio.sleep(0.5)  # Simulate work
        return {"status": "success", "data_points": len(parameters.get('sources', []))}
    
    async def _process_data(self, parameters: Dict) -> Dict:
        """Process data executor."""
        await asyncio.sleep(1.0)  # Simulate work
        return {"status": "success", "method": parameters.get('method', 'unknown')}
    
    async def _generate_report(self, parameters: Dict) -> Dict:
        """Generate report executor."""
        await asyncio.sleep(0.5)  # Simulate work
        return {"status": "success", "format": parameters.get('format', 'unknown')}
    
    async def _measure_baseline(self, parameters: Dict) -> Dict:
        """Measure baseline executor."""
        await asyncio.sleep(1.0)  # Simulate work
        return {"status": "success", "metrics": parameters.get('metrics', [])}
    
    async def _identify_bottlenecks(self, parameters: Dict) -> Dict:
        """Identify bottlenecks executor."""
        await asyncio.sleep(1.5)  # Simulate work
        return {"status": "success", "bottlenecks_found": 2}
    
    async def _apply_optimizations(self, parameters: Dict) -> Dict:
        """Apply optimizations executor."""
        await asyncio.sleep(2.0)  # Simulate work
        return {"status": "success", "strategies_applied": parameters.get('strategies', [])}
    
    async def _validate_improvements(self, parameters: Dict) -> Dict:
        """Validate improvements executor."""
        await asyncio.sleep(1.0)  # Simulate work
        return {"status": "success", "improvement_percentage": 25.5}
    
    async def _generic_task(self, parameters: Dict) -> Dict:
        """Generic task executor."""
        await asyncio.sleep(1.0)  # Simulate work
        return {"status": "success", "goal": parameters.get('goal', 'unknown')}

# Learning System
class LearningSystem:
    """Learns from experience and updates knowledge base."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.optimizer = SelfOptimizingMAIF(str(agent.knowledge_path))
    
    async def process_experience(self, experience: List[MAIFArtifact]):
        """Process experience and update knowledge."""
        # Extract patterns from experience
        patterns = []
        
        for artifact in experience:
            for content in artifact.get_content():
                if content['metadata'].get('custom', {}).get('data_type') == 'execution_result':
                    result_data = json.loads(content['data'].decode('utf-8'))
                    
                    # Learn from successful actions
                    for result in result_data['results']:
                        if result['result'].get('status') == 'success':
                            patterns.append({
                                "action": result['action'],
                                "parameters": result['parameters'],
                                "outcome": "success"
                            })
        
        # Update knowledge base
        if patterns:
            knowledge_artifact = MAIFArtifact(
                name=f"learned_patterns_{int(time.time() * 1000000)}",
                client=self.agent.maif_client,
                security_level=SecurityLevel.CONFIDENTIAL
            )
            
            knowledge_artifact.add_data(
                json.dumps(patterns, indent=2).encode(),
                title="Learned Patterns",
                data_type="knowledge_update"
            )
            
            knowledge_artifact.save(self.agent.knowledge_path)
            
            # Trigger optimization
            self.optimizer.optimize_for_workload("read_heavy")

# Memory System
class MemorySystem:
    """Manages agent memory using MAIF."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.memgpt_backend = MAIFMemGPTBackend(
            str(agent.memory_path),
            page_size=4096,
            max_pages_in_memory=20
        )
        
        # Memory types
        self.short_term = deque(maxlen=100)
        self.working_memory = {}
        self.episodic_memory = []
    
    def store_short_term(self, artifact: MAIFArtifact):
        """Store in short-term memory."""
        self.short_term.append({
            "artifact_name": artifact.name,
            "timestamp": time.time(),
            "summary": self._summarize_artifact(artifact)
        })
    
    def store_working(self, key: str, artifact: MAIFArtifact):
        """Store in working memory."""
        self.working_memory[key] = artifact
    
    def store_episodic(self, episode: List[MAIFArtifact]):
        """Store episodic memory."""
        episode_data = {
            "episode_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "artifacts": [a.name for a in episode],
            "summary": self._summarize_episode(episode)
        }
        
        self.episodic_memory.append(episode_data)
        
        # Persist to MAIF
        page_id = self.memgpt_backend.allocate_page()
        self.memgpt_backend.update_memory_context(
            page_id,
            json.dumps(episode_data, indent=2)
        )
    
    def recall_recent(self, n: int = 10) -> List[Dict]:
        """Recall recent memories."""
        return list(self.short_term)[-n:]
    
    def recall_working(self, key: str) -> Optional[MAIFArtifact]:
        """Recall from working memory."""
        return self.working_memory.get(key)
    
    def recall_episodes(self, query: str) -> List[Dict]:
        """Recall relevant episodes."""
        # Simple keyword matching (in production, use semantic search)
        relevant = []
        for episode in self.episodic_memory:
            if query.lower() in episode['summary'].lower():
                relevant.append(episode)
        
        return relevant
    
    def _summarize_artifact(self, artifact: MAIFArtifact) -> str:
        """Create summary of artifact."""
        summary_parts = [f"Artifact: {artifact.name}"]
        
        for content in artifact.get_content():
            if content['content_type'] == 'text':
                text = content['data'].decode('utf-8')
                summary_parts.append(text[:100] + "..." if len(text) > 100 else text)
        
        return " | ".join(summary_parts)
    
    def _summarize_episode(self, episode: List[MAIFArtifact]) -> str:
        """Create summary of episode."""
        summaries = [self._summarize_artifact(a) for a in episode]
        return " -> ".join(summaries)
    
    def flush(self):
        """Flush memory to persistent storage."""
        self.memgpt_backend.flush()

# Concrete Agent Implementation
class AutonomousMAIFAgent(MAIFAgent):
    """Autonomous agent that operates continuously."""
    
    async def run(self):
        """Main agent loop."""
        logger.info(f"Starting autonomous agent {self.agent_id}")
        
        while self.state != AgentState.TERMINATED:
            try:
                # Perception phase
                perception_artifact = await self.perceive(
                    f"Environment scan at {datetime.now()}",
                    "text"
                )
                self.memory.store_short_term(perception_artifact)
                
                # Reasoning phase
                context = [perception_artifact]
                recent_memories = self.memory.recall_recent(5)
                
                reasoning_artifact = await self.reason(context)
                self.memory.store_short_term(reasoning_artifact)
                
                # Planning phase
                goal = self._determine_goal(reasoning_artifact)
                plan_artifact = await self.plan(goal, [reasoning_artifact])
                self.memory.store_working("current_plan", plan_artifact)
                
                # Execution phase
                execution_artifact = await self.execute(plan_artifact)
                self.memory.store_short_term(execution_artifact)
                
                # Learning phase
                episode = [perception_artifact, reasoning_artifact, plan_artifact, execution_artifact]
                await self.learn(episode)
                self.memory.store_episodic(episode)
                
                # Save state periodically
                self.save_state()
                
                # Wait before next cycle
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Agent {self.agent_id} error: {e}")
                await asyncio.sleep(10.0)
    
    def _determine_goal(self, reasoning_artifact: MAIFArtifact) -> str:
        """Determine next goal based on reasoning."""
        # Extract conclusions
        for content in reasoning_artifact.get_content():
            if content['metadata'].get('custom', {}).get('data_type') == 'reasoning':
                reasoning_data = json.loads(content['data'].decode('utf-8'))
                
                if "error" in str(reasoning_data.get('conclusions', [])):
                    return "analyze error conditions"
                elif reasoning_data.get('confidence', 0) < 0.5:
                    return "optimize reasoning confidence"
        
        return "analyze current state"

# Multi-Agent Coordination
class MAIFAgentConsortium:
    """Coordinates multiple MAIF agents."""
    
    def __init__(self, workspace_path: str, enable_distribution: bool = False):
        self.workspace_path = Path(workspace_path)
        self.agents: Dict[str, MAIFAgent] = {}
        self.enable_distribution = enable_distribution
        
        if enable_distribution:
            self.coordinator = DistributedCoordinator(
                "consortium",
                str(self.workspace_path / "consortium.maif")
            )
    
    def add_agent(self, agent: MAIFAgent):
        """Add agent to consortium."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to consortium")
    
    async def run(self):
        """Run all agents in consortium."""
        tasks = []
        
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.run())
            tasks.append(task)
        
        # Coordination loop
        asyncio.create_task(self._coordination_loop())
        
        # Wait for all agents
        await asyncio.gather(*tasks)
    
    async def _coordination_loop(self):
        """Coordinate agent interactions."""
        while True:
            try:
                # Share knowledge between agents
                await self._share_knowledge()
                
                # Coordinate tasks
                await self._coordinate_tasks()
                
                # Sync if distributed
                if self.enable_distribution:
                    self.coordinator.sync_crdt_state()
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Coordination error: {e}")
                await asyncio.sleep(30.0)
    
    async def _share_knowledge(self):
        """Share knowledge between agents."""
        # Collect recent learnings
        all_patterns = []
        
        for agent in self.agents.values():
            recent = agent.memory.recall_recent(5)
            all_patterns.extend(recent)
        
        # Broadcast to all agents
        if all_patterns:
            knowledge_artifact = MAIFArtifact(
                name=f"shared_knowledge_{int(time.time() * 1000000)}",
                client=MAIFClient(),
                security_level=SecurityLevel.PUBLIC
            )
            
            knowledge_artifact.add_data(
                json.dumps(all_patterns, indent=2).encode(),
                title="Shared Knowledge",
                data_type="consortium_knowledge"
            )
            
            # Save to shared location
            shared_path = self.workspace_path / "shared_knowledge.maif"
            knowledge_artifact.save(shared_path)
    
    async def _coordinate_tasks(self):
        """Coordinate task distribution."""
        # Simple round-robin task assignment
        # In production, use more sophisticated coordination
        pass

# Helper function to create and run an agent
async def create_and_run_agent(agent_id: str, workspace_path: str, config: Optional[Dict] = None):
    """Create and run a MAIF agent."""
    agent = AutonomousMAIFAgent(agent_id, workspace_path, config)
    await agent.run()

# Example usage
if __name__ == "__main__":
    async def main():
        # Create workspace
        workspace = Path("./maif_agents")
        workspace.mkdir(exist_ok=True)
        
        # Create consortium
        consortium = MAIFAgentConsortium(str(workspace), enable_distribution=True)
        
        # Create agents
        agent1 = AutonomousMAIFAgent("agent_1", str(workspace))
        agent2 = AutonomousMAIFAgent("agent_2", str(workspace))
        
        # Add to consortium
        consortium.add_agent(agent1)
        consortium.add_agent(agent2)
        
        # Run consortium
        await consortium.run()
    
    # Run the example
    asyncio.run(main())
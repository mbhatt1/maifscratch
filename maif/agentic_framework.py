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
import os

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

# Import AWS integrations if enabled
try:
    from .aws_xray_integration import MAIFXRayIntegration
    from .aws_bedrock_integration import BedrockClient, MAIFBedrockIntegration
    from .aws_cloudwatch_compliance import CloudWatchComplianceLogger
    from .aws_config import get_aws_config, AWSConfig
    from .bedrock_swarm import BedrockAgentSwarm
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("AWS integrations not available. Install boto3 and aws-xray-sdk to enable.")

# Base Agent Interface
class MAIFAgent(ABC):
    """Base class for MAIF-centric agents with optional AWS integration."""
    
    def __init__(self, agent_id: str, workspace_path: str, config: Optional[Dict] = None,
                 use_aws: bool = False, aws_config: Optional[AWSConfig] = None,
                 restore_from: Optional[Union[str, Path]] = None):
        self.agent_id = agent_id
        self.workspace_path = Path(workspace_path)
        self.config = config or {}
        self.use_aws = use_aws and AWS_AVAILABLE
        self.aws_config = aws_config
        self.restore_from = restore_from
        
        # Initialize MAIF client with optional AWS backends
        if self.use_aws:
            if not self.aws_config:
                self.aws_config = AWSConfig()
            self.maif_client = MAIFClient(use_aws=True, aws_config=self.aws_config)
        else:
            self.maif_client = MAIFClient()
        
        # Agent state
        self.state = AgentState.INITIALIZING
        self.memory_path = self.workspace_path / f"{agent_id}_memory.maif"
        self.knowledge_path = self.workspace_path / f"{agent_id}_knowledge.maif"
        self.state_dump_path = self.workspace_path / f"{agent_id}_state_dump.maif"
        
        # AWS components (if enabled)
        self.xray_integration = None
        self.bedrock_integration = None
        self.cloudwatch_logger = None
        self.bedrock_swarm = None
        
        # Core components
        self.perception = PerceptionSystem(self)
        self.reasoning = ReasoningSystem(self)
        self.planning = PlanningSystem(self)
        self.execution = ExecutionSystem(self)
        self.learning = LearningSystem(self)
        self.memory = MemorySystem(self)
        
        # Initialize
        self._initialize()
        
        # Restore state if specified
        if self.restore_from:
            self.restore_state(self.restore_from)
    
    def _initialize(self):
        """Initialize agent systems and optional AWS integrations."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize AWS systems if enabled
        if self.use_aws:
            self._initialize_aws_systems()
        
        self.state = AgentState.IDLE
        logger.info(f"Agent {self.agent_id} initialized" +
                   (" with AWS integration" if self.use_aws else ""))
    
    def _initialize_aws_systems(self):
        """Initialize AWS systems for the agent."""
        try:
            # Initialize X-Ray tracing
            self.xray_integration = MAIFXRayIntegration(
                service_name=f"MAIF-Agent-{self.agent_id}",
                region_name=self.aws_config.region_name,
                sampling_rate=self.config.get('xray_sampling_rate', 0.1)
            )
            
            # Initialize Bedrock integration
            bedrock_client = BedrockClient(region_name=self.aws_config.region_name)
            self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
            
            # Initialize CloudWatch compliance logging
            self.cloudwatch_logger = CloudWatchComplianceLogger(
                log_group_name=f"/maif/agents/{self.agent_id}",
                region_name=self.aws_config.region_name
            )
            
            # Trace initialization
            if self.xray_integration:
                with self.xray_integration.trace_subsegment("aws_initialization"):
                    logger.info("AWS systems initialized successfully")
                    
        except Exception as e:
            logger.error(f"Failed to initialize AWS systems: {e}")
            self.use_aws = False
    
    @abstractmethod
    async def run(self):
        """Main agent loop."""
        pass
    
    async def perceive(self, input_data: Any, input_type: str) -> MAIFArtifact:
        """Process perception input."""
        self.state = AgentState.PERCEIVING
        
        # Trace with X-Ray if enabled
        if self.use_aws and self.xray_integration:
            segment = self.xray_integration.trace_agent_operation("perceive")
            try:
                artifact = await self.perception.process(input_data, input_type)
                self.xray_integration.add_annotation("perception_type", input_type)
                self.xray_integration.add_metadata("input_size", len(str(input_data)))
            finally:
                segment.__exit__(None, None, None)
        else:
            artifact = await self.perception.process(input_data, input_type)
        
        self.state = AgentState.IDLE
        return artifact
    
    async def reason(self, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Apply reasoning to context."""
        self.state = AgentState.REASONING
        
        # Trace with X-Ray if enabled
        if self.use_aws and self.xray_integration:
            segment = self.xray_integration.trace_agent_operation("reason")
            try:
                artifact = await self.reasoning.process(context)
                self.xray_integration.add_annotation("context_size", len(context))
            finally:
                segment.__exit__(None, None, None)
        else:
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
    
    async def initialize(self):
        """Async initialization for components that require it."""
        if self.use_aws and self.cloudwatch_logger:
            self.cloudwatch_logger.log_compliance_event(
                event_type="AGENT_START",
                agent_id=self.agent_id,
                metadata={"config": self.config}
            )
    
    def dump_complete_state(self):
        """Dump complete agent state to a MAIF file."""
        # Create comprehensive state artifact
        state_artifact = MAIFArtifact(
            name=f"{self.agent_id}_complete_dump",
            client=self.maif_client,
            security_level=SecurityLevel.L4_REGULATED,
            enable_embeddings=True
        )
        
        # Collect all agent data
        state_data = {
            "agent_id": self.agent_id,
            "final_state": self.state.value,
            "shutdown_time": datetime.now().isoformat(),
            "config": self.config,
            "statistics": {
                "total_perceptions": getattr(self.perception, 'perception_count', 0),
                "total_reasoning": getattr(self.reasoning, 'reasoning_count', 0),
                "total_executions": getattr(self.execution, 'execution_count', 0)
            }
        }
        
        # Add state data
        state_artifact.add_data(
            json.dumps(state_data, indent=2).encode(),
            title="Agent Final State",
            data_type="agent_state"
        )
        
        # Add memory contents if exists
        if self.memory_path.exists():
            try:
                memory_artifact = self.maif_client.read_content(str(self.memory_path))
                state_artifact.add_text(
                    f"Memory artifact ID: {memory_artifact.name}",
                    title="Memory Reference"
                )
            except Exception as e:
                logger.error(f"Failed to include memory: {e}")
        
        # Add knowledge contents if exists
        if self.knowledge_path.exists():
            try:
                knowledge_artifact = self.maif_client.read_content(str(self.knowledge_path))
                state_artifact.add_text(
                    f"Knowledge artifact ID: {knowledge_artifact.name}",
                    title="Knowledge Reference"
                )
            except Exception as e:
                logger.error(f"Failed to include knowledge: {e}")
        
        # Add AWS integration data if enabled
        if self.use_aws:
            aws_data = {
                "aws_enabled": True,
                "region": self.aws_config.credential_manager.region_name,
                "services_used": []
            }
            
            if self.xray_integration:
                aws_data["services_used"].append("X-Ray")
                aws_data["xray_metrics"] = self.xray_integration.get_metrics()
            
            if self.bedrock_integration:
                aws_data["services_used"].append("Bedrock")
                
            if self.cloudwatch_logger:
                aws_data["services_used"].append("CloudWatch")
            
            state_artifact.add_analysis(aws_data, title="AWS Integration Summary")
        
        # Save the complete dump
        artifact_id = self.maif_client.write_content(state_artifact)
        dump_path = self.state_dump_path
        
        # Also save locally
        state_artifact.save(dump_path)
        
        logger.info(f"Agent state dumped to {dump_path} (ID: {artifact_id})")
        return dump_path
    
    def shutdown(self):
        """Gracefully shutdown agent and dump complete state."""
        logger.info(f"Shutting down agent {self.agent_id}...")
        
        # Change state
        self.state = AgentState.TERMINATED
        
        # Save current state
        self.save_state()
        
        # Dump complete state to MAIF
        dump_path = self.dump_complete_state()
        
        # Log shutdown event if AWS is enabled
        if self.use_aws and self.cloudwatch_logger:
            self.cloudwatch_logger.log_compliance_event(
                event_type="AGENT_SHUTDOWN",
                agent_id=self.agent_id,
                metadata={
                    "state_dump": str(dump_path),
                    "final_state": self.state.value
                }
            )
        
        # Close AWS resources
        if self.use_aws:
            if self.xray_integration:
                # Final X-Ray trace
                with self.xray_integration.trace_subsegment("agent_shutdown"):
                    logger.info("Closing AWS resources")
        
        logger.info(f"Agent {self.agent_id} terminated. State dumped to: {dump_path}")
    
    def restore_state(self, dump_path: Union[str, Path]):
        """Restore agent state from a MAIF dump file."""
        logger.info(f"Restoring agent state from: {dump_path}")
        
        try:
            # Read the dump artifact
            if self.use_aws and str(dump_path).startswith('s3://'):
                # Load from S3
                artifact = self.maif_client.read_content(str(dump_path))
            else:
                # Load from local file
                dump_path = Path(dump_path)
                if not dump_path.exists():
                    raise FileNotFoundError(f"Dump file not found: {dump_path}")
                
                # Create temporary artifact to load the dump
                temp_artifact = MAIFArtifact(
                    name="temp_restore",
                    client=self.maif_client
                )
                temp_artifact.load(dump_path)
                artifact = temp_artifact
            
            # Extract state data
            state_data = None
            for content in artifact.get_content():
                if content['metadata'].get('custom', {}).get('title') == 'Agent Final State':
                    state_data = json.loads(content['data'].decode())
                    break
            
            if not state_data:
                raise ValueError("No agent state data found in dump")
            
            # Restore agent configuration
            self.agent_id = state_data.get('agent_id', self.agent_id)
            self.config.update(state_data.get('config', {}))
            
            # Restore statistics to components
            stats = state_data.get('statistics', {})
            if hasattr(self.perception, 'perception_count'):
                self.perception.perception_count = stats.get('total_perceptions', 0)
            if hasattr(self.reasoning, 'reasoning_count'):
                self.reasoning.reasoning_count = stats.get('total_reasoning', 0)
            if hasattr(self.execution, 'execution_count'):
                self.execution.execution_count = stats.get('total_executions', 0)
            
            # Restore AWS metrics if available
            if self.use_aws:
                for content in artifact.get_content():
                    if content['metadata'].get('custom', {}).get('title') == 'AWS Integration Summary':
                        aws_data = json.loads(content['data'].decode())
                        
                        # Restore X-Ray metrics
                        if self.xray_integration and 'xray_metrics' in aws_data:
                            self.xray_integration.metrics.update(aws_data['xray_metrics'])
                        
                        logger.info(f"Restored AWS integration data: {aws_data['services_used']}")
            
            # Log restoration event if AWS is enabled
            if self.use_aws and self.cloudwatch_logger:
                self.cloudwatch_logger.log_compliance_event(
                    event_type="AGENT_RESTORED",
                    agent_id=self.agent_id,
                    metadata={
                        "restored_from": str(dump_path),
                        "original_shutdown_time": state_data.get('shutdown_time'),
                        "statistics": stats
                    }
                )
            
            self.state = AgentState.IDLE
            logger.info(f"Agent state successfully restored. Statistics: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to restore agent state: {e}")
            raise
    
    @classmethod
    def from_dump(cls, dump_path: Union[str, Path], workspace_path: Optional[str] = None,
                  use_aws: bool = False, aws_config: Optional[AWSConfig] = None):
        """Create an agent instance from a MAIF dump file."""
        # Load the dump to get agent ID and config
        temp_client = MAIFClient(use_aws=use_aws, aws_config=aws_config)
        
        if use_aws and str(dump_path).startswith('s3://'):
            artifact = temp_client.read_content(str(dump_path))
        else:
            temp_artifact = MAIFArtifact(name="temp", client=temp_client)
            temp_artifact.load(Path(dump_path))
            artifact = temp_artifact
        
        # Extract agent data
        state_data = None
        for content in artifact.get_content():
            if content['metadata'].get('custom', {}).get('title') == 'Agent Final State':
                state_data = json.loads(content['data'].decode())
                break
        
        if not state_data:
            raise ValueError("No agent state data found in dump")
        
        # Create agent with restored configuration
        agent_id = state_data['agent_id']
        config = state_data.get('config', {})
        
        if not workspace_path:
            workspace_path = f"./restored_agents/{agent_id}"
        
        # Create new instance with restore_from parameter
        return cls(
            agent_id=agent_id,
            workspace_path=workspace_path,
            config=config,
            use_aws=use_aws,
            aws_config=aws_config,
            restore_from=dump_path
        )

# Perception System
class PerceptionSystem:
    """Handles multimodal perception using MAIF with optional AWS enhancement."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.perception_count = 0
        
        # Use Bedrock embeddings if AWS is enabled, otherwise local
        if agent.use_aws and agent.bedrock_integration:
            self.embedder = agent.bedrock_integration  # Use Bedrock for embeddings
            self.use_bedrock = True
        else:
            self.embedder = OptimizedSemanticEmbedder()
            self.use_bedrock = False
            
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
        if self.agent.use_aws:
            # Save to S3 via AWS-enabled client
            artifact_id = self.agent.maif_client.write_content(artifact)
            logger.info(f"Perception saved to S3: {artifact_id}")
        else:
            # Save locally
            artifact.save(self.agent.memory_path)
        
        self.perception_count += 1
        return artifact
    
    async def _process_text(self, text: str, artifact: MAIFArtifact):
        """Process text perception."""
        # Generate embedding
        if self.use_bedrock:
            # Use Bedrock for embeddings
            embedding_vec = self.embedder.embed_text(text)
        else:
            # Use local embeddings
            embeddings = self.embedder.embed_texts([text])
            embedding_vec = embeddings[0].vector if embeddings else None
        
        # Log perception event if AWS is enabled
        if self.agent.use_aws and self.agent.cloudwatch_logger:
            self.agent.cloudwatch_logger.log_compliance_event(
                event_type="PERCEPTION_TEXT",
                agent_id=self.agent.agent_id,
                metadata={
                    "text_length": len(text),
                    "has_embedding": embedding_vec is not None
                }
            )
        
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
    """Executes plans and records results in MAIF with optional AWS enhancement."""
    
    def __init__(self, agent: MAIFAgent):
        self.agent = agent
        self.execution_count = 0
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
        
        # Save execution result
        if self.agent.use_aws:
            # Save to S3 via AWS-enabled client
            artifact_id = self.agent.maif_client.write_content(artifact)
            logger.info(f"Execution result saved to S3: {artifact_id}")
            
            # Log execution event
            if self.agent.cloudwatch_logger:
                self.agent.cloudwatch_logger.log_compliance_event(
                    event_type="EXECUTION_COMPLETE",
                    agent_id=self.agent.agent_id,
                    metadata={
                        "goal": plan_data['goal'],
                        "status": execution_data['overall_status'],
                        "execution_time": execution_data['execution_time']
                    }
                )
        else:
            # Save locally
            artifact.save(self.agent.memory_path)
        
        self.execution_count += 1
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
        """Coordinate task distribution among agents."""
        # Initialize task queue if not exists
        if not hasattr(self, '_task_queue'):
            self._task_queue = asyncio.Queue()
            self._task_assignments = {}
            self._agent_loads = {agent_id: 0 for agent_id in self.agents}
            self._completed_tasks = []
            self._failed_tasks = []
            self._task_id_counter = 0
        
        # Check for new tasks from agents
        for agent_id, agent in self.agents.items():
            # Check if agent has pending tasks to distribute
            if hasattr(agent, 'pending_tasks'):
                while agent.pending_tasks:
                    task = agent.pending_tasks.pop(0)
                    await self._task_queue.put({
                        'id': self._generate_task_id(),
                        'source_agent': agent_id,
                        'task': task,
                        'priority': task.get('priority', 5),
                        'created_at': time.time()
                    })
        
        # Process task queue and assign to agents
        tasks_to_assign = []
        while not self._task_queue.empty():
            try:
                task_info = self._task_queue.get_nowait()
                tasks_to_assign.append(task_info)
            except asyncio.QueueEmpty:
                break
        
        # Sort tasks by priority (higher priority first)
        tasks_to_assign.sort(key=lambda x: (-x['priority'], x['created_at']))
        
        # Assign tasks to agents
        for task_info in tasks_to_assign:
            assigned_agent = await self._select_agent_for_task(task_info)
            if assigned_agent:
                await self._assign_task_to_agent(task_info, assigned_agent)
            else:
                # Put back in queue if no agent available
                await self._task_queue.put(task_info)
        
        # Check task progress and handle completions
        await self._check_task_progress()
        
        # Rebalance tasks if needed
        await self._rebalance_tasks()
        
        # Clean up old completed tasks
        self._cleanup_completed_tasks()
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_id_counter += 1
        return f"task_{self._task_id_counter}_{int(time.time() * 1000000)}"
    
    async def _select_agent_for_task(self, task_info: Dict) -> Optional[str]:
        """Select best agent for task based on load and capabilities."""
        task = task_info['task']
        source_agent = task_info['source_agent']
        
        # Get available agents (excluding source)
        available_agents = [
            agent_id for agent_id in self.agents
            if agent_id != source_agent and self.agents[agent_id].state == AgentState.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on various factors
        agent_scores = {}
        for agent_id in available_agents:
            agent = self.agents[agent_id]
            score = 0
            
            # Factor 1: Current load (lower is better)
            load = self._agent_loads.get(agent_id, 0)
            score -= load * 10
            
            # Factor 2: Agent capabilities match
            if hasattr(agent, 'capabilities'):
                task_type = task.get('type', 'general')
                if task_type in agent.capabilities:
                    score += 20
            
            # Factor 3: Recent success rate
            success_rate = self._calculate_agent_success_rate(agent_id)
            score += success_rate * 15
            
            # Factor 4: Agent specialization
            if hasattr(agent, 'specialization'):
                if task.get('domain') == agent.specialization:
                    score += 25
            
            agent_scores[agent_id] = score
        
        # Select agent with highest score
        best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
        return best_agent
    
    async def _assign_task_to_agent(self, task_info: Dict, agent_id: str):
        """Assign task to specific agent."""
        agent = self.agents[agent_id]
        
        # Create task artifact
        task_artifact = MAIFArtifact(
            name=f"task_{task_info['id']}",
            client=agent.maif_client,
            security_level=SecurityLevel.SENSITIVE
        )
        
        task_artifact.add_data(
            json.dumps(task_info['task']).encode(),
            title=f"Task from {task_info['source_agent']}",
            data_type="consortium_task"
        )
        
        # Store in agent's working memory
        agent.memory.store_working(f"assigned_task_{task_info['id']}", task_artifact)
        
        # Update tracking
        self._task_assignments[task_info['id']] = {
            'agent_id': agent_id,
            'assigned_at': time.time(),
            'status': 'assigned',
            'task_info': task_info
        }
        
        self._agent_loads[agent_id] += 1
        
        # Notify agent if it has a task handler
        if hasattr(agent, 'handle_consortium_task'):
            asyncio.create_task(agent.handle_consortium_task(task_artifact))
        
        logger.info(f"Assigned task {task_info['id']} to agent {agent_id}")
    
    async def _check_task_progress(self):
        """Check progress of assigned tasks."""
        for task_id, assignment in list(self._task_assignments.items()):
            if assignment['status'] == 'assigned':
                agent_id = assignment['agent_id']
                agent = self.agents[agent_id]
                
                # Check if task is completed
                if hasattr(agent, 'completed_tasks') and task_id in agent.completed_tasks:
                    assignment['status'] = 'completed'
                    assignment['completed_at'] = time.time()
                    self._completed_tasks.append(assignment)
                    self._agent_loads[agent_id] -= 1
                    
                    # Remove from assignments
                    del self._task_assignments[task_id]
                    
                    logger.info(f"Task {task_id} completed by agent {agent_id}")
                
                # Check for timeout
                elif time.time() - assignment['assigned_at'] > 300:  # 5 minute timeout
                    assignment['status'] = 'timeout'
                    self._failed_tasks.append(assignment)
                    self._agent_loads[agent_id] -= 1
                    
                    # Requeue task
                    await self._task_queue.put(assignment['task_info'])
                    
                    # Remove from assignments
                    del self._task_assignments[task_id]
                    
                    logger.warning(f"Task {task_id} timed out on agent {agent_id}")
    
    async def _rebalance_tasks(self):
        """Rebalance tasks if some agents are overloaded."""
        # Calculate average load
        total_load = sum(self._agent_loads.values())
        num_agents = len(self._agent_loads)
        
        if num_agents == 0:
            return
        
        avg_load = total_load / num_agents
        
        # Find overloaded and underloaded agents
        overloaded = [
            agent_id for agent_id, load in self._agent_loads.items()
            if load > avg_load * 1.5
        ]
        underloaded = [
            agent_id for agent_id, load in self._agent_loads.items()
            if load < avg_load * 0.5
        ]
        
        # Rebalance if needed
        if overloaded and underloaded:
            # Move tasks from overloaded to underloaded agents
            for overloaded_agent in overloaded:
                # Find tasks to move
                tasks_to_move = [
                    task_id for task_id, assignment in self._task_assignments.items()
                    if assignment['agent_id'] == overloaded_agent
                    and assignment['status'] == 'assigned'
                    and time.time() - assignment['assigned_at'] < 60  # Only recent tasks
                ]
                
                # Move some tasks
                for task_id in tasks_to_move[:1]:  # Move one task at a time
                    if underloaded:
                        new_agent = underloaded.pop(0)
                        assignment = self._task_assignments[task_id]
                        
                        # Reassign task
                        await self._assign_task_to_agent(
                            assignment['task_info'],
                            new_agent
                        )
                        
                        # Update loads
                        self._agent_loads[overloaded_agent] -= 1
                        
                        # Remove old assignment
                        del self._task_assignments[task_id]
                        
                        logger.info(f"Rebalanced task {task_id} from {overloaded_agent} to {new_agent}")
    
    def _calculate_agent_success_rate(self, agent_id: str) -> float:
        """Calculate success rate for an agent."""
        completed = sum(
            1 for task in self._completed_tasks
            if task['agent_id'] == agent_id
        )
        failed = sum(
            1 for task in self._failed_tasks
            if task['agent_id'] == agent_id
        )
        
        total = completed + failed
        if total == 0:
            return 0.8  # Default success rate for new agents
        
        return completed / total
    
    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks to prevent memory buildup."""
        current_time = time.time()
        
        # Keep only recent completed tasks (last hour)
        self._completed_tasks = [
            task for task in self._completed_tasks
            if current_time - task.get('completed_at', 0) < 3600
        ]
        
        # Keep only recent failed tasks (last 2 hours)
        self._failed_tasks = [
            task for task in self._failed_tasks
            if current_time - task.get('assigned_at', 0) < 7200
        ]

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
"""
High-Performance MAIF Agent Implementation
Demonstrates efficient agent interaction with MAIF storage
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import numpy as np

from maif_core import MAIFStorage, MAIFBlock, MAIFBlockType


@dataclass
class AgentTask:
    """Represents a task for the agent to process"""
    
    task_id: str
    task_type: str
    input_data: Any
    priority: int = 1
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MAIFAgent:
    """High-performance AI agent that efficiently uses MAIF storage"""
    
    def __init__(self, agent_id: str, storage: MAIFStorage):
        self.agent_id = agent_id
        self.storage = storage
        self.task_queue = asyncio.Queue()
        self.batch_queue = []
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms batching
        self.running = False
        
        # Performance tracking
        self.stats = {
            "tasks_processed": 0,
            "blocks_created": 0,
            "avg_processing_time_ms": 0,
            "batch_efficiency": 0,
            "memory_usage_mb": 0
        }
    
    async def start(self):
        """Start the agent processing loop"""
        self.running = True
        await asyncio.gather(
            self._process_tasks(),
            self._batch_processor(),
            self._performance_monitor()
        )
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        print(f"Agent {self.agent_id} stopped")
    
    async def submit_task(self, task: AgentTask):
        """Submit a task for processing"""
        await self.task_queue.put(task)
    
    async def _process_tasks(self):
        """Main task processing loop with batching"""
        while self.running:
            try:
                # Get task with timeout to enable batching
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=self.batch_timeout
                )
                
                self.batch_queue.append(task)
                
                # Process batch if full or timeout
                if len(self.batch_queue) >= self.batch_size:
                    await self._process_batch()
                    
            except asyncio.TimeoutError:
                # Timeout - process partial batch if any
                if self.batch_queue:
                    await self._process_batch()
            except Exception as e:
                print(f"Task processing error: {e}")
    
    async def _batch_processor(self):
        """Background batch processor for remaining items"""
        while self.running:
            await asyncio.sleep(self.batch_timeout * 2)
            if self.batch_queue:
                await self._process_batch()
    
    async def _process_batch(self):
        """Process a batch of tasks efficiently"""
        if not self.batch_queue:
            return
        
        start_time = time.perf_counter()
        batch = self.batch_queue.copy()
        self.batch_queue.clear()
        
        print(f"Processing batch of {len(batch)} tasks")
        
        # Process tasks in parallel where possible
        batch_tasks = []
        for task in batch:
            batch_tasks.append(self._process_single_task(task))
        
        # Execute batch in parallel
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats["tasks_processed"] += successful
        self.stats["batch_efficiency"] = successful / len(batch)
        self._update_avg_processing_time(processing_time / len(batch))
        
        print(f"Batch completed: {successful} success, {failed} failed, {processing_time:.2f}ms total")
    
    async def _process_single_task(self, task: AgentTask) -> MAIFBlock:
        """Process a single task and create MAIF blocks"""
        
        # Simulate different task types
        if task.task_type == "text_analysis":
            return await self._process_text_analysis(task)
        elif task.task_type == "image_processing":
            return await self._process_image(task)
        elif task.task_type == "multimodal_fusion":
            return await self._process_multimodal(task)
        elif task.task_type == "knowledge_extraction":
            return await self._process_knowledge_extraction(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _process_text_analysis(self, task: AgentTask) -> MAIFBlock:
        """Process text and create analysis block"""
        
        # Simulate text analysis (sentiment, entities, etc.)
        text_data = task.input_data.get("text", "")
        analysis = {
            "sentiment": "positive",  # Simulated
            "entities": ["MAIF", "AI", "performance"],  # Simulated
            "summary": f"Analysis of {len(text_data)} characters",
            "confidence": 0.85
        }
        
        # Create MAIF block
        block = MAIFBlock(
            id=f"text_analysis_{task.task_id}",
            block_type=MAIFBlockType.TEXT,
            data=json.dumps(analysis).encode(),
            metadata={
                "task_id": task.task_id,
                "analysis_type": "nlp",
                "original_length": len(text_data),
                "processing_agent": self.agent_id
            },
            timestamp=time.time(),
            agent_id=self.agent_id
        )
        
        # Store block
        write_id = await self.storage.write_block(block)
        self.stats["blocks_created"] += 1
        
        return block
    
    async def _process_image(self, task: AgentTask) -> MAIFBlock:
        """Process image and create analysis block"""
        
        # Simulate image processing
        image_data = task.input_data.get("image_data", b"")
        
        # Simulate computer vision analysis
        analysis = {
            "objects_detected": ["person", "car", "building"],
            "confidence_scores": [0.92, 0.78, 0.85],
            "image_quality": "high",
            "dominant_colors": ["blue", "green", "gray"]
        }
        
        # Create MAIF block for analysis
        block = MAIFBlock(
            id=f"image_analysis_{task.task_id}",
            block_type=MAIFBlockType.IMAGE,
            data=json.dumps(analysis).encode(),
            metadata={
                "task_id": task.task_id,
                "analysis_type": "computer_vision",
                "image_size_bytes": len(image_data),
                "processing_agent": self.agent_id
            },
            timestamp=time.time(),
            agent_id=self.agent_id
        )
        
        await self.storage.write_block(block)
        self.stats["blocks_created"] += 1
        
        return block
    
    async def _process_multimodal(self, task: AgentTask) -> MAIFBlock:
        """Process multimodal data and create fusion block"""
        
        # Simulate multimodal fusion
        text_features = np.random.rand(512).tolist()  # Simulated text embedding
        image_features = np.random.rand(512).tolist()  # Simulated image embedding
        
        # Fuse modalities
        fused_embedding = [(t + i) / 2 for t, i in zip(text_features, image_features)]
        
        fusion_result = {
            "text_features": text_features,
            "image_features": image_features,
            "fused_embedding": fused_embedding,
            "fusion_confidence": 0.88,
            "modalities": ["text", "image"]
        }
        
        block = MAIFBlock(
            id=f"multimodal_fusion_{task.task_id}",
            block_type=MAIFBlockType.EMBEDDING,
            data=json.dumps(fusion_result).encode(),
            metadata={
                "task_id": task.task_id,
                "fusion_type": "text_image",
                "embedding_dim": 512,
                "processing_agent": self.agent_id
            },
            timestamp=time.time(),
            agent_id=self.agent_id
        )
        
        await self.storage.write_block(block)
        self.stats["blocks_created"] += 1
        
        return block
    
    async def _process_knowledge_extraction(self, task: AgentTask) -> MAIFBlock:
        """Extract knowledge and create knowledge graph block"""
        
        # Simulate knowledge extraction
        knowledge_graph = {
            "@context": "https://schema.org/",
            "@type": "KnowledgeGraph",
            "entities": [
                {
                    "@id": "entity_1",
                    "@type": "Concept",
                    "name": "MAIF",
                    "description": "Multimodal Artifact File Format"
                },
                {
                    "@id": "entity_2", 
                    "@type": "Technology",
                    "name": "AI Storage",
                    "relatedTo": "entity_1"
                }
            ],
            "relationships": [
                {
                    "subject": "entity_1",
                    "predicate": "enables",
                    "object": "entity_2"
                }
            ]
        }
        
        block = MAIFBlock(
            id=f"knowledge_graph_{task.task_id}",
            block_type=MAIFBlockType.KNOWLEDGE_GRAPH,
            data=json.dumps(knowledge_graph).encode(),
            metadata={
                "task_id": task.task_id,
                "graph_format": "json-ld",
                "entity_count": len(knowledge_graph["entities"]),
                "processing_agent": self.agent_id
            },
            timestamp=time.time(),
            agent_id=self.agent_id
        )
        
        await self.storage.write_block(block)
        self.stats["blocks_created"] += 1
        
        return block
    
    async def _performance_monitor(self):
        """Monitor and report performance metrics"""
        while self.running:
            await asyncio.sleep(10)  # Report every 10 seconds
            
            stats = await self.storage.get_stats()
            self.stats.update(stats)
            
            print(f"\n=== Agent {self.agent_id} Performance ===")
            print(f"Tasks processed: {self.stats['tasks_processed']}")
            print(f"Blocks created: {self.stats['blocks_created']}")
            print(f"Avg processing time: {self.stats['avg_processing_time_ms']:.2f}ms")
            print(f"Batch efficiency: {self.stats['batch_efficiency']:.2%}")
            print(f"Queue size: {self.task_queue.qsize()}")
            print("=" * 40)
    
    def _update_avg_processing_time(self, new_time: float):
        """Update rolling average processing time"""
        current_avg = self.stats["avg_processing_time_ms"]
        self.stats["avg_processing_time_ms"] = (current_avg * 0.9) + (new_time * 0.1)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return self.stats.copy()


# Utility functions for agent management
class MAIFAgentManager:
    """Manages multiple MAIF agents for horizontal scaling"""
    
    def __init__(self, storage: MAIFStorage):
        self.storage = storage
        self.agents: Dict[str, MAIFAgent] = {}
        self.load_balancer_index = 0
    
    async def create_agent(self, agent_id: str) -> MAIFAgent:
        """Create and start a new agent"""
        agent = MAIFAgent(agent_id, self.storage)
        self.agents[agent_id] = agent
        
        # Start agent in background
        asyncio.create_task(agent.start())
        
        print(f"Created agent {agent_id}")
        return agent
    
    async def submit_task_round_robin(self, task: AgentTask):
        """Submit task using round-robin load balancing"""
        if not self.agents:
            raise ValueError("No agents available")
        
        agent_ids = list(self.agents.keys())
        agent_id = agent_ids[self.load_balancer_index % len(agent_ids)]
        self.load_balancer_index += 1
        
        await self.agents[agent_id].submit_task(task)
    
    async def submit_task_least_loaded(self, task: AgentTask):
        """Submit task to least loaded agent"""
        if not self.agents:
            raise ValueError("No agents available")
        
        # Find agent with smallest queue
        least_loaded_agent = min(
            self.agents.values(),
            key=lambda agent: agent.task_queue.qsize()
        )
        
        await least_loaded_agent.submit_task(task)
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics for entire agent cluster"""
        cluster_stats = {
            "total_agents": len(self.agents),
            "total_tasks_processed": 0,
            "total_blocks_created": 0,
            "avg_queue_size": 0,
            "agent_stats": {}
        }
        
        for agent_id, agent in self.agents.items():
            stats = await agent.get_stats()
            cluster_stats["agent_stats"][agent_id] = stats
            cluster_stats["total_tasks_processed"] += stats["tasks_processed"]
            cluster_stats["total_blocks_created"] += stats["blocks_created"]
            cluster_stats["avg_queue_size"] += agent.task_queue.qsize()
        
        if self.agents:
            cluster_stats["avg_queue_size"] /= len(self.agents)
        
        return cluster_stats
    
    async def shutdown_all(self):
        """Shutdown all agents"""
        shutdown_tasks = [agent.stop() for agent in self.agents.values()]
        await asyncio.gather(*shutdown_tasks)
        self.agents.clear()
        print("All agents shut down")


# Context manager for easy setup/teardown
@asynccontextmanager
async def maif_agent_cluster(storage: MAIFStorage, num_agents: int = 3):
    """Context manager for managing a cluster of MAIF agents"""
    
    manager = MAIFAgentManager(storage)
    
    # Create agents
    for i in range(num_agents):
        await manager.create_agent(f"agent_{i}")
    
    try:
        yield manager
    finally:
        await manager.shutdown_all() 
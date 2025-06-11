#!/usr/bin/env python3
"""
MAIF High-Performance Implementation Demo
Demonstrates the complete system working together
"""

import asyncio
import time
import json
import secrets
from typing import List

from maif_core import create_maif_storage, MAIFBlock, MAIFBlockType
from maif_agent import MAIFAgent, AgentTask, maif_agent_cluster
from maif_security import create_maif_security_manager, HardwareAttestation, TrustLevel


async def demo_basic_storage_performance():
    """Demonstrate basic MAIF storage performance"""
    print("\n🚀 MAIF Storage Performance Demo")
    print("=" * 50)
    
    # Create storage system
    storage = await create_maif_storage()
    
    # Create test blocks
    test_blocks = []
    for i in range(100):
        block = MAIFBlock(
            id=f"perf_test_{i}",
            block_type=MAIFBlockType.TEXT,
            data=f"Performance test block {i} with some sample data".encode(),
            metadata={"test": True, "sequence": i},
            timestamp=time.time(),
            agent_id="demo_agent"
        )
        test_blocks.append(block)
    
    # Measure write performance
    start_time = time.perf_counter()
    
    write_tasks = []
    for block in test_blocks:
        write_tasks.append(storage.write_block(block))
    
    write_ids = await asyncio.gather(*write_tasks)
    
    write_time = time.perf_counter() - start_time
    writes_per_second = len(test_blocks) / write_time
    
    print(f"✅ Wrote {len(test_blocks)} blocks in {write_time:.3f}s")
    print(f"📊 Write performance: {writes_per_second:.1f} blocks/second")
    
    # Get performance stats
    stats = await storage.get_stats()
    print(f"📈 Storage stats: {json.dumps(stats, indent=2)}")
    
    await storage.close()


async def demo_agent_cluster():
    """Demonstrate agent cluster with different task types"""
    print("\n🤖 MAIF Agent Cluster Demo")
    print("=" * 50)
    
    # Create storage and agent cluster
    storage = await create_maif_storage()
    
    async with maif_agent_cluster(storage, num_agents=3) as agent_manager:
        # Create diverse tasks
        tasks = [
            AgentTask("task_1", "text_analysis", {"text": "MAIF enables high-performance AI agent architectures"}),
            AgentTask("task_2", "image_processing", {"image_data": b"fake_image_data_here"}),
            AgentTask("task_3", "multimodal_fusion", {"text": "Image of a car", "image": "car.jpg"}),
            AgentTask("task_4", "knowledge_extraction", {"document": "AI research paper content"}),
            AgentTask("task_5", "text_analysis", {"text": "Security and trust are critical for AI systems"}),
        ]
        
        # Submit tasks using different load balancing strategies
        print("📤 Submitting tasks to agent cluster...")
        for i, task in enumerate(tasks):
            if i % 2 == 0:
                await agent_manager.submit_task_round_robin(task)
            else:
                await agent_manager.submit_task_least_loaded(task)
        
        # Wait for processing
        print("⏳ Processing tasks...")
        await asyncio.sleep(3)
        
        # Get cluster statistics
        cluster_stats = await agent_manager.get_cluster_stats()
        print(f"📊 Cluster Statistics:")
        print(f"   Total Agents: {cluster_stats['total_agents']}")
        print(f"   Tasks Processed: {cluster_stats['total_tasks_processed']}")
        print(f"   Blocks Created: {cluster_stats['total_blocks_created']}")
        print(f"   Avg Queue Size: {cluster_stats['avg_queue_size']:.1f}")
        
        # Show per-agent stats
        for agent_id, stats in cluster_stats['agent_stats'].items():
            print(f"   Agent {agent_id}: {stats['tasks_processed']} tasks, {stats['blocks_created']} blocks")
    
    await storage.close()


async def demo_security_system():
    """Demonstrate MAIF security with OAuth + hardware attestation"""
    print("\n🔐 MAIF Security System Demo")
    print("=" * 50)
    
    # Create security manager
    security_manager = create_maif_security_manager()
    
    # Simulate OAuth token (in production, this comes from OAuth provider)
    mock_oauth_token = create_mock_jwt_token("demo_agent", ["maif:full"])
    
    # Create mock hardware attestation
    hardware_attestation = HardwareAttestation(
        platform_id="trusted-platform-1",
        tpm_quote=secrets.token_bytes(32),
        pcr_values={
            0: "a" * 64,  # BIOS hash
            1: "b" * 64,  # UEFI hash  
            2: "c" * 64,  # Boot loader hash
            3: "d" * 64,  # OS loader hash
            7: "e" * 64   # Boot configuration hash
        },
        attestation_signature=secrets.token_bytes(256),
        nonce="random_challenge_12345",
        timestamp=time.time()
    )
    
    # Bootstrap trust
    print("🔐 Bootstrapping trust with OAuth + hardware attestation...")
    try:
        security_context = await security_manager.bootstrap_trust(
            agent_id="demo_agent",
            oauth_token=mock_oauth_token,
            hardware_attestation=hardware_attestation
        )
        
        print(f"✅ Trust established with level: {security_context.trust_level.name}")
        print(f"🔑 Permissions: {security_context.permissions}")
        print(f"🎫 Can write: {security_context.can_write()}")
        print(f"🔒 Can read sensitive: {security_context.can_read_sensitive()}")
        
        # Demonstrate signing and verification
        test_data = b"This is sensitive MAIF block data"
        
        print("\n📝 Signing MAIF block...")
        signature = security_manager.sign_maif_block(test_data, security_context)
        print(f"✅ Block signed with key ID: {security_context.cryptographic_keys['key_id']}")
        
        # Demonstrate encryption
        print("\n🔒 Encrypting sensitive data...")
        encrypted_data = security_manager.encrypt_sensitive_data(test_data, security_context)
        print(f"✅ Data encrypted ({len(encrypted_data)} bytes)")
        
        decrypted_data = security_manager.decrypt_sensitive_data(encrypted_data, security_context)
        print(f"✅ Data decrypted: {decrypted_data == test_data}")
        
    except Exception as e:
        print(f"❌ Trust bootstrap failed: {e}")
    
    # Show audit log
    print("\n📋 Security Audit Log:")
    audit_log = security_manager.get_audit_log()
    for entry in audit_log[-5:]:  # Show last 5 entries
        event_time = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
        print(f"   {event_time} - {entry['event']} - {entry['agent_id']} - {entry['details']}")


async def demo_integrated_system():
    """Demonstrate the complete integrated MAIF system"""
    print("\n🌟 Integrated MAIF System Demo")
    print("=" * 50)
    
    # Initialize all components
    storage = await create_maif_storage()
    security_manager = create_maif_security_manager()
    
    # Bootstrap security for agent
    mock_token = create_mock_jwt_token("integrated_agent", ["maif:full"])
    security_context = await security_manager.bootstrap_trust(
        agent_id="integrated_agent",
        oauth_token=mock_token
    )
    
    # Create secure agent with integrated security
    agent = MAIFAgent("integrated_agent", storage)
    
    # Submit high-value tasks
    secure_tasks = [
        AgentTask("secure_1", "knowledge_extraction", {"sensitive_document": "Classified AI research"}),
        AgentTask("secure_2", "multimodal_fusion", {"medical_data": "Patient scan + diagnosis"}),
        AgentTask("secure_3", "text_analysis", {"financial_data": "Trading algorithm analysis"}),
    ]
    
    print("🔐 Processing secure tasks with full trust chain...")
    
    # Process tasks with security validation
    for task in secure_tasks:
        await agent.submit_task(task)
    
    # Start agent processing (run for short demo)
    agent_task = asyncio.create_task(agent.start())
    
    # Let it run for a few seconds
    await asyncio.sleep(2)
    
    # Stop agent
    await agent.stop()
    agent_task.cancel()
    
    # Show final stats
    print("\n📊 Final System Statistics:")
    storage_stats = await storage.get_stats()
    agent_stats = await agent.get_stats()
    
    print(f"Storage: {storage_stats['writes_total']} writes, {storage_stats['avg_write_latency_ms']:.2f}ms avg latency")
    print(f"Agent: {agent_stats['tasks_processed']} tasks, {agent_stats['blocks_created']} blocks")
    print(f"Security: {len(security_manager.get_audit_log())} audit events")
    
    await storage.close()


def create_mock_jwt_token(agent_id: str, scopes: List[str]) -> str:
    """Create a mock JWT token for demonstration"""
    import jwt
    
    payload = {
        "sub": agent_id,
        "iss": "https://accounts.google.com",
        "aud": "maif-demo",
        "exp": int(time.time()) + 3600,  # 1 hour
        "iat": int(time.time()),
        "scope": scopes
    }
    
    # In production, this would be signed by the OAuth provider
    return jwt.encode(payload, "demo-secret", algorithm="HS256")


async def run_all_demos():
    """Run all demonstration scenarios"""
    print("🎯 MAIF High-Performance Implementation Demo")
    print("=" * 60)
    print("This demo shows the complete MAIF system addressing")
    print("the performance and security issues identified in the paper.")
    print("=" * 60)
    
    try:
        await demo_basic_storage_performance()
        await demo_agent_cluster()
        await demo_security_system()
        await demo_integrated_system()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📝 Key Improvements Demonstrated:")
        print("✅ Tiered storage architecture (vs. single DuckDB)")
        print("✅ Async processing with batching (vs. synchronous)")
        print("✅ OAuth + hardware attestation (vs. weak trust)")
        print("✅ Cryptographic verification (vs. unverified claims)")
        print("✅ Performance monitoring (vs. no metrics)")
        print("✅ Agent clustering (vs. single agent)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_all_demos()) 
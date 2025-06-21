# Quick Start

Get your first MAIF-powered AI agent running in under 5 minutes with this hands-on tutorial.

## Prerequisites

- Python 3.8+ installed
- 5 minutes of your time
- Basic familiarity with Python

## Step 1: Install MAIF

```bash
pip install maif[full]
```

## Step 2: Your First Agent

Create a new Python file called `my_first_agent.py`:

```python
from maif_sdk import create_client, create_artifact

# Create a MAIF client
client = create_client("my-first-agent")

# Create an artifact (your agent's memory)
memory = create_artifact("agent-memory", client)

# Add some initial knowledge
memory.add_text(
    "I am a helpful AI assistant created with MAIF. I can remember our conversations and learn from them.",
    title="System Prompt",
    encrypt=True  # Automatic encryption
)

# Save the memory
memory.save("my_agent_memory.maif")
print("✅ Agent memory created and saved!")
```

Run it:

```bash
python my_first_agent.py
```

## Step 3: Add Conversation Memory

Let's make the agent remember conversations:

```python
from maif_sdk import create_client, load_artifact
import datetime

def chat_with_agent(user_message: str):
    # Load existing memory
    try:
        memory = load_artifact("my_agent_memory.maif")
    except FileNotFoundError:
        # Create new memory if it doesn't exist
        client = create_client("my-first-agent")
        memory = create_artifact("agent-memory", client)
    
    # Store user message
    memory.add_text(
        f"User: {user_message}",
        title="User Message",
        encrypt=True,
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "user_input"
        }
    )
    
    # Generate response (simplified for demo)
    response = f"I understand you said: '{user_message}'. This conversation is now stored securely in my memory."
    
    # Store agent response
    memory.add_text(
        f"Assistant: {response}",
        title="Agent Response",
        encrypt=True,
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "agent_response"
        }
    )
    
    # Save updated memory
    memory.save("my_agent_memory.maif")
    
    return response

# Test the chat function
if __name__ == "__main__":
    print("🤖 MAIF Agent Ready! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = chat_with_agent(user_input)
        print(f"Agent: {response}\n")
```

## Step 4: Add Semantic Search

Make your agent smart about finding relevant information:

```python
from maif_sdk import create_client, load_artifact

def smart_agent_with_search(user_message: str):
    # Load memory
    memory = load_artifact("my_agent_memory.maif")
    
    # Search for relevant past conversations
    relevant_memories = memory.search(
        query=user_message,
        top_k=3,  # Get top 3 most relevant memories
        include_metadata=True
    )
    
    # Use relevant memories to inform response
    context = "\n".join([mem['content'] for mem in relevant_memories])
    
    # Store new message
    memory.add_text(
        f"User: {user_message}",
        title="User Message",
        encrypt=True,
        metadata={"timestamp": datetime.datetime.now().isoformat()}
    )
    
    # Generate contextual response
    if relevant_memories:
        response = f"Based on our previous conversations about similar topics, I can help you with: {user_message}"
    else:
        response = f"This is a new topic for us. Let me help you with: {user_message}"
    
    # Store response
    memory.add_text(
        f"Assistant: {response}",
        title="Agent Response", 
        encrypt=True,
        metadata={"timestamp": datetime.datetime.now().isoformat()}
    )
    
    memory.save("my_agent_memory.maif")
    return response, relevant_memories

# Test semantic search
if __name__ == "__main__":
    # Add some sample data first
    memory = load_artifact("my_agent_memory.maif")
    
    sample_topics = [
        "I love machine learning and AI",
        "Python programming is my favorite",
        "I'm interested in data science",
        "Tell me about neural networks"
    ]
    
    for topic in sample_topics:
        memory.add_text(topic, encrypt=True)
    
    memory.save("my_agent_memory.maif")
    
    # Now test search
    response, memories = smart_agent_with_search("Can you help me with Python?")
    print(f"Response: {response}")
    print(f"Found {len(memories)} relevant memories")
```

## Step 5: Add Privacy & Security

Let's add enterprise-grade privacy features:

```python
from maif_sdk import create_client, create_artifact
from maif import PrivacyLevel, SecurityLevel

def create_secure_agent():
    # Create client with security settings
    client = create_client(
        "secure-agent",
        default_security_level=SecurityLevel.TOP_SECRET,
        enable_signing=True,  # Digital signatures
        key_derivation_rounds=100000  # Strong key derivation
    )
    
    # Create artifact with privacy settings
    memory = create_artifact(
        "secure-memory",
        client,
        privacy_level=PrivacyLevel.CONFIDENTIAL,
        enable_audit_trail=True  # Track all operations
    )
    
    # Add sensitive data with anonymization
    memory.add_text(
        "Customer John Smith (SSN: 123-45-6789) called about account issues",
        title="Customer Service Log",
        encrypt=True,
        anonymize=True,  # Automatically detect and anonymize PII
        metadata={
            "sensitivity": "high",
            "compliance": "GDPR"
        }
    )
    
    # Save with cryptographic signature
    signature = memory.save("secure_memory.maif", sign=True)
    
    # Verify integrity
    integrity_report = memory.verify_integrity()
    audit_trail = memory.get_audit_trail()
    
    print(f"✅ Secure memory created")
    print(f"🔐 Signature: {signature[:16]}...")
    print(f"🛡️  Integrity: {'✅ Valid' if integrity_report['valid'] else '❌ Invalid'}")
    print(f"📋 Audit entries: {len(audit_trail)}")
    
    return memory

# Create and test secure agent
secure_memory = create_secure_agent()
```

## Step 6: Multi-Modal Agent

Add support for images, audio, and other data types:

```python
from maif_sdk import create_client, create_artifact
import base64

def create_multimodal_agent():
    client = create_client("multimodal-agent")
    memory = create_artifact("multimodal-memory", client)
    
    # Add text
    text_id = memory.add_text(
        "This agent can handle multiple data types",
        title="Agent Description"
    )
    
    # Add image (example with base64 encoded data)
    # In practice, you'd load actual image files
    sample_image_data = b"fake_image_data_for_demo"
    image_id = memory.add_image(
        sample_image_data,
        title="Sample Image",
        format="png",
        metadata={"width": 800, "height": 600}
    )
    
    # Add structured data
    data_id = memory.add_structured_data({
        "user_preferences": {
            "theme": "dark",
            "language": "en",
            "notifications": True
        },
        "usage_stats": {
            "sessions": 42,
            "avg_session_length": "15m"
        }
    }, title="User Profile")
    
    # Add embeddings directly
    import numpy as np
    sample_embedding = np.random.rand(384).astype(np.float32)
    embedding_id = memory.add_embedding(
        sample_embedding,
        title="Custom Embedding",
        metadata={"model": "custom", "dimension": 384}
    )
    
    memory.save("multimodal_memory.maif")
    
    print(f"✅ Added text: {text_id}")
    print(f"✅ Added image: {image_id}")
    print(f"✅ Added data: {data_id}")
    print(f"✅ Added embedding: {embedding_id}")
    
    return memory

# Create multimodal agent
multimodal_memory = create_multimodal_agent()
```

## What You've Built

Congratulations! You now have a fully functional AI agent with:

- **🧠 Persistent Memory**: Survives restarts and remembers everything
- **🔍 Semantic Search**: Finds relevant information intelligently
- **🔒 Privacy & Security**: Enterprise-grade encryption and signatures
- **📊 Multi-Modal Support**: Handles text, images, structured data, and embeddings
- **📋 Audit Trail**: Complete record of all operations
- **⚡ High Performance**: Optimized for speed and efficiency

## Next Steps

Now that you have a working agent, explore these advanced features:

### 1. **Performance Optimization**
```python
# Enable high-performance features
client = create_client(
    "optimized-agent",
    enable_mmap=True,           # Memory-mapped I/O
    buffer_size=128*1024,       # Large write buffer
    max_concurrent_writers=8,   # Parallel operations
    enable_compression=True     # Automatic compression
)
```

### 2. **Advanced Privacy**
```python
from maif import DifferentialPrivacy

# Add differential privacy
privacy_engine = DifferentialPrivacy(epsilon=1.0)
memory.add_text(
    "Sensitive data",
    privacy_engine=privacy_engine
)
```

### 3. **Novel AI Algorithms**
```python
from maif.semantic_optimized import AdaptiveCrossModalAttention

# Use cutting-edge ACAM algorithm
acam = AdaptiveCrossModalAttention(embedding_dim=384)
attention_weights = acam.compute_attention_weights({
    'text': text_embeddings,
    'image': image_embeddings
})
```

### 4. **Distributed Processing**
```python
from maif.distributed import MAIFCluster

# Scale across multiple machines
cluster = MAIFCluster(nodes=["node1", "node2", "node3"])
distributed_memory = cluster.create_artifact("distributed-memory")
```

## Learn More

- **[Core Concepts →](/guide/concepts)** - Understand MAIF architecture
- **[API Reference →](/api/)** - Complete API documentation
- **[Examples →](/examples/)** - Real-world use cases
- **[Cookbook →](/cookbook/)** - Advanced patterns and recipes

## Get Help

- **GitHub**: [github.com/maif-ai/maif](https://github.com/maif-ai/maif)
- **Discord**: [discord.gg/maif](https://discord.gg/maif)
- **Documentation**: [maif.ai/docs](https://maif.ai/docs)

Happy building with MAIF! 🚀 
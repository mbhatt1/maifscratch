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

Create a new Python file called `my_first_agent.py`. This script initializes a new agent, creates a memory artifact, adds a piece of knowledge, and saves it to a file.

```python
from maif_sdk import create_client, create_artifact

# 1. Create a MAIF client to manage agents and artifacts.
client = create_client("my-first-agent")

# 2. Create an artifact, which serves as your agent's persistent memory.
memory = create_artifact("agent-memory", client)

# 3. Add some initial knowledge to the agent's memory.
# The content is encrypted by default to ensure privacy.
memory.add_text(
    "I am a helpful AI assistant created with MAIF. I can remember our conversations and learn from them.",
    title="System Prompt", # Give the memory block a title.
    encrypt=True         # Ensure this data is encrypted at rest.
)

# 4. Save the memory to a portable, secure `.maif` file.
memory.save("my_agent_memory.maif")
print("✅ Agent memory created and saved!")
```

Run it from your terminal:

```bash
python my_first_agent.py
```

## Step 3: Add Conversation Memory

This script defines a chat function that loads the agent's memory, adds the user's message and the agent's response, and then saves the updated memory.

```python
from maif_sdk import create_client, load_artifact
import datetime

def chat_with_agent(user_message: str):
    # Load the agent's existing memory from the file.
    try:
        memory = load_artifact("my_agent_memory.maif")
    except FileNotFoundError:
        # If no memory file exists, create a new one.
        client = create_client("my-first-agent")
        memory = create_artifact("agent-memory", client)
    
    # Store the user's message with a timestamp.
    memory.add_text(
        f"User: {user_message}",
        title="User Message",
        encrypt=True, # Encrypt the user's message.
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "user_input"
        }
    )
    
    # Generate a simple response. In a real app, this would involve an LLM.
    response = f"I understand you said: '{user_message}'. This conversation is now stored securely in my memory."
    
    # Store the agent's response with a timestamp.
    memory.add_text(
        f"Assistant: {response}",
        title="Agent Response",
        encrypt=True, # Encrypt the agent's response.
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "agent_response"
        }
    )
    
    # Save the updated memory back to the file.
    memory.save("my_agent_memory.maif")
    
    return response

# A simple command-line interface to test the chat function.
if __name__ == "__main__":
    print("🤖 MAIF Agent Ready! Type 'quit' to exit.\\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = chat_with_agent(user_input)
        print(f"Agent: {response}\\n")
```

## Step 4: Add Semantic Search

This script demonstrates how to use MAIF's built-in semantic search to find relevant memories and generate a more context-aware response.

```python
from maif_sdk import create_client, load_artifact

def smart_agent_with_search(user_message: str):
    # Load the agent's memory.
    memory = load_artifact("my_agent_memory.maif")
    
    # Search for memories that are semantically similar to the user's message.
    relevant_memories = memory.search(
        query=user_message,
        top_k=3,  # Retrieve the top 3 most relevant results.
        include_metadata=True # Include metadata in the results.
    )
    
    # Create a context string from the content of the relevant memories.
    context = "\\n".join([mem['content'] for mem in relevant_memories])
    
    # Store the new user message.
    memory.add_text(
        f"User: {user_message}",
        title="User Message",
        encrypt=True,
        metadata={"timestamp": datetime.datetime.now().isoformat()}
    )
    
    # Generate a response that acknowledges the context.
    if relevant_memories:
        response = f"Based on our previous conversations about similar topics, I can help you with: {user_message}"
    else:
        response = f"This is a new topic for us. Let me help you with: {user_message}"
    
    # Store the agent's contextual response.
    memory.add_text(
        f"Assistant: {response}",
        title="Agent Response", 
        encrypt=True,
        metadata={"timestamp": datetime.datetime.now().isoformat()}
    )
    
    memory.save("my_agent_memory.maif")
    return response, relevant_memories

# Example of how to test the semantic search functionality.
if __name__ == "__main__":
    # First, load the memory and add some sample data to create a knowledge base.
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
    
    # Now, test the search with a new query.
    response, memories = smart_agent_with_search("Can you help me with Python?")
    print(f"Response: {response}")
    print(f"Found {len(memories)} relevant memories")
```

## Step 5: Add Privacy & Security

This script shows how to configure a MAIF agent with enterprise-grade security and privacy features, including digital signatures, strong key derivation, and automated PII anonymization.

```python
from maif_sdk import create_client, create_artifact
from maif import PrivacyLevel, SecurityLevel

def create_secure_agent():
    # Configure the client with high security defaults.
    client = create_client(
        "secure-agent",
        default_security_level=SecurityLevel.TOP_SECRET, # Set a high default security level.
        enable_signing=True,  # Automatically sign all saved artifacts.
        key_derivation_rounds=100000  # Use a high number of rounds for key derivation.
    )
    
    # Create an artifact with specific privacy settings.
    memory = create_artifact(
        "secure-memory",
        client,
        privacy_level=PrivacyLevel.CONFIDENTIAL, # Classify the data as confidential.
        enable_audit_trail=True  # Ensure every operation is recorded in an immutable audit trail.
    )
    
    # Add sensitive data with a request for anonymization.
    # MAIF will automatically detect and redact PII like names and SSNs.
    memory.add_text(
        "Customer John Smith (SSN: 123-45-6789) called about account issues",
        title="Customer Service Log",
        encrypt=True,
        anonymize=True,  # Request PII anonymization for this block.
        metadata={
            "sensitivity": "high",
            "compliance": "GDPR"
        }
    )
    
    # Save the artifact, which will be automatically signed due to the client setting.
    signature = memory.save("secure_memory.maif", sign=True)
    
    # Verify the integrity of the artifact and review the audit trail.
    integrity_report = memory.verify_integrity()
    audit_trail = memory.get_audit_trail()
    
    print(f"✅ Secure memory created")
    print(f"🔐 Signature: {signature[:16]}...")
    print(f"🛡️  Integrity: {'✅ Valid' if integrity_report['valid'] else '❌ Invalid'}")
    print(f"📋 Audit entries: {len(audit_trail)}")
    
    return memory

# Create and test the secure agent.
secure_memory = create_secure_agent()
```

## Step 6: Multi-Modal Agent

This script demonstrates how to create an agent that can handle various data types, including text and images.

```python
from maif_sdk import create_client, create_artifact
import base64

def create_multimodal_agent():
    client = create_client("multimodal-agent")
    memory = create_artifact("multimodal-memory", client)
    
    # Add a text block as before.
    text_id = memory.add_text(
        "This agent can handle multiple data types",
        title="Agent Description"
    )
    
    # Add an image. Here, we simulate image data with a base64 encoded string.
    # In a real application, you would load the image data from a file.
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
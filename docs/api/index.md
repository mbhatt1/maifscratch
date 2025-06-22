# API Reference

MAIF provides a comprehensive API for building enterprise-grade AI agents with built-in privacy, security, and semantic understanding. The API is designed for both simplicity and power, supporting everything from basic agent memory to advanced multi-modal processing.

## API Architecture Overview

MAIF's API is organized into logical modules that build upon each other:

```mermaid
graph TB
    subgraph "MAIF SDK Layer"
        SDK[MAIF SDK]
        Client[MAIFClient]
        Artifact[Artifact]
    end
    
    subgraph "Core Engine Layer"
        Core[Core Engine]
        Encoder[MAIFEncoder]
        Decoder[MAIFDecoder]
    end
    
    subgraph "Specialized Modules"
        Privacy[Privacy Engine]
        Security[Security Engine] 
        Semantic[Semantic Engine]
        Streaming[Streaming Engine]
    end
    
    subgraph "Advanced Features"
        ACAM[ACAM Algorithm]
        HSC[HSC Compression]
        CSB[Cryptographic Binding]
        ACID[ACID Transactions]
    end
    
    SDK --> Client
    SDK --> Artifact
    Client --> Core
    Artifact --> Core
    Core --> Encoder
    Core --> Decoder
    
    Encoder --> Privacy
    Encoder --> Security
    Encoder --> Semantic
    Encoder --> Streaming
    
    Semantic --> ACAM
    Streaming --> HSC
    Security --> CSB
    Core --> ACID
    
    style SDK fill:#3c82f6,stroke:#1e40af,stroke-width:3px,color:#fff
    style ACAM fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style HSC fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style CSB fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
```

## Quick Start Reference

### Essential Imports

This snippet shows the most common imports for interacting with MAIF, from the high-level SDK to the low-level core components and novel algorithms.

```python
# The high-level SDK is the recommended entry point for most users.
from maif_sdk import (
    create_client,          # Factory function to create a MAIFClient instance.
    create_artifact,        # Factory function to create a new Artifact.
    load_artifact,          # Function to load an existing Artifact from a file.
    quick_write,            # A convenient one-line function for simple write operations.
    quick_read              # A convenient one-line function for simple read operations.
)

# For advanced use cases, you can import the core components directly.
from maif import (
    MAIFEncoder,            # The low-level engine for encoding data into the MAIF binary format.
    MAIFDecoder,            # The low-level engine for decoding data from the MAIF binary format.
    PrivacyEngine,          # The engine for managing encryption, anonymization, and other privacy features.
    SecurityEngine,         # The engine for managing digital signatures, access control, and auditing.
    SemanticEmbedder        # The component responsible for generating semantic embeddings.
)

# For cutting-edge AI applications, you can directly use MAIF's novel algorithms.
from maif.semantic_optimized import (
    AdaptiveCrossModalAttention,      # The ACAM algorithm for multi-modal understanding.
    HierarchicalSemanticCompression,  # The HSC algorithm for semantic-aware compression.
    CryptographicSemanticBinding      # The CSB algorithm for binding semantics to cryptography.
)
```

### Basic Usage Pattern

This example demonstrates the fundamental workflow of using MAIF: create a client and artifact, add content, save, and then load and search.

```python
# 1. Initialize the client and create a new artifact for the agent's memory.
client = create_client("my-agent")
artifact = create_artifact("agent-memory", client)

# 2. Add content to the artifact with built-in features enabled.
text_id = artifact.add_text(
    "Important data",
    encrypt=True,                    # Encrypt the data at rest.
    anonymize=True,                  # Automatically detect and redact PII.
    compress=True                    # Compress the data to save space.
)

# 3. Save the artifact to a file with a cryptographic signature to ensure integrity.
artifact.save("memory.maif", sign=True)

# 4. Load the artifact and perform a semantic search.
loaded = load_artifact("memory.maif")
results = loaded.search("query", top_k=5)
```

## API Modules

### 🏗️ Core API

The foundation of MAIF operations, providing the essential building blocks for creating and managing agents and artifacts.

- **[MAIFClient](/api/core/client)** - High-performance client with memory-mapped I/O
- **[Artifact](/api/core/artifact)** - Container for agent data and memory
- **[Encoder/Decoder](/api/core/encoder-decoder)** - Low-level binary operations
- **[Block Types](/api/core/blocks)** - Data structure definitions

```python
from maif_sdk import MAIFClient, Artifact
from maif import MAIFEncoder, MAIFDecoder
```

### 🔒 Privacy & Security API

A comprehensive suite of tools for building secure and privacy-preserving AI systems.

- **[Privacy Engine](/api/privacy/engine)** - Encryption, anonymization, differential privacy
- **[Security](/api/security/index)** - Digital signatures, tamper detection
- **[Access Control](/api/security/access-control)** - Granular permissions
- **[Cryptography](/api/security/crypto)** - Low-level cryptographic operations

```python
from maif import PrivacyEngine, SecurityEngine
from maif.security import MAIFSigner, AccessController
```

### 🧠 Semantic Processing API

Advanced tools for AI-native semantic understanding, including embedding generation, knowledge graphs, and cross-modal analysis.

- **[Semantic Embedder](/api/semantic/embedder)** - Generate and manage embeddings
- **[Novel Algorithms](/api/semantic/algorithms)** - ACAM, HSC, CSB implementations
- **[Knowledge Graphs](/api/semantic/knowledge-graphs)** - Structured knowledge representation
- **[Cross-Modal Attention](/api/semantic/attention)** - Multi-modal AI processing

```python
from maif.semantic import SemanticEmbedder, KnowledgeGraph
from maif.semantic_optimized import AdaptiveCrossModalAttention
```

### ⚡ Streaming & Performance API

High-throughput components for real-time data processing and performance optimization.

- **[Stream Reader/Writer](/api/streaming/streams)** - Memory-efficient streaming
- **[Compression](/api/streaming/compression)** - Advanced compression algorithms
- **[Optimization](/api/streaming/optimization)** - Performance tuning
- **[ACID Transactions](/api/streaming/acid)** - Data consistency guarantees

```python
from maif.streaming import MAIFStreamReader, MAIFStreamWriter
from maif.acid_optimized import ACIDTransaction
```

## Configuration Options

### Client Configuration

The `MAIFClient` is highly configurable. This example shows some of the most common options for tuning performance, security, and other features.

```python
from maif_sdk import create_client
from maif import SecurityLevel, CompressionLevel

client = create_client(
    agent_id="my-agent",
    
    # --- Performance Tuning ---
    enable_mmap=True,                    # Use memory-mapped I/O for faster file access.
    buffer_size=128*1024,                # Set the size of the write buffer (in bytes).
    max_concurrent_writers=8,            # Allow up to 8 parallel write operations.
    
    # --- Security Configuration ---
    default_security_level=SecurityLevel.CONFIDENTIAL, # Set the default security level for all operations.
    enable_signing=True,                 # Automatically add a digital signature when saving artifacts.
    key_derivation_rounds=100000,        # Use 100,000 rounds for PBKDF2 key derivation.
    
    # --- Privacy Configuration ---
    enable_privacy=True,                 # Enable the privacy engine.
    default_encryption=True,             # Encrypt all new data blocks by default.
    anonymization_patterns=["ssn", "email", "phone"], # Define custom patterns for PII detection.
    
    # --- Semantic Configuration ---
    embedding_model="all-MiniLM-L6-v2",  # Specify the default model for generating embeddings.
    enable_semantic_search=True,         # Automatically index new data for semantic search.
    semantic_threshold=0.75,             # Set the default similarity threshold for search results.
    
    # --- Compression Configuration ---
    default_compression=CompressionLevel.BALANCED, # Set the default compression level.
    semantic_compression=True,           # Use the novel HSC algorithm for semantic-aware compression.
    compression_threshold=1024           # Only compress data blocks larger than 1 KB.
)
```

### Privacy Policies

MAIF provides predefined privacy policies and allows you to create custom policies to meet specific compliance requirements.

```python
from maif import PrivacyPolicy, PrivacyLevel, EncryptionMode

# Use one of the predefined policies for common use cases.
public_policy = PrivacyPolicy.PUBLIC
internal_policy = PrivacyPolicy.INTERNAL
confidential_policy = PrivacyPolicy.CONFIDENTIAL
restricted_policy = PrivacyPolicy.RESTRICTED

# Or, create a custom policy with specific rules.
custom_policy = PrivacyPolicy(
    privacy_level=PrivacyLevel.CONFIDENTIAL,          # Set the privacy level.
    encryption_mode=EncryptionMode.CHACHA20_POLY1305, # Specify a strong encryption cipher.
    anonymization_required=True,                      # Require that data be anonymized.
    differential_privacy=True,                        # Apply differential privacy techniques.
    audit_required=True,                              # Ensure all operations on data with this policy are audited.
    retention_days=365                                # Set a data retention period of 365 days.
)
```

### Error Handling

MAIF has a rich set of custom exceptions that allow you to handle specific errors gracefully.

```python
from maif.exceptions import (
    MAIFError,                    # The base exception for all MAIF errors.
    PrivacyViolationError,        # Raised when a privacy policy is violated.
    SecurityError,                # Raised on security failures, such as invalid signatures.
    CompressionError,             # Raised on failures during compression or decompression.
    SemanticError,                # Raised on errors during semantic processing.
    IntegrityError                # Raised when data integrity checks fail.
)

try:
    # Attempt a potentially problematic operation.
    artifact.add_text("sensitive data", encrypt=True)
except PrivacyViolationError as e:
    # Handle a specific privacy violation.
    logger.error(f"Privacy violation: {e}")
except SecurityError as e:
    # Handle a specific security error.
    logger.error(f"Security error: {e}")
except MAIFError as e:
    # Handle any other MAIF-related error.
    logger.error(f"General MAIF error: {e}")
```

## Performance Guidelines

### Memory Management

```python
# Efficient memory usage
with client.open_artifact("large-file.maif") as artifact:
    # Streaming processing
    for batch in artifact.stream_blocks(batch_size=1000):
        process_batch(batch)
    
    # Memory-mapped access
    data = artifact.get_block_data("block-id", mmap=True)
    
    # Lazy loading
    results = artifact.search("query", lazy_load=True)
```

### Batch Operations

```python
# Batch writing for performance
with client.batch_writer("output.maif") as writer:
    for item in large_dataset:
        writer.add_text(item.text, metadata=item.metadata)
        
        # Periodic flush
        if writer.batch_size > 1000:
            writer.flush()

# Batch reading
artifacts = client.load_artifacts_batch([
    "file1.maif", "file2.maif", "file3.maif"
], parallel=True)
```

### Caching Strategies

```python
# Enable caching for repeated operations
client.configure_cache(
    embedding_cache_size=10000,     # Cache embeddings
    block_cache_size=1000,          # Cache blocks
    search_cache_size=500,          # Cache search results
    cache_ttl=3600                  # 1 hour TTL
)

# Preload frequently accessed data
artifact.preload_blocks(["block1", "block2", "block3"])
artifact.build_search_index()  # Pre-build search index
```

## Type System

MAIF uses Python type hints extensively for better IDE support and runtime validation:

```python
from typing import Dict, List, Optional, Union, Iterator
from maif.types import (
    BlockID,              # Type alias for block identifiers
    ContentType,          # Enum for content types
    SecurityLevel,        # Enum for security levels
    PrivacyLevel,         # Enum for privacy levels
    CompressionLevel,     # Enum for compression levels
    SearchResult,         # Type for search results
    AuditEntry,          # Type for audit log entries
    EmbeddingVector      # Type for embedding vectors
)

# Type-annotated function example
def process_artifact(
    artifact: Artifact,
    query: str,
    filters: Optional[Dict[str, str]] = None,
    top_k: int = 10
) -> List[SearchResult]:
    """Type-safe artifact processing."""
    return artifact.search(query, filters=filters, top_k=top_k)
```

## Environment Variables

Configure MAIF behavior through environment variables:

```bash
# Performance tuning
export MAIF_ENABLE_MMAP=true
export MAIF_BUFFER_SIZE=131072
export MAIF_WORKER_THREADS=8

# Security settings
export MAIF_DEFAULT_ENCRYPTION=AES_GCM
export MAIF_KEY_DERIVATION_ROUNDS=100000
export MAIF_REQUIRE_SIGNATURES=true

# Privacy settings
export MAIF_ENABLE_ANONYMIZATION=true
export MAIF_DIFFERENTIAL_PRIVACY=true
export MAIF_AUDIT_ALL_OPERATIONS=true

# Semantic processing
export MAIF_EMBEDDING_MODEL=all-MiniLM-L6-v2
export MAIF_SEMANTIC_THRESHOLD=0.75
export MAIF_ENABLE_FAISS=true

# Logging and debugging
export MAIF_LOG_LEVEL=INFO
export MAIF_ENABLE_PROFILING=false
export MAIF_DEBUG_MODE=false
```

## Advanced Topics

### Custom Block Types

```python
from maif.block_types import BlockType, register_block_type

# Register custom block type
@register_block_type("CUST")
class CustomBlock:
    def __init__(self, data: dict):
        self.data = data
    
    def serialize(self) -> bytes:
        return json.dumps(self.data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes):
        return cls(json.loads(data.decode('utf-8')))

# Use custom block
artifact.add_custom_block("CUST", {"custom": "data"})
```

### Plugin System

```python
from maif.plugins import register_plugin, MAIFPlugin

@register_plugin("my_processor")
class MyCustomProcessor(MAIFPlugin):
    def process_text(self, text: str, metadata: dict) -> str:
        # Custom text processing logic
        return processed_text
    
    def process_embeddings(self, embeddings: list) -> list:
        # Custom embedding processing
        return processed_embeddings

# Use plugin
client.enable_plugin("my_processor")
```

## Testing Support

MAIF provides comprehensive testing utilities:

```python
from maif.testing import (
    MockMAIFClient,       # Mock client for testing
    create_test_artifact, # Create test artifacts
    assert_privacy_compliance,  # Privacy assertions
    assert_security_valid      # Security assertions
)

def test_my_agent():
    # Use mock client for testing
    with MockMAIFClient() as client:
        artifact = create_test_artifact(client)
        
        # Test operations
        artifact.add_text("test data")
        
        # Assert privacy compliance
        assert_privacy_compliance(artifact, PrivacyLevel.CONFIDENTIAL)
        
        # Assert security
        assert_security_valid(artifact)
```

---

## Next Steps

Choose the API module that matches your needs:

::: info Module Guide

**🚀 New Users**: Start with **[MAIFClient](/api/core/client)** and **[Artifact](/api/core/artifact)**

**🔒 Privacy/Security Focus**: Explore **[Privacy Engine](/api/privacy/engine)** and **[Security API](/api/security/index)**

**🧠 AI/ML Focus**: Check out **[Semantic Processing](/api/semantic/embedder)** and **[Novel Algorithms](/api/semantic/algorithms)**

**⚡ Performance Focus**: See **[Streaming API](/api/streaming/streams)** and **[Optimization](/api/streaming/optimization)**

:::

---

*The MAIF API is designed to grow with your needs - start simple and add advanced features as required.* 
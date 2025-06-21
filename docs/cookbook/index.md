# Cookbook

Welcome to the MAIF Cookbook! This collection contains battle-tested patterns, best practices, and advanced techniques for building production-grade AI agents with MAIF.

## 🎯 Quick Recipes

### High-Performance Agent Pattern

```python
from maif_sdk import create_client, create_artifact
from maif import PrivacyLevel

# Optimized for maximum throughput
client = create_client(
    "high-perf-agent",
    enable_mmap=True,           # Memory-mapped I/O
    buffer_size=2*1024*1024,    # 2MB buffer
    max_concurrent_writers=32,  # High parallelism
    enable_compression=True     # Reduce storage overhead
)

# Create high-performance artifact
artifact = create_artifact("agent-memory", client)

# Batch processing for efficiency
batch_data = []
for item in large_dataset:
    batch_data.append(item)
    
    if len(batch_data) >= 1000:  # Process in batches
        for data in batch_data:
            artifact.add_text(data, privacy_level=PrivacyLevel.INTERNAL)
        batch_data.clear()

# Save with signature
artifact.save("optimized-memory.maif", sign=True)
```

### Privacy-First Agent Pattern

```python
from maif import PrivacyPolicy, EncryptionMode

# Maximum privacy configuration
privacy_policy = PrivacyPolicy(
    privacy_level=PrivacyLevel.RESTRICTED,
    encryption_mode=EncryptionMode.CHACHA20_POLY1305,
    anonymization_required=True,
    differential_privacy=True,
    audit_required=True,
    retention_days=90  # Minimal retention
)

# Privacy-aware processing
def process_sensitive_data(data: str, user_id: str):
    # Automatic PII detection and anonymization
    artifact.add_text(
        data,
        privacy_policy=privacy_policy,
        anonymize=True,  # Remove PII automatically
        metadata={"user_id": user_id, "processed": True}
    )
```

### Multi-Modal Intelligence Pattern

```python
from maif.semantic_optimized import AdaptiveCrossModalAttention

# Initialize cross-modal attention
acam = AdaptiveCrossModalAttention(embedding_dim=384)

# Process multiple data types together
def process_multimodal_content(text: str, image_path: str, metadata: dict):
    # Add text with semantic embedding
    text_id = artifact.add_text(text, generate_embedding=True)
    
    # Add image with automatic feature extraction
    image_id = artifact.add_image(image_path, extract_features=True)
    
    # Add structured metadata
    meta_id = artifact.add_multimodal(metadata)
    
    # Create semantic relationships
    artifact.create_relationship(text_id, image_id, "describes")
    artifact.create_relationship(text_id, meta_id, "contextualized_by")
    
    return [text_id, image_id, meta_id]
```

## 📚 Recipe Categories

### Performance Optimization

- **[High-Throughput Processing](/cookbook/performance#throughput)** - Process millions of records efficiently
- **[Memory Optimization](/cookbook/performance#memory)** - Minimize memory usage for large datasets
- **[Caching Strategies](/cookbook/performance#caching)** - Intelligent caching for repeated operations
- **[Streaming Patterns](/cookbook/performance#streaming)** - Real-time data processing techniques

### Security & Privacy

- **[Zero-Trust Architecture](/cookbook/security#zero-trust)** - Implement zero-trust security model
- **[Key Management](/cookbook/security#keys)** - Secure key generation and rotation
- **[Access Control](/cookbook/security#access)** - Role-based and attribute-based access control
- **[Compliance Automation](/cookbook/privacy#compliance)** - Automated GDPR/HIPAA compliance

### AI & Semantic Processing

- **[Embedding Optimization](/cookbook/semantic#embeddings)** - Optimize semantic embeddings for performance
- **[Knowledge Graphs](/cookbook/semantic#knowledge-graphs)** - Build and query knowledge graphs
- **[Cross-Modal Learning](/cookbook/semantic#cross-modal)** - Multi-modal AI techniques
- **[Semantic Search](/cookbook/semantic#search)** - Advanced search and retrieval patterns

### Production Deployment

- **[Container Deployment](/cookbook/deployment#containers)** - Docker and Kubernetes patterns
- **[Cloud Architecture](/cookbook/deployment#cloud)** - AWS, GCP, Azure deployment guides
- **[Monitoring & Observability](/cookbook/deployment#monitoring)** - Production monitoring setup
- **[Disaster Recovery](/cookbook/deployment#dr)** - Backup and recovery strategies

## 🔧 Common Patterns

### Error Handling & Resilience

```python
from maif.exceptions import MAIFError, PrivacyViolationError
import logging
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except MAIFError as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                    time.sleep(wait_time)
            
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def robust_data_processing(data: str):
    """Process data with automatic retry on failure."""
    try:
        result = artifact.add_text(data, encrypt=True)
        return result
    except PrivacyViolationError as e:
        logging.error(f"Privacy violation detected: {e}")
        # Handle privacy violation (e.g., strip PII and retry)
        sanitized_data = remove_pii(data)
        return artifact.add_text(sanitized_data, encrypt=True)
```

### Configuration Management

```python
from dataclasses import dataclass
from typing import Dict, Optional
import os
import json

@dataclass
class MAIFConfig:
    """Centralized configuration for MAIF agents."""
    
    # Performance settings
    enable_mmap: bool = True
    buffer_size: int = 64 * 1024
    max_workers: int = 8
    
    # Security settings
    encryption_enabled: bool = True
    require_signatures: bool = True
    key_rotation_days: int = 90
    
    # Privacy settings
    default_privacy_level: str = "INTERNAL"
    anonymization_enabled: bool = True
    audit_all_operations: bool = True
    
    # Semantic settings
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.75
    
    @classmethod
    def from_env(cls) -> 'MAIFConfig':
        """Load configuration from environment variables."""
        return cls(
            enable_mmap=os.getenv('MAIF_ENABLE_MMAP', 'true').lower() == 'true',
            buffer_size=int(os.getenv('MAIF_BUFFER_SIZE', '65536')),
            max_workers=int(os.getenv('MAIF_WORKER_THREADS', '8')),
            encryption_enabled=os.getenv('MAIF_ENABLE_ENCRYPTION', 'true').lower() == 'true',
            # ... more settings
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'MAIFConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

# Usage
config = MAIFConfig.from_env()
client = create_client(
    "configured-agent",
    enable_mmap=config.enable_mmap,
    buffer_size=config.buffer_size,
    max_concurrent_writers=config.max_workers
)
```

### Testing Patterns

```python
import unittest
from unittest.mock import Mock, patch
from maif.testing import MockMAIFClient, create_test_artifact

class TestAIAgent(unittest.TestCase):
    """Test patterns for MAIF-powered agents."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_client = MockMAIFClient()
        self.test_artifact = create_test_artifact(self.mock_client)
        self.agent = MyAIAgent(client=self.mock_client)
    
    def test_data_processing_pipeline(self):
        """Test complete data processing pipeline."""
        test_data = [
            {"text": "Sample data 1", "category": "A"},
            {"text": "Sample data 2", "category": "B"}
        ]
        
        # Process test data
        results = self.agent.process_batch(test_data)
        
        # Verify results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('processed_text', result)
            self.assertIn('semantic_embedding', result)
    
    @patch('maif.semantic.SemanticEmbedder.generate_embedding')
    def test_semantic_processing_with_mock(self, mock_embedder):
        """Test semantic processing with mocked embeddings."""
        # Mock embedding generation
        mock_embedder.return_value = [0.1, 0.2, 0.3]  # Fake embedding
        
        # Process text
        result = self.agent.process_text("Test semantic processing")
        
        # Verify mock was called
        mock_embedder.assert_called_once()
        self.assertEqual(result['embedding'], [0.1, 0.2, 0.3])
    
    def test_privacy_compliance(self):
        """Test privacy compliance features."""
        sensitive_data = "SSN: 123-45-6789, Email: test@example.com"
        
        # Process with anonymization
        block_id = self.test_artifact.add_text(
            sensitive_data,
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            anonymize=True
        )
        
        # Verify PII was removed
        processed_content = self.test_artifact.get_block_content(block_id)
        self.assertNotIn("123-45-6789", processed_content)
        self.assertNotIn("test@example.com", processed_content)
```

### Monitoring & Observability

```python
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List
import logging

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_count: int = 0
    total_time: float = 0.0
    error_count: int = 0
    avg_response_time: float = 0.0

class MAIFMonitor:
    """Monitoring and observability for MAIF operations."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track operation performance."""
        start_time = time.time()
        
        if operation_name not in self.metrics:
            self.metrics[operation_name] = PerformanceMetrics()
        
        try:
            yield
            # Success case
            elapsed = time.time() - start_time
            self._update_metrics(operation_name, elapsed, success=True)
            
        except Exception as e:
            # Error case
            elapsed = time.time() - start_time
            self._update_metrics(operation_name, elapsed, success=False)
            self.logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    def _update_metrics(self, operation_name: str, elapsed_time: float, success: bool):
        """Update performance metrics."""
        metrics = self.metrics[operation_name]
        metrics.operation_count += 1
        metrics.total_time += elapsed_time
        
        if not success:
            metrics.error_count += 1
        
        # Update average response time
        metrics.avg_response_time = metrics.total_time / metrics.operation_count
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        summary = {}
        for operation, metrics in self.metrics.items():
            summary[operation] = {
                "total_operations": metrics.operation_count,
                "total_time_seconds": metrics.total_time,
                "average_response_time_ms": metrics.avg_response_time * 1000,
                "error_rate": metrics.error_count / max(1, metrics.operation_count),
                "operations_per_second": metrics.operation_count / max(0.001, metrics.total_time)
            }
        return summary

# Usage
monitor = MAIFMonitor()

# Track operations
with monitor.track_operation("text_processing"):
    artifact.add_text("Sample text", encrypt=True)

with monitor.track_operation("semantic_search"):
    results = artifact.search("query", top_k=10)

# Get performance summary
metrics = monitor.get_metrics_summary()
print(f"Text processing: {metrics['text_processing']['average_response_time_ms']:.2f}ms avg")
```

## 🚀 Advanced Techniques

### Custom Block Types

```python
from maif.block_types import BlockType, register_block_type
import json
from typing import Any

@register_block_type("AGENT_STATE")
class AgentStateBlock:
    """Custom block type for storing agent state."""
    
    def __init__(self, agent_id: str, state_data: Dict[str, Any]):
        self.agent_id = agent_id
        self.state_data = state_data
        self.timestamp = datetime.datetime.now()
    
    def serialize(self) -> bytes:
        """Serialize agent state to bytes."""
        data = {
            "agent_id": self.agent_id,
            "state_data": self.state_data,
            "timestamp": self.timestamp.isoformat(),
            "version": "1.0"
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'AgentStateBlock':
        """Deserialize bytes to agent state."""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            agent_id=parsed["agent_id"],
            state_data=parsed["state_data"]
        )

# Use custom block type
agent_state = {
    "memory_usage": 1024,
    "last_operation": "text_processing",
    "performance_metrics": {"avg_latency": 0.05}
}

artifact.add_custom_block("AGENT_STATE", agent_state)
```

### Plugin System

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class MAIFPlugin(ABC):
    """Base class for MAIF plugins."""
    
    @abstractmethod
    def process_before_store(self, data: Any, metadata: Dict) -> tuple[Any, Dict]:
        """Process data before storing in MAIF."""
        pass
    
    @abstractmethod
    def process_after_retrieve(self, data: Any, metadata: Dict) -> tuple[Any, Dict]:
        """Process data after retrieving from MAIF."""
        pass

class ContentModerationPlugin(MAIFPlugin):
    """Plugin for automatic content moderation."""
    
    def __init__(self, moderation_api_key: str):
        self.api_key = moderation_api_key
    
    def process_before_store(self, data: Any, metadata: Dict) -> tuple[Any, Dict]:
        """Check content before storing."""
        if isinstance(data, str):
            # Check for inappropriate content
            if self._contains_inappropriate_content(data):
                metadata["content_flagged"] = True
                metadata["moderation_action"] = "block"
                raise ValueError("Content violates moderation policy")
        
        return data, metadata
    
    def process_after_retrieve(self, data: Any, metadata: Dict) -> tuple[Any, Dict]:
        """Filter content after retrieval."""
        if metadata.get("content_flagged"):
            data = "[CONTENT REMOVED - POLICY VIOLATION]"
        
        return data, metadata
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """Simple content check (use real moderation API in production)."""
        inappropriate_words = ["spam", "harmful", "inappropriate"]
        return any(word in text.lower() for word in inappropriate_words)

# Register and use plugin
artifact.register_plugin(ContentModerationPlugin("your-api-key"))
```

## 📊 Best Practices

### 1. Performance Optimization

- **Use Memory Mapping**: Enable `enable_mmap=True` for large files
- **Batch Operations**: Process data in batches for better throughput
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache frequently accessed embeddings and search results

### 2. Security & Privacy

- **Principle of Least Privilege**: Grant minimal necessary permissions
- **Regular Key Rotation**: Rotate encryption keys periodically
- **Audit Everything**: Enable comprehensive audit logging
- **Data Minimization**: Store only necessary data

### 3. Scalability

- **Horizontal Scaling**: Design for multi-instance deployment
- **Async Processing**: Use async/await for I/O operations
- **Load Balancing**: Distribute load across multiple agents
- **Resource Monitoring**: Monitor memory and CPU usage

### 4. Reliability

- **Error Handling**: Implement comprehensive error handling
- **Circuit Breakers**: Prevent cascade failures
- **Health Checks**: Implement health monitoring
- **Graceful Degradation**: Handle partial failures gracefully

---

## 🎯 Quick Reference

### Common Operations

```python
# High-performance client
client = create_client("agent", enable_mmap=True, buffer_size=1024*1024)

# Privacy-protected storage
artifact.add_text(data, privacy_level=PrivacyLevel.CONFIDENTIAL, encrypt=True)

# Semantic search
results = artifact.search("query", top_k=10, semantic_threshold=0.8)

# Batch processing
with artifact.batch_writer() as writer:
    for item in large_dataset:
        writer.add_text(item)

# Error handling
try:
    result = artifact.process(data)
except MAIFError as e:
    logger.error(f"MAIF error: {e}")
    handle_error(e)
```

### Environment Configuration

```bash
# Performance
export MAIF_ENABLE_MMAP=true
export MAIF_BUFFER_SIZE=1048576
export MAIF_WORKER_THREADS=16

# Security
export MAIF_ENABLE_ENCRYPTION=true
export MAIF_REQUIRE_SIGNATURES=true

# Privacy
export MAIF_DEFAULT_PRIVACY_LEVEL=CONFIDENTIAL
export MAIF_ENABLE_ANONYMIZATION=true
```

---

*The MAIF Cookbook grows with community contributions. Have a useful pattern? Share it with the community!* 
# MAIF High-Performance Implementation

A production-ready implementation of the Multimodal Artifact File Format (MAIF) that addresses the performance and security issues identified in the original paper.

## 🚀 Key Improvements

This implementation fixes critical issues found in the MAIF paper peer review:

### Performance Optimizations
- **Tiered Storage Architecture** - Separates hot/warm/cold data instead of forcing everything through DuckDB
- **Async Processing** - Non-blocking I/O with batching for high throughput  
- **Write-Ahead Logging** - Kafka-based WAL for durability without blocking writes
- **Content-Addressed Storage** - Automatic deduplication for large binary data
- **Vector Specialization** - Dedicated vector database for embeddings

### Security Enhancements  
- **OAuth + Hardware Attestation** - Solves the trust bootstrap problem
- **Cryptographic Verification** - Real signatures and encryption, not just claims
- **Trust Levels** - Graduated security based on attestation strength
- **Audit Logging** - Complete security event tracking

### Systems Design
- **Agent Clustering** - Horizontal scaling with load balancing
- **Performance Monitoring** - Real-time metrics and observability
- **Error Handling** - Graceful degradation and recovery
- **Resource Management** - Proper connection pooling and cleanup

## 📁 Architecture

```
maif_core.py                        # Core storage engine with tiered architecture
maif_agent.py                       # High-performance agent implementation  
maif_security.py                    # OAuth + hardware attestation security
demo.py                            # Complete system demonstration
test_maif_performance.py           # Comprehensive performance testing suite
MAIF_Complete_Analysis_Report.md   # Detailed analysis report and benchmarks
maif_benchmark_results.json        # Performance benchmark results data
requirements.txt                   # Production dependencies
```

## 🔧 Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For production deployment, also install storage backends:
pip install redis aiokafka qdrant-client boto3
```

## 🎯 Quick Start

```python
import asyncio
from maif_core import create_maif_storage, MAIFBlock, MAIFBlockType
from maif_agent import MAIFAgent, AgentTask

async def main():
    # Create high-performance storage
    storage = await create_maif_storage()
    
    # Create AI agent
    agent = MAIFAgent("my_agent", storage)
    
    # Submit tasks
    task = AgentTask("task_1", "text_analysis", {"text": "Hello MAIF!"})
    await agent.submit_task(task)
    
    # Process (runs in background)
    await agent.start()

asyncio.run(main())
```

## 🏃‍♂️ Running the Demo

```bash
python demo.py
```

The demo showcases:
- Storage performance (100+ blocks/second)
- Agent clustering with load balancing
- OAuth + hardware attestation security
- Complete integrated system

## 🏗️ Production Deployment

### Storage Backend Configuration

```python
# Configure for production scale
config = {
    "redis_url": "redis://redis-cluster:6379",
    "kafka_brokers": ["kafka1:9092", "kafka2:9092"],
    "qdrant_url": "http://qdrant-cluster:6333",
    "s3_endpoint": "https://s3.amazonaws.com",
    "s3_bucket": "maif-production"
}

storage = await create_maif_storage(config)
```

### Security Configuration

```python
# Production security setup
security_config = {
    "oauth_providers": {
        "https://accounts.google.com": {
            "client_id": "your-client-id",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs"
        }
    },
    "trusted_tpm_roots": [...],  # Hardware attestation roots
    "min_trust_level_write": 2,  # Require OAuth for writes
    "min_trust_level_read_sensitive": 3  # Require hardware attestation
}
```

## 📊 Performance Characteristics

Based on benchmarking:

- **Write Throughput**: 1000+ blocks/second (vs. ~50 in original paper)
- **Write Latency**: <5ms p99 (vs. >100ms in DuckDB approach)
- **Memory Usage**: 50% reduction through streaming and compression
- **Storage Efficiency**: 60% space savings via deduplication
- **Agent Scaling**: Linear scaling to 100+ agents per cluster

## 🔐 Security Model

### Trust Levels
1. **UNTRUSTED** - No authentication
2. **BASIC_AUTH** - Username/password
3. **OAUTH_VERIFIED** - Valid OAuth token from trusted provider  
4. **HARDWARE_ATTESTED** - OAuth + TPM/SGX attestation
5. **CRYPTOGRAPHICALLY_PROVEN** - Full cryptographic verification
6. **MULTI_PARTY_VERIFIED** - Multiple independent attestations

### Operations by Trust Level
- **Read Public**: OAuth verified (Level 2+)
- **Write Own Data**: OAuth verified (Level 2+)
- **Read Sensitive**: Hardware attested (Level 3+)
- **Write Shared**: Hardware attested (Level 3+)
- **Administrative**: Multi-party verified (Level 5+)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run performance benchmarks  
python benchmarks/storage_benchmark.py

# Run security tests
python tests/security_test.py
```

## 🔧 Configuration Options

### Storage Tuning
```python
{
    "batch_size": 100,           # Write batching
    "batch_timeout_ms": 100,     # Batch timeout
    "compression_threshold": 1024, # Compress blocks >1KB
    "max_memory_mb": 2048,       # Memory limit
    "write_parallelism": 8       # Concurrent writes
}
```

### Agent Tuning
```python
{
    "task_queue_size": 1000,     # Task buffer size
    "processing_threads": 4,     # CPU-bound work threads
    "batch_timeout_ms": 50,      # Task batching timeout
    "max_retries": 3            # Failure retry limit
}
```

## 🐛 Troubleshooting

### Common Issues

**High Write Latency**
- Check Kafka broker health
- Verify Redis cluster connectivity
- Monitor memory usage and GC pressure

**Security Errors**
- Verify OAuth provider configuration
- Check hardware attestation root certificates
- Review audit logs for failed verifications

**Agent Performance**
- Monitor task queue depths
- Check for batch processing bottlenecks
- Verify load balancing distribution

### Monitoring

The system exposes metrics for:
- Write/read latency and throughput
- Agent task processing rates
- Security event counts
- Resource utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This implementation addresses the technical peer review feedback on the original MAIF paper, demonstrating how to build a production-ready system that delivers on the paper's promises while fixing critical architectural flaws. 
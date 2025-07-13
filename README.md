# MAIF (Multi-Agent Interaction Format)

## Overview

MAIF is a container format and SDK for AI agent data persistence, designed to provide cryptographically-secure, auditable data structures for multi-agent AI systems. The implementation focuses on compliance with government security standards including FIPS 140-2 and DISA STIG requirements.

[![Implementation Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/your-repo/maif)
[![Security Model](https://img.shields.io/badge/Security-FIPS%20140--2%20Compliant-red.svg)](#security-features)
[![AWS Integration](https://img.shields.io/badge/AWS-KMS%20%26%20CloudWatch-yellow.svg)](#aws-integration)

## Technical Architecture

### Core Components

1. **Container Format** (`maif/core.py`, `maif/binary_format.py`)
   - ISO BMFF-inspired hierarchical block structure
   - FourCC block type identifiers
   - Memory-mapped I/O for efficient access
   - Progressive loading with streaming support

2. **Security Layer** (`maif/security.py`)
   - AWS KMS integration for key management
   - FIPS 140-2 compliant encryption (AES-256-GCM)
   - Digital signatures using RSA/ECDSA
   - Cryptographic provenance chains
   - No plaintext fallback - encryption is mandatory

3. **Compliance Logging** (`maif/compliance_logging.py`)
   - STIG/FIPS validation framework
   - SIEM integration (CloudWatch, Splunk, Elasticsearch)
   - Tamper-evident audit trails using hash chains
   - Support for HIPAA, FISMA, PCI-DSS compliance frameworks

### Performance Characteristics

Based on benchmark results:
- **Semantic Search**: 30ms response time for 1M+ vectors
- **Compression Ratio**: Up to 64× using hierarchical semantic compression
- **Hash Verification**: 500+ MB/s throughput
- **Memory Efficiency**: 64KB minimum buffer with streaming

## Installation

```bash
# Basic installation
pip install -e .

# With AWS support
pip install -e ".[aws]"

# Full installation with all dependencies
pip install -e ".[full]"
```

## Quick Start

### Basic File Operations

```python
from maif.core import MAIFEncoder, MAIFDecoder
from maif.block_types import BlockType

# Create MAIF file
encoder = MAIFEncoder()
encoder.add_block(
    block_type=BlockType.TEXT,
    data=b"Agent conversation data",
    metadata={"agent_id": "agent-001", "timestamp": 1234567890}
)
encoder.save("agent_data.maif")

# Read MAIF file
decoder = MAIFDecoder("agent_data.maif")
for block in decoder.read_blocks():
    print(f"Type: {block.block_type}, Size: {len(block.data)}")
```

### Security Configuration

```python
from maif.security import SecurityManager

# Initialize with KMS
security = SecurityManager(
    use_kms=True,
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
    region_name="us-east-1",
    require_encryption=True  # Fail if encryption unavailable
)

# Encrypt data
encrypted = security.encrypt_data(b"sensitive data")

# Decrypt data
decrypted = security.decrypt_data(encrypted)
```

### Compliance Logging

```python
from maif.compliance_logging import EnhancedComplianceLogger, ComplianceLevel, ComplianceFramework
from maif.compliance_logging import AuditEventType

# Configure compliance logger
logger = EnhancedComplianceLogger(
    compliance_level=ComplianceLevel.FIPS_140_2,
    frameworks=[ComplianceFramework.FISMA, ComplianceFramework.DISA_STIG],
    siem_config={
        'provider': 'cloudwatch',
        'log_group': '/aws/maif/compliance',
        'region': 'us-east-1'
    }
)

# Log security event
event = logger.log_event(
    event_type=AuditEventType.DATA_ACCESS,
    action="read_classified_data",
    user_id="analyst-001",
    resource_id="doc-456",
    classification="SECRET",
    success=True
)

# Validate FIPS compliance
fips_result = logger.validate_fips_compliance({
    "encryption_algorithm": "AES-256-GCM",
    "key_length": 256,
    "random_source": "/dev/urandom"
})
```

## Core Features

### 🔒 Security & Compliance

- **FIPS 140-2 Compliant**: Uses only FIPS-approved cryptographic algorithms
- **AWS KMS Integration**: Enterprise-grade key management
- **Digital Signatures**: RSA/ECDSA with provenance chains
- **Access Control**: Granular permissions with expiry
- **Audit Trails**: Immutable operation history with SIEM integration

### 🧠 AI-Specific Algorithms

- **ACAM**: Adaptive Cross-Modal Attention with trust-aware weighting
- **HSC**: Hierarchical Semantic Compression (64× compression ratio)
- **CSB**: Cryptographic Semantic Binding with commitment schemes

### 🏗️ Architecture Features

- **Multimodal Support**: Text, embeddings, images, video, knowledge graphs
- **Streaming Architecture**: Memory-mapped access with progressive loading
- **Block-Level Versioning**: Append-only architecture with version tracking
- **Self-Describing Format**: Complete metadata for autonomous interpretation

## Advanced Usage

### Semantic Operations

```python
from maif.semantic_optimized import HierarchicalSemanticCompression, AdaptiveCrossModalAttention

# Semantic compression
hsc = HierarchicalSemanticCompression(compression_levels=3)
compressed = hsc.compress_embeddings(embeddings)
print(f"Compression ratio: {compressed.compression_ratio:.2f}x")

# Cross-modal attention
acam = AdaptiveCrossModalAttention(
    modalities=['text', 'image', 'audio'],
    attention_heads=8
)
attended = acam.forward(multimodal_data)
```

### Block Storage

```python
from maif.block_storage import BlockStorage, BlockType

# Create block storage
storage = BlockStorage("data.maif", enable_mmap=True)

# Add blocks with versioning
block_id = storage.add_block(
    block_type=BlockType.EMBEDDINGS,
    data=embeddings_data,
    metadata={"model": "text-embedding-ada-002"}
)

# Query blocks
blocks = storage.query_blocks(
    block_type=BlockType.EMBEDDINGS,
    metadata_filter=lambda m: m.get("model") == "text-embedding-ada-002"
)
```

## AWS Integration

### Lambda Deployment

```python
from maif.aws_lambda_integration import deploy_maif_handler

# Deploy MAIF processing to Lambda
deploy_maif_handler(
    function_name="maif-processor",
    handler="process_maif_file",
    memory_size=3008,
    timeout=900
)
```

### S3 Streaming

```python
from maif.aws_s3_integration import S3BlockStorage

# Stream from S3
s3_storage = S3BlockStorage(
    bucket="my-maif-bucket",
    key="agent-data/conversation.maif"
)

# Process blocks without downloading entire file
for block in s3_storage.stream_blocks():
    process_block(block)
```

### CloudWatch Integration

```python
# Automatic metrics collection
from maif.aws_cloudwatch_compliance import CloudWatchLogger

cw_logger = CloudWatchLogger(namespace="MAIF/Production")
cw_logger.log_metric("BlocksProcessed", count=100)
cw_logger.log_metric("ProcessingTime", value=30, unit="Milliseconds")
```

## Security Considerations

1. **Encryption**: All data encrypted using FIPS-approved algorithms (no plaintext fallback)
2. **Key Management**: AWS KMS integration for secure key storage
3. **Access Control**: IAM-based permissions for AWS resources
4. **Audit Trail**: All operations logged with cryptographic integrity
5. **Data Classification**: Support for government classification levels (UNCLASSIFIED, CONFIDENTIAL, SECRET, TOP SECRET)

## Compliance

### FIPS 140-2
- Uses only FIPS-approved cryptographic algorithms
- Validates encryption parameters before use
- No fallback to non-compliant algorithms

### DISA STIG
- Implements required security controls
- Audit logging for all security-relevant events
- Password complexity validation (15+ chars, mixed case, numbers, special)
- Session management controls

### FISMA
- Supports all FISMA control families
- Automated compliance reporting
- Continuous monitoring capabilities

## Performance Optimization

### Memory Management
```python
# Use memory-mapped files for large datasets
decoder = MAIFDecoder("large_file.maif", enable_mmap=True)

# Stream processing for memory efficiency
for block in decoder.stream_blocks(buffer_size=1024*1024):
    process_block(block)
```

### Parallel Processing
```python
from maif.batch_processor import BatchProcessor

processor = BatchProcessor(
    process_func=analyze_block,
    batch_size=100,
    num_workers=8
)
results = processor.process_blocks(blocks)
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or use the Makefile
make docker-build
make docker-run

# View logs
make docker-logs
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_security_kms.py -v
python -m pytest tests/test_compliance_logging.py -v

# Run with coverage
python -m pytest --cov=maif tests/
```

## Project Structure

```
maif/
├── maif/                    # Core library
│   ├── core.py             # Core encoder/decoder
│   ├── security.py         # Security with KMS integration
│   ├── compliance_logging.py # STIG/FIPS compliance
│   ├── semantic_optimized.py # AI algorithms (ACAM, HSC, CSB)
│   └── aws_*.py            # AWS service integrations
├── maif_sdk/               # High-level SDK
│   ├── client.py          # Client interface
│   └── artifact.py        # Artifact management
├── tests/                  # Comprehensive test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Limitations

1. **Performance**: Semantic search achieves 30ms (not sub-3ms as originally claimed)
2. **Compression**: Maximum compression ratio of 64× (not 225× as originally claimed)
3. **Streaming**: Limited to sequential access patterns
4. **Scalability**: Single-file format not suitable for distributed systems
5. **Block Size**: Maximum block size of 2GB due to 32-bit addressing

## Contributing

Please ensure all contributions:
1. Include comprehensive unit tests
2. Pass FIPS compliance validation
3. Include security impact analysis
4. Update technical documentation
5. Follow PEP 8 style guidelines

## References

- [FIPS 140-2 Standards](https://csrc.nist.gov/publications/detail/fips/140/2/final)
- [DISA STIG Requirements](https://public.cyber.mil/stigs/)
- [NIST 800-53 Controls](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- [ISO BMFF Specification](https://www.iso.org/standard/68960.html)

## License

MIT License - See LICENSE file for details

---

*MAIF: Enabling trustworthy AI through cryptographically-secure, compliant data structures.*

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

## Agentic Framework

MAIF includes a sophisticated agentic framework that enables autonomous AI agents with perception, reasoning, planning, and execution capabilities. The framework seamlessly integrates with AWS services for cloud-scale multi-agent orchestration.

### Core Agent Architecture

```python
from maif.agentic_framework import MAIFAgent

# Create an autonomous agent
agent = MAIFAgent(
    agent_id="medical-assistant-001",
    workspace_path="/data/agents/medical",
    use_aws=True,  # Enable cloud capabilities
    config={
        "memory_size": 10000,
        "learning_rate": 0.1,
        "xray_sampling_rate": 0.1
    }
)

# Agent follows Perception → Reasoning → Planning → Execution cycle
await agent.initialize()
```

### Perception-Reasoning-Planning-Execution (PRPE) Cycle

The agent framework implements a sophisticated cognitive cycle:

```python
# 1. PERCEPTION - Process multimodal inputs
perception = await agent.perceive(
    data="Patient shows symptoms of fever and cough",
    input_type="text"
)

# 2. REASONING - Analyze and understand
reasoning = await agent.reason([perception])

# 3. PLANNING - Create action plans
plan = await agent.plan(
    goal="Diagnose patient condition",
    context=[reasoning]
)

# 4. EXECUTION - Take actions
result = await agent.execute(plan)
```

### Multi-Agent Orchestration

Coordinate multiple agents for complex tasks:

```python
from maif.agentic_framework import MultiAgentOrchestrator

# Create orchestrator
orchestrator = MultiAgentOrchestrator(agent)

# Define multi-agent task
task = {
    'id': 'diagnosis-001',
    'description': 'Collaborative medical diagnosis',
    'requirements': ['medical-knowledge', 'diagnostic-skills'],
    'priority': 10,
    'strategy': 'consensus'  # or 'vote', 'weighted'
}

# Execute with multiple agents (uses Bedrock swarm if available)
result = await orchestrator.execute_multi_agent_task(task)
```

### AWS Bedrock Swarm Integration

When AWS is enabled, the framework automatically leverages Bedrock for scalable multi-agent execution:

```python
# Automatic Bedrock swarm for cloud-scale agents
from maif.bedrock_swarm import BedrockAgentSwarm

swarm = BedrockAgentSwarm(
    max_agents=10,
    enable_consensus=True,
    enable_security=True,  # Auto-encrypt sensitive data
    enable_compliance_logging=True  # Track AI governance
)

# Execute task across multiple Bedrock models
result = swarm.execute_task(
    task="Analyze patient data for treatment options",
    strategy="consensus",
    agents=["Claude", "Titan", "AI21"]
)
```

### Memory and State Management

Agents maintain persistent memory across sessions:

```python
# Save agent state
dump_path = await agent.dump_state("medical-agent-backup.maif")

# Restore agent from previous state
restored_agent = MAIFAgent.from_dump(
    dump_path,
    workspace_path="/data/agents/medical-restored",
    use_aws=True
)

# Agent retains all memories and learned patterns
```

### Agent Specialization

Create specialized agents for different domains:

```python
# Medical Diagnosis Agent
@maif_agent(
    specialization="medical-diagnosis",
    required_knowledge=["medical", "diagnostic-procedures"],
    compliance_frameworks=["HIPAA"]
)
class MedicalDiagnosisAgent(MAIFAgent):
    async def diagnose(self, symptoms, patient_history):
        # Specialized diagnostic logic
        perception = await self.perceive(symptoms)
        reasoning = await self.reason([perception, patient_history])
        return await self.execute_diagnosis(reasoning)

# Financial Analysis Agent
@maif_agent(
    specialization="financial-analysis",
    required_knowledge=["finance", "risk-assessment"],
    compliance_frameworks=["SOX", "PCI-DSS"]
)
class FinancialAnalysisAgent(MAIFAgent):
    async def analyze_portfolio(self, holdings, market_data):
        # Specialized financial logic
        pass
```

### Task Orchestration

Complex task management with dependencies:

```python
# Create task manager
task_manager = TaskManager(agent)

# Define task hierarchy
task_manager.create_task({
    'id': 'patient-treatment-plan',
    'subtasks': [
        {'id': 'diagnosis', 'agent_type': 'medical'},
        {'id': 'medication-check', 'agent_type': 'pharmacy'},
        {'id': 'insurance-verification', 'agent_type': 'billing'}
    ],
    'dependencies': {
        'medication-check': ['diagnosis'],
        'insurance-verification': ['diagnosis', 'medication-check']
    }
})

# Execute with automatic task distribution
results = await task_manager.execute_all()
```

### Real-Time Learning

Agents learn and adapt from interactions:

```python
# Enable continuous learning
agent.enable_learning(mode="reinforcement")

# Agent improves from feedback
feedback = await agent.process_feedback({
    'result_quality': 0.95,
    'user_satisfaction': 'high',
    'corrections': []
})

# Learned patterns persist across sessions
await agent.consolidate_learning()
```

### Production-Ready Features

- **Automatic AWS Integration**: Seamlessly uses Bedrock, KMS, CloudWatch
- **Compliance Tracking**: All agent actions logged for governance
- **Security by Default**: Sensitive data automatically encrypted
- **Scalable Architecture**: From single agent to cloud swarm
- **State Persistence**: Full agent state saved/restored
- **Error Recovery**: Graceful handling of failures
- **Monitoring**: Built-in metrics and tracing

### Example: Medical Assistant Agent

```python
# Complete example of a production medical assistant
agent = MAIFAgent(
    agent_id="medical-ai-prod",
    workspace_path="/secure/medical-ai",
    use_aws=True,
    config={
        "compliance_frameworks": ["HIPAA", "HITECH"],
        "encryption_required": True,
        "audit_all_actions": True
    }
)

# Process patient query
patient_query = "I have severe headaches and blurred vision"

# Full cognitive cycle with security
async with agent:
    # Perceive symptoms (auto-encrypted if sensitive)
    perception = await agent.perceive(patient_query, "text")
    
    # Reason about condition (logged for compliance)
    reasoning = await agent.reason([perception])
    
    # Plan diagnostic steps
    plan = await agent.plan("provide initial assessment", [reasoning])
    
    # Execute with safety checks
    result = await agent.execute(plan)
    
    # Save interaction for medical records
    await agent.save_interaction("patient-123", result)
```

## AWS Integration

### Tightly Integrated AWS Credential Management

MAIF features a revolutionary centralized credential system - configure AWS once and ALL services automatically use those credentials:

```python
from maif.aws_config import configure_aws

# ONE-TIME SETUP: Configure globally for ALL services
configure_aws(
    environment="production",
    profile="my-aws-profile",  # Or use IAM role, env vars, etc.
    region="us-east-1"
)

# THAT'S IT! Every AWS service now uses these credentials automatically
```

#### Zero-Configuration Service Integration

Once configured, ALL AWS services in MAIF automatically use the centralized credentials:

```python
# No credentials needed - they're already configured globally!
from maif.aws_bedrock_integration import BedrockClient
from maif.aws_s3_integration import S3Client
from maif.aws_dynamodb_integration import DynamoDBClient
from maif.aws_kms_integration import KMSClient
from maif.aws_lambda_integration import LambdaClient

bedrock = BedrockClient()      # ✓ Automatically uses global config
s3 = S3Client()                # ✓ Automatically uses global config
dynamodb = DynamoDBClient()    # ✓ Automatically uses global config
kms = KMSClient()              # ✓ Automatically uses global config
lambda_client = LambdaClient() # ✓ Automatically uses global config

# Even security features automatically use the global KMS config!
from maif.security import SecurityManager
security = SecurityManager(use_kms=True)  # ✓ Uses global KMS automatically
```

The credential manager supports:
- **Environment variables** - Auto-detected from AWS_* vars
- **IAM roles** - EC2, ECS, Lambda instance roles
- **AWS profiles** - From ~/.aws/credentials
- **Explicit credentials** - For CI/CD pipelines
- **Role assumption** - Cross-account access
- **Container credentials** - ECS task roles
- **Instance metadata** - EC2 IMDS v2
- **Automatic refresh** - Temporary credentials refresh before expiry

Benefits of tight integration:
- **Zero redundancy** - Configure once, use everywhere
- **Thread-safe** - Safe for concurrent operations
- **Auto-refresh** - Credentials never expire
- **Environment-aware** - Different settings per environment
- **Service-optimized** - Each service gets ideal retry/timeout config

### S3 Integration

```python
from maif.aws_s3_integration import S3Client
from maif.aws_config import get_aws_config

# S3 client automatically uses configured credentials
s3_client = S3Client()  # Uses global AWS configuration

# Or provide custom configuration
from maif.aws_credentials import AWSCredentialManager
custom_creds = AWSCredentialManager.from_profile("custom-profile")
s3_client = S3Client(credential_manager=custom_creds)

# Upload MAIF file
s3_client.upload_file(
    local_path="agent_data.maif",
    bucket="my-bucket",
    key="data/agent_data.maif"
)

# Stream blocks from S3
for block in s3_client.stream_blocks("my-bucket", "data/agent_data.maif"):
    process_block(block)
```

### Lambda Deployment

```python
from maif.aws_lambda_integration import deploy_maif_handler

# Deploy MAIF processing to Lambda (uses configured credentials)
deploy_maif_handler(
    function_name="maif-processor",
    handler="process_maif_file",
    memory_size=3008,
    timeout=900
)
```

### CloudWatch Integration

```python
from maif.aws_cloudwatch_compliance import CloudWatchLogger

# CloudWatch logger uses configured credentials
cw_logger = CloudWatchLogger(namespace="MAIF/Production")
cw_logger.log_metric("BlocksProcessed", count=100)
cw_logger.log_metric("ProcessingTime", value=30, unit="Milliseconds")
```

### Environment-Specific Configuration

```python
from maif.aws_config import configure_aws, AWSEnvironment

# Development environment
configure_aws(
    environment="development",
    enable_debug=True,
    enable_monitoring=False
)

# Production environment with strict settings
configure_aws(
    environment="production",
    enable_monitoring=True,
    enable_debug=False
)

# GovCloud environment
from maif.aws_config import AWSConfig
config = AWSConfig.for_govcloud()
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

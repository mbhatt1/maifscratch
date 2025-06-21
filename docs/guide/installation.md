# Installation Guide

This guide covers all installation methods for MAIF, from basic setup to advanced deployment configurations.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 2GB RAM available
- **Storage**: 500MB free disk space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.11+ (for optimal performance)
- **Memory**: 8GB+ RAM (for large-scale operations)
- **Storage**: 2GB+ free disk space (for caching and temp files)
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **CPU**: Multi-core processor with AVX2 support

## Installation Methods

### 1. Basic Installation

For most users, the basic installation provides core MAIF functionality:

```bash
pip install maif
```

This includes:
- Core MAIF SDK
- Basic encryption and security
- Standard semantic embeddings
- File-based artifact storage

### 2. Full Installation (Recommended)

For production use and advanced features:

```bash
pip install maif[full]
```

This includes everything from basic installation plus:
- Novel AI algorithms (ACAM, HSC, CSB)
- Advanced privacy features
- High-performance optimizations
- Distributed processing capabilities
- Enterprise security features

### 3. Development Installation

For contributors and advanced users:

```bash
# Clone the repository
git clone https://github.com/maif-ai/maif.git
cd maif

# Install in development mode
pip install -e .[dev,full]

# Run tests to verify installation
pytest tests/
```

### 4. Docker Installation

For containerized deployments:

```bash
# Pull the official image
docker pull maif/maif:latest

# Run with volume mounting
docker run -v $(pwd)/data:/app/data maif/maif:latest
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3.11 python3.11-pip python3.11-venv -y

# Create virtual environment
python3.11 -m venv maif-env
source maif-env/bin/activate

# Install MAIF
pip install --upgrade pip
pip install maif[full]
```

### Linux (CentOS/RHEL)

```bash
# Install Python 3.11
sudo dnf install python3.11 python3.11-pip -y

# Create virtual environment
python3.11 -m venv maif-env
source maif-env/bin/activate

# Install MAIF
pip install --upgrade pip
pip install maif[full]
```

### macOS

```bash
# Install Python via Homebrew (recommended)
brew install python@3.11

# Create virtual environment
python3.11 -m venv maif-env
source maif-env/bin/activate

# Install MAIF
pip install --upgrade pip
pip install maif[full]
```

### Windows

```powershell
# Install Python from python.org or Microsoft Store
# Then open PowerShell as Administrator

# Create virtual environment
python -m venv maif-env
.\maif-env\Scripts\Activate.ps1

# Install MAIF
pip install --upgrade pip
pip install maif[full]
```

## Optional Dependencies

### High-Performance Computing

For maximum performance on large datasets:

```bash
pip install maif[hpc]
```

Includes:
- Intel MKL optimizations
- CUDA support for GPU acceleration
- Distributed computing libraries

### Cloud Integrations

For cloud-native deployments:

```bash
pip install maif[cloud]
```

Includes:
- AWS S3 integration
- Google Cloud Storage
- Azure Blob Storage
- Kubernetes operators

### Enterprise Features

For enterprise deployments:

```bash
pip install maif[enterprise]
```

Includes:
- Advanced audit logging
- LDAP/Active Directory integration
- Hardware security module (HSM) support
- Compliance reporting tools

## Verification

After installation, verify MAIF is working correctly:

```python
import maif
from maif_sdk import create_client, create_artifact

# Check version
print(f"MAIF version: {maif.__version__}")

# Test basic functionality
client = create_client("test-agent")
artifact = create_artifact("test-artifact", client)

# Add some test data
text_id = artifact.add_text("Hello, MAIF!")
print(f"✅ Successfully created artifact with ID: {text_id}")

# Test encryption
encrypted_id = artifact.add_text("Secret data", encrypt=True)
print(f"✅ Successfully encrypted data with ID: {encrypted_id}")

# Test semantic search
results = artifact.search("Hello", top_k=1)
print(f"✅ Semantic search returned {len(results)} results")

print("🎉 MAIF installation verified successfully!")
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# On Linux/macOS
pip install --user maif[full]

# Or use virtual environment
python -m venv maif-env
source maif-env/bin/activate  # Linux/macOS
# .\maif-env\Scripts\Activate.ps1  # Windows
pip install maif[full]
```

#### 2. Compilation Errors

Some dependencies require compilation. Install build tools:

```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# CentOS/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

#### 3. Memory Issues

For systems with limited memory:

```bash
# Install with reduced memory usage
pip install --no-cache-dir maif[full]

# Or install basic version first
pip install maif
```

#### 4. Network Issues

If you're behind a corporate firewall:

```bash
# Use corporate proxy
pip install --proxy http://proxy.company.com:8080 maif[full]

# Or download and install offline
pip download maif[full]
pip install --no-index --find-links . maif[full]
```

### Performance Optimization

#### 1. Enable Memory Mapping

```python
from maif_sdk import create_client

client = create_client(
    "optimized-agent",
    enable_mmap=True,
    buffer_size=128*1024  # 128KB buffer
)
```

#### 2. Configure Parallel Processing

```python
client = create_client(
    "parallel-agent",
    max_concurrent_writers=8,
    enable_compression=True
)
```

#### 3. GPU Acceleration

```python
from maif.semantic import SemanticEmbedder

embedder = SemanticEmbedder(
    device="cuda",  # Use GPU if available
    batch_size=32   # Optimize batch size
)
```

## Next Steps

After successful installation:

1. **[Quick Start →](/guide/quick-start)** - Build your first agent in 5 minutes
2. **[Core Concepts →](/guide/concepts)** - Understand MAIF fundamentals
3. **[Examples →](/examples/)** - See real-world implementations
4. **[API Reference →](/api/)** - Explore the complete API

## Getting Help

If you encounter issues:

- **Documentation**: Check our [troubleshooting guide](/guide/troubleshooting)
- **GitHub Issues**: Report bugs at [github.com/maif-ai/maif/issues](https://github.com/maif-ai/maif/issues)
- **Community**: Join our [Discord server](https://discord.gg/maif)
- **Enterprise Support**: Contact [support@maif.ai](mailto:support@maif.ai) 
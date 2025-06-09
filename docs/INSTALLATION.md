# MAIF Installation Guide

This guide covers installing and using the MAIF (Multimodal Artifact File Format) package from PyPI.

## Installation

### Basic Installation

Install MAIF with basic functionality:

```bash
pip install maif
```

This installs the core library with minimal dependencies for basic MAIF operations.

### Full Installation (Recommended)

Install MAIF with all features enabled:

```bash
pip install maif[full]
```

This includes all optional dependencies for:
- Advanced semantic processing
- Computer vision capabilities
- High-performance compression
- CLI tools
- Validation frameworks

### Selective Installation

Install only the features you need:

```bash
# For machine learning features
pip install maif[ml]

# For CLI tools
pip install maif[cli]

# For compression features
pip install maif[compression]

# For computer vision
pip install maif[vision]

# For development
pip install maif[dev]

# Multiple feature sets
pip install maif[ml,cli,compression]
```

### Development Installation

For development or contributing:

```bash
git clone https://github.com/maif-ai/maif.git
cd maif
pip install -e .[dev,full]
```

## Quick Start

### Using the Simple API

```python
import maif

# Create a new MAIF
artifact = maif.create_maif("my_agent")

# Add content
artifact.add_text("Hello, MAIF world!", title="Greeting")
artifact.add_multimodal({
    "text": "A beautiful sunset",
    "description": "Nature photography"
}, title="Sunset Scene")

# Save
artifact.save("my_artifact.maif")

# Load and verify
loaded = maif.load_maif("my_artifact.maif")
print(f"Integrity: {loaded.verify_integrity()}")
```

### Using the Advanced API

```python
from maif import MAIFEncoder, MAIFDecoder, MAIFSigner

# Create encoder
encoder = MAIFEncoder(agent_id="advanced_agent")

# Add content with full control
text_id = encoder.add_text_block(
    "Advanced MAIF usage",
    metadata={"category": "documentation"}
)

# Save with manifest
encoder.build_maif("advanced.maif", "advanced_manifest.json")

# Sign for provenance
signer = MAIFSigner(agent_id="advanced_agent")
signer.add_provenance_entry("create", text_id)
```

## Command Line Interface

After installation, MAIF provides several CLI commands:

### Main CLI

```bash
# General help
maif --help

# Create a MAIF file
maif create --input data.txt --output result.maif

# Verify a MAIF file
maif verify result.maif

# Analyze MAIF contents
maif analyze result.maif
```

### Specialized Commands

```bash
# Create MAIF from various inputs
maif-create --text "Hello world" --output hello.maif

# Verify integrity and signatures
maif-verify --file artifact.maif --check-signatures

# Analyze content and metadata
maif-analyze --file artifact.maif --detailed

# Extract content from MAIF
maif-extract --file artifact.maif --output-dir extracted/
```

## Feature Overview

### Core Features (Always Available)

- ✅ Basic MAIF creation and reading
- ✅ Text content handling
- ✅ Binary data storage
- ✅ Basic integrity verification
- ✅ Simple API access

### ML Features (`pip install maif[ml]`)

- 🧠 Semantic embeddings with sentence-transformers
- 🔍 Fast similarity search with FAISS
- 🎯 Cross-modal attention mechanisms (ACAM)
- 📊 Hierarchical semantic compression (HSC)
- 🔐 Cryptographic semantic binding (CSB)

### Vision Features (`pip install maif[vision]`)

- 🖼️ Image processing with OpenCV
- 📷 Automatic metadata extraction
- 🎬 Video content handling
- 🖼️ Image format support (JPEG, PNG, etc.)

### CLI Features (`pip install maif[cli]`)

- 💻 Command-line tools
- 📊 Progress bars and status updates
- 🔧 Batch processing capabilities
- 📋 Detailed analysis reports

### Compression Features (`pip install maif[compression]`)

- 🗜️ Advanced compression algorithms (Brotli, Zstandard)
- 📈 Optimal compression selection
- 💾 Space-efficient storage
- ⚡ Fast decompression

### Performance Features (`pip install maif[performance]`)

- ⚡ Fast hashing with xxHash
- 📦 Efficient serialization with MessagePack
- 📊 System monitoring with psutil
- 🚀 Optimized data structures

## Usage Examples

### Example 1: Document Processing

```python
import maif

# Create document processor
processor = maif.create_maif("document_processor", enable_privacy=True)

# Add encrypted document
processor.add_text(
    "Confidential business plan...",
    title="Business Plan Q4",
    encrypt=True,
    anonymize=True
)

# Save securely
processor.save("business_plan.maif", sign=True)
```

### Example 2: Multimodal AI Dataset

```python
import maif

# Create dataset
dataset = maif.create_maif("dataset_creator")

# Add training examples
for i, (text, image_path) in enumerate(training_data):
    dataset.add_multimodal({
        "text": text,
        "image_path": image_path,
        "label": labels[i]
    }, title=f"Training Example {i}")

# Save with compression
dataset.save("training_dataset.maif")
```

### Example 3: Research Archive

```python
import maif

# Create research archive
archive = maif.create_maif("research_archive")

# Add papers with embeddings
for paper in research_papers:
    # Add paper text
    text_id = archive.add_text(paper.content, title=paper.title)
    
    # Add semantic embeddings
    embeddings = generate_embeddings(paper.content)
    archive.add_embeddings(
        embeddings, 
        model_name="scientific-bert",
        compress=True
    )

# Save archive
archive.save("research_archive.maif")

# Later: search the archive
loaded_archive = maif.load_maif("research_archive.maif")
results = loaded_archive.search("machine learning", top_k=5)
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'sentence_transformers'**
```bash
pip install maif[ml]
```

**Import Error: No module named 'cv2'**
```bash
pip install maif[vision]
```

**Command not found: maif**
```bash
pip install maif[cli]
```

**Performance Issues with Large Files**
```bash
pip install maif[performance]
```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Create a virtual environment
python -m venv maif_env
source maif_env/bin/activate  # On Windows: maif_env\Scripts\activate

# Install MAIF
pip install maif[full]
```

### Memory Issues

For large files, use streaming:

```python
from maif import MAIFStreamReader

# Stream large MAIF files
with MAIFStreamReader("large_file.maif") as reader:
    for block in reader:
        process_block(block)
```

## Verification

Test your installation:

```python
import maif

# Check version
print(f"MAIF version: {maif.__version__}")

# Check available features
print(f"Enhanced algorithms: {maif.ENHANCED_ALGORITHMS_AVAILABLE}")
print(f"Simple API: {maif.SIMPLE_API_AVAILABLE}")

# Run basic test
test_maif = maif.create_maif("test")
test_maif.add_text("Installation test")
success = test_maif.save("test.maif")
print(f"Installation test: {'✅ PASSED' if success else '❌ FAILED'}")
```

## Next Steps

- 📖 Read the [Simple API Guide](SIMPLE_API_GUIDE.md)
- 🔬 Explore [Novel Algorithms](NOVEL_ALGORITHMS_IMPLEMENTATION.md)
- 🛡️ Learn about [Security Features](MAIF_Security_Verifications_Table.md)
- 📊 Check [Performance Benchmarks](BENCHMARK_SUMMARY.md)
- 🎯 Try the [Examples](../examples/)

## Support

- 🐛 Report bugs: [GitHub Issues](https://github.com/maif-ai/maif/issues)
- 📚 Documentation: [ReadTheDocs](https://maif.readthedocs.io/)
- 💬 Discussions: [GitHub Discussions](https://github.com/maif-ai/maif/discussions)
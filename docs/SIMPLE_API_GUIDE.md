# MAIF Simple API Guide

The MAIF SDK provides an easy-to-use interface for working with Multimodal Artifact File Format (MAIF) files. This guide covers both the standard SDK and the new classified security API.

## Quick Start

### Installation

```bash
pip install maif
```

### Basic Usage with SDK

```python
from maif_sdk.client import MAIFClient
from maif_sdk.artifact import Artifact

# Create client and artifact
client = MAIFClient()
artifact = Artifact(name="my_agent", client=client)

# Add content
artifact.add_text("Hello world!")
artifact.add_multimodal({
    "text": "A beautiful sunset",
    "description": "Nature photography"
})

# Save
artifact.save("my_artifact.maif")

# Load existing
loaded = Artifact.load("my_artifact.maif", client=client)
```

### Classified Data API (NEW)

For working with classified data, use the simplified SecureMAIF API:

```python
from maif.classified_api import SecureMAIF

# Create secure instance
maif = SecureMAIF(classification="SECRET")

# Grant clearance
maif.grant_clearance("user.001", "SECRET")

# Store classified data
doc_id = maif.store_classified_data(
    data={"mission": "OPERATION_X"},
    classification="SECRET"
)

# Retrieve with access control
if maif.can_access("user.001", doc_id):
    data = maif.retrieve_classified_data(doc_id)
```

## API Reference

### SDK API: `Artifact` Class

The main class for creating and manipulating MAIF files.

#### Constructor

```python
Artifact(name: str, client: MAIFClient,
         security_level: SecurityLevel = SecurityLevel.PUBLIC)
```

- `name`: Name of the artifact
- `client`: MAIFClient instance
- `security_level`: Security level (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)

#### Key Methods

##### `add_text(text: str, metadata: dict = None)`

Add text content to the artifact.

**Example:**
```python
artifact.add_text(
    "This is confidential information",
    metadata={"title": "Secret Document"}
)
```

##### `add_binary(data: bytes, content_type: str, metadata: dict = None)`

Add binary data (images, documents, etc).

**Example:**
```python
with open("image.jpg", "rb") as f:
    artifact.add_binary(f.read(), "image/jpeg", {"title": "Photo"})
```

### Classified Security API: `SecureMAIF` Class

Simple API for handling classified data with built-in security.

#### Constructor

```python
SecureMAIF(classification: str = "UNCLASSIFIED",
          region: str = None, use_fips: bool = True)
```

- `classification`: Default classification level
- `region`: AWS region (auto-detected if not specified)
- `use_fips`: Enable FIPS 140-2 compliant mode

#### Key Methods

##### Authentication

```python
# PKI/CAC authentication
user_id = maif.authenticate_with_pki(certificate_pem)

# Hardware MFA
challenge = maif.authenticate_with_mfa(user_id, token_serial)
verified = maif.verify_mfa(challenge, token_code)
```

##### User Management

```python
# Grant clearance
maif.grant_clearance(
    user_id="analyst.001",
    level="SECRET",
    compartments=["CRYPTO", "SIGINT"],
    caveats=["NOFORN"]
)

# Check clearance
level = maif.check_clearance("analyst.001")
```

##### Data Operations

```python
# Store classified data
doc_id = maif.store_classified_data(
    data={"content": "classified"},
    classification="SECRET",
    compartments=["CRYPTO"]
)

# Access control
if maif.can_access("analyst.001", doc_id):
    data = maif.retrieve_classified_data(doc_id)
    
# Audit logging
maif.log_access("analyst.001", doc_id, "read")
```

##### `add_image(image_path, title=None, extract_metadata=True)`

Add image to the MAIF.

**Parameters:**
- `image_path` (str): Path to image file
- `title` (str, optional): Title for the image
- `extract_metadata` (bool): Whether to extract image metadata

**Returns:** Block ID (str)

**Example:**
```python
block_id = maif.add_image("photo.jpg", title="Vacation Photo")
```

##### `add_video(video_path, title=None, extract_metadata=True)`

Add video to the MAIF.

**Parameters:**
- `video_path` (str): Path to video file
- `title` (str, optional): Title for the video
- `extract_metadata` (bool): Whether to extract video metadata

**Returns:** Block ID (str)

##### `add_multimodal(content, title=None, use_acam=True)`

Add multimodal content using ACAM (Adaptive Cross-Modal Attention) processing.

**Parameters:**
- `content` (dict): Dictionary with different modality data
- `title` (str, optional): Title for the content
- `use_acam` (bool): Whether to use ACAM processing

**Returns:** Block ID (str)

**Example:**
```python
block_id = maif.add_multimodal({
    "text": "A mountain landscape",
    "image_description": "Snow-capped peaks",
    "location": "Swiss Alps",
    "weather": "Clear sky"
}, title="Alpine Scene")
```

##### `add_embeddings(embeddings, model_name="custom", compress=True)`

Add embeddings with optional HSC (Hierarchical Semantic Compression).

**Parameters:**
- `embeddings` (List[List[float]]): List of embedding vectors
- `model_name` (str): Name of the model that generated embeddings
- `compress` (bool): Whether to use HSC compression

**Returns:** Block ID (str)

##### `save(filepath, sign=True)`

Save MAIF to file.

**Parameters:**
- `filepath` (str): Output file path
- `sign` (bool): Whether to cryptographically sign the file

**Returns:** bool (success status)

##### `get_content_list()`

Get list of all content blocks.

**Returns:** List[Dict] - List of content metadata

##### `search(query, top_k=5)`

Search content using semantic similarity.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return

**Returns:** List[Dict] - Search results with similarity scores

##### `verify_integrity()`

Verify MAIF file integrity.

**Returns:** bool - True if integrity check passes

##### `get_privacy_report()`

Get privacy report for the MAIF.

**Returns:** Dict - Privacy status and statistics

### Class Methods

##### `MAIF.load(filepath)`

Load existing MAIF file.

**Parameters:**
- `filepath` (str): Path to MAIF file

**Returns:** MAIF instance

## Convenience Functions

### `create_maif(agent_id="default_agent", enable_privacy=False)`

Create a new MAIF instance.

### `load_maif(filepath)`

Load existing MAIF file.

### `quick_text_maif(text, output_path, title=None)`

Quickly create a MAIF with just text content.

### `quick_multimodal_maif(content, output_path, title=None)`

Quickly create a MAIF with multimodal content.

## Examples

### Example 1: Simple Document

```python
from maif.integration import create_maif

# Create MAIF
maif = create_maif("document_agent")

# Add content
maif.add_text("This is my document content.", title="My Document")

# Save
maif.save("document.maif")
```

### Example 2: Multimodal Content with ACAM

```python
from maif.integration import create_maif

# Create MAIF
maif = create_maif("multimodal_agent")

# Add cross-modal content
maif.add_multimodal({
    "text": "Product description: High-quality headphones",
    "image_description": "Black over-ear headphones on white background",
    "price": "$199.99",
    "category": "Electronics",
    "features": ["Noise cancelling", "Wireless", "30-hour battery"]
}, title="Product Listing", use_acam=True)

# Save
maif.save("product.maif")
```

### Example 3: Privacy-Enabled MAIF

```python
from maif.integration import create_maif

# Create MAIF with privacy
maif = create_maif("secure_agent", enable_privacy=True)

# Add encrypted content
maif.add_text(
    "Patient John Doe has condition XYZ",
    title="Medical Record",
    encrypt=True,
    anonymize=True
)

# Save with signing
maif.save("medical_record.maif", sign=True)

# Check privacy report
report = maif.get_privacy_report()
print(f"Encrypted blocks: {report.get('encrypted_blocks', 0)}")
```

### Example 4: Search and Retrieval

```python
from maif.integration import load_maif

# Load existing MAIF
maif = load_maif("knowledge_base.maif")

# Search content
results = maif.search("machine learning algorithms", top_k=3)

for result in results:
    print(f"Found: {result}")
```

### Example 5: Working with Embeddings

```python
from maif.integration import create_maif

# Create MAIF
maif = create_maif("embedding_agent")

# Add embeddings with compression
embeddings = [
    [0.1, 0.2, 0.3] * 128,  # 384-dimensional
    [0.4, 0.5, 0.6] * 128   # 384-dimensional
]

maif.add_embeddings(
    embeddings,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    compress=True  # Uses HSC compression
)

# Save
maif.save("embeddings.maif")
```

## Advanced Features

### Novel Algorithms

The API automatically uses MAIF's novel algorithms:

- **ACAM (Adaptive Cross-Modal Attention)**: Used in `add_multimodal()` when `use_acam=True`
- **HSC (Hierarchical Semantic Compression)**: Used in `add_embeddings()` when `compress=True`
- **CSB (Cryptographic Semantic Binding)**: Automatically applied for integrity verification

### Security Features

- **Digital Signatures**: Enabled by default when saving (`sign=True`)
- **Encryption**: Available for text content (`encrypt=True`)
- **Anonymization**: Automatic PII detection and replacement (`anonymize=True`)
- **Access Control**: Granular permissions (advanced usage)

### Performance Optimizations

- **Streaming**: Large files processed with bounded memory usage
- **Compression**: Multiple algorithms automatically selected for optimal results
- **Indexing**: Efficient search with sub-50ms response times
- **Validation**: Multi-level integrity checking with error recovery

## Error Handling

The API includes comprehensive error handling:

```python
try:
    maif = create_maif("my_agent")
    maif.add_text("Hello world!")
    success = maif.save("output.maif")
    
    if not success:
        print("Failed to save MAIF")
        
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Use descriptive agent IDs**: Choose meaningful identifiers for your agents
2. **Add titles to content**: Makes content easier to identify and search
3. **Enable privacy for sensitive data**: Use `enable_privacy=True` for confidential content
4. **Verify integrity**: Always check `verify_integrity()` after loading files
5. **Use compression for embeddings**: Enable `compress=True` for large embedding sets
6. **Sign important files**: Keep `sign=True` for audit trails and provenance

## Integration with Existing Code

The Simple API is designed to work alongside the existing MAIF library:

```python
# You can still use the full library
from maif.core import MAIFEncoder
from maif.integration import create_maif

# Mix and match as needed
simple_maif = create_maif("agent1")
advanced_encoder = MAIFEncoder("agent2")
```

This allows gradual migration to the simpler API while maintaining access to advanced features when needed.
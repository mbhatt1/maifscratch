# MAIF File Format vs AWS Backend Parity Analysis

## Executive Summary

This document identifies critical parity issues between the MAIF local file format and AWS S3 backend implementation that could cause interoperability problems.

## Key Parity Issues Identified

### 1. Block Header Structure Mismatch

**Local File Format (`binary_format.py`):**
- Header size: 80 bytes
- Structure: `size(4) + block_type(4) + version(4) + flags(4) + timestamp(8) + block_id(16) + previous_hash(32) + reserved(8)`
- Uses big-endian byte ordering (`>`)

**Local Block Storage (`block_storage.py`):**
- Header size: 72 bytes  
- Structure: `size(4) + block_type(4) + version(4) + timestamp(8) + uuid(16) + previous_hash(32) + flags(4)`
- Uses little-endian byte ordering (`<`)

**AWS S3 Backend:**
- Stores header as JSON metadata in S3 object tags
- No binary header serialization
- Header fields stored as string key-value pairs

**Impact:** Files created locally cannot be directly read by AWS backend and vice versa.

### 2. Block Type Representation

**Local File Format:**
- Uses integer constants (e.g., `TEXT = 0x54455854`)
- Stored as 4-byte integers in binary format

**Block Storage & AWS:**
- Uses string FourCC codes (e.g., `"TEXT"`)
- AWS stores as string metadata

**Impact:** Type conversion needed between implementations.

### 3. Data Storage Location

**Local File:**
```
[File Header][Block1 Header][Block1 Data][Block2 Header][Block2 Data]...
```
- Sequential storage in single file
- Headers and data interleaved

**AWS S3:**
```
s3://bucket/prefix/block_index.json          # Index file
s3://bucket/prefix/data/{block_uuid}        # Individual block files
```
- Distributed storage across multiple objects
- Headers stored as metadata, data as object content

**Impact:** Completely different access patterns and retrieval mechanisms.

### 4. Block Versioning & Chaining

**Local Implementation:**
- Creates new blocks for updates (immutable)
- Links via `previous_hash` field
- Maintains version history in-file

**AWS Implementation:**
- Updates stored as new S3 objects
- Version linking through metadata
- No built-in version history traversal

**Impact:** Version history reconstruction differs between implementations.

### 5. Metadata Storage

**Local File:**
- Limited metadata in fixed header structure
- Additional metadata requires separate storage

**AWS S3:**
- Extensive metadata as S3 object tags
- User metadata prefixed with `user_`
- Signature data stored as JSON in metadata

**Impact:** Metadata capacity and access patterns differ significantly.

### 6. Signature Storage

**Local Storage:**
- Signatures stored in separate in-memory dictionary
- Not persisted with block data

**AWS S3:**
- Signatures stored as S3 object metadata
- Persisted with each block
- JSON serialized in metadata

**Impact:** Signature verification workflows differ.

### 7. Index Management

**Local File:**
- In-memory index built during file parsing
- No persistent index structure

**AWS S3:**
- Persistent `block_index.json` file
- Contains full block metadata
- Must be kept synchronized

**Impact:** Index corruption in AWS can prevent block access.

## Recommended Solutions

### 1. Unified Block Header Format
Create a standardized header structure used by both implementations:
```python
@dataclass
class UnifiedBlockHeader:
    magic: bytes = b'MAIF'           # 4 bytes
    version: int                     # 4 bytes  
    size: int                       # 8 bytes
    block_type: str                 # 4 bytes (FourCC)
    uuid: str                       # 36 bytes (string representation)
    timestamp: float                # 8 bytes
    previous_hash: Optional[str]    # 64 bytes (hex string)
    flags: int                      # 4 bytes
    reserved: bytes                 # 32 bytes
    # Total: 164 bytes
```

### 2. Standardized Serialization
- Use consistent byte ordering (big-endian recommended)
- Define canonical JSON representation for metadata
- Create conversion utilities between formats

### 3. Abstraction Layer
Implement a storage abstraction layer that:
- Provides unified API for both backends
- Handles format conversions transparently
- Maintains compatibility with existing data

### 4. Migration Tools
Develop tools to:
- Convert between local and S3 formats
- Validate format compliance
- Migrate existing data

### 5. Format Version Negotiation
- Add format version field to identify storage backend
- Implement format detection and auto-conversion
- Support multiple format versions

## Implementation Priority

1. **High Priority:** Fix header structure inconsistencies
2. **High Priority:** Standardize block type representation  
3. **Medium Priority:** Unify metadata storage approach
4. **Medium Priority:** Implement format conversion utilities
5. **Low Priority:** Optimize for specific backend characteristics

## Testing Requirements

1. Round-trip conversion tests between formats
2. Cross-backend interoperability tests
3. Performance benchmarks for conversion overhead
4. Data integrity validation across backends

## Conclusion

The current implementations have significant structural differences that prevent interoperability. A unified approach with proper abstraction and conversion utilities is needed to maintain compatibility while leveraging backend-specific optimizations.
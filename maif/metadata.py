"""
MAIF metadata management and standards compliance.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

class MAIFVersion(Enum):
    """MAIF format versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"

class ContentType(Enum):
    """Standard content types for MAIF blocks."""
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    BINARY = "application/octet-stream"
    IMAGE = "image/*"
    AUDIO = "audio/*"
    VIDEO = "video/*"
    EMBEDDING = "application/x-embedding"
    KNOWLEDGE_GRAPH = "application/x-knowledge-graph"

class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    BROTLI = "brotli"
    CUSTOM = "custom"

@dataclass
class MAIFHeader:
    """MAIF file header metadata."""
    magic: str = "MAIF"
    version: str = MAIFVersion.V2_0.value
    created: str = ""
    modified: str = ""
    file_id: str = ""
    creator_agent: str = ""
    format_flags: int = 0
    block_count: int = 0
    total_size: int = 0
    checksum: str = ""
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if not self.modified:
            self.modified = self.created
        if not self.file_id:
            self.file_id = str(uuid.uuid4())

@dataclass
class BlockMetadata:
    """Metadata for individual MAIF blocks."""
    block_id: str
    block_type: str
    content_type: str
    size: int
    offset: int
    checksum: str
    compression: str = CompressionType.NONE.value
    encryption: Optional[str] = None
    created: str = ""
    agent_id: str = ""
    version: int = 1
    parent_block: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}

@dataclass
class ProvenanceRecord:
    """Provenance tracking for MAIF operations."""
    operation_id: str
    operation_type: str
    timestamp: str
    agent_id: str
    block_ids: List[str]
    operation_data: Dict[str, Any]
    signature: Optional[str] = None
    
    def __post_init__(self):
        if not self.operation_id:
            self.operation_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class MAIFMetadataManager:
    """Manages MAIF file metadata and standards compliance."""
    
    def __init__(self, version: str = MAIFVersion.V2_0.value):
        self.version = version
        self.header = MAIFHeader(version=version)
        self.blocks: Dict[str, BlockMetadata] = {}
        self.provenance: List[ProvenanceRecord] = []
        self.custom_schemas: Dict[str, Dict] = {}
        
    def create_header(self, creator_agent: str, **kwargs) -> MAIFHeader:
        """Create a new MAIF header."""
        self.header = MAIFHeader(
            version=self.version,
            creator_agent=creator_agent,
            **kwargs
        )
        return self.header
    
    def add_block_metadata(self, 
                          block_id: str,
                          block_type: str,
                          content_type: str,
                          size: int,
                          offset: int,
                          checksum: str,
                          **kwargs) -> BlockMetadata:
        """Add metadata for a new block."""
        metadata = BlockMetadata(
            block_id=block_id,
            block_type=block_type,
            content_type=content_type,
            size=size,
            offset=offset,
            checksum=checksum,
            **kwargs
        )
        self.blocks[block_id] = metadata
        self.header.block_count = len(self.blocks)
        return metadata
    
    def update_block_metadata(self, block_id: str, **updates) -> bool:
        """Update existing block metadata."""
        if block_id not in self.blocks:
            return False
        
        metadata = self.blocks[block_id]
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        self.header.modified = datetime.now(timezone.utc).isoformat()
        return True
    
    def add_provenance_record(self, 
                            operation_type: str,
                            agent_id: str,
                            block_ids: List[str],
                            operation_data: Dict[str, Any],
                            **kwargs) -> ProvenanceRecord:
        """Add a provenance record."""
        record = ProvenanceRecord(
            operation_type=operation_type,
            agent_id=agent_id,
            block_ids=block_ids,
            operation_data=operation_data,
            **kwargs
        )
        self.provenance.append(record)
        return record
    
    def get_block_dependencies(self, block_id: str) -> List[str]:
        """Get all dependencies for a block."""
        if block_id not in self.blocks:
            return []
        
        dependencies = []
        metadata = self.blocks[block_id]
        
        # Direct dependencies
        dependencies.extend(metadata.dependencies)
        
        # Parent block dependency
        if metadata.parent_block:
            dependencies.append(metadata.parent_block)
        
        return list(set(dependencies))
    
    def get_block_dependents(self, block_id: str) -> List[str]:
        """Get all blocks that depend on this block."""
        dependents = []
        
        for bid, metadata in self.blocks.items():
            if (block_id in metadata.dependencies or 
                metadata.parent_block == block_id):
                dependents.append(bid)
        
        return dependents
    
    def validate_dependencies(self) -> List[str]:
        """Validate all block dependencies."""
        errors = []
        
        for block_id, metadata in self.blocks.items():
            # Check if dependencies exist
            for dep_id in metadata.dependencies:
                if dep_id not in self.blocks:
                    errors.append(f"Block {block_id} depends on non-existent block {dep_id}")
            
            # Check parent block
            if metadata.parent_block and metadata.parent_block not in self.blocks:
                errors.append(f"Block {block_id} has non-existent parent {metadata.parent_block}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(block_id):
                errors.append(f"Block {block_id} has circular dependency")
        
        return errors
    
    def _has_circular_dependency(self, block_id: str, visited: set = None) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()
        
        if block_id in visited:
            return True
        
        if block_id not in self.blocks:
            return False
        
        visited.add(block_id)
        metadata = self.blocks[block_id]
        
        # Check dependencies
        for dep_id in metadata.dependencies:
            if self._has_circular_dependency(dep_id, visited.copy()):
                return True
        
        # Check parent
        if metadata.parent_block:
            if self._has_circular_dependency(metadata.parent_block, visited.copy()):
                return True
        
        return False
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get a summary of all metadata."""
        content_types = {}
        compression_types = {}
        total_size = 0
        
        for metadata in self.blocks.values():
            # Count content types
            content_types[metadata.content_type] = content_types.get(metadata.content_type, 0) + 1
            
            # Count compression types
            compression_types[metadata.compression] = compression_types.get(metadata.compression, 0) + 1
            
            # Sum sizes
            total_size += metadata.size
        
        return {
            "header": asdict(self.header),
            "block_count": len(self.blocks),
            "total_size": total_size,
            "content_types": content_types,
            "compression_types": compression_types,
            "provenance_records": len(self.provenance),
            "dependency_errors": self.validate_dependencies()
        }
    
    def export_manifest(self) -> Dict[str, Any]:
        """Export complete metadata as manifest."""
        return {
            "header": asdict(self.header),
            "blocks": {bid: asdict(metadata) for bid, metadata in self.blocks.items()},
            "provenance": [asdict(record) for record in self.provenance],
            "custom_schemas": self.custom_schemas,
            "exported": datetime.now(timezone.utc).isoformat()
        }
    
    def import_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Import metadata from manifest."""
        try:
            # Import header
            if "header" in manifest:
                self.header = MAIFHeader(**manifest["header"])
            
            # Import blocks
            if "blocks" in manifest:
                self.blocks = {}
                for block_id, block_data in manifest["blocks"].items():
                    self.blocks[block_id] = BlockMetadata(**block_data)
            
            # Import provenance
            if "provenance" in manifest:
                self.provenance = []
                for record_data in manifest["provenance"]:
                    self.provenance.append(ProvenanceRecord(**record_data))
            
            # Import custom schemas
            if "custom_schemas" in manifest:
                self.custom_schemas = manifest["custom_schemas"]
            
            return True
            
        except Exception as e:
            print(f"Error importing manifest: {e}")
            return False
    
    def add_custom_schema(self, schema_name: str, schema: Dict[str, Any]) -> bool:
        """Add a custom metadata schema."""
        try:
            # Basic schema validation
            if not isinstance(schema, dict):
                return False
            
            if "type" not in schema or "properties" not in schema:
                return False
            
            self.custom_schemas[schema_name] = schema
            return True
            
        except Exception:
            return False
    
    def validate_custom_metadata(self, block_id: str, schema_name: str) -> List[str]:
        """Validate custom metadata against schema."""
        errors = []
        
        if block_id not in self.blocks:
            errors.append(f"Block {block_id} not found")
            return errors
        
        if schema_name not in self.custom_schemas:
            errors.append(f"Schema {schema_name} not found")
            return errors
        
        metadata = self.blocks[block_id]
        schema = self.custom_schemas[schema_name]
        custom_data = metadata.custom_metadata
        
        # Basic validation (simplified JSON Schema validation)
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if prop in custom_data:
                    value = custom_data[prop]
                    expected_type = prop_schema.get("type")
                    
                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Property {prop} should be string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Property {prop} should be number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Property {prop} should be boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Property {prop} should be array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Property {prop} should be object")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the MAIF file."""
        stats = {
            "file_info": {
                "version": self.header.version,
                "created": self.header.created,
                "modified": self.header.modified,
                "creator": self.header.creator_agent,
                "file_id": self.header.file_id
            },
            "blocks": {
                "total": len(self.blocks),
                "by_type": {},
                "by_content_type": {},
                "by_compression": {},
                "total_size": 0,
                "average_size": 0
            },
            "provenance": {
                "total_records": len(self.provenance),
                "by_operation": {},
                "by_agent": {},
                "time_span": None
            },
            "dependencies": {
                "total_dependencies": 0,
                "blocks_with_dependencies": 0,
                "dependency_errors": len(self.validate_dependencies())
            }
        }
        
        # Block statistics
        total_size = 0
        dependency_count = 0
        
        for metadata in self.blocks.values():
            # Block types
            stats["blocks"]["by_type"][metadata.block_type] = \
                stats["blocks"]["by_type"].get(metadata.block_type, 0) + 1
            
            # Content types
            stats["blocks"]["by_content_type"][metadata.content_type] = \
                stats["blocks"]["by_content_type"].get(metadata.content_type, 0) + 1
            
            # Compression
            stats["blocks"]["by_compression"][metadata.compression] = \
                stats["blocks"]["by_compression"].get(metadata.compression, 0) + 1
            
            # Size
            total_size += metadata.size
            
            # Dependencies
            if metadata.dependencies:
                dependency_count += len(metadata.dependencies)
                stats["dependencies"]["blocks_with_dependencies"] += 1
        
        stats["blocks"]["total_size"] = total_size
        stats["blocks"]["average_size"] = total_size / len(self.blocks) if self.blocks else 0
        stats["dependencies"]["total_dependencies"] = dependency_count
        
        # Provenance statistics
        timestamps = []
        for record in self.provenance:
            # Operation types
            stats["provenance"]["by_operation"][record.operation_type] = \
                stats["provenance"]["by_operation"].get(record.operation_type, 0) + 1
            
            # Agents
            stats["provenance"]["by_agent"][record.agent_id] = \
                stats["provenance"]["by_agent"].get(record.agent_id, 0) + 1
            
            # Timestamps
            try:
                timestamps.append(datetime.fromisoformat(record.timestamp.replace('Z', '+00:00')))
            except:
                pass
        
        if timestamps:
            timestamps.sort()
            stats["provenance"]["time_span"] = {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "duration_seconds": (timestamps[-1] - timestamps[0]).total_seconds()
            }
        
        return stats
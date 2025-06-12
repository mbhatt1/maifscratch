"""
MAIF SDK Artifact - High-level interface for working with MAIF artifacts.

Provides a convenient object-oriented interface for creating and managing
MAIF artifacts while leveraging the high-performance client underneath.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass, asdict

from .types import ContentType, SecurityLevel, CompressionLevel, ContentMetadata, SecurityOptions, ProcessingOptions


@dataclass
class ContentItem:
    """Represents a single content item within an artifact."""
    content_type: ContentType
    data: bytes
    metadata: Optional[ContentMetadata] = None
    block_id: Optional[str] = None
    size: Optional[int] = None
    hash_value: Optional[str] = None
    
    def __post_init__(self):
        if self.size is None:
            self.size = len(self.data)


class Artifact:
    """
    High-level interface for MAIF artifacts.
    
    An artifact represents a logical collection of related content items
    that can be efficiently stored and retrieved using the MAIF format.
    """
    
    def __init__(self, name: str = "untitled", 
                 client=None,
                 security_level: SecurityLevel = SecurityLevel.PUBLIC,
                 compression_level: CompressionLevel = CompressionLevel.BALANCED,
                 enable_embeddings: bool = True):
        self.name = name
        self.client = client
        self.security_level = security_level
        self.compression_level = compression_level
        self.enable_embeddings = enable_embeddings
        
        # Artifact metadata
        self.created_at = time.time()
        self.modified_at = self.created_at
        self.version = "1.0.0"
        self.description = ""
        self.tags = []
        self.custom_metadata = {}
        
        # Content tracking
        self._content_items: List[ContentItem] = []
        self._filepath: Optional[Path] = None
        self._is_loaded = False
    
    def add_text(self, text: str, 
                title: Optional[str] = None,
                description: Optional[str] = None,
                language: str = "en",
                **kwargs) -> str:
        """
        Add text content to the artifact.
        
        Args:
            text: Text content to add
            title: Optional title for the text
            description: Optional description
            language: Language code (default: "en")
            **kwargs: Additional metadata
            
        Returns:
            Block ID of the added content
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            language=language,
            custom=kwargs
        )
        
        return self._add_content(
            data=text.encode('utf-8'),
            content_type=ContentType.TEXT,
            metadata=metadata
        )
    
    def add_image(self, image_data: bytes,
                 title: Optional[str] = None,
                 description: Optional[str] = None,
                 format: str = "unknown",
                 **kwargs) -> str:
        """
        Add image content to the artifact.
        
        Args:
            image_data: Raw image bytes
            title: Optional title for the image
            description: Optional description
            format: Image format (e.g., "jpeg", "png")
            **kwargs: Additional metadata
            
        Returns:
            Block ID of the added content
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            custom={"format": format, **kwargs}
        )
        
        return self._add_content(
            data=image_data,
            content_type=ContentType.IMAGE,
            metadata=metadata
        )
    
    def add_video(self, video_data: bytes,
                 title: Optional[str] = None,
                 description: Optional[str] = None,
                 format: str = "unknown",
                 duration: Optional[float] = None,
                 **kwargs) -> str:
        """
        Add video content to the artifact.
        
        Args:
            video_data: Raw video bytes
            title: Optional title for the video
            description: Optional description
            format: Video format (e.g., "mp4", "avi")
            duration: Video duration in seconds
            **kwargs: Additional metadata
            
        Returns:
            Block ID of the added content
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            custom={"format": format, "duration": duration, **kwargs}
        )
        
        return self._add_content(
            data=video_data,
            content_type=ContentType.VIDEO,
            metadata=metadata
        )
    
    def add_document(self, document_data: bytes,
                    title: Optional[str] = None,
                    description: Optional[str] = None,
                    format: str = "unknown",
                    **kwargs) -> str:
        """
        Add document content to the artifact.
        
        Args:
            document_data: Raw document bytes
            title: Optional title for the document
            description: Optional description
            format: Document format (e.g., "pdf", "docx")
            **kwargs: Additional metadata
            
        Returns:
            Block ID of the added content
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            custom={"format": format, **kwargs}
        )
        
        return self._add_content(
            data=document_data,
            content_type=ContentType.DOCUMENT,
            metadata=metadata
        )
    
    def add_data(self, data: bytes,
                title: Optional[str] = None,
                description: Optional[str] = None,
                data_type: str = "binary",
                **kwargs) -> str:
        """
        Add arbitrary data content to the artifact.
        
        Args:
            data: Raw data bytes
            title: Optional title for the data
            description: Optional description
            data_type: Type of data (e.g., "json", "binary", "csv")
            **kwargs: Additional metadata
            
        Returns:
            Block ID of the added content
        """
        metadata = ContentMetadata(
            title=title,
            description=description,
            custom={"data_type": data_type, **kwargs}
        )
        
        return self._add_content(
            data=data,
            content_type=ContentType.DATA,
            metadata=metadata
        )
    
    def _add_content(self, data: bytes, content_type: ContentType, 
                    metadata: Optional[ContentMetadata] = None) -> str:
        """Internal method to add content with proper configuration."""
        
        # Create security options
        security_options = SecurityOptions(
            level=self.security_level,
            encrypt=(self.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]),
            sign=True,
            anonymize=(self.security_level == SecurityLevel.RESTRICTED)
        )
        
        # Create processing options
        processing_options = ProcessingOptions(
            compression=self.compression_level,
            generate_embeddings=self.enable_embeddings,
            extract_knowledge=True,
            enable_search=True,
            use_acam=True
        )
        
        # If we have a client and filepath, write directly
        if self.client and self._filepath:
            block_id = self.client.write_content(
                filepath=self._filepath,
                content=data,
                content_type=content_type,
                metadata=metadata,
                security_options=security_options,
                processing_options=processing_options
            )
        else:
            # Store in memory for later saving
            from uuid import uuid4
            block_id = str(uuid4())
        
        # Track content item
        content_item = ContentItem(
            content_type=content_type,
            data=data,
            metadata=metadata,
            block_id=block_id
        )
        self._content_items.append(content_item)
        self.modified_at = time.time()
        
        return block_id
    
    def get_content(self, content_type: Optional[ContentType] = None,
                   block_id: Optional[str] = None) -> Iterator[Dict]:
        """
        Get content from the artifact.
        
        Args:
            content_type: Filter by content type (optional)
            block_id: Get specific block by ID (optional)
            
        Yields:
            Dictionary with content data and metadata
        """
        if self.client and self._filepath and self._is_loaded:
            # Read from file using client
            yield from self.client.read_content(
                filepath=self._filepath,
                content_type=content_type,
                block_id=block_id
            )
        else:
            # Read from memory
            for item in self._content_items:
                if content_type and item.content_type != content_type:
                    continue
                if block_id and item.block_id != block_id:
                    continue
                
                yield {
                    'block_id': item.block_id,
                    'content_type': item.content_type.value,
                    'data': item.data,
                    'metadata': asdict(item.metadata) if item.metadata else {},
                    'size': item.size,
                    'hash': item.hash_value
                }
    
    def get_text_content(self) -> Iterator[str]:
        """Get all text content as strings."""
        for content in self.get_content(ContentType.TEXT):
            yield content['data'].decode('utf-8')
    
    def get_metadata(self) -> Dict:
        """Get artifact metadata."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'tags': self.tags,
            'security_level': self.security_level.value,
            'compression_level': self.compression_level.value,
            'enable_embeddings': self.enable_embeddings,
            'content_count': len(self._content_items),
            'content_types': list(set(item.content_type.value for item in self._content_items)),
            'total_size': sum(item.size or 0 for item in self._content_items),
            'custom_metadata': self.custom_metadata
        }
    
    def save(self, filepath: Union[str, Path], client=None) -> Path:
        """
        Save the artifact to a MAIF file.
        
        Args:
            filepath: Path where to save the artifact
            client: Optional client to use (uses self.client if not provided)
            
        Returns:
            Path to the saved file
        """
        filepath = Path(filepath)
        client = client or self.client
        
        if not client:
            # Import here to avoid circular imports
            from .client import MAIFClient
            client = MAIFClient()
        
        # Save artifact metadata as a special block
        metadata_block = self._create_metadata_block()
        client.write_content(
            filepath=filepath,
            content=json.dumps(metadata_block).encode('utf-8'),
            content_type=ContentType.DATA,
            metadata=ContentMetadata(
                title="Artifact Metadata",
                description=f"Metadata for artifact: {self.name}",
                custom={"artifact_metadata": True}
            )
        )
        
        # Save all content items if not already saved
        if not self._is_loaded:
            for item in self._content_items:
                if not item.block_id:  # Not yet saved
                    security_options = SecurityOptions(
                        level=self.security_level,
                        encrypt=(self.security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]),
                        sign=True
                    )
                    
                    processing_options = ProcessingOptions(
                        compression=self.compression_level,
                        generate_embeddings=self.enable_embeddings
                    )
                    
                    item.block_id = client.write_content(
                        filepath=filepath,
                        content=item.data,
                        content_type=item.content_type,
                        metadata=item.metadata,
                        security_options=security_options,
                        processing_options=processing_options
                    )
        
        self._filepath = filepath
        self._is_loaded = True
        return filepath
    
    def load(self, filepath: Union[str, Path], client=None) -> 'Artifact':
        """
        Load an artifact from a MAIF file.
        
        Args:
            filepath: Path to the MAIF file
            client: Optional client to use
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        client = client or self.client
        
        if not client:
            from .client import MAIFClient
            client = MAIFClient()
        
        self.client = client
        self._filepath = filepath
        
        # Load artifact metadata
        for content in client.read_content(filepath):
            metadata = content.get('metadata', {})
            if metadata.get('artifact_metadata'):
                try:
                    artifact_meta = json.loads(content['data'].decode('utf-8'))
                    self._load_metadata(artifact_meta)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        
        self._is_loaded = True
        return self
    
    def _create_metadata_block(self) -> Dict:
        """Create metadata block for saving."""
        return {
            'artifact_name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'tags': self.tags,
            'security_level': self.security_level.value,
            'compression_level': self.compression_level.value,
            'enable_embeddings': self.enable_embeddings,
            'custom_metadata': self.custom_metadata
        }
    
    def _load_metadata(self, metadata: Dict):
        """Load metadata from saved block."""
        self.name = metadata.get('artifact_name', self.name)
        self.description = metadata.get('description', '')
        self.version = metadata.get('version', '1.0.0')
        self.created_at = metadata.get('created_at', time.time())
        self.modified_at = metadata.get('modified_at', time.time())
        self.tags = metadata.get('tags', [])
        self.custom_metadata = metadata.get('custom_metadata', {})
        
        # Load enum values safely
        try:
            self.security_level = SecurityLevel(metadata.get('security_level', 'public'))
        except ValueError:
            self.security_level = SecurityLevel.PUBLIC
            
        try:
            self.compression_level = CompressionLevel(metadata.get('compression_level', 'balanced'))
        except ValueError:
            self.compression_level = CompressionLevel.BALANCED
        
        self.enable_embeddings = metadata.get('enable_embeddings', True)
    
    def __len__(self) -> int:
        """Return number of content items."""
        return len(self._content_items)
    
    def __str__(self) -> str:
        """String representation of the artifact."""
        return f"Artifact(name='{self.name}', items={len(self._content_items)}, size={sum(item.size or 0 for item in self._content_items)} bytes)"
    
    def __repr__(self) -> str:
        """Detailed representation of the artifact."""
        return (f"Artifact(name='{self.name}', description='{self.description}', "
                f"items={len(self._content_items)}, security={self.security_level.value}, "
                f"compression={self.compression_level.value})")
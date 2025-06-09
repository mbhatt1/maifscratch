"""
MAIF SDK Types - Simple enums and data classes for the SDK
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class ContentType(Enum):
    """Simplified content types for SDK users."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    DATA = "data"

class SecurityLevel(Enum):
    """Security levels for content protection."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class CompressionLevel(Enum):
    """Compression levels for storage optimization."""
    NONE = "none"
    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"

@dataclass
class ContentMetadata:
    """Metadata for content items."""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list] = None
    source: Optional[str] = None
    language: str = "en"
    custom: Optional[Dict[str, Any]] = None

@dataclass
class SecurityOptions:
    """Security configuration for content."""
    level: SecurityLevel = SecurityLevel.PUBLIC
    encrypt: bool = False
    sign: bool = True
    anonymize: bool = False
    access_control: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingOptions:
    """Processing options for content."""
    compression: CompressionLevel = CompressionLevel.BALANCED
    generate_embeddings: bool = True
    extract_knowledge: bool = True
    enable_search: bool = True
    use_acam: bool = True  # Use Adaptive Cross-Modal Attention
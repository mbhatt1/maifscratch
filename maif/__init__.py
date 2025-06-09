"""
MAIF (Multimodal Artifact File Format) Library

A comprehensive library for creating, managing, and analyzing MAIF files.
MAIF is an AI-native file format designed for multimodal content with
embedded security, semantics, and provenance tracking.
"""

from .core import MAIFEncoder, MAIFDecoder, MAIFParser, MAIFBlock, MAIFVersion
from .security import MAIFSigner
from .privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode, AccessRule, DifferentialPrivacy, SecureMultipartyComputation, ZeroKnowledgeProof
from .semantic import (
    SemanticEmbedder, SemanticEmbedding, KnowledgeTriple,
    CrossModalAttention, HierarchicalSemanticCompression,
    CryptographicSemanticBinding, DeepSemanticUnderstanding
)

# Import enhanced algorithms from semantic_optimized
try:
    from .semantic_optimized import (
        AdaptiveCrossModalAttention,
        HierarchicalSemanticCompression as EnhancedHierarchicalSemanticCompression,
        CryptographicSemanticBinding as EnhancedCryptographicSemanticBinding,
        AttentionWeights
    )
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ENHANCED_ALGORITHMS_AVAILABLE = False
from .forensics import ForensicAnalyzer
from .compression import MAIFCompressor, CompressionMetadata
from .binary_format import MAIFBinaryParser, MAIFBinaryWriter
from .validation import MAIFValidator, MAIFRepairTool
from .metadata import MAIFMetadataManager
from .streaming import MAIFStreamReader, MAIFStreamWriter
from .integration_enhanced import EnhancedMAIFProcessor, ConversionResult

# Import simple API for easy access
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from maif_api import MAIF, create_maif, load_maif, quick_text_maif, quick_multimodal_maif
    SIMPLE_API_AVAILABLE = True
except ImportError:
    SIMPLE_API_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "MAIF Development Team"
__license__ = "MIT"

__all__ = [
    # Core functionality
    'MAIFEncoder',
    'MAIFDecoder',
    'MAIFParser',
    'MAIFBlock',
    'MAIFVersion',
    
    # Security
    'MAIFSigner',
    
    # Privacy
    'PrivacyEngine',
    'PrivacyPolicy',
    'PrivacyLevel',
    'EncryptionMode',
    'AccessRule',
    'DifferentialPrivacy',
    'SecureMultipartyComputation',
    'ZeroKnowledgeProof',
    
    # Semantics
    'SemanticEmbedder',
    'SemanticEmbedding',
    'KnowledgeTriple',
    'CrossModalAttention',
    'HierarchicalSemanticCompression',
    'CryptographicSemanticBinding',
    'DeepSemanticUnderstanding',
    
    # Enhanced Novel Algorithms (if available)
    'AdaptiveCrossModalAttention',
    'EnhancedHierarchicalSemanticCompression',
    'EnhancedCryptographicSemanticBinding',
    'AttentionWeights',
    'ENHANCED_ALGORITHMS_AVAILABLE',
    
    # Forensics
    'ForensicAnalyzer',
    
    # Compression
    'MAIFCompressor',
    'CompressionMetadata',
    
    # Binary Format
    'MAIFBinaryParser',
    'MAIFBinaryWriter',
    
    # Validation
    'MAIFValidator',
    'MAIFRepairTool',
    
    # Metadata
    'MAIFMetadataManager',
    
    # Streaming
    'MAIFStreamReader',
    'MAIFStreamWriter',
    
    # Integration
    'EnhancedMAIFProcessor',
    'ConversionResult',
    
    # Simple API (if available)
    'MAIF',
    'create_maif',
    'load_maif',
    'quick_text_maif',
    'quick_multimodal_maif',
    'SIMPLE_API_AVAILABLE',
]
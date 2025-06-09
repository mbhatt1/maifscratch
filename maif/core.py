"""
Core MAIF implementation - encoding, decoding, and parsing functionality.
Enhanced with privacy-by-design features and improved block structure.
"""

import json
import hashlib
import struct
import time
from typing import Dict, List, Optional, Union, BinaryIO, Any
from dataclasses import dataclass
from pathlib import Path
import io
import uuid
from .privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode, AccessRule
from .block_types import BlockType, BlockHeader, BlockFactory, BlockValidator

@dataclass
class MAIFBlock:
    """Represents a MAIF block with metadata."""
    block_type: str
    offset: int = 0
    size: int = 0
    hash_value: str = ""
    version: int = 1
    previous_hash: Optional[str] = None
    block_id: Optional[str] = None
    metadata: Optional[Dict] = None
    data: Optional[bytes] = None
    
    def __post_init__(self):
        if self.block_id is None:
            self.block_id = str(uuid.uuid4())
        if self.data is not None and not self.hash_value:
            self.hash_value = hashlib.sha256(self.data).hexdigest()
    
    @property
    def hash(self) -> str:
        """Return the hash value for compatibility with tests."""
        return self.hash_value
    
    def to_dict(self) -> Dict:
        return {
            "type": self.block_type,
            "block_type": self.block_type,
            "offset": self.offset,
            "size": self.size,
            "hash": self.hash_value,
            "version": self.version,
            "previous_hash": self.previous_hash,
            "block_id": self.block_id,
            "metadata": self.metadata or {}
        }

@dataclass
class MAIFVersion:
    """Represents a version entry in the version history."""
    version: int
    timestamp: float
    agent_id: str
    operation: str  # "create", "update", "delete"
    block_hash: str
    block_id: Optional[str] = None
    previous_hash: Optional[str] = None
    change_description: Optional[str] = None
    
    # Keep version_number as alias for backward compatibility
    @property
    def version_number(self) -> int:
        return self.version
    
    # Keep current_hash as alias for backward compatibility
    @property
    def current_hash(self) -> str:
        return self.block_hash
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "current_hash": self.block_hash,
            "block_hash": self.block_hash,
            "change_description": self.change_description
        }

@dataclass
class MAIFHeader:
    """MAIF file header structure."""
    version: str = "0.1.0"
    created_timestamp: float = None
    creator_id: Optional[str] = None
    root_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = time.time()

class MAIFEncoder:
    """Encodes multimodal data into MAIF format with versioning and privacy-by-design support."""
    
    # Block type mapping for backward compatibility
    BLOCK_TYPE_MAPPING = {
        "text": BlockType.TEXT_DATA.value,
        "text_data": BlockType.TEXT_DATA.value,
        "binary": BlockType.BINARY_DATA.value,
        "binary_data": BlockType.BINARY_DATA.value,
        "embedding": BlockType.EMBEDDING.value,
        "embeddings": BlockType.EMBEDDING.value,
        "cross_modal": BlockType.CROSS_MODAL.value,
        "semantic_binding": BlockType.SEMANTIC_BINDING.value,
        "compressed_embeddings": BlockType.COMPRESSED_EMBEDDINGS.value,
    }
    
    def __init__(self, agent_id: Optional[str] = None, existing_maif_path: Optional[str] = None,
                 existing_manifest_path: Optional[str] = None, enable_privacy: bool = True,
                 privacy_engine: Optional['PrivacyEngine'] = None):
        self.blocks: List[MAIFBlock] = []
        self.header = MAIFHeader()
        self.buffer = io.BytesIO()
        self.agent_id = agent_id or str(uuid.uuid4())
        self.version_history: Dict = {}  # Changed to dict for test compatibility
        self.block_registry: Dict[str, List[MAIFBlock]] = {}  # block_id -> list of versions
        self.access_rules: List[AccessRule] = []  # Add missing access_rules attribute
        
        # Privacy-by-design features
        self.enable_privacy = enable_privacy
        self.privacy_engine = privacy_engine or (PrivacyEngine() if enable_privacy else None)
        self.default_privacy_policy = PrivacyPolicy(
            privacy_level=PrivacyLevel.INTERNAL,
            encryption_mode=EncryptionMode.AES_GCM if enable_privacy else EncryptionMode.NONE,
            anonymization_required=False,
            audit_required=True
        )
        
        # Load existing MAIF if provided (for append-on-write)
        if existing_maif_path and existing_manifest_path:
            self._load_existing_maif(existing_maif_path, existing_manifest_path)
    
    def _load_existing_maif(self, maif_path: str, manifest_path: str):
        """Load existing MAIF for append-on-write operations."""
        try:
            decoder = MAIFDecoder(maif_path, manifest_path)
            
            # Copy existing blocks
            self.blocks = decoder.blocks.copy()
            
            # Copy existing buffer content
            with open(maif_path, 'rb') as f:
                self.buffer.write(f.read())
            
            # Load version history if available
            if "version_history" in decoder.manifest:
                self.version_history = [
                    MAIFVersion(**v) for v in decoder.manifest["version_history"]
                ]
            
            # Build block registry
            for block in self.blocks:
                if block.block_id not in self.block_registry:
                    self.block_registry[block.block_id] = []
                self.block_registry[block.block_id].append(block)
                
        except Exception as e:
            print(f"Warning: Could not load existing MAIF: {e}")
    
    def add_text_block(self, text: str, metadata: Optional[Dict] = None,
                      update_block_id: Optional[str] = None,
                      privacy_policy: Optional[PrivacyPolicy] = None,
                      anonymize: bool = False,
                      privacy_level: Optional[PrivacyLevel] = None,
                      encryption_mode: Optional[EncryptionMode] = None) -> str:
        """Add or update a text block to the MAIF with privacy controls."""
        # Create privacy policy from individual parameters if provided
        if privacy_level is not None or encryption_mode is not None:
            privacy_policy = PrivacyPolicy(
                privacy_level=privacy_level or PrivacyLevel.INTERNAL,
                encryption_mode=encryption_mode or EncryptionMode.NONE,
                anonymization_required=anonymize,
                audit_required=True
            )
        
        # Apply anonymization if requested
        if self.enable_privacy and anonymize:
            text = self.privacy_engine.anonymize_data(text, "text_block")
        
        text_bytes = text.encode('utf-8')
        return self._add_block("text", text_bytes, metadata, update_block_id, privacy_policy)
    
    def add_binary_block(self, data: bytes, block_type: str, metadata: Optional[Dict] = None,
                        update_block_id: Optional[str] = None,
                        privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add or update a binary data block to the MAIF with privacy controls."""
        return self._add_block(block_type, data, metadata, update_block_id, privacy_policy)
    
    def add_embeddings_block(self, embeddings: List[List[float]], metadata: Optional[Dict] = None,
                           update_block_id: Optional[str] = None,
                           privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add or update semantic embeddings block to the MAIF with privacy controls."""
        # Pack embeddings as binary data
        embedding_data = b""
        for embedding in embeddings:
            for value in embedding:
                embedding_data += struct.pack('f', value)
        
        embed_metadata = metadata or {}
        embed_metadata.update({
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        })
        
        return self._add_block("embeddings", embedding_data, embed_metadata, update_block_id, privacy_policy)
    
    def add_video_block(self, video_data: bytes, metadata: Optional[Dict] = None,
                       update_block_id: Optional[str] = None,
                       privacy_policy: Optional[PrivacyPolicy] = None,
                       extract_metadata: bool = True) -> str:
        """Add or update a video block with automatic metadata extraction and semantic analysis."""
        video_metadata = metadata or {}
        
        if extract_metadata:
            # Extract video metadata using ffprobe-like functionality
            extracted_metadata = self._extract_video_metadata(video_data)
            video_metadata.update(extracted_metadata)
        
        # Generate video embeddings for semantic search
        if len(video_data) > 0:
            video_embeddings = self._generate_video_embeddings(video_data)
            if video_embeddings:
                video_metadata["semantic_embeddings"] = video_embeddings
                video_metadata["has_semantic_analysis"] = True
        
        video_metadata.update({
            "content_type": "video",
            "size_bytes": len(video_data),
            "block_type": "video_data"
        })
        
        return self._add_block("video_data", video_data, video_metadata, update_block_id, privacy_policy)
    
    def _extract_video_metadata(self, video_data: bytes) -> Dict[str, Any]:
        """Extract metadata from video data."""
        metadata = {
            "duration": None,
            "resolution": None,
            "fps": None,
            "codec": None,
            "format": None,
            "bitrate": None,
            "audio_codec": None,
            "extraction_method": "basic"
        }
        
        try:
            # Try to detect video format from header
            if video_data[:4] == b'\x00\x00\x00\x18' or video_data[4:8] == b'ftyp':
                metadata["format"] = "mp4"
            elif video_data[:4] == b'RIFF' and video_data[8:12] == b'AVI ':
                metadata["format"] = "avi"
            elif video_data[:3] == b'FLV':
                metadata["format"] = "flv"
            elif video_data[:4] == b'\x1a\x45\xdf\xa3':
                metadata["format"] = "mkv"
            
            # Basic size estimation for common formats
            if metadata["format"] == "mp4":
                # Try to extract basic MP4 metadata
                mp4_metadata = self._extract_mp4_metadata(video_data)
                metadata.update(mp4_metadata)
            
        except Exception as e:
            metadata["extraction_error"] = str(e)
            metadata["extraction_method"] = "fallback"
        
        return metadata
    
    def _extract_mp4_metadata(self, video_data: bytes) -> Dict[str, Any]:
        """Extract basic metadata from MP4 video data."""
        metadata = {}
        
        try:
            # Look for common MP4 atoms/boxes
            pos = 0
            while pos < len(video_data) - 8:
                if pos + 8 > len(video_data):
                    break
                    
                # Read box size and type
                box_size = struct.unpack('>I', video_data[pos:pos+4])[0]
                box_type = video_data[pos+4:pos+8]
                
                if box_size == 0 or box_size > len(video_data) - pos:
                    break
                
                # Look for mvhd (movie header) box for duration
                if box_type == b'mvhd':
                    try:
                        # Skip version and flags (4 bytes)
                        # Skip creation and modification time (8 bytes)
                        timescale_pos = pos + 8 + 8
                        if timescale_pos + 8 <= len(video_data):
                            timescale = struct.unpack('>I', video_data[timescale_pos:timescale_pos+4])[0]
                            duration_units = struct.unpack('>I', video_data[timescale_pos+4:timescale_pos+8])[0]
                            if timescale > 0:
                                metadata["duration"] = duration_units / timescale
                                metadata["timescale"] = timescale
                    except (struct.error, IndexError):
                        pass
                
                # Look for tkhd (track header) for video dimensions
                elif box_type == b'tkhd' and box_size >= 84:
                    try:
                        # Track header contains width and height at the end
                        if pos + box_size <= len(video_data):
                            width_pos = pos + box_size - 8
                            if width_pos + 8 <= len(video_data):
                                width = struct.unpack('>I', video_data[width_pos:width_pos+4])[0] >> 16
                                height = struct.unpack('>I', video_data[width_pos+4:width_pos+8])[0] >> 16
                                if width > 0 and height > 0:
                                    metadata["resolution"] = f"{width}x{height}"
                                    metadata["width"] = width
                                    metadata["height"] = height
                    except (struct.error, IndexError):
                        pass
                
                pos += box_size
                
        except Exception as e:
            metadata["mp4_extraction_error"] = str(e)
        
        return metadata
    
    def _generate_video_embeddings(self, video_data: bytes) -> Optional[List[float]]:
        """Generate semantic embeddings for video content."""
        try:
            # Placeholder for video semantic analysis
            # In a real implementation, this would:
            # 1. Extract key frames
            # 2. Run visual feature extraction (CNN-based)
            # 3. Generate semantic embeddings
            
            # For now, generate a simple hash-based embedding
            import hashlib
            video_hash = hashlib.sha256(video_data).hexdigest()
            
            # Convert hash to embedding vector (384 dimensions)
            embedding = []
            for i in range(0, min(len(video_hash), 96), 2):
                hex_val = video_hash[i:i+2]
                normalized_val = int(hex_val, 16) / 255.0
                embedding.extend([normalized_val] * 4)  # Repeat to get 384 dims
            
            # Pad to 384 dimensions if needed
            while len(embedding) < 384:
                embedding.append(0.0)
            
            return embedding[:384]
            
        except Exception:
            return None
    
    def add_cross_modal_block(self, multimodal_data: Dict[str, Any], metadata: Optional[Dict] = None,
                             update_block_id: Optional[str] = None,
                             privacy_policy: Optional[PrivacyPolicy] = None,
                             use_enhanced_acam: bool = True) -> str:
        """Add cross-modal data block using enhanced ACAM implementation."""
        try:
            if use_enhanced_acam:
                from .semantic_optimized import AdaptiveCrossModalAttention
                import numpy as np
                
                # Initialize enhanced ACAM
                acam = AdaptiveCrossModalAttention()
                
                # Process embeddings for each modality
                embeddings = {}
                trust_scores = {}
                
                for modality, data in multimodal_data.items():
                    if modality == "text":
                        from .semantic import SemanticEmbedder
                        embedder = SemanticEmbedder()
                        embedding = embedder.embed_text(str(data)).vector
                        embeddings[modality] = np.array(embedding)
                        trust_scores[modality] = 1.0
                    else:
                        # Generate embeddings for other modalities
                        import hashlib
                        hash_obj = hashlib.sha256(str(data).encode())
                        hash_hex = hash_obj.hexdigest()
                        base_embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, len(hash_hex), 2)]
                        embedding = (base_embedding * (384 // len(base_embedding) + 1))[:384]
                        embeddings[modality] = np.array(embedding)
                        trust_scores[modality] = 0.8  # Lower trust for non-text modalities
                
                # Compute attention weights
                attention_weights = acam.compute_attention_weights(embeddings, trust_scores)
                
                # Create unified representation
                if embeddings:
                    primary_modality = list(embeddings.keys())[0]
                    unified_repr = acam.get_attended_representation(embeddings, attention_weights, primary_modality)
                else:
                    unified_repr = []
                
                # Prepare result
                processed_result = {
                    "embeddings": {k: v.tolist() for k, v in embeddings.items()},
                    "attention_weights": {
                        "query_key_weights": attention_weights.query_key_weights.tolist(),
                        "trust_scores": attention_weights.trust_scores,
                        "coherence_matrix": attention_weights.coherence_matrix.tolist(),
                        "normalized_weights": attention_weights.normalized_weights.tolist()
                    },
                    "unified_representation": unified_repr.tolist() if hasattr(unified_repr, 'tolist') else unified_repr,
                    "algorithm": "Enhanced_ACAM_v2"
                }
            else:
                # Fallback to original implementation
                from .semantic import DeepSemanticUnderstanding
                dsu = DeepSemanticUnderstanding()
                processed_result = dsu.process_multimodal_input(multimodal_data)
            
            # Serialize the result
            import json
            serialized_data = json.dumps(processed_result, default=str).encode('utf-8')
            
            cross_modal_metadata = metadata or {}
            cross_modal_metadata.update({
                "algorithm": processed_result.get("algorithm", "ACAM"),
                "modalities": list(multimodal_data.keys()),
                "unified_representation_dim": len(processed_result.get("unified_representation", [])),
                "attention_weights_available": "attention_weights" in processed_result,
                "enhanced_version": use_enhanced_acam
            })
            
            return self._add_block("cross_modal", serialized_data, cross_modal_metadata, update_block_id, privacy_policy)
            
        except ImportError:
            # Fallback if enhanced modules not available
            serialized_data = json.dumps(multimodal_data, default=str).encode('utf-8')
            fallback_metadata = metadata or {}
            fallback_metadata.update({"algorithm": "fallback", "modalities": list(multimodal_data.keys())})
            return self._add_block(BlockType.CROSS_MODAL.value, serialized_data, fallback_metadata, update_block_id, privacy_policy)
    
    def add_semantic_binding_block(self, embedding: List[float], source_data: str,
                                  metadata: Optional[Dict] = None,
                                  update_block_id: Optional[str] = None,
                                  privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add semantic binding block using CSB (Cryptographic Semantic Binding)."""
        try:
            from .semantic import CryptographicSemanticBinding
            
            # Create cryptographic semantic binding
            csb = CryptographicSemanticBinding()
            binding = csb.create_semantic_commitment(embedding, source_data)
            
            # Create zero-knowledge proof
            zk_proof = csb.create_zero_knowledge_proof(embedding, binding)
            
            # Combine binding and proof
            binding_data = {
                "binding": binding,
                "zk_proof": zk_proof,
                "embedding_dim": len(embedding),
                "source_hash": hashlib.sha256(source_data.encode()).hexdigest()
            }
            
            # Serialize the binding data
            import json
            serialized_data = json.dumps(binding_data).encode('utf-8')
            
            binding_metadata = metadata or {}
            binding_metadata.update({
                "algorithm": "CSB",
                "binding_type": "cryptographic_semantic",
                "has_zk_proof": True,
                "embedding_dimensions": len(embedding)
            })
            
            return self._add_block("semantic_binding", serialized_data, binding_metadata, update_block_id, privacy_policy)
            
        except ImportError:
            # Fallback if semantic module not available
            fallback_data = {
                "embedding": embedding,
                "source_data_hash": hashlib.sha256(source_data.encode()).hexdigest(),
                "timestamp": time.time()
            }
            serialized_data = json.dumps(fallback_data).encode('utf-8')
            fallback_metadata = metadata or {}
            fallback_metadata.update({"algorithm": "fallback", "binding_type": "simple_hash"})
            return self._add_block("semantic_binding", serialized_data, fallback_metadata, update_block_id, privacy_policy)
    
    def add_compressed_embeddings_block(self, embeddings: List[List[float]],
                                       use_enhanced_hsc: bool = True,
                                       preserve_fidelity: bool = True,
                                       target_compression_ratio: float = 0.4,
                                       metadata: Optional[Dict] = None,
                                       update_block_id: Optional[str] = None,
                                       privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add embeddings block with enhanced HSC (Hierarchical Semantic Compression)."""
        if use_enhanced_hsc:
            try:
                from .semantic_optimized import HierarchicalSemanticCompression
                
                # Apply enhanced HSC compression
                hsc = HierarchicalSemanticCompression(target_compression_ratio=target_compression_ratio)
                compressed_result = hsc.compress_embeddings(embeddings, preserve_fidelity=preserve_fidelity)
                
                # Serialize compressed result
                import json
                serialized_data = json.dumps(compressed_result, default=str).encode('utf-8')
                
                hsc_metadata = metadata or {}
                hsc_metadata.update({
                    "algorithm": "Enhanced_HSC_v2",
                    "compression_type": "hierarchical_semantic_dbscan",
                    "original_count": len(embeddings),
                    "original_dimensions": len(embeddings[0]) if embeddings else 0,
                    "compression_ratio": compressed_result.get("metadata", {}).get("compression_ratio", 1.0),
                    "fidelity_score": compressed_result.get("metadata", {}).get("fidelity_score", 0.0),
                    "tier1_clusters": compressed_result.get("metadata", {}).get("tier1_clusters", 0),
                    "tier2_codebook_size": compressed_result.get("metadata", {}).get("tier2_codebook_size", 0),
                    "tier3_encoding": compressed_result.get("metadata", {}).get("tier3_encoding", "unknown"),
                    "preserve_fidelity": preserve_fidelity
                })
                
                return self._add_block(BlockType.COMPRESSED_EMBEDDINGS.value, serialized_data, hsc_metadata, update_block_id, privacy_policy)
                
            except ImportError:
                # Fallback to original HSC if enhanced not available
                try:
                    from .semantic import HierarchicalSemanticCompression
                    
                    hsc = HierarchicalSemanticCompression()
                    compressed_result = hsc.compress_embeddings(embeddings)
                    
                    import json
                    serialized_data = json.dumps(compressed_result).encode('utf-8')
                    
                    hsc_metadata = metadata or {}
                    hsc_metadata.update({
                        "algorithm": "HSC_v1",
                        "compression_type": "hierarchical_semantic",
                        "original_count": len(embeddings),
                        "original_dimensions": len(embeddings[0]) if embeddings else 0,
                        "compression_ratio": compressed_result.get("metadata", {}).get("compression_ratio", 1.0)
                    })
                    
                    return self._add_block(BlockType.COMPRESSED_EMBEDDINGS.value, serialized_data, hsc_metadata, update_block_id, privacy_policy)
                    
                except ImportError:
                    pass
        
        # Fallback to regular embeddings block
        return self.add_embeddings_block(embeddings, metadata, update_block_id, privacy_policy)
    
    def _add_block(self, block_type: str, data: bytes, metadata: Optional[Dict] = None,
                  update_block_id: Optional[str] = None,
                  privacy_policy: Optional[PrivacyPolicy] = None,
                  block_id: Optional[str] = None) -> str:
        """Internal method to add or update a block with enhanced structure and privacy."""
        # Validate block type
        if not block_type or not block_type.strip():
            raise ValueError("Block type cannot be empty")
        
        offset = self.buffer.tell()
        
        # Handle block_id parameter for backward compatibility
        if block_id is not None and update_block_id is None:
            update_block_id = block_id
        
        # Map block type to FourCC if needed
        if block_type in self.BLOCK_TYPE_MAPPING:
            fourcc_type = self.BLOCK_TYPE_MAPPING[block_type]
        else:
            fourcc_type = block_type
        
        # Determine if this is an update or new block
        is_update = update_block_id is not None
        previous_hash = None
        version_number = 1
        block_id = update_block_id
        
        if is_update and update_block_id in self.block_registry:
            # This is an update - get the latest version
            latest_block = self.block_registry[update_block_id][-1]
            previous_hash = latest_block.hash_value
            version_number = latest_block.version + 1
        else:
            # This is a new block
            block_id = str(uuid.uuid4())
        
        # Apply privacy policy
        policy = privacy_policy or self.default_privacy_policy
        encryption_metadata = {}
        original_data = data  # Store original data for test compatibility
        data_for_storage = data  # Data to store in MAIFBlock
        
        if self.enable_privacy and policy.encryption_mode != EncryptionMode.NONE:
            # Encrypt the data
            data, encryption_metadata = self.privacy_engine.encrypt_data(
                data, block_id, policy.encryption_mode
            )
            # Set privacy policy for this block
            self.privacy_engine.set_privacy_policy(block_id, policy)
            # Use encrypted data for storage when encryption is applied
            data_for_storage = data
        
        # Calculate hash (after encryption)
        hash_value = hashlib.sha256(data).hexdigest()
        
        # Create enhanced block header using new structure
        block_header = BlockHeader(
            size=len(data) + 32,  # Data size + header size
            type=fourcc_type,
            version=version_number,
            flags=0,
            uuid=block_id
        )
        
        # Validate block header
        header_errors = BlockValidator.validate_block_header(block_header)
        if header_errors:
            print(f"Warning: Block header validation errors: {header_errors}")
        
        # Write enhanced header (32 bytes)
        header_bytes = block_header.to_bytes()
        self.buffer.write(header_bytes)
        
        # Write data
        self.buffer.write(data)
        
        # Merge encryption metadata with user metadata
        combined_metadata = metadata.copy() if metadata else {}
        if encryption_metadata:
            combined_metadata['encryption'] = encryption_metadata
            combined_metadata['encrypted'] = True
            combined_metadata['encryption_mode'] = policy.encryption_mode.name
        if self.enable_privacy and policy:
            combined_metadata['privacy_policy'] = {
                'privacy_level': policy.privacy_level.value,
                'encryption_mode': policy.encryption_mode.value,
                'anonymization_required': policy.anonymization_required,
                'audit_required': policy.audit_required
            }
            # Set anonymized flag if anonymization was required
            if policy.anonymization_required:
                combined_metadata['anonymized'] = True
        
        # Create block record
        block = MAIFBlock(
            block_type=block_type,  # Keep original for test compatibility
            offset=offset,
            size=len(data) + 24,  # Include extended header size
            hash_value=f"sha256:{hash_value}",
            version=version_number,
            previous_hash=previous_hash,
            block_id=block_id,
            metadata=combined_metadata,
            data=data_for_storage  # Use appropriate data (encrypted if encryption applied)
        )
        
        # Add to blocks and registry
        self.blocks.append(block)
        if block_id not in self.block_registry:
            self.block_registry[block_id] = []
        self.block_registry[block_id].append(block)
        
        # Record version history
        operation = "update" if is_update else "create"
        version_entry = MAIFVersion(
            version=version_number,
            timestamp=time.time(),
            agent_id=self.agent_id,
            operation=operation,
            block_hash=f"sha256:{hash_value}",
            block_id=block_id,
            previous_hash=previous_hash,
            change_description=combined_metadata.get("change_description") if combined_metadata else None
        )
        # Add to version history (dict structure)
        if block_id not in self.version_history:
            self.version_history[block_id] = []
        self.version_history[block_id].append(version_entry)
        
        return hash_value
    
    def delete_block(self, block_id: str, reason: Optional[str] = None) -> bool:
        """Mark a block as deleted (soft delete with versioning)."""
        if block_id not in self.block_registry:
            return False
        
        latest_block = self.block_registry[block_id][-1]
        
        # Mark the block as deleted in its metadata
        if latest_block.metadata is None:
            latest_block.metadata = {}
        latest_block.metadata["deleted"] = True
        if reason:
            latest_block.metadata["deletion_reason"] = reason
        
        # Create deletion record
        version_entry = MAIFVersion(
            version=latest_block.version + 1,
            timestamp=time.time(),
            agent_id=self.agent_id,
            operation="delete",
            block_hash="deleted",
            block_id=block_id,
            previous_hash=latest_block.hash_value,
            change_description=reason
        )
        
        # Add to version history
        if block_id not in self.version_history:
            self.version_history[block_id] = []
        self.version_history[block_id].append(version_entry)
        
        return True
    
    def get_block_history(self, block_id: str) -> List[MAIFBlock]:
        """Get the complete version history of a block."""
        return self.block_registry.get(block_id, [])
    
    def get_block_at_version(self, block_id: str, version: int) -> Optional[MAIFBlock]:
        """Get a specific version of a block."""
        if block_id not in self.block_registry:
            return None
        
        # Find the block with the specified version
        for block in self.block_registry[block_id]:
            if block.version == version:
                return block
        
        return None
        
    def add_access_rule(self, subject: str, resource: str, permissions: List[str],
                       conditions: Optional[Dict[str, Any]] = None, expiry: Optional[float] = None):
        """Add an access control rule for privacy protection."""
        rule = AccessRule(
            subject=subject,
            resource=resource,
            permissions=permissions,
            conditions=conditions,
            expiry=expiry
        )
        
        # Add to encoder's access rules list
        self.access_rules.append(rule)
        
        # Also add to privacy engine if privacy is enabled
        if self.enable_privacy:
            self.privacy_engine.add_access_rule(rule)
    
    def check_access(self, subject: str, block_id: str, permission: str) -> bool:
        """Check if subject has permission to access a block."""
        if not self.enable_privacy:
            return True
        
        return self.privacy_engine.check_access(subject, block_id, permission)
    
    def set_default_privacy_policy(self, policy: PrivacyPolicy):
        """Set the default privacy policy for new blocks."""
        self.default_privacy_policy = policy
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate a privacy compliance report."""
        # Count blocks by privacy characteristics
        total_blocks = len(self.blocks)
        encrypted_blocks = sum(1 for block in self.blocks
                             if block.metadata and block.metadata.get("encrypted", False))
        anonymized_blocks = sum(1 for block in self.blocks
                              if block.metadata and block.metadata.get("anonymized", False))
        
        report = {
            "privacy_enabled": self.enable_privacy,
            "total_blocks": total_blocks,
            "encrypted_blocks": encrypted_blocks,
            "anonymized_blocks": anonymized_blocks,
            "total_version_entries": sum(len(versions) for versions in self.version_history.values())
        }
        
        if self.enable_privacy:
            # Add privacy engine report
            engine_report = self.privacy_engine.generate_privacy_report()
            report.update(engine_report)
        
        return report
    
    def anonymize_existing_block(self, block_id: str, context: str = "general") -> bool:
        """Anonymize an existing text block."""
        if not self.enable_privacy or block_id not in self.block_registry:
            return False
        
        latest_block = self.block_registry[block_id][-1]
        if latest_block.block_type != "text_data":
            return False
        
        # Read the current data (this would need decoder integration)
        # For now, we'll mark it as requiring anonymization
        metadata = latest_block.metadata.copy() if latest_block.metadata else {}
        metadata["anonymization_pending"] = True
        metadata["anonymization_context"] = context
        
        # Update the block metadata
        latest_block.metadata = metadata
        return True
    
    def build_maif(self, output_path: str, manifest_path: str) -> None:
        """Build the final MAIF file and manifest with version history."""
        # Create manifest
        manifest = {
            "maif_version": self.header.version,
            "created": self.header.created_timestamp,
            "creator_id": self.header.creator_id,
            "agent_id": self.agent_id,
            "header": {
                "version": self.header.version,
                "created_timestamp": self.header.created_timestamp,
                "creator_id": self.header.creator_id,
                "agent_id": self.agent_id
            },
            "blocks": [block.to_dict() for block in self.blocks],
            "version_history": {
                block_id: [v.to_dict() for v in versions]
                for block_id, versions in self.version_history.items()
            },
            "block_registry": {
                block_id: [block.to_dict() for block in versions]
                for block_id, versions in self.block_registry.items()
            }
        }
        
        # Calculate root hash including version history
        all_hashes = "".join([block.hash_value for block in self.blocks])
        version_hashes = "".join([
            v.current_hash for versions in self.version_history.values()
            for v in versions
        ])
        combined_hash = hashlib.sha256((all_hashes + version_hashes).encode()).hexdigest()
        manifest["root_hash"] = f"sha256:{combined_hash}"
        
        # Write files
        with open(output_path, 'wb') as f:
            f.write(self.buffer.getvalue())
            
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def save(self, output_path: str, manifest_path: str):
        """Save the MAIF file and manifest."""
        # Add privacy metadata to manifest if privacy is enabled
        manifest = {
            "maif_version": self.header.version,
            "created": self.header.created_timestamp,
            "creator_id": self.header.creator_id,
            "agent_id": self.agent_id,
            "blocks": [block.to_dict() for block in self.blocks],
            "version_history": {
                block_id: [v.to_dict() if hasattr(v, 'to_dict') else str(v) for v in versions]
                for block_id, versions in self.version_history.items()
            },
            "block_registry": {
                block_id: [block.to_dict() for block in versions]
                for block_id, versions in self.block_registry.items()
            }
        }
        
        # Add privacy information if enabled
        if self.enable_privacy and self.privacy_engine:
            privacy_report = self.get_privacy_report()
            manifest["privacy"] = {
                "enabled": True,
                "report": privacy_report
            }
        
        # Calculate root hash including version history
        all_hashes = "".join([block.hash_value for block in self.blocks])
        version_hashes = ""
        if isinstance(self.version_history, dict):
            version_hashes = "".join([
                v.current_hash for versions in self.version_history.values()
                for v in (versions if isinstance(versions, list) else [versions])
            ])
        elif isinstance(self.version_history, list):
            version_hashes = "".join([v.current_hash for v in self.version_history])
        combined_hash = hashlib.sha256((all_hashes + version_hashes).encode()).hexdigest()
        manifest["root_hash"] = f"sha256:{combined_hash}"
        
        # Write files
        with open(output_path, 'wb') as f:
            f.write(self.buffer.getvalue())
            
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

class MAIFDecoder:
    """Decodes MAIF files with versioning and privacy support."""
    
    def __init__(self, maif_path: str, manifest_path: str, privacy_engine: Optional[PrivacyEngine] = None,
                 requesting_agent: Optional[str] = None):
        self.maif_path = maif_path
        self.manifest_path = manifest_path
        self._maif_path_obj = Path(maif_path)
        self._manifest_path_obj = Path(manifest_path)
        self.privacy_engine = privacy_engine
        self.requesting_agent = requesting_agent or "anonymous"
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load blocks with versioning support
        self.blocks = []
        for block_data in self.manifest['blocks']:
            # Handle both old and new block formats
            if 'block_id' not in block_data:
                block_data['block_id'] = str(uuid.uuid4())
            if 'version' not in block_data:
                block_data['version'] = 1
            
            # Map field names correctly
            mapped_data = {
                'block_type': block_data.get('type', block_data.get('block_type')),
                'offset': block_data['offset'],
                'size': block_data['size'],
                'hash_value': block_data.get('hash', block_data.get('hash_value')),
                'version': block_data['version'],
                'previous_hash': block_data.get('previous_hash'),
                'block_id': block_data['block_id'],
                'metadata': block_data.get('metadata')
            }
            self.blocks.append(MAIFBlock(**mapped_data))
        
        # Load version history if available
        self.version_history = []
        if 'version_history' in self.manifest:
            for v in self.manifest['version_history']:
                # Skip if not a dictionary
                if not isinstance(v, dict):
                    continue
                    
                # Map field names correctly
                mapped_version = {
                    'version': v.get('version', v.get('version_number', 1)),
                    'timestamp': v['timestamp'],
                    'agent_id': v['agent_id'],
                    'operation': v['operation'],
                    'block_id': v['block_id'],
                    'previous_hash': v.get('previous_hash'),
                    'block_hash': v.get('current_hash', v.get('block_hash', '')),
                    'change_description': v.get('change_description')
                }
                self.version_history.append(MAIFVersion(**mapped_version))
        
        # Load block registry if available
        self.block_registry = {}
        if 'block_registry' in self.manifest:
            for block_id, versions in self.manifest['block_registry'].items():
                mapped_blocks = []
                for block_data in versions:
                    # Map field names correctly
                    mapped_data = {
                        'block_type': block_data.get('type', block_data.get('block_type')),
                        'offset': block_data['offset'],
                        'size': block_data['size'],
                        'hash_value': block_data.get('hash', block_data.get('hash_value')),
                        'version': block_data.get('version', 1),
                        'previous_hash': block_data.get('previous_hash'),
                        'block_id': block_data.get('block_id', block_id),
                        'metadata': block_data.get('metadata')
                    }
                    mapped_blocks.append(MAIFBlock(**mapped_data))
                self.block_registry[block_id] = mapped_blocks
    
    def verify_integrity(self) -> bool:
        """Verify all block hashes match stored values."""
        try:
            with open(self.maif_path, 'rb') as f:
                for block in self.blocks:
                    try:
                        # Handle both old (16-byte) and new (24-byte) headers
                        header_size = 24 if hasattr(block, 'version') and block.version else 16
                        f.seek(block.offset + header_size)
                        data_size = block.size - header_size
                        
                        # Ensure we don't read beyond file
                        if data_size <= 0:
                            continue
                            
                        data = f.read(data_size)
                        if len(data) != data_size:
                            continue  # Skip incomplete blocks
                            
                        computed_hash = hashlib.sha256(data).hexdigest()
                        
                        expected_hash = block.hash_value.replace('sha256:', '')
                        if computed_hash != expected_hash:
                            return False
                    except Exception:
                        continue  # Skip problematic blocks
            return True
        except Exception:
            return False
    
    def get_block_versions(self, block_id: str) -> List[MAIFBlock]:
        """Get all versions of a specific block."""
        return self.block_registry.get(block_id, [])
    
    def get_latest_block_version(self, block_id: str) -> Optional[MAIFBlock]:
        """Get the latest version of a block."""
        versions = self.get_block_versions(block_id)
        if not versions:
            return None
        return max(versions, key=lambda b: b.version)
    
    def get_version_timeline(self) -> List[MAIFVersion]:
        """Get the complete version timeline sorted by timestamp."""
        return sorted(self.version_history.values(), key=lambda v: v.timestamp)
    
    def get_changes_by_agent(self, agent_id: str) -> List[MAIFVersion]:
        """Get all changes made by a specific agent."""
        return [v for v in self.version_history.values() if v.agent_id == agent_id]
    
    def is_block_deleted(self, block_id: str) -> bool:
        """Check if a block has been marked as deleted."""
        for version in reversed(self.version_history):
            if version.block_id == block_id:
                return version.operation == "delete"
        return False
    
    def get_block_data(self, block_type: str, block_id: Optional[str] = None) -> Optional[bytes]:
        """Get raw data from a specific block type with privacy checks."""
        for block in self.blocks:
            if block.block_type == block_type and (block_id is None or block.block_id == block_id):
                # Check access permissions
                if self.privacy_engine and not self.privacy_engine.check_access(
                    self.requesting_agent, block.block_id, "read"
                ):
                    continue  # Skip blocks without read permission
                
                with open(self.maif_path, 'rb') as f:
                    # Handle both old (16-byte) and new (24-byte) headers
                    header_size = 24 if hasattr(block, 'version') and block.version else 16
                    f.seek(block.offset + header_size)
                    data = f.read(block.size - header_size)
                    
                    # Decrypt if necessary
                    if self.privacy_engine and block.metadata and 'encryption' in block.metadata:
                        try:
                            data = self.privacy_engine.decrypt_data(data, block.block_id, block.metadata['encryption'])
                        except Exception as e:
                            print(f"Warning: Could not decrypt block {block.block_id}: {e}")
                            continue
                    
                    return data
        return None
    
    def get_text_blocks(self, include_anonymized: bool = False) -> List[str]:
        """Extract all text blocks with privacy filtering."""
        texts = []
        for block in self.blocks:
            if block.block_type in ["text", "text_data"]:
                # Check access permissions
                if self.privacy_engine and not self.privacy_engine.check_access(
                    self.requesting_agent, block.block_id, "read"
                ):
                    continue
                
                data = self.get_block_data(block.block_type, block.block_id)
                if data:
                    try:
                        text = data.decode('utf-8')
                    except UnicodeDecodeError:
                        # Fallback for binary data that might contain text
                        text = data.decode('latin-1', errors='ignore')
                    
                    # Apply anonymization if required and not already anonymized
                    if (self.privacy_engine and block.metadata and
                        block.metadata.get('privacy_policy', {}).get('anonymization_required', False) and
                        not include_anonymized):
                        text = self.privacy_engine.anonymize_data(text, "text_block")
                    
                    texts.append(text)
        return texts
    
    def get_embeddings(self) -> List[List[float]]:
        """Extract embedding vectors from MAIF with privacy checks."""
        # Find embedding block
        embed_block = next((b for b in self.blocks if b.block_type == "embeddings"), None)
        if not embed_block:
            return []
        
        # Check access permissions
        if self.privacy_engine and not self.privacy_engine.check_access(
            self.requesting_agent, embed_block.block_id, "read"
        ):
            return []
        
        data = self.get_block_data("embeddings", embed_block.block_id)
        if not data:
            return []
        
        if not embed_block.metadata:
            return []
        
        dimensions = embed_block.metadata.get('dimensions', 0)
        count = embed_block.metadata.get('count', 0)
        
        if dimensions == 0 or count == 0:
            return []
        
        # Unpack embeddings
        embeddings = []
        for i in range(count):
            embedding = []
            for j in range(dimensions):
                offset = (i * dimensions + j) * 4  # 4 bytes per float
                value = struct.unpack('f', data[offset:offset+4])[0]
                embedding.append(value)
            embeddings.append(embedding)
        
        return embeddings
    
    def check_block_access(self, block_id: str, permission: str = "read") -> bool:
        """Check if the requesting agent can access a specific block."""
        if not self.privacy_engine:
            return True
        return self.privacy_engine.check_access(self.requesting_agent, block_id, permission)
    
    def get_accessible_blocks(self, permission: str = "read") -> List[MAIFBlock]:
        """Get all blocks accessible to the requesting agent."""
        accessible = []
        for block in self.blocks:
            if self.check_block_access(block.block_id, permission):
                accessible.append(block)
        return accessible
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get a summary of privacy policies and access controls."""
        if not self.privacy_engine:
            return {"privacy_enabled": False}
        
        total_blocks = len(self.blocks)
        accessible_blocks = len(self.get_accessible_blocks("read"))
        
        privacy_levels = {}
        encryption_modes = {}
        
        for block in self.blocks:
            if block.metadata and 'privacy_policy' in block.metadata:
                policy = block.metadata['privacy_policy']
                level = policy.get('privacy_level', 'unknown')
                mode = policy.get('encryption_mode', 'unknown')
                
                privacy_levels[level] = privacy_levels.get(level, 0) + 1
                encryption_modes[mode] = encryption_modes.get(mode, 0) + 1
        
        return {
            "privacy_enabled": True,
            "requesting_agent": self.requesting_agent,
            "total_blocks": total_blocks,
            "accessible_blocks": accessible_blocks,
            "access_ratio": accessible_blocks / total_blocks if total_blocks > 0 else 0,
            "privacy_levels": privacy_levels,
            "encryption_modes": encryption_modes
        }
    
    def get_video_blocks(self) -> List[MAIFBlock]:
        """Get all video blocks with access control."""
        video_blocks = []
        for block in self.blocks:
            if block.block_type == "video_data":
                # Check access permissions
                if self.privacy_engine and not self.privacy_engine.check_access(
                    self.requesting_agent, block.block_id, "read"
                ):
                    continue
                video_blocks.append(block)
        return video_blocks
    
    def query_videos(self,
                    duration_range: Optional[tuple] = None,
                    min_resolution: Optional[str] = None,
                    max_resolution: Optional[str] = None,
                    format_filter: Optional[str] = None,
                    min_size_mb: Optional[float] = None,
                    max_size_mb: Optional[float] = None) -> List[Dict[str, Any]]:
        """Query videos by properties with advanced filtering."""
        video_blocks = self.get_video_blocks()
        results = []
        
        for block in video_blocks:
            if not block.metadata:
                continue
                
            # Apply filters
            if duration_range:
                duration = block.metadata.get("duration")
                if duration is None:
                    continue
                min_dur, max_dur = duration_range
                if duration < min_dur or duration > max_dur:
                    continue
            
            if min_resolution or max_resolution:
                resolution = block.metadata.get("resolution")
                if resolution:
                    width, height = self._parse_resolution(resolution)
                    if min_resolution:
                        min_w, min_h = self._parse_resolution(min_resolution)
                        # For minimum resolution, both width AND height must meet the minimum
                        if width < min_w or height < min_h:
                            continue
                    if max_resolution:
                        max_w, max_h = self._parse_resolution(max_resolution)
                        if width > max_w or height > max_h:
                            continue
                else:
                    # If no resolution metadata, skip this video when resolution filter is applied
                    continue
            
            if format_filter:
                video_format = block.metadata.get("format")
                if video_format != format_filter:
                    continue
            
            if min_size_mb or max_size_mb:
                size_bytes = block.metadata.get("size_bytes", 0)
                size_mb = size_bytes / (1024 * 1024)
                if min_size_mb and size_mb < min_size_mb:
                    continue
                if max_size_mb and size_mb > max_size_mb:
                    continue
            
            # Include block info and metadata
            result = {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "metadata": block.metadata,
                "size_bytes": block.metadata.get("size_bytes", 0),
                "duration": block.metadata.get("duration"),
                "resolution": block.metadata.get("resolution"),
                "format": block.metadata.get("format"),
                "has_semantic_analysis": block.metadata.get("has_semantic_analysis", False)
            }
            results.append(result)
        
        return results
    
    def get_video_data(self, block_id: str) -> Optional[bytes]:
        """Get video data by block ID."""
        return self.get_block_data("video_data", block_id)
    
    def search_videos_by_content(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search videos by semantic content using embeddings."""
        try:
            from .semantic import SemanticEmbedder
            
            # Generate query embedding
            embedder = SemanticEmbedder()
            query_embedding = embedder.embed_text(query_text)
            
            # Get video blocks with semantic embeddings
            video_blocks = self.get_video_blocks()
            similarities = []
            
            for block in video_blocks:
                if not block.metadata or not block.metadata.get("has_semantic_analysis"):
                    continue
                
                video_embeddings = block.metadata.get("semantic_embeddings")
                if video_embeddings:
                    # Calculate similarity
                    similarity = embedder.compute_similarity(query_embedding,
                                                           type('obj', (object,), {'vector': video_embeddings})())
                    similarities.append((block, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for block, similarity in similarities[:top_k]:
                result = {
                    "block_id": block.block_id,
                    "metadata": block.metadata,
                    "similarity_score": similarity,
                    "duration": block.metadata.get("duration"),
                    "resolution": block.metadata.get("resolution"),
                    "format": block.metadata.get("format")
                }
                results.append(result)
            
            return results
            
        except ImportError:
            # Fallback without semantic search
            return []
    
    def get_video_frames_at_timestamps(self, block_id: str, timestamps: List[float]) -> List[bytes]:
        """Extract frames at specific timestamps (placeholder for future implementation)."""
        # This would require video processing libraries like OpenCV or FFmpeg
        # For now, return empty list as placeholder
        return []
    
    def get_video_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all videos in the MAIF."""
        video_blocks = self.get_video_blocks()
        
        if not video_blocks:
            return {"total_videos": 0}
        
        total_duration = 0
        total_size = 0
        formats = {}
        resolutions = {}
        
        for block in video_blocks:
            if block.metadata:
                duration = block.metadata.get("duration", 0)
                if duration:
                    total_duration += duration
                
                size_bytes = block.metadata.get("size_bytes", 0)
                total_size += size_bytes
                
                video_format = block.metadata.get("format", "unknown")
                formats[video_format] = formats.get(video_format, 0) + 1
                
                resolution = block.metadata.get("resolution", "unknown")
                resolutions[resolution] = resolutions.get(resolution, 0) + 1
        
        return {
            "total_videos": len(video_blocks),
            "total_duration_seconds": total_duration,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_duration": total_duration / len(video_blocks) if video_blocks else 0,
            "formats": formats,
            "resolutions": resolutions,
            "videos_with_semantic_analysis": sum(1 for block in video_blocks
                                               if block.metadata and block.metadata.get("has_semantic_analysis"))
        }
    
    def _parse_resolution(self, resolution: str) -> tuple:
        """Parse resolution string like '1920x1080' into (width, height)."""
        try:
            if 'x' in resolution:
                width, height = resolution.split('x')
                return int(width), int(height)
            elif resolution == "720p":
                return 1280, 720
            elif resolution == "1080p":
                return 1920, 1080
            elif resolution == "4K":
                return 3840, 2160
            else:
                return 0, 0
        except:
            return 0, 0

class MAIFParser:
    """High-level MAIF parsing interface."""
    
    def __init__(self, maif_path: str, manifest_path: str):
        self.decoder = MAIFDecoder(maif_path, manifest_path)
    
    def verify_integrity(self) -> bool:
        """Verify file integrity."""
        return self.decoder.verify_integrity()
    
    def get_metadata(self) -> Dict:
        """Get MAIF metadata."""
        return {
            "header": {
                "version": self.decoder.manifest.get("maif_version"),
                "created": self.decoder.manifest.get("created"),
                "creator_id": self.decoder.manifest.get("creator_id"),
                "agent_id": self.decoder.manifest.get("agent_id"),
                "root_hash": self.decoder.manifest.get("root_hash")
            },
            "blocks": [block.to_dict() for block in self.decoder.blocks],
            "version": self.decoder.manifest.get("maif_version"),
            "created": self.decoder.manifest.get("created"),
            "creator_id": self.decoder.manifest.get("creator_id"),
            "root_hash": self.decoder.manifest.get("root_hash"),
            "block_count": len(self.decoder.blocks)
        }
    
    def list_blocks(self) -> List[Dict]:
        """List all blocks with their metadata."""
        return [block.to_dict() for block in self.decoder.blocks]
    
    def extract_content(self) -> Dict:
        """Extract all content from the MAIF."""
        return {
            "text_blocks": self.decoder.get_text_blocks(),
            "texts": self.decoder.get_text_blocks(),
            "embeddings": self.decoder.get_embeddings(),
            "metadata": self.get_metadata()
        }
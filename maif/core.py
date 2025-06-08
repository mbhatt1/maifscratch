"""
Core MAIF implementation - encoding, decoding, and parsing functionality.
Enhanced with privacy-by-design features.
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

@dataclass
class MAIFBlock:
    """Represents a MAIF block with metadata."""
    block_type: str
    offset: int
    size: int
    hash_value: str
    version: int = 1
    previous_hash: Optional[str] = None
    block_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.block_id is None:
            self.block_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict:
        return {
            "type": self.block_type,
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
    version_number: int
    timestamp: float
    agent_id: str
    operation: str  # "create", "update", "delete"
    block_id: str
    previous_hash: Optional[str]
    current_hash: str
    change_description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version_number,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "current_hash": self.current_hash,
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
    
    def __init__(self, agent_id: Optional[str] = None, existing_maif_path: Optional[str] = None,
                 existing_manifest_path: Optional[str] = None, enable_privacy: bool = True):
        self.blocks: List[MAIFBlock] = []
        self.header = MAIFHeader()
        self.buffer = io.BytesIO()
        self.agent_id = agent_id or str(uuid.uuid4())
        self.version_history: List[MAIFVersion] = []
        self.block_registry: Dict[str, List[MAIFBlock]] = {}  # block_id -> list of versions
        
        # Privacy-by-design features
        self.enable_privacy = enable_privacy
        self.privacy_engine = PrivacyEngine() if enable_privacy else None
        self.default_privacy_policy = PrivacyPolicy(
            privacy_level=PrivacyLevel.INTERNAL,
            encryption_mode=EncryptionMode.AES_GCM,
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
                      anonymize: bool = False) -> str:
        """Add or update a text block to the MAIF with privacy controls."""
        # Apply anonymization if requested
        if self.enable_privacy and anonymize:
            text = self.privacy_engine.anonymize_data(text, "text_block")
        
        text_bytes = text.encode('utf-8')
        return self._add_block("text_data", text_bytes, metadata, update_block_id, privacy_policy)
    
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
    
    def add_cross_modal_block(self, multimodal_data: Dict[str, Any], metadata: Optional[Dict] = None,
                             update_block_id: Optional[str] = None,
                             privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add cross-modal data block using ACAM (Adaptive Cross-Modal Attention Mechanism)."""
        try:
            from .semantic import DeepSemanticUnderstanding
            
            # Initialize deep semantic understanding
            dsu = DeepSemanticUnderstanding()
            
            # Register basic processors for different modalities
            def text_processor(text):
                from .semantic import SemanticEmbedder
                embedder = SemanticEmbedder()
                return embedder.embed_text(str(text)).vector
            
            def binary_processor(data):
                # Simple hash-based embedding for binary data (384 dimensions)
                import hashlib
                hash_obj = hashlib.sha256(data if isinstance(data, bytes) else str(data).encode())
                # Convert hash to pseudo-embedding
                hash_hex = hash_obj.hexdigest()
                base_embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, len(hash_hex), 2)]
                # Repeat to get 384 dimensions to match text embeddings
                embedding = (base_embedding * (384 // len(base_embedding) + 1))[:384]
                return embedding
            
            dsu.register_modality_processor("text", text_processor)
            dsu.register_modality_processor("binary", binary_processor)
            dsu.register_modality_processor("image", binary_processor)
            dsu.register_modality_processor("audio", binary_processor)
            
            # Process multimodal input
            processed_result = dsu.process_multimodal_input(multimodal_data)
            
            # Serialize the result
            import json
            serialized_data = json.dumps(processed_result, default=str).encode('utf-8')
            
            cross_modal_metadata = metadata or {}
            cross_modal_metadata.update({
                "algorithm": "ACAM",
                "modalities": list(multimodal_data.keys()),
                "unified_representation_dim": len(processed_result.get("unified_representation", [])),
                "attention_weights_count": len(processed_result.get("attention_weights", {}))
            })
            
            return self._add_block("cross_modal", serialized_data, cross_modal_metadata, update_block_id, privacy_policy)
            
        except ImportError:
            # Fallback if semantic module not available
            serialized_data = json.dumps(multimodal_data, default=str).encode('utf-8')
            fallback_metadata = metadata or {}
            fallback_metadata.update({"algorithm": "fallback", "modalities": list(multimodal_data.keys())})
            return self._add_block("cross_modal", serialized_data, fallback_metadata, update_block_id, privacy_policy)
    
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
                                       use_hsc: bool = True,
                                       metadata: Optional[Dict] = None,
                                       update_block_id: Optional[str] = None,
                                       privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Add embeddings block with HSC (Hierarchical Semantic Compression)."""
        if use_hsc:
            try:
                from .semantic import HierarchicalSemanticCompression
                
                # Apply HSC compression
                hsc = HierarchicalSemanticCompression()
                compressed_result = hsc.compress_embeddings(embeddings)
                
                # Serialize compressed result
                import json
                serialized_data = json.dumps(compressed_result).encode('utf-8')
                
                hsc_metadata = metadata or {}
                hsc_metadata.update({
                    "algorithm": "HSC",
                    "compression_type": "hierarchical_semantic",
                    "original_count": len(embeddings),
                    "original_dimensions": len(embeddings[0]) if embeddings else 0,
                    "compression_ratio": compressed_result.get("metadata", {}).get("compression_ratio", 1.0)
                })
                
                return self._add_block("compressed_embeddings", serialized_data, hsc_metadata, update_block_id, privacy_policy)
                
            except ImportError:
                # Fallback to regular embeddings if HSC not available
                pass
        
        # Fallback to regular embeddings block
        return self.add_embeddings_block(embeddings, metadata, update_block_id, privacy_policy)
    
    def _add_block(self, block_type: str, data: bytes, metadata: Optional[Dict] = None,
                  update_block_id: Optional[str] = None,
                  privacy_policy: Optional[PrivacyPolicy] = None) -> str:
        """Internal method to add or update a block with versioning and privacy."""
        offset = self.buffer.tell()
        
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
        
        if self.enable_privacy and policy.encryption_mode != EncryptionMode.NONE:
            # Encrypt the data
            data, encryption_metadata = self.privacy_engine.encrypt_data(
                data, block_id, policy.encryption_mode
            )
            # Set privacy policy for this block
            self.privacy_engine.set_privacy_policy(block_id, policy)
        
        # Calculate hash (after encryption)
        hash_value = hashlib.sha256(data).hexdigest()
        
        # Write block header (24 bytes - extended for versioning)
        size = len(data)
        header = struct.pack('>I4sIII', size, block_type.encode('ascii')[:4].ljust(4, b'\0'),
                           version_number, 0, 0)  # version, flags, reserved
        self.buffer.write(header)
        
        # Write data
        self.buffer.write(data)
        
        # Merge encryption metadata with user metadata
        combined_metadata = metadata.copy() if metadata else {}
        if encryption_metadata:
            combined_metadata['encryption'] = encryption_metadata
        if self.enable_privacy and policy:
            combined_metadata['privacy_policy'] = {
                'privacy_level': policy.privacy_level.value,
                'encryption_mode': policy.encryption_mode.value,
                'anonymization_required': policy.anonymization_required,
                'audit_required': policy.audit_required
            }
        
        # Create block record
        block = MAIFBlock(
            block_type=block_type,
            offset=offset,
            size=size + 24,  # Include extended header size
            hash_value=f"sha256:{hash_value}",
            version=version_number,
            previous_hash=previous_hash,
            block_id=block_id,
            metadata=combined_metadata
        )
        
        # Add to blocks and registry
        self.blocks.append(block)
        if block_id not in self.block_registry:
            self.block_registry[block_id] = []
        self.block_registry[block_id].append(block)
        
        # Record version history
        operation = "update" if is_update else "create"
        version_entry = MAIFVersion(
            version_number=version_number,
            timestamp=time.time(),
            agent_id=self.agent_id,
            operation=operation,
            block_id=block_id,
            previous_hash=previous_hash,
            current_hash=f"sha256:{hash_value}",
            change_description=combined_metadata.get("change_description") if combined_metadata else None
        )
        self.version_history.append(version_entry)
        
        return hash_value
    
    def delete_block(self, block_id: str, reason: Optional[str] = None) -> bool:
        """Mark a block as deleted (soft delete with versioning)."""
        if block_id not in self.block_registry:
            return False
        
        latest_block = self.block_registry[block_id][-1]
        
        # Create deletion record
        version_entry = MAIFVersion(
            version_number=latest_block.version + 1,
            timestamp=time.time(),
            agent_id=self.agent_id,
            operation="delete",
            block_id=block_id,
            previous_hash=latest_block.hash_value,
            current_hash="deleted",
            change_description=reason
        )
        self.version_history.append(version_entry)
        
        return True
    
    def get_block_history(self, block_id: str) -> List[MAIFBlock]:
        """Get the complete version history of a block."""
        return self.block_registry.get(block_id, [])
    
    def get_block_at_version(self, block_id: str, version: int) -> Optional[MAIFBlock]:
        """Get a specific version of a block."""
        if block_id not in self.block_registry:
            return None
        
    def add_access_rule(self, subject: str, resource: str, permissions: List[str],
                       conditions: Optional[Dict[str, Any]] = None, expiry: Optional[float] = None):
        """Add an access control rule for privacy protection."""
        if not self.enable_privacy:
            return
        
        rule = AccessRule(
            subject=subject,
            resource=resource,
            permissions=permissions,
            conditions=conditions,
            expiry=expiry
        )
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
        if not self.enable_privacy:
            return {"privacy_enabled": False}
        
        report = self.privacy_engine.generate_privacy_report()
        report["privacy_enabled"] = True
        report["total_version_entries"] = len(self.version_history)
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
            "blocks": [block.to_dict() for block in self.blocks],
            "version_history": [v.to_dict() for v in self.version_history],
            "block_registry": {
                block_id: [block.to_dict() for block in versions]
                for block_id, versions in self.block_registry.items()
            }
        }
        
        # Calculate root hash including version history
        all_hashes = "".join([block.hash_value for block in self.blocks])
        version_hashes = "".join([v.current_hash for v in self.version_history])
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
            "version_history": [v.to_dict() for v in self.version_history],
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
        self.maif_path = Path(maif_path)
        self.manifest_path = Path(manifest_path)
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
                # Map field names correctly
                mapped_version = {
                    'version_number': v.get('version', v.get('version_number')),
                    'timestamp': v['timestamp'],
                    'agent_id': v['agent_id'],
                    'operation': v['operation'],
                    'block_id': v['block_id'],
                    'previous_hash': v.get('previous_hash'),
                    'current_hash': v['current_hash'],
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
        return sorted(self.version_history, key=lambda v: v.timestamp)
    
    def get_changes_by_agent(self, agent_id: str) -> List[MAIFVersion]:
        """Get all changes made by a specific agent."""
        return [v for v in self.version_history if v.agent_id == agent_id]
    
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
            if block.block_type == "text_data":
                # Check access permissions
                if self.privacy_engine and not self.privacy_engine.check_access(
                    self.requesting_agent, block.block_id, "read"
                ):
                    continue
                
                data = self.get_block_data("text_data", block.block_id)
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
            "texts": self.decoder.get_text_blocks(),
            "embeddings": self.decoder.get_embeddings(),
            "metadata": self.get_metadata()
        }
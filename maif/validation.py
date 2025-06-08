"""
Validation and repair functionality for MAIF files.
"""

import json
import hashlib
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .core import MAIFDecoder, MAIFEncoder

@dataclass
class ValidationResult:
    """Result of MAIF validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    file_size: int
    block_count: int
    
class MAIFValidator:
    """Validates MAIF files for integrity and compliance."""
    
    def __init__(self):
        self.validation_rules = [
            self._validate_file_structure,
            self._validate_block_integrity,
            self._validate_manifest_consistency,
            self._validate_signatures,
            self._validate_provenance_chain
        ]
    
    def validate_file(self, maif_path: str, manifest_path: str) -> ValidationResult:
        """Validate a MAIF file and its manifest."""
        errors = []
        warnings = []
        
        try:
            # Check file existence
            if not os.path.exists(maif_path):
                errors.append(f"MAIF file not found: {maif_path}")
                return ValidationResult(False, errors, warnings, 0, 0)
            
            if not os.path.exists(manifest_path):
                errors.append(f"Manifest file not found: {manifest_path}")
                return ValidationResult(False, errors, warnings, 0, 0)
            
            # Get file info
            file_size = os.path.getsize(maif_path)
            
            # Load and validate
            decoder = MAIFDecoder(maif_path, manifest_path)
            block_count = len(decoder.blocks)
            
            # Run validation rules
            for rule in self.validation_rules:
                rule_errors, rule_warnings = rule(decoder, maif_path, manifest_path)
                errors.extend(rule_errors)
                warnings.extend(rule_warnings)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(is_valid, errors, warnings, file_size, block_count)
            
        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, errors, warnings, 0, 0)
    
    def _validate_file_structure(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Validate basic file structure."""
        errors = []
        warnings = []
        
        # Check manifest structure
        required_fields = ["maif_version", "blocks", "root_hash"]
        for field in required_fields:
            if field not in decoder.manifest:
                errors.append(f"Missing required manifest field: {field}")
        
        # Check block structure
        for i, block in enumerate(decoder.blocks):
            if not hasattr(block, 'block_type') or not block.block_type:
                errors.append(f"Block {i} missing block_type")
            if not hasattr(block, 'hash_value') or not block.hash_value:
                errors.append(f"Block {i} missing hash_value")
        
        return errors, warnings
    
    def _validate_block_integrity(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Validate block integrity."""
        errors = []
        warnings = []
        
        try:
            if not decoder.verify_integrity():
                errors.append("Block integrity verification failed")
        except Exception as e:
            errors.append(f"Integrity check error: {str(e)}")
        
        return errors, warnings
    
    def _validate_manifest_consistency(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Validate manifest consistency."""
        errors = []
        warnings = []
        
        # Check block count consistency
        manifest_block_count = len(decoder.manifest.get("blocks", []))
        actual_block_count = len(decoder.blocks)
        
        if manifest_block_count != actual_block_count:
            errors.append(f"Block count mismatch: manifest={manifest_block_count}, actual={actual_block_count}")
        
        return errors, warnings
    
    def _validate_signatures(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Validate digital signatures if present."""
        errors = []
        warnings = []
        
        if "signature" in decoder.manifest:
            try:
                from .security import MAIFVerifier
                verifier = MAIFVerifier()
                if not verifier.verify_maif_signature(decoder.manifest):
                    errors.append("Invalid digital signature")
            except Exception as e:
                warnings.append(f"Could not verify signature: {str(e)}")
        
        return errors, warnings
    
    def _validate_provenance_chain(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> Tuple[List[str], List[str]]:
        """Validate provenance chain if present."""
        errors = []
        warnings = []
        
        if "version_history" in decoder.manifest:
            version_history = decoder.manifest["version_history"]
            
            # Check chain linkage
            for i in range(1, len(version_history)):
                current = version_history[i]
                previous = version_history[i-1]
                
                if current.get("previous_hash") != previous.get("current_hash"):
                    errors.append(f"Provenance chain break at version {i}")
        
        return errors, warnings


class MAIFRepairTool:
    """Repairs corrupted or inconsistent MAIF files."""
    
    def __init__(self):
        self.repair_strategies = [
            self._repair_manifest_consistency,
            self._repair_block_metadata,
            self._repair_hash_mismatches
        ]
    
    def repair_file(self, maif_path: str, manifest_path: str) -> bool:
        """Attempt to repair a MAIF file."""
        try:
            # First validate to identify issues
            validator = MAIFValidator()
            result = validator.validate_file(maif_path, manifest_path)
            
            if result.is_valid:
                return True  # Nothing to repair
            
            # Load the file
            decoder = MAIFDecoder(maif_path, manifest_path)
            
            # Apply repair strategies
            repairs_made = False
            for strategy in self.repair_strategies:
                if strategy(decoder, maif_path, manifest_path):
                    repairs_made = True
            
            # If repairs were made, rebuild the file
            if repairs_made:
                self._rebuild_file(decoder, maif_path, manifest_path)
            
            # Validate again
            final_result = validator.validate_file(maif_path, manifest_path)
            return final_result.is_valid
            
        except Exception as e:
            print(f"Repair failed: {e}")
            return False
    
    def _repair_manifest_consistency(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> bool:
        """Repair manifest consistency issues."""
        repairs_made = False
        
        # Ensure all blocks are in manifest
        manifest_blocks = decoder.manifest.get("blocks", [])
        if len(manifest_blocks) != len(decoder.blocks):
            # Rebuild blocks list from actual blocks
            decoder.manifest["blocks"] = [block.to_dict() for block in decoder.blocks]
            repairs_made = True
        
        return repairs_made
    
    def _repair_block_metadata(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> bool:
        """Repair block metadata issues."""
        repairs_made = False
        
        for block in decoder.blocks:
            if not hasattr(block, 'block_id') or not block.block_id:
                import uuid
                block.block_id = str(uuid.uuid4())
                repairs_made = True
            
            if not hasattr(block, 'version') or not block.version:
                block.version = 1
                repairs_made = True
        
        return repairs_made
    
    def _repair_hash_mismatches(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str) -> bool:
        """Repair hash mismatches by recalculating."""
        repairs_made = False
        
        with open(maif_path, 'rb') as f:
            for block in decoder.blocks:
                try:
                    # Read block data
                    header_size = 24 if hasattr(block, 'version') and block.version else 16
                    f.seek(block.offset + header_size)
                    data = f.read(block.size - header_size)
                    
                    # Recalculate hash
                    computed_hash = hashlib.sha256(data).hexdigest()
                    expected_hash = block.hash_value.replace('sha256:', '')
                    
                    if computed_hash != expected_hash:
                        # Update hash
                        block.hash_value = f"sha256:{computed_hash}"
                        repairs_made = True
                        
                except Exception:
                    continue  # Skip problematic blocks
        
        return repairs_made
    
    def _rebuild_file(self, decoder: MAIFDecoder, maif_path: str, manifest_path: str):
        """Rebuild the MAIF file with repaired data."""
        # Create new encoder with repaired data
        encoder = MAIFEncoder()
        
        # Copy blocks (this is a simplified rebuild)
        encoder.blocks = decoder.blocks
        encoder.header.version = decoder.manifest.get("maif_version", "0.1.0")
        
        # Rebuild manifest
        manifest = decoder.manifest.copy()
        manifest["blocks"] = [block.to_dict() for block in decoder.blocks]
        
        # Recalculate root hash
        all_hashes = "".join([block.hash_value for block in decoder.blocks])
        root_hash = hashlib.sha256(all_hashes.encode()).hexdigest()
        manifest["root_hash"] = f"sha256:{root_hash}"
        
        # Save repaired files
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
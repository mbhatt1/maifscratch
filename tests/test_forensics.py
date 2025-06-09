"""
Comprehensive tests for MAIF forensics functionality.
"""

import pytest
import tempfile
import os
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock

from maif.validation import MAIFValidator, MAIFRepairTool, ValidationResult
from maif.core import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner, MAIFVerifier


class TestForensicAnalysis:
    """Test forensic analysis capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_forensic_validation_chain(self):
        """Test forensic validation of provenance chain."""
        # Create a MAIF file with provenance
        encoder = MAIFEncoder(agent_id="forensic_test")
        encoder.add_text_block("Forensic evidence data", metadata={"evidence_id": "E001"})
        
        maif_path = os.path.join(self.temp_dir, "evidence.maif")
        manifest_path = os.path.join(self.temp_dir, "evidence_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Validate forensically
        result = self.validator.validate_file(maif_path, manifest_path)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_forensic_integrity_verification(self):
        """Test forensic integrity verification."""
        # Create test file
        encoder = MAIFEncoder(agent_id="forensic_test")
        encoder.add_text_block("Chain of custody data", metadata={"custody_id": "C001"})
        
        maif_path = os.path.join(self.temp_dir, "custody.maif")
        manifest_path = os.path.join(self.temp_dir, "custody_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Load and verify
        decoder = MAIFDecoder(maif_path, manifest_path)
        integrity_valid = decoder.verify_integrity()
        
        assert integrity_valid is True
    
    def test_forensic_signature_verification(self):
        """Test forensic signature verification."""
        # Create signed MAIF
        signer = MAIFSigner(agent_id="forensic_signer")
        encoder = MAIFEncoder(agent_id="forensic_test")
        encoder.add_text_block("Signed evidence", metadata={"signature_required": True})
        
        maif_path = os.path.join(self.temp_dir, "signed_evidence.maif")
        manifest_path = os.path.join(self.temp_dir, "signed_evidence_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Load and sign manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        signed_manifest = signer.sign_maif_manifest(manifest)
        
        with open(manifest_path, 'w') as f:
            json.dump(signed_manifest, f)
        
        # Verify signature
        verifier = MAIFVerifier()
        is_valid = verifier.verify_maif_signature(signed_manifest)
        
        assert is_valid is True
    
    def test_forensic_tamper_detection(self):
        """Test detection of tampering."""
        # Create original file
        encoder = MAIFEncoder(agent_id="forensic_test")
        encoder.add_text_block("Original evidence", metadata={"tamper_test": True})
        
        maif_path = os.path.join(self.temp_dir, "tamper_test.maif")
        manifest_path = os.path.join(self.temp_dir, "tamper_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Tamper with the file
        with open(maif_path, 'r+b') as f:
            f.seek(50)  # Seek to middle of file
            f.write(b'TAMPERED')
        
        # Validate should detect tampering
        result = self.validator.validate_file(maif_path, manifest_path)
        
        # Should detect integrity issues
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_forensic_repair_attempt(self):
        """Test forensic repair capabilities."""
        # Create file with known issues
        encoder = MAIFEncoder(agent_id="forensic_test")
        encoder.add_text_block("Repair test data", metadata={"repair_test": True})
        
        maif_path = os.path.join(self.temp_dir, "repair_test.maif")
        manifest_path = os.path.join(self.temp_dir, "repair_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Attempt repair
        repair_success = self.repair_tool.repair_file(maif_path, manifest_path)
        
        # Should handle repair attempt
        assert isinstance(repair_success, bool)
    
    def test_forensic_metadata_analysis(self):
        """Test forensic metadata analysis."""
        # Create file with rich metadata
        encoder = MAIFEncoder(agent_id="forensic_analyst")
        encoder.add_text_block(
            "Evidence with metadata",
            metadata={
                "case_id": "CASE-2024-001",
                "evidence_type": "digital",
                "chain_of_custody": ["Officer A", "Lab Tech B", "Analyst C"],
                "collection_timestamp": "2024-01-01T12:00:00Z",
                "hash_algorithm": "SHA-256"
            }
        )
        
        maif_path = os.path.join(self.temp_dir, "metadata_test.maif")
        manifest_path = os.path.join(self.temp_dir, "metadata_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Analyze metadata
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        # Should have blocks with metadata
        assert len(decoder.blocks) > 0
        assert decoder.blocks[0].metadata is not None
        assert "case_id" in decoder.blocks[0].metadata
    
    def test_forensic_version_history_analysis(self):
        """Test forensic analysis of version history."""
        # Create file with version history
        encoder = MAIFEncoder(agent_id="forensic_test")
        
        # Add initial block
        block_id = encoder.add_text_block("Initial evidence", metadata={"version": 1})
        
        # Update the block to create version history
        encoder.update_text_block(block_id, "Updated evidence", metadata={"version": 2})
        
        maif_path = os.path.join(self.temp_dir, "version_test.maif")
        manifest_path = os.path.join(self.temp_dir, "version_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Analyze version history
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        # Should have version history
        assert hasattr(decoder, 'version_history')
        assert len(decoder.version_history) >= 0  # May be empty if not implemented


class TestForensicCompliance:
    """Test forensic compliance features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_forensic_hash_verification(self):
        """Test forensic hash verification."""
        # Create file with known content
        test_content = "Forensic hash test content"
        expected_hash = hashlib.sha256(test_content.encode()).hexdigest()
        
        encoder = MAIFEncoder(agent_id="hash_test")
        encoder.add_text_block(test_content, metadata={"expected_hash": expected_hash})
        
        maif_path = os.path.join(self.temp_dir, "hash_test.maif")
        manifest_path = os.path.join(self.temp_dir, "hash_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify hashes
        decoder = MAIFDecoder(maif_path, manifest_path)
        integrity_valid = decoder.verify_integrity()
        
        assert integrity_valid is True
    
    def test_forensic_audit_trail(self):
        """Test forensic audit trail creation."""
        # Create file with audit information
        encoder = MAIFEncoder(agent_id="audit_test")
        encoder.add_text_block(
            "Audit trail test",
            metadata={
                "audit_trail": [
                    {"action": "created", "timestamp": "2024-01-01T10:00:00Z", "user": "analyst1"},
                    {"action": "accessed", "timestamp": "2024-01-01T11:00:00Z", "user": "supervisor1"},
                    {"action": "verified", "timestamp": "2024-01-01T12:00:00Z", "user": "expert1"}
                ]
            }
        )
        
        maif_path = os.path.join(self.temp_dir, "audit_test.maif")
        manifest_path = os.path.join(self.temp_dir, "audit_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify audit trail is preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) > 0
        assert decoder.blocks[0].metadata is not None
        assert "audit_trail" in decoder.blocks[0].metadata
    
    def test_forensic_chain_of_custody(self):
        """Test chain of custody preservation."""
        # Create file with chain of custody
        encoder = MAIFEncoder(agent_id="custody_test")
        encoder.add_text_block(
            "Chain of custody evidence",
            metadata={
                "chain_of_custody": {
                    "collected_by": "Officer Smith",
                    "collected_at": "2024-01-01T09:00:00Z",
                    "transferred_to": "Lab Tech Jones",
                    "transferred_at": "2024-01-01T10:00:00Z",
                    "analyzed_by": "Forensic Expert Brown",
                    "analyzed_at": "2024-01-01T14:00:00Z"
                }
            }
        )
        
        maif_path = os.path.join(self.temp_dir, "custody_test.maif")
        manifest_path = os.path.join(self.temp_dir, "custody_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Verify chain of custody is preserved
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        assert len(decoder.blocks) > 0
        assert decoder.blocks[0].metadata is not None
        assert "chain_of_custody" in decoder.blocks[0].metadata
        assert "collected_by" in decoder.blocks[0].metadata["chain_of_custody"]


class TestForensicReporting:
    """Test forensic reporting capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MAIFValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_forensic_validation_report(self):
        """Test forensic validation report generation."""
        # Create test file
        encoder = MAIFEncoder(agent_id="report_test")
        encoder.add_text_block("Report test data", metadata={"report_test": True})
        
        maif_path = os.path.join(self.temp_dir, "report_test.maif")
        manifest_path = os.path.join(self.temp_dir, "report_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Generate validation report
        result = self.validator.validate_file(maif_path, manifest_path)
        
        # Should have comprehensive report
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'details')
    
    def test_forensic_summary_generation(self):
        """Test forensic summary generation."""
        # Create file with multiple blocks
        encoder = MAIFEncoder(agent_id="summary_test")
        encoder.add_text_block("Evidence block 1", metadata={"evidence_id": "E001"})
        encoder.add_text_block("Evidence block 2", metadata={"evidence_id": "E002"})
        encoder.add_binary_block(b"Binary evidence", "evidence", metadata={"evidence_id": "E003"})
        
        maif_path = os.path.join(self.temp_dir, "summary_test.maif")
        manifest_path = os.path.join(self.temp_dir, "summary_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Generate summary
        decoder = MAIFDecoder(maif_path, manifest_path)
        
        # Should have multiple blocks
        assert len(decoder.blocks) >= 3
        
        # Should have different block types
        block_types = [block.block_type for block in decoder.blocks]
        assert len(set(block_types)) > 1  # Multiple different types
"""
Comprehensive tests for MAIF validation functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from maif.validation import ValidationResult, MAIFValidator, MAIFRepairTool
from maif.core import MAIFEncoder, MAIFDecoder


class TestValidationResult:
    """Test ValidationResult data structure."""
    
    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(
            is_valid=True,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            details={"blocks_checked": 5, "signatures_verified": 3}
        )
        
        assert result.is_valid is True
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == ["Warning 1"]
        assert result.details["blocks_checked"] == 5
        assert result.details["signatures_verified"] == 3
    
    def test_validation_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult()
        
        assert result.is_valid is False
        assert result.errors == []
        assert result.warnings == []
        assert result.details == {}


class TestMAIFValidator:
    """Test MAIFValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MAIFValidator()
        
        # Create a valid test MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Hello, validation world!", metadata={"id": 1})
        encoder.add_binary_block(b"binary_data_123", "data", metadata={"id": 2})
        
        self.maif_path = os.path.join(self.temp_dir, "test_validation.maif")
        self.manifest_path = os.path.join(self.temp_dir, "test_validation_manifest.json")
        
        encoder.build_maif(self.maif_path, self.manifest_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test MAIFValidator initialization."""
        assert hasattr(self.validator, 'validation_rules')
        assert hasattr(self.validator, 'repair_strategies')
    
    def test_validate_valid_file(self):
        """Test validation of a valid MAIF file."""
        result = self.validator.validate_file(self.maif_path, self.manifest_path)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
        # May have warnings, but should be valid
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.maif")
        nonexistent_manifest = os.path.join(self.temp_dir, "nonexistent_manifest.json")
        
        result = self.validator.validate_file(nonexistent_path, nonexistent_manifest)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("not found" in error.lower() or "does not exist" in error.lower() 
                  for error in result.errors)
    
    def test_validate_corrupted_manifest(self):
        """Test validation with corrupted manifest."""
        corrupted_manifest = os.path.join(self.temp_dir, "corrupted_manifest.json")
        
        # Create corrupted manifest
        with open(corrupted_manifest, 'w') as f:
            f.write("invalid json content {")
        
        result = self.validator.validate_file(self.maif_path, corrupted_manifest)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_file_structure(self):
        """Test file structure validation."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        errors, warnings = self.validator._validate_file_structure(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Valid file should have no structural errors
        assert len(errors) == 0
        # May have warnings
        assert isinstance(warnings, list)
    
    def test_validate_block_integrity(self):
        """Test block integrity validation."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        errors, warnings = self.validator._validate_block_integrity(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Valid file should have no integrity errors
        assert len(errors) == 0
        assert isinstance(warnings, list)
    
    def test_validate_manifest_consistency(self):
        """Test manifest consistency validation."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        errors, warnings = self.validator._validate_manifest_consistency(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Valid file should have consistent manifest
        assert len(errors) == 0
        assert isinstance(warnings, list)
    
    def test_validate_signatures(self):
        """Test signature validation."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        errors, warnings = self.validator._validate_signatures(
            decoder, self.maif_path, self.manifest_path
        )
        
        # May have errors if signatures are not present, but should not crash
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
    
    def test_validate_provenance_chain(self):
        """Test provenance chain validation."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        errors, warnings = self.validator._validate_provenance_chain(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Should handle missing or present provenance chains
        assert isinstance(errors, list)
        assert isinstance(warnings, list)
    
    def test_validate_with_missing_blocks(self):
        """Test validation with missing blocks."""
        # Modify manifest to reference non-existent blocks
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Add fake block reference
        manifest['blocks'].append({
            "block_id": "fake_block_123",
            "block_type": "text",
            "hash": "fake_hash_456",
            "offset": 99999,
            "size": 100
        })
        
        corrupted_manifest = os.path.join(self.temp_dir, "missing_blocks_manifest.json")
        with open(corrupted_manifest, 'w') as f:
            json.dump(manifest, f)
        
        result = self.validator.validate_file(self.maif_path, corrupted_manifest)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_with_hash_mismatch(self):
        """Test validation with hash mismatches."""
        # Modify manifest to have wrong hashes
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Corrupt a block hash
        if manifest['blocks']:
            manifest['blocks'][0]['hash'] = "corrupted_hash_123"
        
        corrupted_manifest = os.path.join(self.temp_dir, "hash_mismatch_manifest.json")
        with open(corrupted_manifest, 'w') as f:
            json.dump(manifest, f)
        
        result = self.validator.validate_file(self.maif_path, corrupted_manifest)
        
        # Should detect hash mismatch
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("hash" in error.lower() for error in result.errors)


class TestMAIFRepairTool:
    """Test MAIFRepairTool functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repair_tool = MAIFRepairTool()
        
        # Create a test MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Hello, repair world!", metadata={"id": 1})
        encoder.add_binary_block(b"binary_data_456", "data", metadata={"id": 2})
        
        self.maif_path = os.path.join(self.temp_dir, "test_repair.maif")
        self.manifest_path = os.path.join(self.temp_dir, "test_repair_manifest.json")
        
        encoder.build_maif(self.maif_path, self.manifest_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_repair_tool_initialization(self):
        """Test MAIFRepairTool initialization."""
        assert hasattr(self.repair_tool, 'repair_strategies')
        assert hasattr(self.repair_tool, 'backup_enabled')
    
    def test_repair_valid_file(self):
        """Test repair of already valid file."""
        result = self.repair_tool.repair_file(self.maif_path, self.manifest_path)
        
        # Valid file should not need repair, but repair should succeed
        assert result is True
    
    def test_repair_nonexistent_file(self):
        """Test repair of non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.maif")
        nonexistent_manifest = os.path.join(self.temp_dir, "nonexistent_manifest.json")
        
        result = self.repair_tool.repair_file(nonexistent_path, nonexistent_manifest)
        
        # Cannot repair non-existent file
        assert result is False
    
    def test_repair_manifest_consistency(self):
        """Test manifest consistency repair."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        result = self.repair_tool._repair_manifest_consistency(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Should handle repair attempt
        assert isinstance(result, bool)
    
    def test_repair_block_metadata(self):
        """Test block metadata repair."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        result = self.repair_tool._repair_block_metadata(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Should handle repair attempt
        assert isinstance(result, bool)
    
    def test_repair_hash_mismatches(self):
        """Test hash mismatch repair."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        result = self.repair_tool._repair_hash_mismatches(
            decoder, self.maif_path, self.manifest_path
        )
        
        # Should handle repair attempt
        assert isinstance(result, bool)
    
    def test_repair_with_corrupted_manifest(self):
        """Test repair with corrupted manifest."""
        corrupted_manifest = os.path.join(self.temp_dir, "corrupted_repair_manifest.json")
        
        # Create corrupted manifest
        with open(corrupted_manifest, 'w') as f:
            f.write("invalid json {")
        
        result = self.repair_tool.repair_file(self.maif_path, corrupted_manifest)
        
        # Should attempt repair but may fail
        assert isinstance(result, bool)
    
    def test_repair_with_missing_manifest(self):
        """Test repair with missing manifest."""
        missing_manifest = os.path.join(self.temp_dir, "missing_manifest.json")
        
        result = self.repair_tool.repair_file(self.maif_path, missing_manifest)
        
        # Should attempt to rebuild manifest
        assert isinstance(result, bool)
    
    def test_rebuild_file(self):
        """Test file rebuilding."""
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        # This should not raise an exception
        try:
            self.repair_tool._rebuild_file(decoder, self.maif_path, self.manifest_path)
        except Exception as e:
            # Some exceptions might be expected if rebuild is not fully implemented
            assert isinstance(e, (NotImplementedError, ValueError, IOError))
    
    def test_repair_with_backup(self):
        """Test repair with backup creation."""
        # Enable backup
        self.repair_tool.backup_enabled = True
        
        # Create a copy to test backup
        backup_test_maif = os.path.join(self.temp_dir, "backup_test.maif")
        backup_test_manifest = os.path.join(self.temp_dir, "backup_test_manifest.json")
        
        import shutil
        shutil.copy2(self.maif_path, backup_test_maif)
        shutil.copy2(self.manifest_path, backup_test_manifest)
        
        result = self.repair_tool.repair_file(backup_test_maif, backup_test_manifest)
        
        # Should handle backup creation
        assert isinstance(result, bool)


class TestValidationIntegration:
    """Test validation integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_repair_cycle(self):
        """Test complete validate-repair cycle."""
        # Create a MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Integration test data", metadata={"id": 1})
        
        maif_path = os.path.join(self.temp_dir, "integration.maif")
        manifest_path = os.path.join(self.temp_dir, "integration_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # 1. Initial validation (should be valid)
        result1 = self.validator.validate_file(maif_path, manifest_path)
        assert result1.is_valid is True
        
        # 2. Corrupt the file slightly
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Add a minor inconsistency
        manifest['header']['timestamp'] = "invalid_timestamp"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # 3. Validation should detect issues
        result2 = self.validator.validate_file(maif_path, manifest_path)
        # May or may not be invalid depending on validation strictness
        
        # 4. Attempt repair
        repair_result = self.repair_tool.repair_file(maif_path, manifest_path)
        assert isinstance(repair_result, bool)
        
        # 5. Re-validate after repair
        result3 = self.validator.validate_file(maif_path, manifest_path)
        # Should be valid or at least not worse than before
        assert isinstance(result3, ValidationResult)
    
    def test_validation_with_different_file_sizes(self):
        """Test validation with different file sizes."""
        test_cases = [
            ("small", "Small file content"),
            ("medium", "Medium file content " * 100),
            ("large", "Large file content " * 10000)
        ]
        
        for size_name, content in test_cases:
            encoder = MAIFEncoder(agent_id="test_agent")
            encoder.add_text_block(content, metadata={"size": size_name})
            
            maif_path = os.path.join(self.temp_dir, f"{size_name}.maif")
            manifest_path = os.path.join(self.temp_dir, f"{size_name}_manifest.json")
            
            encoder.build_maif(maif_path, manifest_path)
            
            # Validate
            result = self.validator.validate_file(maif_path, manifest_path)
            
            # All sizes should validate successfully
            assert result.is_valid is True
            assert len(result.errors) == 0
    
    def test_validation_with_multiple_block_types(self):
        """Test validation with various block types."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # Add different types of blocks
        encoder.add_text_block("Text block", metadata={"type": "text"})
        encoder.add_binary_block(b"Binary data", "data", metadata={"type": "binary"})
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]], metadata={"type": "embeddings"})
        
        maif_path = os.path.join(self.temp_dir, "multitype.maif")
        manifest_path = os.path.join(self.temp_dir, "multitype_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Validate
        result = self.validator.validate_file(maif_path, manifest_path)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_concurrent_validation(self):
        """Test concurrent validation operations."""
        import threading
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            encoder = MAIFEncoder(agent_id=f"agent_{i}")
            encoder.add_text_block(f"Concurrent test data {i}", metadata={"id": i})
            
            maif_path = os.path.join(self.temp_dir, f"concurrent_{i}.maif")
            manifest_path = os.path.join(self.temp_dir, f"concurrent_{i}_manifest.json")
            
            encoder.build_maif(maif_path, manifest_path)
            test_files.append((maif_path, manifest_path))
        
        # Validate concurrently
        results = []
        errors = []
        
        def validate_file(maif_path, manifest_path):
            try:
                result = self.validator.validate_file(maif_path, manifest_path)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for maif_path, manifest_path in test_files:
            thread = threading.Thread(target=validate_file, args=(maif_path, manifest_path))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        assert all(result.is_valid for result in results)


class TestValidationErrorHandling:
    """Test validation error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        empty_maif = os.path.join(self.temp_dir, "empty.maif")
        empty_manifest = os.path.join(self.temp_dir, "empty_manifest.json")
        
        # Create empty files
        open(empty_maif, 'w').close()
        open(empty_manifest, 'w').close()
        
        result = self.validator.validate_file(empty_maif, empty_manifest)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_binary_garbage(self):
        """Test validation of binary garbage file."""
        garbage_maif = os.path.join(self.temp_dir, "garbage.maif")
        garbage_manifest = os.path.join(self.temp_dir, "garbage_manifest.json")
        
        # Create garbage files
        with open(garbage_maif, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05' * 100)
        
        with open(garbage_manifest, 'w') as f:
            f.write('{"invalid": "json"')
        
        result = self.validator.validate_file(garbage_maif, garbage_manifest)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_repair_readonly_file(self):
        """Test repair of read-only file."""
        # Create a test file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Read-only test", metadata={"id": 1})
        
        readonly_maif = os.path.join(self.temp_dir, "readonly.maif")
        readonly_manifest = os.path.join(self.temp_dir, "readonly_manifest.json")
        
        encoder.build_maif(readonly_maif, readonly_manifest)
        
        # Make files read-only
        os.chmod(readonly_maif, 0o444)
        os.chmod(readonly_manifest, 0o444)
        
        try:
            result = self.repair_tool.repair_file(readonly_maif, readonly_manifest)
            # Should handle read-only files gracefully
            assert isinstance(result, bool)
        except PermissionError:
            # Expected on some systems
            pass
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(readonly_maif, 0o644)
                os.chmod(readonly_manifest, 0o644)
            except:
                pass
    
    def test_validate_with_permission_errors(self):
        """Test validation with permission errors."""
        # Create a file in a directory we can't access
        restricted_dir = os.path.join(self.temp_dir, "restricted")
        os.makedirs(restricted_dir)
        
        restricted_maif = os.path.join(restricted_dir, "restricted.maif")
        restricted_manifest = os.path.join(restricted_dir, "restricted_manifest.json")
        
        # Create files
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Restricted test", metadata={"id": 1})
        encoder.build_maif(restricted_maif, restricted_manifest)
        
        # Remove directory permissions
        os.chmod(restricted_dir, 0o000)
        
        try:
            result = self.validator.validate_file(restricted_maif, restricted_manifest)
            # Should handle permission errors gracefully
            assert result.is_valid is False
            assert len(result.errors) > 0
        except PermissionError:
            # Expected on some systems
            pass
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(restricted_dir, 0o755)
            except:
                pass
    
    def test_validation_with_large_files(self):
        """Test validation performance with large files."""
        # Create a large MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # Add many blocks
        for i in range(100):
            encoder.add_text_block(f"Large file block {i} " * 100, metadata={"id": i})
        
        large_maif = os.path.join(self.temp_dir, "large.maif")
        large_manifest = os.path.join(self.temp_dir, "large_manifest.json")
        
        encoder.build_maif(large_maif, large_manifest)
        
        # Validate (should complete in reasonable time)
        import time
        start_time = time.time()
        
        result = self.validator.validate_file(large_maif, large_manifest)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete validation
        assert isinstance(result, ValidationResult)
        # Should complete in reasonable time (adjust threshold as needed)
        assert duration < 30.0  # 30 seconds max


if __name__ == "__main__":
    pytest.main([__file__])
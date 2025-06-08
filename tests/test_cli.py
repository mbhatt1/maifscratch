"""
Comprehensive tests for MAIF CLI functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from maif.cli import (
    create_privacy_maif, access_privacy_maif, manage_privacy,
    create_maif, verify_maif, analyze_maif, extract_content, main
)


class TestCLICommands:
    """Test CLI command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_privacy_maif_command(self):
        """Test create-privacy-maif command."""
        # Create test input file
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for privacy MAIF creation")
        
        output_file = os.path.join(self.temp_dir, "output.maif")
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        
        # Test command with minimal arguments
        result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--privacy-level', 'medium',
            '--agent-id', 'test_agent'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        # Output files should be created
        assert os.path.exists(output_file)
        assert os.path.exists(manifest_file)
    
    def test_access_privacy_maif_command(self):
        """Test access-privacy-maif command."""
        # First create a privacy MAIF file
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for privacy access")
        
        maif_file = os.path.join(self.temp_dir, "test.maif")
        manifest_file = os.path.join(self.temp_dir, "test_manifest.json")
        
        # Create the MAIF file first
        create_result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--privacy-level', 'low',
            '--agent-id', 'test_agent'
        ])
        
        assert create_result.exit_code == 0
        
        # Now test access command
        result = self.runner.invoke(access_privacy_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--user-id', 'test_user',
            '--permission', 'read'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
    
    def test_manage_privacy_command(self):
        """Test manage-privacy command."""
        # Create a test MAIF file first
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for privacy management")
        
        maif_file = os.path.join(self.temp_dir, "manage_test.maif")
        manifest_file = os.path.join(self.temp_dir, "manage_test_manifest.json")
        
        # Create the MAIF file
        create_result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--privacy-level', 'medium',
            '--agent-id', 'test_agent'
        ])
        
        assert create_result.exit_code == 0
        
        # Test privacy management
        result = self.runner.invoke(manage_privacy, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--action', 'report'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
    
    def test_create_maif_command(self):
        """Test create-maif command."""
        # Create test input file
        input_file = os.path.join(self.temp_dir, "create_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for MAIF creation")
        
        output_file = os.path.join(self.temp_dir, "created.maif")
        manifest_file = os.path.join(self.temp_dir, "created_manifest.json")
        
        # Test command
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert os.path.exists(manifest_file)
    
    def test_verify_maif_command(self):
        """Test verify-maif command."""
        # Create a test MAIF file first
        input_file = os.path.join(self.temp_dir, "verify_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for verification")
        
        maif_file = os.path.join(self.temp_dir, "verify_test.maif")
        manifest_file = os.path.join(self.temp_dir, "verify_test_manifest.json")
        
        # Create the MAIF file
        create_result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        assert create_result.exit_code == 0
        
        # Test verification
        result = self.runner.invoke(verify_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
    
    def test_analyze_maif_command(self):
        """Test analyze-maif command."""
        # Create a test MAIF file first
        input_file = os.path.join(self.temp_dir, "analyze_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for analysis")
        
        maif_file = os.path.join(self.temp_dir, "analyze_test.maif")
        manifest_file = os.path.join(self.temp_dir, "analyze_test_manifest.json")
        
        # Create the MAIF file
        create_result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        assert create_result.exit_code == 0
        
        # Test analysis
        result = self.runner.invoke(analyze_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--analysis-type', 'basic'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
    
    def test_extract_content_command(self):
        """Test extract-content command."""
        # Create a test MAIF file first
        input_file = os.path.join(self.temp_dir, "extract_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for extraction")
        
        maif_file = os.path.join(self.temp_dir, "extract_test.maif")
        manifest_file = os.path.join(self.temp_dir, "extract_test_manifest.json")
        
        # Create the MAIF file
        create_result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        assert create_result.exit_code == 0
        
        # Test content extraction
        output_dir = os.path.join(self.temp_dir, "extracted")
        
        result = self.runner.invoke(extract_content, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--output-dir', output_dir,
            '--format', 'text'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        assert os.path.exists(output_dir)
    
    def test_main_cli_help(self):
        """Test main CLI help command."""
        result = self.runner.invoke(main, ['--help'])
        
        # Should show help without errors
        assert result.exit_code == 0
        assert "Usage:" in result.output
    
    def test_invalid_input_file(self):
        """Test CLI with invalid input file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.txt")
        output_file = os.path.join(self.temp_dir, "output.maif")
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        
        result = self.runner.invoke(create_maif, [
            '--input', nonexistent_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        # Should handle error gracefully
        assert result.exit_code != 0
    
    def test_invalid_output_directory(self):
        """Test CLI with invalid output directory."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content")
        
        # Try to write to non-existent directory
        invalid_output = "/nonexistent/directory/output.maif"
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', invalid_output,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        # Should handle error gracefully
        assert result.exit_code != 0
    
    def test_cli_with_json_input(self):
        """Test CLI with JSON input format."""
        # Create test JSON file
        json_data = {
            "title": "Test Document",
            "content": "This is test content",
            "metadata": {"author": "Test Author"}
        }
        
        input_file = os.path.join(self.temp_dir, "input.json")
        with open(input_file, 'w') as f:
            json.dump(json_data, f)
        
        output_file = os.path.join(self.temp_dir, "json_output.maif")
        manifest_file = os.path.join(self.temp_dir, "json_manifest.json")
        
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'json'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert os.path.exists(manifest_file)
    
    def test_cli_with_compression(self):
        """Test CLI with compression options."""
        input_file = os.path.join(self.temp_dir, "compress_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for compression " * 100)  # Repetitive for better compression
        
        output_file = os.path.join(self.temp_dir, "compressed.maif")
        manifest_file = os.path.join(self.temp_dir, "compressed_manifest.json")
        
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text',
            '--compression', 'zlib'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert os.path.exists(manifest_file)
    
    def test_cli_with_encryption(self):
        """Test CLI with encryption options."""
        input_file = os.path.join(self.temp_dir, "encrypt_input.txt")
        with open(input_file, 'w') as f:
            f.write("Sensitive content for encryption")
        
        output_file = os.path.join(self.temp_dir, "encrypted.maif")
        manifest_file = os.path.join(self.temp_dir, "encrypted_manifest.json")
        
        result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--privacy-level', 'high',
            '--encryption', 'aes-gcm'
        ])
        
        # Should execute without errors
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert os.path.exists(manifest_file)


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_required_arguments(self):
        """Test CLI with missing required arguments."""
        # Test create-maif without required arguments
        result = self.runner.invoke(create_maif, [])
        
        # Should show error for missing arguments
        assert result.exit_code != 0
    
    def test_invalid_privacy_level(self):
        """Test CLI with invalid privacy level."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content")
        
        output_file = os.path.join(self.temp_dir, "output.maif")
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        
        result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--privacy-level', 'invalid_level'
        ])
        
        # Should handle invalid privacy level
        assert result.exit_code != 0
    
    def test_invalid_format(self):
        """Test CLI with invalid format."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content")
        
        output_file = os.path.join(self.temp_dir, "output.maif")
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'invalid_format'
        ])
        
        # Should handle invalid format
        assert result.exit_code != 0
    
    def test_corrupted_maif_file(self):
        """Test CLI with corrupted MAIF file."""
        # Create corrupted MAIF file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.maif")
        with open(corrupted_file, 'wb') as f:
            f.write(b"corrupted binary data")
        
        # Create valid manifest
        manifest_file = os.path.join(self.temp_dir, "manifest.json")
        with open(manifest_file, 'w') as f:
            json.dump({"version": "2.0", "blocks": []}, f)
        
        result = self.runner.invoke(verify_maif, [
            '--maif-file', corrupted_file,
            '--manifest', manifest_file
        ])
        
        # Should handle corrupted file gracefully
        assert result.exit_code != 0
    
    def test_permission_denied(self):
        """Test CLI with permission denied scenarios."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content")
        
        # Try to write to root directory (should fail)
        output_file = "/root/output.maif"
        manifest_file = "/root/manifest.json"
        
        result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', output_file,
            '--manifest', manifest_file,
            '--agent-id', 'test_agent',
            '--format', 'text'
        ])
        
        # Should handle permission error gracefully
        assert result.exit_code != 0


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_verify_analyze_workflow(self):
        """Test complete create -> verify -> analyze workflow."""
        # Step 1: Create MAIF file
        input_file = os.path.join(self.temp_dir, "workflow_input.txt")
        with open(input_file, 'w') as f:
            f.write("Test content for complete workflow")
        
        maif_file = os.path.join(self.temp_dir, "workflow.maif")
        manifest_file = os.path.join(self.temp_dir, "workflow_manifest.json")
        
        create_result = self.runner.invoke(create_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--agent-id', 'workflow_agent',
            '--format', 'text'
        ])
        
        assert create_result.exit_code == 0
        
        # Step 2: Verify MAIF file
        verify_result = self.runner.invoke(verify_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file
        ])
        
        assert verify_result.exit_code == 0
        
        # Step 3: Analyze MAIF file
        analyze_result = self.runner.invoke(analyze_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--analysis-type', 'basic'
        ])
        
        assert analyze_result.exit_code == 0
    
    def test_privacy_workflow(self):
        """Test privacy-focused workflow."""
        # Create privacy MAIF
        input_file = os.path.join(self.temp_dir, "privacy_input.txt")
        with open(input_file, 'w') as f:
            f.write("Sensitive information for privacy testing")
        
        maif_file = os.path.join(self.temp_dir, "privacy.maif")
        manifest_file = os.path.join(self.temp_dir, "privacy_manifest.json")
        
        # Create with high privacy
        create_result = self.runner.invoke(create_privacy_maif, [
            '--input', input_file,
            '--output', maif_file,
            '--manifest', manifest_file,
            '--agent-id', 'privacy_agent',
            '--privacy-level', 'high',
            '--encryption', 'aes-gcm'
        ])
        
        assert create_result.exit_code == 0
        
        # Test access control
        access_result = self.runner.invoke(access_privacy_maif, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--user-id', 'authorized_user',
            '--permission', 'read'
        ])
        
        assert access_result.exit_code == 0
        
        # Generate privacy report
        manage_result = self.runner.invoke(manage_privacy, [
            '--maif-file', maif_file,
            '--manifest', manifest_file,
            '--action', 'report'
        ])
        
        assert manage_result.exit_code == 0
    
    def test_batch_processing(self):
        """Test batch processing multiple files."""
        # Create multiple input files
        input_files = []
        for i in range(3):
            input_file = os.path.join(self.temp_dir, f"batch_input_{i}.txt")
            with open(input_file, 'w') as f:
                f.write(f"Batch content {i}")
            input_files.append(input_file)
        
        # Process each file
        for i, input_file in enumerate(input_files):
            maif_file = os.path.join(self.temp_dir, f"batch_{i}.maif")
            manifest_file = os.path.join(self.temp_dir, f"batch_{i}_manifest.json")
            
            result = self.runner.invoke(create_maif, [
                '--input', input_file,
                '--output', maif_file,
                '--manifest', manifest_file,
                '--agent-id', f'batch_agent_{i}',
                '--format', 'text'
            ])
            
            assert result.exit_code == 0
            assert os.path.exists(maif_file)
            assert os.path.exists(manifest_file)


if __name__ == "__main__":
    pytest.main([__file__])
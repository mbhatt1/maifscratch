#!/usr/bin/env python3
"""
Test suite for verifying Fernet removal and AES-GCM replacement.
Tests both positive cases (expected functionality) and negative cases (error handling).
"""

import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import maif
sys.path.insert(0, str(Path(__file__).parent))

from maif.core import MAIFEncoder, MAIFDecoder
from maif.privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode
import tempfile
import json

class TestEncryptionModeChanges:
    """Test suite for encryption mode changes."""
    
    def test_fernet_enum_removed(self):
        """Positive test: Verify Fernet is no longer in EncryptionMode enum."""
        available_modes = [mode.value for mode in EncryptionMode]
        assert 'fernet' not in available_modes
        print("✓ Fernet successfully removed from EncryptionMode enum")
    
    def test_aes_gcm_available(self):
        """Positive test: Verify AES-GCM is available."""
        available_modes = [mode.value for mode in EncryptionMode]
        assert 'aes_gcm' in available_modes
        print("✓ AES-GCM is available in EncryptionMode enum")
    
    def test_aes_gcm_encryption_decryption(self):
        """Positive test: Verify AES-GCM encryption and decryption works."""
        privacy_engine = PrivacyEngine()
        test_data = b"This is test data for AES-GCM encryption"
        block_id = "test_block_001"
        
        # Encrypt data
        encrypted_data, metadata = privacy_engine.encrypt_data(
            test_data, block_id, EncryptionMode.AES_GCM
        )
        
        # Verify encryption worked
        assert encrypted_data != test_data
        assert metadata['algorithm'] == 'AES-GCM'
        assert 'iv' in metadata
        assert 'tag' in metadata
        
        # Decrypt data
        decrypted_data = privacy_engine.decrypt_data(encrypted_data, block_id, metadata)
        
        # Verify decryption worked
        assert decrypted_data == test_data
        print("✓ AES-GCM encryption and decryption working correctly")
    
    def test_fernet_encryption_raises_error(self):
        """Negative test: Verify attempting to use Fernet raises an error."""
        # Try to access FERNET attribute (should fail)
        try:
            fernet_mode = EncryptionMode.FERNET
            # If we get here, the test should fail
            assert False, "FERNET should not exist in EncryptionMode enum"
        except AttributeError:
            # This is expected - FERNET should not exist
            print("✓ Fernet enum access correctly raises AttributeError")
    
    def test_maif_encoder_with_aes_gcm(self):
        """Positive test: Verify MAIFEncoder works with AES-GCM."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create encoder with AES-GCM encryption
            encoder = MAIFEncoder(enable_privacy=True)
            
            # Set AES-GCM policy
            policy = PrivacyPolicy(
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                encryption_mode=EncryptionMode.AES_GCM,
                anonymization_required=False,
                audit_required=False
            )
            encoder.set_default_privacy_policy(policy)
            
            # Add test data
            test_data = b"Test data for MAIF encoder with AES-GCM"
            encoder.add_binary_block(test_data, "test_binary")
            
            # Save to file
            output_path = os.path.join(temp_dir, "test_aes_gcm.maif")
            manifest_path = os.path.join(temp_dir, "test_aes_gcm_manifest.json")
            encoder.save(output_path, manifest_path)
            
            # Verify files were created
            assert os.path.exists(output_path)
            assert os.path.exists(manifest_path)
            print("✓ MAIFEncoder successfully creates files with AES-GCM encryption")
    
    def test_default_encryption_is_aes_gcm(self):
        """Positive test: Verify default encryption mode is AES-GCM."""
        encoder = MAIFEncoder(enable_privacy=True)
        default_policy = encoder.default_privacy_policy
        assert default_policy.encryption_mode == EncryptionMode.AES_GCM
        print("✓ Default encryption mode is correctly set to AES-GCM")
    
    def test_chacha20_still_available(self):
        """Positive test: Verify ChaCha20 is still available as alternative."""
        privacy_engine = PrivacyEngine()
        test_data = b"Test data for ChaCha20"
        block_id = "test_block_chacha20"
        
        # Encrypt with ChaCha20
        encrypted_data, metadata = privacy_engine.encrypt_data(
            test_data, block_id, EncryptionMode.CHACHA20_POLY1305
        )
        
        # Verify encryption worked
        assert encrypted_data != test_data
        assert metadata['algorithm'] == 'ChaCha20-Poly1305'
        assert 'nonce' in metadata
        
        # Decrypt data
        decrypted_data = privacy_engine.decrypt_data(encrypted_data, block_id, metadata)
        assert decrypted_data == test_data
        print("✓ ChaCha20-Poly1305 encryption still works as alternative")
    
    def test_unsupported_algorithm_decryption(self):
        """Negative test: Verify unsupported algorithm in decryption raises error."""
        privacy_engine = PrivacyEngine()
        test_data = b"Test data"
        block_id = "test_block_unsupported"
        
        # Manually create encryption key
        key = privacy_engine.derive_key(f"block:{block_id}")
        privacy_engine.encryption_keys[block_id] = key
        
        # Try to decrypt with unsupported algorithm metadata
        fake_metadata = {'algorithm': 'Fernet'}  # This should fail
        
        with pytest.raises(ValueError, match="Unsupported decryption algorithm"):
            privacy_engine.decrypt_data(test_data, block_id, fake_metadata)
        print("✓ Unsupported decryption algorithm correctly raises error")
    
    def test_no_encryption_mode_still_works(self):
        """Positive test: Verify NONE encryption mode still works."""
        privacy_engine = PrivacyEngine()
        test_data = b"Test data without encryption"
        block_id = "test_block_none"
        
        # Encrypt with NONE mode
        encrypted_data, metadata = privacy_engine.encrypt_data(
            test_data, block_id, EncryptionMode.NONE
        )
        
        # Verify no encryption occurred
        assert encrypted_data == test_data
        assert metadata == {}
        print("✓ NONE encryption mode works correctly")
    
    def test_large_data_aes_gcm_performance(self):
        """Positive test: Verify AES-GCM works with larger data."""
        privacy_engine = PrivacyEngine()
        # Create 1MB of test data
        test_data = b"A" * (1024 * 1024)
        block_id = "test_block_large"
        
        # Encrypt large data
        encrypted_data, metadata = privacy_engine.encrypt_data(
            test_data, block_id, EncryptionMode.AES_GCM
        )
        
        # Verify encryption worked
        assert len(encrypted_data) > 0
        assert encrypted_data != test_data
        assert metadata['algorithm'] == 'AES-GCM'
        
        # Decrypt large data
        decrypted_data = privacy_engine.decrypt_data(encrypted_data, block_id, metadata)
        assert decrypted_data == test_data
        print("✓ AES-GCM handles large data (1MB) correctly")

class TestCLIChanges:
    """Test CLI argument changes."""
    
    def test_cli_fernet_option_removed(self):
        """Negative test: Verify fernet is not in CLI choices."""
        # Read the CLI file content to check encryption choices
        import inspect
        from maif import cli
        
        # Get the source code of the CLI module
        cli_source = inspect.getsource(cli)
        
        # Check that fernet is not in the encryption choices
        assert "'fernet'" not in cli_source
        assert "'aes_gcm'" in cli_source
        print("✓ CLI correctly removes 'fernet' option and keeps 'aes_gcm'")

def run_tests():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("TESTING FERNET REMOVAL AND AES-GCM REPLACEMENT")
    print("=" * 60)
    
    test_encryption = TestEncryptionModeChanges()
    test_cli = TestCLIChanges()
    
    # Run positive tests
    print("\n--- POSITIVE TESTS (Expected Functionality) ---")
    try:
        test_encryption.test_fernet_enum_removed()
        test_encryption.test_aes_gcm_available()
        test_encryption.test_aes_gcm_encryption_decryption()
        test_encryption.test_maif_encoder_with_aes_gcm()
        test_encryption.test_default_encryption_is_aes_gcm()
        test_encryption.test_chacha20_still_available()
        test_encryption.test_no_encryption_mode_still_works()
        test_encryption.test_large_data_aes_gcm_performance()
        test_cli.test_cli_fernet_option_removed()
        print("\n✓ All positive tests PASSED")
    except Exception as e:
        print(f"\n✗ Positive test FAILED: {e}")
        return False
    
    # Run negative tests
    print("\n--- NEGATIVE TESTS (Error Handling) ---")
    try:
        test_encryption.test_fernet_encryption_raises_error()
        test_encryption.test_unsupported_algorithm_decryption()
        print("\n✓ All negative tests PASSED")
    except Exception as e:
        print(f"\n✗ Negative test FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Fernet successfully removed from codebase")
    print("✓ AES-GCM is now the default encryption method")
    print("✓ All encryption/decryption functionality works correctly")
    print("✓ Error handling works for unsupported algorithms")
    print("✓ CLI options updated correctly")
    print("✓ Alternative encryption modes (ChaCha20) still available")
    print("\nAll tests PASSED! The Fernet to AES-GCM migration is successful.")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
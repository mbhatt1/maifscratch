# Cryptography API Reference

The Cryptography module provides low-level cryptographic operations including encryption, hashing, key derivation, and digital signatures with support for industry-standard algorithms and MAIF's novel cryptographic innovations.

## Overview

Cryptographic features:
- **Symmetric Encryption**: AES-GCM, ChaCha20-Poly1305, XChaCha20-Poly1305
- **Asymmetric Encryption**: RSA-OAEP, ECDH, X25519
- **Digital Signatures**: RSA-PSS, ECDSA, EdDSA (Ed25519/Ed448)
- **Hash Functions**: SHA-2, SHA-3, Blake2, Argon2
- **Key Derivation**: PBKDF2, scrypt, Argon2, HKDF
- **Novel Algorithms**: CSB (Cryptographic Semantic Binding)

```mermaid
graph TB
    subgraph "Cryptography Module"
        Crypto[Cryptography Engine]
        
        subgraph "Symmetric Crypto"
            AES[AES-GCM]
            ChaCha[ChaCha20-Poly1305]
            XChaCha[XChaCha20-Poly1305]
        end
        
        subgraph "Asymmetric Crypto"
            RSA[RSA-OAEP]
            ECDH[ECDH]
            X25519[X25519]
        end
        
        subgraph "Signatures"
            RSAPSS[RSA-PSS]
            ECDSA[ECDSA]
            EdDSA[EdDSA]
        end
        
        subgraph "Hash & KDF"
            SHA[SHA-2/3]
            Blake[Blake2]
            PBKDF2[PBKDF2]
            Argon2[Argon2]
        end
        
        subgraph "Novel Algorithms"
            CSB[CSB Algorithm]
            SemanticCrypto[Semantic Crypto]
        end
    end
    
    Crypto --> AES
    Crypto --> ChaCha
    Crypto --> XChaCha
    Crypto --> RSA
    Crypto --> ECDH
    Crypto --> X25519
    Crypto --> RSAPSS
    Crypto --> ECDSA
    Crypto --> EdDSA
    Crypto --> SHA
    Crypto --> Blake
    Crypto --> PBKDF2
    Crypto --> Argon2
    Crypto --> CSB
    Crypto --> SemanticCrypto
    
    style Crypto fill:#3c82f6,stroke:#1e40af,stroke-width:3px,color:#fff
    style CSB fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style SemanticCrypto fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
```

## Quick Start

```python
from maif.crypto import CryptographyEngine, SymmetricKey, AsymmetricKeyPair

# Create crypto engine
crypto = CryptographyEngine()

# Symmetric encryption
key = crypto.generate_symmetric_key("ChaCha20-Poly1305")
ciphertext = crypto.encrypt_symmetric(b"secret data", key)
plaintext = crypto.decrypt_symmetric(ciphertext, key)

# Asymmetric encryption
keypair = crypto.generate_asymmetric_keypair("RSA", key_size=2048)
encrypted = crypto.encrypt_asymmetric(b"secret", keypair.public_key)
decrypted = crypto.decrypt_asymmetric(encrypted, keypair.private_key)

# Digital signatures
signature = crypto.sign(b"document", keypair.private_key)
valid = crypto.verify_signature(b"document", signature, keypair.public_key)

# Cryptographic hashing
hash_value = crypto.hash(b"data", algorithm="SHA-256")
```

## CryptographyEngine Class

### Constructor

```python
crypto = CryptographyEngine(
    # Default algorithms
    default_symmetric_algorithm="ChaCha20-Poly1305",
    default_asymmetric_algorithm="RSA",
    default_hash_algorithm="SHA-256",
    default_signature_algorithm="RSA-PSS",
    
    # Security settings
    secure_random_source="system",  # or "hardware", "fortuna"
    constant_time_operations=True,
    side_channel_protection=True,
    
    # Key management
    key_derivation_algorithm="Argon2id",
    key_derivation_iterations=100000,
    key_derivation_memory_kb=65536,
    
    # Performance
    hardware_acceleration=True,     # Use AES-NI, etc.
    parallel_operations=True,
    
    # Compliance
    fips_mode=True,                # FIPS 140-2 compliance
    quantum_resistant_mode=False,   # Post-quantum algorithms
    
    # Novel features
    enable_semantic_binding=True,   # CSB algorithm
    semantic_binding_strength=128   # bits
)
```

## Symmetric Encryption

### Key Generation

#### `generate_symmetric_key(algorithm, **options) -> SymmetricKey`

```python
# ChaCha20-Poly1305 key
chacha_key = crypto.generate_symmetric_key(
    algorithm="ChaCha20-Poly1305",
    key_size=256,                   # bits
    
    # Key metadata
    key_id="data-encryption-key",
    purpose="data_encryption",
    
    # Security options
    secure_generation=True,
    hardware_random=True,
    
    # Key derivation (if from password)
    derive_from_password=False,
    password="strong-password",
    salt_size=32,
    iterations=100000
)

# AES-GCM key
aes_key = crypto.generate_symmetric_key(
    algorithm="AES-GCM",
    key_size=256,
    key_id="file-encryption-key"
)

# XChaCha20-Poly1305 key (extended nonce)
xchacha_key = crypto.generate_symmetric_key(
    algorithm="XChaCha20-Poly1305",
    key_size=256,
    key_id="stream-encryption-key"
)
```

### Encryption Operations

#### `encrypt_symmetric(data, key, **options) -> SymmetricCiphertext`

```python
# Basic encryption
plaintext = b"Sensitive information"
ciphertext = crypto.encrypt_symmetric(plaintext, chacha_key)

# Advanced encryption with options
ciphertext = crypto.encrypt_symmetric(
    data=plaintext,
    key=chacha_key,
    
    # Encryption options
    associated_data=b"metadata",    # Authenticated but not encrypted
    nonce=None,                     # Auto-generate if None
    
    # Security options
    constant_time=True,
    secure_memory=True,
    
    # Compression
    compress_before_encrypt=True,
    compression_algorithm="zstd",
    
    # Metadata
    include_metadata=True,
    custom_metadata={"purpose": "data_protection"}
)

print(f"Algorithm: {ciphertext.algorithm}")
print(f"Ciphertext size: {len(ciphertext.ciphertext)} bytes")
print(f"Nonce: {ciphertext.nonce.hex()}")
print(f"Tag: {ciphertext.tag.hex()}")
```

#### `decrypt_symmetric(ciphertext, key, **options) -> bytes`

```python
# Basic decryption
decrypted = crypto.decrypt_symmetric(ciphertext, chacha_key)

# Advanced decryption with verification
decrypted = crypto.decrypt_symmetric(
    ciphertext=ciphertext,
    key=chacha_key,
    
    # Verification options
    verify_associated_data=True,
    expected_associated_data=b"metadata",
    
    # Security options
    constant_time=True,
    secure_memory=True,
    clear_plaintext_after_use=False,
    
    # Decompression
    decompress_after_decrypt=True,
    
    # Validation
    verify_metadata=True,
    expected_metadata={"purpose": "data_protection"}
)
```

### Stream Encryption

#### `create_symmetric_stream_cipher(key, **options) -> StreamCipher`

```python
# Create stream cipher for large data
stream_cipher = crypto.create_symmetric_stream_cipher(
    key=xchacha_key,
    algorithm="XChaCha20-Poly1305",
    
    # Stream options
    chunk_size=64*1024,             # 64KB chunks
    buffer_size=1024*1024,          # 1MB buffer
    
    # Security
    unique_nonce_per_chunk=True,
    authenticate_chunks=True
)

# Encrypt stream
with open("large_file.bin", "rb") as input_file:
    with open("encrypted_file.bin", "wb") as output_file:
        for chunk in input_file:
            encrypted_chunk = stream_cipher.encrypt_chunk(chunk)
            output_file.write(encrypted_chunk)

# Finalize stream
final_tag = stream_cipher.finalize()
```

## Asymmetric Encryption

### Key Pair Generation

#### `generate_asymmetric_keypair(algorithm, **options) -> AsymmetricKeyPair`

```python
# RSA key pair
rsa_keypair = crypto.generate_asymmetric_keypair(
    algorithm="RSA",
    key_size=2048,                  # or 3072, 4096
    public_exponent=65537,
    
    # Key metadata
    key_id="document-encryption-key",
    usage=["encryption", "signature"],
    
    # Security options
    secure_generation=True,
    hardware_backed=False,
    
    # Storage options
    password_protect_private_key=True,
    private_key_password="strong-password"
)

# ECDH key pair (Elliptic Curve Diffie-Hellman)
ecdh_keypair = crypto.generate_asymmetric_keypair(
    algorithm="ECDH",
    curve="P-256",                  # or "P-384", "P-521"
    key_id="key-exchange-key"
)

# X25519 key pair (Curve25519)
x25519_keypair = crypto.generate_asymmetric_keypair(
    algorithm="X25519",
    key_id="modern-encryption-key"
)

# Ed25519 key pair (EdDSA)
ed25519_keypair = crypto.generate_asymmetric_keypair(
    algorithm="Ed25519",
    key_id="signature-key"
)
```

### Asymmetric Encryption

#### `encrypt_asymmetric(data, public_key, **options) -> AsymmetricCiphertext`

```python
# RSA-OAEP encryption
plaintext = b"Secret message"
encrypted = crypto.encrypt_asymmetric(
    data=plaintext,
    public_key=rsa_keypair.public_key,
    
    # Encryption options
    padding="OAEP",                 # or "PKCS1v15"
    hash_algorithm="SHA-256",       # for OAEP
    mgf_hash="SHA-256",            # Mask generation function
    
    # Hybrid encryption for large data
    use_hybrid_encryption=True,     # RSA + AES
    symmetric_algorithm="AES-GCM",
    
    # Security options
    secure_memory=True
)
```

#### `decrypt_asymmetric(ciphertext, private_key, **options) -> bytes`

```python
# RSA-OAEP decryption
decrypted = crypto.decrypt_asymmetric(
    ciphertext=encrypted,
    private_key=rsa_keypair.private_key,
    
    # Decryption options
    padding="OAEP",
    hash_algorithm="SHA-256",
    
    # Security options
    constant_time=True,
    secure_memory=True,
    
    # Private key protection
    private_key_password="strong-password"
)
```

### Key Exchange

#### `perform_key_exchange(private_key, public_key, **options) -> SharedSecret`

```python
# ECDH key exchange
alice_keypair = crypto.generate_asymmetric_keypair("ECDH", curve="P-256")
bob_keypair = crypto.generate_asymmetric_keypair("ECDH", curve="P-256")

# Alice computes shared secret
alice_shared = crypto.perform_key_exchange(
    private_key=alice_keypair.private_key,
    public_key=bob_keypair.public_key,
    
    # Key derivation
    derive_key=True,
    key_derivation_function="HKDF",
    hash_algorithm="SHA-256",
    salt=b"unique-salt",
    info=b"key-exchange-context",
    derived_key_length=32
)

# Bob computes the same shared secret
bob_shared = crypto.perform_key_exchange(
    private_key=bob_keypair.private_key,
    public_key=alice_keypair.public_key,
    derive_key=True,
    key_derivation_function="HKDF",
    hash_algorithm="SHA-256",
    salt=b"unique-salt",
    info=b"key-exchange-context",
    derived_key_length=32
)

assert alice_shared.key == bob_shared.key  # Same shared secret
```

## Digital Signatures

### Signature Generation

#### `sign(data, private_key, **options) -> Signature`

```python
# RSA-PSS signature
document = b"Important contract terms..."
signature = crypto.sign(
    data=document,
    private_key=rsa_keypair.private_key,
    
    # Signature algorithm
    algorithm="RSA-PSS",
    hash_algorithm="SHA-256",
    salt_length="auto",             # PSS salt length
    
    # Security options
    deterministic=False,            # Use random salt
    secure_memory=True,
    
    # Metadata
    include_timestamp=True,
    custom_attributes={"signer": "alice", "purpose": "approval"}
)

# ECDSA signature
ecdsa_signature = crypto.sign(
    data=document,
    private_key=ecdsa_keypair.private_key,
    algorithm="ECDSA",
    hash_algorithm="SHA-256",
    deterministic=True              # RFC 6979 deterministic ECDSA
)

# EdDSA signature (Ed25519)
ed25519_signature = crypto.sign(
    data=document,
    private_key=ed25519_keypair.private_key,
    algorithm="EdDSA"               # Pure EdDSA (no hashing)
)
```

### Signature Verification

#### `verify_signature(data, signature, public_key, **options) -> bool`

```python
# Verify RSA-PSS signature
is_valid = crypto.verify_signature(
    data=document,
    signature=signature,
    public_key=rsa_keypair.public_key,
    
    # Verification options
    algorithm="RSA-PSS",
    hash_algorithm="SHA-256",
    
    # Security options
    constant_time=True,
    
    # Metadata verification
    verify_timestamp=True,
    verify_attributes=True,
    expected_attributes={"signer": "alice"}
)

print(f"Signature valid: {is_valid}")
```

## Cryptographic Hashing

### Hash Functions

#### `hash(data, algorithm, **options) -> bytes`

```python
# SHA-256 hash
sha256_hash = crypto.hash(
    data=b"Data to hash",
    algorithm="SHA-256"
)

# SHA-3 hash
sha3_hash = crypto.hash(
    data=b"Data to hash",
    algorithm="SHA3-256"
)

# Blake2b hash with key
blake2_hash = crypto.hash(
    data=b"Data to hash",
    algorithm="Blake2b",
    digest_size=32,                 # Output size in bytes
    key=b"secret-key",              # Optional key for MAC
    salt=b"unique-salt",            # Optional salt
    person=b"application-id"        # Optional personalization
)

# Streaming hash for large data
hasher = crypto.create_hasher("SHA-256")
hasher.update(b"chunk 1")
hasher.update(b"chunk 2")
final_hash = hasher.finalize()
```

### Password Hashing

#### `hash_password(password, **options) -> PasswordHash`

```python
# Argon2id password hashing (recommended)
password_hash = crypto.hash_password(
    password="user-password",
    algorithm="Argon2id",
    
    # Argon2 parameters
    time_cost=3,                    # Iterations
    memory_cost=65536,              # Memory in KB
    parallelism=4,                  # Parallel threads
    
    # Salt
    salt_size=32,                   # Random salt size
    salt=None,                      # Auto-generate if None
    
    # Output
    hash_length=32                  # Hash output length
)

# scrypt password hashing
scrypt_hash = crypto.hash_password(
    password="user-password",
    algorithm="scrypt",
    n=32768,                        # CPU/memory cost
    r=8,                            # Block size
    p=1,                            # Parallelization
    salt_size=32,
    hash_length=32
)

# PBKDF2 password hashing
pbkdf2_hash = crypto.hash_password(
    password="user-password",
    algorithm="PBKDF2",
    hash_algorithm="SHA-256",
    iterations=100000,
    salt_size=32,
    hash_length=32
)
```

#### `verify_password(password, password_hash) -> bool`

```python
# Verify password against hash
is_correct = crypto.verify_password(
    password="user-password",
    password_hash=password_hash
)
```

## Key Derivation

### Key Derivation Functions

#### `derive_key(input_key_material, **options) -> DerivedKey`

```python
# HKDF (HMAC-based Key Derivation Function)
derived_key = crypto.derive_key(
    input_key_material=b"shared-secret",
    algorithm="HKDF",
    
    # HKDF parameters
    hash_algorithm="SHA-256",
    salt=b"unique-salt",
    info=b"encryption-key",         # Context information
    length=32                       # Output key length
)

# PBKDF2 key derivation
pbkdf2_key = crypto.derive_key(
    input_key_material=b"password",
    algorithm="PBKDF2",
    hash_algorithm="SHA-256",
    salt=b"random-salt",
    iterations=100000,
    length=32
)

# Argon2 key derivation
argon2_key = crypto.derive_key(
    input_key_material=b"password",
    algorithm="Argon2id",
    salt=b"random-salt",
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    length=32
)
```

## Novel Cryptographic Algorithms

### Cryptographic Semantic Binding (CSB)

#### `create_semantic_binding(content, **options) -> SemanticBinding`

```python
# Create semantic binding for content
semantic_binding = crypto.create_semantic_binding(
    content="Sensitive document content",
    
    # Semantic parameters
    embedding_model="all-MiniLM-L6-v2",
    semantic_threshold=0.8,
    
    # Cryptographic parameters
    binding_strength=128,           # Security level in bits
    encryption_algorithm="ChaCha20-Poly1305",
    
    # Binding options
    bind_to_semantic_meaning=True,
    bind_to_syntactic_structure=False,
    preserve_privacy=True,
    
    # Context binding
    context_data="document-classification",
    temporal_binding=True,
    spatial_binding=False
)

print(f"Binding ID: {semantic_binding.binding_id}")
print(f"Semantic fingerprint: {semantic_binding.semantic_fingerprint}")
```

#### `verify_semantic_binding(content, binding, **options) -> bool`

```python
# Verify semantic binding
is_semantically_bound = crypto.verify_semantic_binding(
    content="Modified document content",
    binding=semantic_binding,
    
    # Verification options
    semantic_threshold=0.8,
    allow_minor_changes=True,
    context_verification=True,
    
    # Security options
    constant_time=True
)

print(f"Semantic binding valid: {is_semantically_bound}")
```

### Privacy-Preserving Cryptography

#### `encrypt_with_semantic_preservation(data, **options) -> SemanticCiphertext`

```python
# Encrypt while preserving semantic searchability
semantic_ciphertext = crypto.encrypt_with_semantic_preservation(
    data="Confidential business document",
    
    # Encryption
    encryption_key=chacha_key,
    encryption_algorithm="ChaCha20-Poly1305",
    
    # Semantic preservation
    preserve_semantic_search=True,
    semantic_granularity="sentence",  # or "word", "paragraph"
    
    # Privacy parameters
    differential_privacy=True,
    epsilon=1.0,                    # Privacy budget
    
    # Search parameters
    enable_fuzzy_search=True,
    search_precision=0.8
)
```

## Random Number Generation

### Secure Random Generation

#### `generate_random_bytes(length, **options) -> bytes`

```python
# Generate cryptographically secure random bytes
random_bytes = crypto.generate_random_bytes(
    length=32,
    
    # Randomness source
    source="system",                # or "hardware", "fortuna"
    
    # Quality requirements
    entropy_requirement="high",
    reseed_interval=1000000,        # Reseed after N bytes
    
    # Testing
    statistical_tests=True,
    fips_tests=True
)

# Generate random integers
random_int = crypto.generate_random_int(
    min_value=1000,
    max_value=9999,
    source="hardware"
)

# Generate random string
random_string = crypto.generate_random_string(
    length=16,
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
```

## Error Handling

```python
from maif.exceptions import (
    CryptographyError,
    EncryptionError,
    DecryptionError,
    SignatureError,
    KeyGenerationError,
    HashError,
    RandomnessError
)

try:
    # Cryptographic operations
    key = crypto.generate_symmetric_key("ChaCha20-Poly1305")
    ciphertext = crypto.encrypt_symmetric(data, key)
    signature = crypto.sign(data, private_key)
    
except KeyGenerationError as e:
    logger.error(f"Key generation failed: {e}")
    # Handle key generation issues
    
except EncryptionError as e:
    logger.error(f"Encryption failed: {e}")
    # Handle encryption failures
    
except DecryptionError as e:
    logger.error(f"Decryption failed: {e}")
    # Handle decryption failures
    
except SignatureError as e:
    logger.error(f"Signature operation failed: {e}")
    # Handle signature issues
    
except HashError as e:
    logger.error(f"Hashing failed: {e}")
    # Handle hashing problems
    
except RandomnessError as e:
    logger.error(f"Random generation failed: {e}")
    # Handle randomness issues
    
except CryptographyError as e:
    logger.error(f"Cryptography error: {e}")
    # Handle general crypto errors
```

## Best Practices

### Algorithm Selection
```python
# Use modern, secure algorithms
crypto = CryptographyEngine(
    default_symmetric_algorithm="ChaCha20-Poly1305",  # Fast, secure
    default_asymmetric_algorithm="X25519",            # Modern ECC
    default_hash_algorithm="SHA-256",                 # Widely supported
    default_signature_algorithm="Ed25519"             # Fast, secure
)
```

### Key Management
```python
# Generate keys securely
key = crypto.generate_symmetric_key(
    "ChaCha20-Poly1305",
    secure_generation=True,
    hardware_random=True
)

# Protect private keys
keypair = crypto.generate_asymmetric_keypair(
    "RSA",
    key_size=2048,
    password_protect_private_key=True,
    hardware_backed=True
)

# Regular key rotation
crypto.schedule_key_rotation(key_id, rotation_interval_days=90)
```

### Security
```python
# Enable constant-time operations
crypto.configure(
    constant_time_operations=True,
    side_channel_protection=True,
    secure_memory=True
)

# Use strong randomness
random_bytes = crypto.generate_random_bytes(
    32,
    source="hardware",
    entropy_requirement="high"
)

# Verify all signatures
is_valid = crypto.verify_signature(
    data, signature, public_key,
    constant_time=True
)
```

## Related APIs

- **[Security](/api/security/index)** - High-level security operations
- **[Privacy Engine](/api/privacy/engine)** - Privacy protection features
- **[Access Control](/api/security/access-control)** - Permission management 
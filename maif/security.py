"""
Security and cryptographic functionality for MAIF.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.exceptions import InvalidSignature
from dataclasses import dataclass
import uuid

@dataclass
class ProvenanceEntry:
    """Represents a single provenance entry."""
    timestamp: float
    agent_id: str
    action: str
    block_hash: str
    signature: str = ""
    previous_hash: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "action": self.action,
            "block_hash": self.block_hash,
            "signature": self.signature,
            "previous_hash": self.previous_hash
        }

class MAIFSigner:
    """Handles digital signing and provenance for MAIF files."""
    
    def __init__(self, private_key_path: Optional[str] = None, agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.provenance_chain: List[ProvenanceEntry] = []
        
        if private_key_path:
            with open(private_key_path, 'rb') as f:
                self.private_key = load_pem_private_key(f.read(), password=None)
        else:
            # Generate new key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
    
    def get_public_key_pem(self) -> bytes:
        """Get the public key in PEM format."""
        public_key = self.private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def sign_data(self, data: bytes) -> str:
        """Sign data and return base64 encoded signature."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        import base64
        return base64.b64encode(signature).decode('ascii')
    
    def add_provenance_entry(self, action: str, block_hash: str) -> ProvenanceEntry:
        """Add a new provenance entry to the chain."""
        timestamp = time.time()
        
        # Create entry data for signing
        entry_data = {
            "timestamp": timestamp,
            "agent_id": self.agent_id,
            "action": action,
            "block_hash": block_hash
        }
        
        # Add previous hash if chain exists
        previous_hash = None
        if self.provenance_chain:
            last_entry = self.provenance_chain[-1]
            previous_hash = hashlib.sha256(
                json.dumps(last_entry.to_dict(), sort_keys=True).encode()
            ).hexdigest()
            entry_data["previous_hash"] = previous_hash
        
        # Sign the entry
        entry_bytes = json.dumps(entry_data, sort_keys=True).encode()
        signature = self.sign_data(entry_bytes)
        
        # Create provenance entry
        entry = ProvenanceEntry(
            timestamp=timestamp,
            agent_id=self.agent_id,
            action=action,
            block_hash=block_hash,
            signature=signature,
            previous_hash=previous_hash
        )
        
        self.provenance_chain.append(entry)
        return entry
    
    def get_provenance_chain(self) -> List[Dict]:
        """Get the complete provenance chain."""
        return [entry.to_dict() for entry in self.provenance_chain]
    
    def sign_maif_manifest(self, manifest: Dict) -> Dict:
        """Sign a MAIF manifest."""
        manifest_copy = manifest.copy()
        
        # Create signature data
        signature_data = {
            "manifest_hash": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "signer_id": self.agent_id,
            "timestamp": time.time(),
            "provenance_chain": [entry.to_dict() for entry in self.provenance_chain]
        }
        
        # Sign the signature data
        signature_bytes = json.dumps(signature_data, sort_keys=True).encode()
        signature = self.sign_data(signature_bytes)
        
        # Add signature and public key to manifest (as expected by tests)
        manifest_copy["signature"] = signature
        manifest_copy["public_key"] = self.get_public_key_pem().decode('ascii')
        
        return manifest_copy


class MAIFVerifier:
    """Handles verification of MAIF signatures and provenance."""
    
    def __init__(self):
        pass
    
    def verify_signature(self, data: bytes, signature: str, public_key_pem: str) -> bool:
        """Verify a signature against data using a public key."""
        try:
            import base64
            signature_bytes = base64.b64decode(signature.encode('ascii'))
            public_key = load_pem_public_key(public_key_pem.encode('ascii'))
            
            public_key.verify(
                signature_bytes,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except (InvalidSignature, Exception):
            return False
    
    def verify_maif_signature(self, signed_manifest: Dict) -> bool:
        """Verify a signed MAIF manifest."""
        try:
            if "signature" not in signed_manifest:
                return False
            
            signature_info = signed_manifest["signature"]
            signature_data = signature_info["data"]
            signature = signature_info["signature"]
            public_key_pem = signature_info["public_key"]
            
            # Load public key
            public_key = load_pem_public_key(public_key_pem.encode('ascii'))
            
            # Verify signature
            signature_bytes = json.dumps(signature_data, sort_keys=True).encode()
            signature_raw = __import__('base64').b64decode(signature)
            
            public_key.verify(
                signature_raw,
                signature_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except (InvalidSignature, Exception):
            return False
    
    def verify_maif_manifest(self, manifest: Dict) -> Tuple[bool, List[str]]:
        """Verify a MAIF manifest and return validation status and errors."""
        errors = []
        
        # Check required fields
        required_fields = ["maif_version", "blocks", "root_hash"]
        for field in required_fields:
            if field not in manifest:
                errors.append(f"Missing required field: {field}")
        
        # Verify signature if present
        if "signature" in manifest:
            if not self.verify_maif_signature(manifest):
                errors.append("Invalid signature")
        
        # Verify block hashes
        if "blocks" in manifest:
            for i, block in enumerate(manifest["blocks"]):
                if "hash" not in block:
                    errors.append(f"Block {i} missing hash")
        
        return len(errors) == 0, errors
    
    def verify_provenance_chain(self, provenance_data: Dict) -> Tuple[bool, List[str]]:
        """Verify the integrity of a provenance chain."""
        errors = []
        
        if "chain" not in provenance_data:
            return False, ["No provenance chain found"]
        
        chain = provenance_data["chain"]
        agent_id = provenance_data.get("agent_id")
        
        if not agent_id:
            errors.append("No agent_id found in provenance")
            return False, errors
        
        previous_hash = None
        for i, entry in enumerate(chain):
            # Verify chain linkage for entries after the first
            if i > 0:
                expected_previous = previous_hash
                actual_previous = entry.get("previous_hash")
                if actual_previous != expected_previous:
                    errors.append(f"Chain link broken at entry {i}")
            
            # Calculate hash for next iteration (use block_hash as the linking hash)
            previous_hash = entry.get("block_hash")
        
        return len(errors) == 0, errors
    


class AccessController:
    """Manages granular access control for MAIF blocks."""
    
    def __init__(self):
        self.permissions: Dict[str, Dict[str, List[str]]] = {}
    
    def set_block_permissions(self, block_hash: str, agent_id: str, permissions: List[str]):
        """Set permissions for a specific block and agent."""
        if block_hash not in self.permissions:
            self.permissions[block_hash] = {}
        self.permissions[block_hash][agent_id] = permissions
    
    def check_permission(self, block_hash: str, agent_id: str, action: str) -> bool:
        """Check if an agent has permission to perform an action on a block."""
        if block_hash not in self.permissions:
            return False
        
        agent_perms = self.permissions[block_hash].get(agent_id, [])
        return action in agent_perms or "admin" in agent_perms
    
    def get_permissions_manifest(self) -> Dict:
        """Get the permissions as a manifest for inclusion in MAIF."""
        return {
            "access_control": self.permissions,
            "version": "1.0"
        }
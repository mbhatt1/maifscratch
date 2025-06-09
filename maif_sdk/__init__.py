"""
MAIF SDK - Simplified Interface for Multimodal Artifact File Format

This SDK provides a simple, intuitive interface for creating and working with MAIF files
while maintaining full alignment with the academic paper specifications.
"""

from .client import MAIFClient
from .artifact import Artifact
from .types import ContentType, SecurityLevel, CompressionLevel

__version__ = "1.0.0"
__all__ = ["MAIFClient", "Artifact", "ContentType", "SecurityLevel", "CompressionLevel"]

# Simple factory function for easy SDK usage
def create_client(agent_id: str = "default_agent", **kwargs) -> MAIFClient:
    """
    Create a new MAIF client with simplified configuration.
    
    Args:
        agent_id: Unique identifier for this agent
        **kwargs: Additional configuration options
        
    Returns:
        Configured MAIF client ready for use
    """
    return MAIFClient(agent_id=agent_id, **kwargs)

def create_artifact(name: str = "untitled") -> Artifact:
    """
    Create a new MAIF artifact with default settings.
    
    Args:
        name: Human-readable name for the artifact
        
    Returns:
        New artifact ready for content addition
    """
    return Artifact(name=name)
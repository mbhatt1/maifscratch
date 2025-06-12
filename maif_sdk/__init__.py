"""
MAIF SDK - High-Performance Interface for Multimodal Artifact File Format

This SDK provides both high-performance native access and convenient POSIX semantics
for MAIF files, implementing the hybrid architecture recommended in the decision memo:

- Native SDK: High-performance "hot path" with direct memory-mapped I/O
- FUSE Filesystem: POSIX interface for exploration, debugging, and legacy tools  
- gRPC Daemon: Multi-writer service for containerized and distributed scenarios

Architecture Overview:
                       ┌──────────────────────────────────────┐
                        │       Orchestration Framework        │
                        │  (LangGraph / MemGPT / CrewAI …)     │
                        └──────────────┬───────────────────────┘
                                       │ SDK call ("hot path")
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
  ┌─────────▼────────┐      ┌──────────▼─────────┐     ┌─────────▼────────┐
  │  MAIF Native SDK │      │  MAIF gRPC daemon   │     │  FUSE Mount      │
  │ (direct mmap I/O)│      │  (multi‑writer)     │     │ (/mnt/maif …)    │
  └─────────┬────────┘      └──────────┬──────────┘     └─────────┬────────┘
            │                          │                          │
            ▼                          ▼                          ▼
        MAIF bundle            Same bundle via IPC          Same bundle
         on NVMe/SSD            (container / remote)          (read‑only)
"""

from .client import MAIFClient, quick_write, quick_read
from .artifact import Artifact, ContentItem
from .types import (
    ContentType, SecurityLevel, CompressionLevel, 
    ContentMetadata, SecurityOptions, ProcessingOptions
)

# Optional components (may not be available on all systems)
try:
    from .fuse_fs import MAIFFilesystem, mount_maif_filesystem, unmount_filesystem
    FUSE_AVAILABLE = True
except ImportError:
    FUSE_AVAILABLE = False
    MAIFFilesystem = None
    mount_maif_filesystem = None
    unmount_filesystem = None

try:
    from .grpc_daemon import MAIFServicer, serve_maif_grpc, MAIFGRPCClient
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    MAIFServicer = None
    serve_maif_grpc = None
    MAIFGRPCClient = None

__version__ = "1.0.0"
__all__ = [
    # Core SDK components
    "MAIFClient", "Artifact", "ContentItem",
    "ContentType", "SecurityLevel", "CompressionLevel",
    "ContentMetadata", "SecurityOptions", "ProcessingOptions",
    "quick_write", "quick_read",
    
    # Optional FUSE components
    "MAIFFilesystem", "mount_maif_filesystem", "unmount_filesystem",
    "FUSE_AVAILABLE",
    
    # Optional gRPC components  
    "MAIFServicer", "serve_maif_grpc", "MAIFGRPCClient",
    "GRPC_AVAILABLE",
    
    # Factory functions
    "create_client", "create_artifact"
]


# Factory functions for easy SDK usage
def create_client(agent_id: str = "default_agent", **kwargs) -> MAIFClient:
    """
    Create a new MAIF client with simplified configuration.
    
    This creates the high-performance native client that provides direct
    memory-mapped I/O and optimized block handling.
    
    Args:
        agent_id: Unique identifier for this agent
        **kwargs: Additional configuration options
        
    Returns:
        Configured MAIF client ready for use
    """
    return MAIFClient(agent_id=agent_id, **kwargs)


def create_artifact(name: str = "untitled", client: MAIFClient = None, **kwargs) -> Artifact:
    """
    Create a new MAIF artifact with default settings.
    
    Args:
        name: Human-readable name for the artifact
        client: Optional client to use (creates default if not provided)
        **kwargs: Additional artifact configuration
        
    Returns:
        New artifact ready for content addition
    """
    return Artifact(name=name, client=client, **kwargs)


def mount_filesystem(maif_root: str, mount_point: str, **kwargs):
    """
    Mount a MAIF filesystem for POSIX access.
    
    This provides the "secondary lens" for humans and legacy tools,
    exposing MAIF bundles as regular files and directories.
    
    Args:
        maif_root: Directory containing MAIF files
        mount_point: Where to mount the filesystem
        **kwargs: Additional FUSE options
        
    Raises:
        RuntimeError: If FUSE is not available
    """
    if not FUSE_AVAILABLE:
        raise RuntimeError(
            "FUSE filesystem not available. Install fusepy: pip install fusepy"
        )
    
    return mount_maif_filesystem(maif_root, mount_point, **kwargs)


def start_grpc_service(host: str = "localhost", port: int = 50051, **kwargs):
    """
    Start a MAIF gRPC service for multi-writer scenarios.
    
    This enables safe multi-writer concurrency and allows containerized
    tasks to share MAIF bundles without root-level FUSE access.
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        **kwargs: Additional service options
        
    Raises:
        RuntimeError: If gRPC is not available
    """
    if not GRPC_AVAILABLE:
        raise RuntimeError(
            "gRPC service not available. Install grpcio: pip install grpcio grpcio-tools"
        )
    
    import asyncio
    return asyncio.run(serve_maif_grpc(host=host, port=port, **kwargs))


# Convenience function for the recommended usage pattern
def get_recommended_interface(workload_pattern: str = "interactive") -> str:
    """
    Get the recommended interface based on workload pattern.
    
    Based on the decision matrix from the memo:
    
    Args:
        workload_pattern: One of:
            - "interactive": Interactive analytics/notebooks
            - "edge_low": Edge node with low write frequency  
            - "chat_medium": Chat agent with medium frequency writes
            - "high_tps": High-TPS multi-agent SaaS
            - "data_exchange": Cross-org data exchange
            
    Returns:
        Recommendation string with interface choice and rationale
    """
    recommendations = {
        "interactive": (
            "Use FUSE mount (default). "
            "Dev convenience outweighs latency; read-heavy workload."
        ),
        "edge_low": (
            "Use FUSE mount. "
            "Overhead dwarfed by network/sensor latency; keeps toolchain simple."
        ),
        "chat_medium": (
            "Use Native SDK with write buffering. "
            "Buffer turns in RAM, commit batches via SDK; FUSE only for reads."
        ),
        "high_tps": (
            "Use Native SDK or gRPC service. "
            "Direct library/service avoids context-switch tax and enables sharding."
        ),
        "data_exchange": (
            "Ship file directly; receiver chooses interface. "
            "Can mount with FUSE or use maif-cli export."
        )
    }
    
    return recommendations.get(workload_pattern, 
        "Unknown workload pattern. Use Native SDK for performance-critical, "
        "FUSE for convenience, gRPC for multi-writer scenarios."
    )


# Module-level configuration
def configure_sdk(enable_mmap: bool = True, buffer_size: int = 64*1024, 
                 cache_timeout: int = 30, **kwargs):
    """
    Configure global SDK settings.
    
    Args:
        enable_mmap: Enable memory mapping for read operations
        buffer_size: Write buffer size for combining operations
        cache_timeout: Cache timeout for FUSE filesystem
        **kwargs: Additional configuration options
    """
    # Store configuration for use by factory functions
    global _sdk_config
    _sdk_config = {
        'enable_mmap': enable_mmap,
        'buffer_size': buffer_size,
        'cache_timeout': cache_timeout,
        **kwargs
    }


# Default configuration
_sdk_config = {
    'enable_mmap': True,
    'buffer_size': 64 * 1024,
    'cache_timeout': 30
}
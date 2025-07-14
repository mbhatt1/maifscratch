"""
MAIF SDK - High-Performance Interface for Multimodal Artifact File Format

This SDK provides both high-performance native access and convenient POSIX semantics
for MAIF files, implementing the hybrid architecture recommended in the decision memo:

- Native SDK: High-performance "hot path" with direct memory-mapped I/O
- FUSE Filesystem: POSIX interface for exploration, debugging, and legacy tools  
- gRPC Daemon: Multi-writer service for containerized and distributed scenarios
- AWS Backend: Seamless cloud integration with S3, DynamoDB, and more

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
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────┐
                        │         AWS Backend Layer           │
                        │  (S3, DynamoDB, Bedrock, Lambda)    │
                        └──────────────────────────────────────┘
"""

# Import core SDK components
try:
    from .client import MAIFClient, quick_write, quick_read
    from .artifact import Artifact, ContentItem
    from .types import (
        ContentType, SecurityLevel, CompressionLevel,
        ContentMetadata, SecurityOptions, ProcessingOptions
    )
    CORE_SDK_AVAILABLE = True
except ImportError as e:
    # If core SDK components are missing, the SDK can't function
    raise ImportError(
        f"Failed to import core SDK components: {e}\n"
        "Please ensure all required dependencies are installed."
    )

# AWS Backend support
try:
    from .aws_backend import AWSConfig, create_aws_backends
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    AWSConfig = None
    create_aws_backends = None

# Optional components (may not be available on all systems)
# These are commented out as they don't exist yet
# When implementing FUSE or gRPC support, uncomment these sections

# try:
#     from .fuse_fs import MAIFFilesystem, mount_maif_filesystem, unmount_filesystem
#     FUSE_AVAILABLE = True
# except ImportError:
#     FUSE_AVAILABLE = False
#     MAIFFilesystem = None
#     mount_maif_filesystem = None
#     unmount_filesystem = None

# try:
#     from .grpc_daemon import MAIFServicer, serve_maif_grpc, MAIFGRPCClient
#     GRPC_AVAILABLE = True
# except ImportError:
#     GRPC_AVAILABLE = False
#     MAIFServicer = None
#     serve_maif_grpc = None
#     MAIFGRPCClient = None

# Set defaults for now
FUSE_AVAILABLE = False
MAIFFilesystem = None
mount_maif_filesystem = None
unmount_filesystem = None

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
    
    # AWS Backend components
    "AWSConfig", "create_aws_backends",
    "AWS_AVAILABLE",
    
    # Optional FUSE components
    "MAIFFilesystem", "mount_maif_filesystem", "unmount_filesystem",
    "FUSE_AVAILABLE",
    
    # Optional gRPC components  
    "MAIFServicer", "serve_maif_grpc", "MAIFGRPCClient",
    "GRPC_AVAILABLE",
    
    # Factory functions
    "create_client", "create_artifact", "create_aws_client"
]


# Factory functions for easy SDK usage
def create_client(agent_id: str = "default_agent", use_aws: bool = False, **kwargs) -> MAIFClient:
    """
    Create a new MAIF client with simplified configuration.
    
    This creates the high-performance native client that provides direct
    memory-mapped I/O and optimized block handling.
    
    Args:
        agent_id: Unique identifier for this agent
        use_aws: Enable AWS backend integration
        **kwargs: Additional configuration options
        
    Returns:
        Configured MAIF client ready for use
    """
    return MAIFClient(agent_id=agent_id, use_aws=use_aws, **kwargs)


def create_aws_client(
    agent_id: str = "aws_agent",
    artifact_name: str = "default",
    region: str = None,
    **kwargs
) -> MAIFClient:
    """
    Create a MAIF client with AWS backend integration enabled.
    
    This provides seamless integration with AWS services:
    - S3 for artifact storage
    - DynamoDB for metadata
    - Bedrock for AI models
    - KMS for encryption
    
    Args:
        agent_id: Unique identifier for this agent
        artifact_name: Name for the artifact
        region: AWS region (uses default if not specified)
        **kwargs: Additional configuration options
        
    Returns:
        MAIF client configured for AWS
        
    Raises:
        RuntimeError: If AWS backend is not available
    """
    if not AWS_AVAILABLE:
        raise RuntimeError(
            "AWS backend not available. Install boto3: pip install boto3"
        )
    
    return MAIFClient(
        agent_id=agent_id,
        use_aws=True,
        aws_config=AWSConfig(region=region) if region else None,
        **kwargs
    )


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
            - "cloud_native": AWS-integrated deployment
            
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
        ),
        "cloud_native": (
            "Use AWS backend with Native SDK. "
            "S3 for storage, DynamoDB for metadata, auto-scaling with Lambda/ECS."
        )
    }
    
    return recommendations.get(workload_pattern, 
        "Unknown workload pattern. Use Native SDK for performance-critical, "
        "FUSE for convenience, gRPC for multi-writer scenarios, "
        "AWS backend for cloud deployments."
    )


# Module-level configuration
def configure_sdk(enable_mmap: bool = True, buffer_size: int = 64*1024, 
                 cache_timeout: int = 30, enable_aws: bool = False, **kwargs):
    """
    Configure global SDK settings.
    
    Args:
        enable_mmap: Enable memory mapping for read operations
        buffer_size: Write buffer size for combining operations
        cache_timeout: Cache timeout for FUSE filesystem
        enable_aws: Enable AWS backend by default
        **kwargs: Additional configuration options
    """
    # Store configuration for use by factory functions
    global _sdk_config
    _sdk_config = {
        'enable_mmap': enable_mmap,
        'buffer_size': buffer_size,
        'cache_timeout': cache_timeout,
        'enable_aws': enable_aws,
        **kwargs
    }


# Default configuration
_sdk_config = {
    'enable_mmap': True,
    'buffer_size': 64 * 1024,
    'cache_timeout': 30,
    'enable_aws': False
}
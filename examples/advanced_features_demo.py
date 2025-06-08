"""
Advanced MAIF Features Demonstration

This example showcases the comprehensive capabilities of MAIF 2.0 including:
- Advanced compression algorithms
- Binary format handling
- Validation and repair
- Streaming operations
- Format conversion
- Performance profiling
"""

import os
import json
import time
from pathlib import Path

# Import MAIF modules
from maif.core import MAIFEncoder, MAIFParser
from maif.security import MAIFSigner, MAIFVerifier
from maif.compression import MAIFCompressor, CompressionAlgorithm
from maif.binary_format import BinaryFormatWriter, BinaryFormatReader
from maif.validation import MAIFValidator, MAIFRepairTool
from maif.streaming import MAIFStreamReader, MAIFStreamWriter, StreamingConfig
from maif.integration import MAIFConverter
from maif.metadata import MAIFMetadataManager
from maif.forensics import ForensicAnalyzer

def demo_compression_algorithms():
    """Demonstrate different compression algorithms."""
    print("=== Compression Algorithms Demo ===")
    
    # Sample data
    sample_text = "This is a sample text that will be compressed using different algorithms. " * 100
    sample_data = sample_text.encode('utf-8')
    
    compressor = MAIFCompressor()
    
    algorithms = [
        CompressionAlgorithm.ZLIB,
        CompressionAlgorithm.LZMA,
        CompressionAlgorithm.BROTLI
    ]
    
    print(f"Original size: {len(sample_data):,} bytes")
    
    for algorithm in algorithms:
        start_time = time.time()
        compressed = compressor.compress(sample_data, algorithm)
        compression_time = time.time() - start_time
        
        start_time = time.time()
        decompressed = compressor.decompress(compressed, algorithm)
        decompression_time = time.time() - start_time
        
        ratio = len(compressed) / len(sample_data)
        
        print(f"\n{algorithm.value}:")
        print(f"  Compressed size: {len(compressed):,} bytes ({ratio:.2%})")
        print(f"  Compression time: {compression_time:.4f}s")
        print(f"  Decompression time: {decompression_time:.4f}s")
        print(f"  Data integrity: {'✓' if decompressed == sample_data else '✗'}")

def demo_binary_format():
    """Demonstrate binary format operations."""
    print("\n=== Binary Format Demo ===")
    
    output_file = "demo_binary.maif"
    
    # Write binary format
    with BinaryFormatWriter(output_file) as writer:
        # Add various block types
        writer.write_block("text_block", b"Hello, MAIF!", "text_data")
        writer.write_block("binary_block", b"\x00\x01\x02\x03\x04", "binary_data")
        writer.write_block("json_block", json.dumps({"key": "value"}).encode(), "json_data")
    
    print(f"✓ Binary MAIF file created: {output_file}")
    
    # Read binary format
    with BinaryFormatReader(output_file) as reader:
        header = reader.read_header()
        print(f"✓ Header read: {header.block_count} blocks")
        
        for block_id in reader.list_blocks():
            block_data = reader.read_block(block_id)
            print(f"  Block {block_id}: {len(block_data)} bytes")
    
    # Cleanup
    if os.path.exists(output_file):
        os.remove(output_file)

def demo_validation_and_repair():
    """Demonstrate validation and repair capabilities."""
    print("\n=== Validation and Repair Demo ===")
    
    # Create a MAIF file
    encoder = MAIFEncoder(agent_id="demo_agent")
    encoder.add_text_block("Sample content for validation")
    encoder.add_metadata_block({"demo": True, "version": "2.0"})
    
    maif_file = "demo_validation.maif"
    manifest_file = f"{maif_file}.manifest.json"
    encoder.build_maif(maif_file, manifest_file)
    
    print(f"✓ Created MAIF file: {maif_file}")
    
    # Validate the file
    validator = MAIFValidator()
    report = validator.validate_file(maif_file, manifest_file)
    
    print(f"✓ Validation complete:")
    print(f"  Valid: {report.is_valid}")
    print(f"  Issues found: {len(report.issues)}")
    print(f"  Statistics: {report.statistics}")
    
    if not report.is_valid:
        # Attempt repair
        repair_tool = MAIFRepairTool()
        if repair_tool.repair_file(maif_file, manifest_file):
            print("✓ File repaired successfully")
        else:
            print("✗ Could not repair all issues")
    
    # Cleanup
    for file_path in [maif_file, manifest_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

def demo_streaming_operations():
    """Demonstrate streaming operations."""
    print("\n=== Streaming Operations Demo ===")
    
    # Create a large MAIF file for streaming
    maif_file = "demo_streaming.maif"
    config = StreamingConfig(chunk_size=4096, enable_compression=True)
    
    # Write using streaming
    with MAIFStreamWriter(maif_file, config) as writer:
        for i in range(10):
            content = f"Block {i}: " + "x" * 1000  # 1KB per block
            writer.write_block(f"block_{i}", content.encode(), "text_data")
    
    print(f"✓ Created streaming MAIF file: {maif_file}")
    
    # Read using streaming
    with MAIFStreamReader(maif_file, config) as reader:
        block_count = 0
        total_size = 0
        
        for block_id, data in reader.stream_blocks():
            block_count += 1
            total_size += len(data)
        
        print(f"✓ Streamed {block_count} blocks, {total_size:,} bytes total")
        
        # Demonstrate parallel reading
        parallel_count = 0
        for block_id, data in reader.stream_blocks_parallel():
            parallel_count += 1
        
        print(f"✓ Parallel streaming: {parallel_count} blocks")
    
    # Cleanup
    if os.path.exists(maif_file):
        os.remove(maif_file)

def demo_format_conversion():
    """Demonstrate format conversion capabilities."""
    print("\n=== Format Conversion Demo ===")
    
    # Create sample JSON file
    json_file = "demo_data.json"
    sample_data = {
        "title": "Sample Document",
        "content": "This is sample content for conversion demo",
        "metadata": {
            "author": "Demo User",
            "created": "2024-01-01",
            "tags": ["demo", "conversion", "maif"]
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Created sample JSON file: {json_file}")
    
    # Convert to MAIF
    converter = MAIFConverter()
    maif_file = "converted.maif"
    
    result = converter.convert_to_maif(json_file, maif_file, "json")
    
    if result.success:
        print(f"✓ Converted to MAIF: {maif_file}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Metadata: {result.metadata}")
        
        # Convert back to JSON
        json_output = "converted_back.json"
        export_result = converter.export_from_maif(maif_file, json_output, "json")
        
        if export_result.success:
            print(f"✓ Exported back to JSON: {json_output}")
        else:
            print(f"✗ Export failed: {export_result.errors}")
        
        # Cleanup export file
        if os.path.exists(json_output):
            os.remove(json_output)
    else:
        print(f"✗ Conversion failed: {result.errors}")
    
    # Cleanup
    for file_path in [json_file, maif_file, f"{maif_file}.manifest.json"]:
        if os.path.exists(file_path):
            os.remove(file_path)

def demo_metadata_management():
    """Demonstrate advanced metadata management."""
    print("\n=== Metadata Management Demo ===")
    
    # Create metadata manager
    metadata_mgr = MAIFMetadataManager()
    
    # Create header
    header = metadata_mgr.create_header("demo_agent", creator_agent="advanced_demo")
    print(f"✓ Created header: {header.file_id}")
    
    # Add block metadata
    block1 = metadata_mgr.add_block_metadata(
        "block_1", "text_data", "text/plain", 1024, 0, "hash123",
        tags=["demo", "text"], custom_metadata={"importance": "high"}
    )
    
    block2 = metadata_mgr.add_block_metadata(
        "block_2", "binary_data", "application/octet-stream", 2048, 1024, "hash456",
        dependencies=["block_1"], tags=["demo", "binary"]
    )
    
    print(f"✓ Added {len(metadata_mgr.blocks)} blocks")
    
    # Add provenance
    provenance = metadata_mgr.add_provenance_record(
        "create_block", "demo_agent", ["block_1", "block_2"],
        {"operation": "demo", "timestamp": time.time()}
    )
    
    print(f"✓ Added provenance record: {provenance.operation_id}")
    
    # Validate dependencies
    errors = metadata_mgr.validate_dependencies()
    print(f"✓ Dependency validation: {len(errors)} errors")
    
    # Get statistics
    stats = metadata_mgr.get_statistics()
    print(f"✓ Statistics: {stats['blocks']['total']} blocks, {stats['provenance']['total_records']} provenance records")

def demo_forensic_analysis():
    """Demonstrate forensic analysis capabilities."""
    print("\n=== Forensic Analysis Demo ===")
    
    # Create a MAIF file with multiple versions
    encoder = MAIFEncoder(agent_id="forensic_demo")
    
    # Add initial content
    encoder.add_text_block("Initial content")
    encoder.add_metadata_block({"version": 1, "author": "user1"})
    
    # Simulate modifications
    encoder.add_text_block("Modified content")
    encoder.add_metadata_block({"version": 2, "author": "user2", "modification": "content_update"})
    
    maif_file = "forensic_demo.maif"
    manifest_file = f"{maif_file}.manifest.json"
    encoder.build_maif(maif_file, manifest_file)
    
    print(f"✓ Created MAIF file for forensic analysis: {maif_file}")
    
    # Perform forensic analysis
    parser = MAIFParser(maif_file, manifest_file)
    verifier = MAIFVerifier()
    analyzer = ForensicAnalyzer()
    
    report = analyzer.analyze_maif(parser, verifier)
    
    print(f"✓ Forensic analysis complete:")
    print(f"  Integrity status: {report.integrity_status}")
    print(f"  Events analyzed: {report.events_analyzed}")
    print(f"  Evidence items: {len(report.evidence)}")
    print(f"  Timeline events: {len(report.timeline)}")
    
    if report.evidence:
        print("  Evidence found:")
        for evidence in report.evidence[:3]:  # Show first 3
            print(f"    - {evidence.severity.upper()}: {evidence.description}")
    
    # Cleanup
    for file_path in [maif_file, manifest_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

def main():
    """Run all demonstrations."""
    print("MAIF 2.0 Advanced Features Demonstration")
    print("=" * 50)
    
    try:
        demo_compression_algorithms()
        demo_binary_format()
        demo_validation_and_repair()
        demo_streaming_operations()
        demo_format_conversion()
        demo_metadata_management()
        demo_forensic_analysis()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
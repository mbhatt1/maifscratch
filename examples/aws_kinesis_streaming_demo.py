"""
AWS Kinesis Streaming Integration Demo
===================================

Demonstrates how to use the AWS Kinesis streaming integration in MAIF.
"""

import json
import time
import logging
import sys
import os
import uuid
from pathlib import Path

from maif.aws_kinesis_streaming import (
    KinesisStreamingConfig, KinesisClient, 
    KinesisStreamReader, KinesisStreamWriter, 
    KinesisMAIFProcessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate AWS Kinesis streaming integration."""
    
    # Kinesis configuration
    region_name = "us-east-1"
    stream_name = "maif-streaming-demo"
    
    # Create streaming configuration
    config = KinesisStreamingConfig(
        region_name=region_name,
        max_retries=3,
        base_delay=0.5,
        max_delay=5.0,
        batch_size=100,  # Smaller batch for demo
        enable_metrics=True
    )
    
    # Initialize Kinesis client
    try:
        kinesis_client = KinesisClient(
            region_name=region_name,
            max_retries=3,
            base_delay=0.5,
            max_delay=5.0
        )
        logger.info("Kinesis client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Kinesis client: {e}")
        return
    
    # Create stream if it doesn't exist
    try:
        # Check if stream exists
        streams = kinesis_client.list_streams()
        
        if stream_name not in streams:
            logger.info(f"Creating Kinesis stream: {stream_name}")
            kinesis_client.create_stream(stream_name, shard_count=1)
            
            # Wait for stream to become active
            logger.info("Waiting for stream to become active...")
            time.sleep(10)  # Simple wait for demo
        else:
            logger.info(f"Stream {stream_name} already exists")
            
    except Exception as e:
        logger.error(f"Error creating stream: {e}")
        return
    
    # Demonstrate writing to Kinesis
    try:
        logger.info("Demonstrating writing to Kinesis...")
        
        with KinesisStreamWriter(stream_name, config) as writer:
            # Write some sample records
            for i in range(10):
                data = {
                    "id": i,
                    "message": f"Sample message {i}",
                    "timestamp": time.time()
                }
                
                # Convert to bytes
                data_bytes = json.dumps(data).encode('utf-8')
                
                # Write to Kinesis
                writer.write_record(data_bytes)
                logger.info(f"Wrote record {i} to Kinesis")
            
            # Get writer stats
            writer_stats = writer.get_stats()
            logger.info(f"Writer stats: {json.dumps(writer_stats, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error writing to Kinesis: {e}")
    
    # Demonstrate reading from Kinesis
    try:
        logger.info("Demonstrating reading from Kinesis...")
        
        with KinesisStreamReader(stream_name, config) as reader:
            # Read records (with a limit to avoid infinite loop in demo)
            record_count = 0
            max_records = 20
            
            logger.info("Reading records from Kinesis...")
            for metadata, data in reader.stream_records_continuous(max_iterations=5):
                record_count += 1
                
                # Parse JSON data
                try:
                    json_data = json.loads(data.decode('utf-8'))
                    logger.info(f"Read record: {json_data}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.info(f"Read raw record: {data[:50]}...")
                
                if record_count >= max_records:
                    break
            
            # Get reader stats
            reader_stats = reader.get_stats()
            logger.info(f"Reader stats: {json.dumps(reader_stats, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error reading from Kinesis: {e}")
    
    # Demonstrate MAIF file to Kinesis
    try:
        logger.info("Demonstrating MAIF file to Kinesis...")
        
        # Create a sample MAIF file
        from maif.streaming import MAIFStreamWriter
        
        maif_path = "sample.maif"
        manifest_path = "sample_manifest.json"
        
        with MAIFStreamWriter(maif_path) as maif_writer:
            # Write some sample blocks
            for i in range(5):
                text_data = f"Sample text block {i}"
                maif_writer.write_text_block(text_data, {"block_id": i})
            
            # Finalize MAIF file
            maif_writer.finalize(manifest_path)
        
        logger.info(f"Created sample MAIF file: {maif_path}")
        
        # Stream MAIF file to Kinesis
        processor = KinesisMAIFProcessor(config)
        result = processor.maif_to_kinesis(maif_path, stream_name)
        
        logger.info(f"MAIF to Kinesis result: {json.dumps(result, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error streaming MAIF to Kinesis: {e}")
    
    # Demonstrate Kinesis to MAIF file
    try:
        logger.info("Demonstrating Kinesis to MAIF file...")
        
        output_path = "kinesis_output.maif"
        
        # Stream Kinesis to MAIF file
        processor = KinesisMAIFProcessor(config)
        result = processor.kinesis_to_maif(stream_name, output_path, max_iterations=5)
        
        logger.info(f"Kinesis to MAIF result: {json.dumps(result, indent=2)}")
        logger.info(f"Created MAIF file from Kinesis: {output_path}")
    
    except Exception as e:
        logger.error(f"Error streaming Kinesis to MAIF: {e}")
    
    # Demonstrate custom processing
    try:
        logger.info("Demonstrating custom processing of Kinesis stream...")
        
        # Define custom processor function
        def process_record(metadata, data):
            try:
                # Parse JSON data
                json_data = json.loads(data.decode('utf-8'))
                
                # Process data (simple transformation for demo)
                processed_data = {
                    "original": json_data,
                    "processed": True,
                    "processing_time": time.time()
                }
                
                logger.info(f"Processed record: {json_data.get('id', 'unknown')}")
                return processed_data
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"Failed to process record: {e}")
                return {"error": "Failed to process record", "raw_data": data[:50]}
        
        # Process stream
        processor = KinesisMAIFProcessor(config)
        result = processor.process_kinesis_stream(stream_name, process_record, max_iterations=5)
        
        logger.info(f"Custom processing result: {json.dumps(result, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error in custom processing: {e}")
    
    # Clean up (optional - commented out to preserve stream)
    # try:
    #     logger.info(f"Deleting Kinesis stream: {stream_name}")
    #     kinesis_client.delete_stream(stream_name)
    # except Exception as e:
    #     logger.error(f"Error deleting stream: {e}")
    
    # Clean up sample files
    try:
        for file_path in [maif_path, manifest_path, output_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")


if __name__ == "__main__":
    main()
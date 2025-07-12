"""
AWS S3 Block Storage Integration Demo
===================================

Demonstrates how to use the AWS S3 block storage integration in MAIF.
"""

import json
import time
import logging
import sys
import os
import uuid
from pathlib import Path

from maif.aws_s3_block_storage import S3BlockStorage, S3StreamingBlockParser
from maif.block_types import BlockFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate AWS S3 block storage integration."""
    
    # S3 configuration
    bucket_name = "maif-block-storage-demo"  # Replace with your bucket name
    region_name = "us-east-1"
    prefix = "blocks/"
    
    # Local cache directory
    cache_dir = "./block_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize S3 block storage
    try:
        block_storage = S3BlockStorage(
            bucket_name=bucket_name,
            prefix=prefix,
            region_name=region_name,
            verify_signatures=True,
            local_cache_dir=cache_dir,
            max_cache_size_mb=100
        )
        
        logger.info(f"S3 block storage initialized with bucket: {bucket_name}")
        
        # Create bucket if it doesn't exist
        logger.info("Creating S3 bucket if needed...")
        block_storage.create_bucket_if_not_exists()
        
        # Demonstrate adding a text block
        logger.info("Adding a text block...")
        text_content = "This is a sample text block for the AWS S3 block storage demo."
        text_block = BlockFactory.create_text_block(
            text=text_content,
            language="en",
            encoding="utf-8"
        )
        
        text_block_uuid = block_storage.add_block(
            block_type=text_block['type'],
            data=text_block['data'],
            metadata=text_block['metadata']
        )
        
        logger.info(f"Added text block with UUID: {text_block_uuid}")
        
        # Demonstrate adding an embedding block
        logger.info("Adding an embedding block...")
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]
        embedding_block = BlockFactory.create_embedding_block(
            embeddings=embeddings,
            model_name="demo-model",
            dimensions=5
        )
        
        embedding_block_uuid = block_storage.add_block(
            block_type=embedding_block['type'],
            data=embedding_block['data'],
            metadata=embedding_block['metadata']
        )
        
        logger.info(f"Added embedding block with UUID: {embedding_block_uuid}")
        
        # Demonstrate adding a knowledge graph block
        logger.info("Adding a knowledge graph block...")
        triples = [
            {"subject": "MAIF", "predicate": "has_feature", "object": "block_storage"},
            {"subject": "block_storage", "predicate": "uses", "object": "AWS_S3"},
            {"subject": "AWS_S3", "predicate": "provides", "object": "durability"}
        ]
        
        kg_block = BlockFactory.create_knowledge_graph_block(
            triples=triples,
            format_type="json-ld"
        )
        
        kg_block_uuid = block_storage.add_block(
            block_type=kg_block['type'],
            data=kg_block['data'],
            metadata=kg_block['metadata']
        )
        
        logger.info(f"Added knowledge graph block with UUID: {kg_block_uuid}")
        
        # List all blocks
        logger.info("Listing all blocks...")
        blocks = block_storage.list_blocks()
        
        for i, header in enumerate(blocks):
            logger.info(f"Block {i+1}: {header.uuid} - Type: {header.block_type}, Size: {header.size} bytes")
        
        # Retrieve a block
        logger.info(f"Retrieving text block {text_block_uuid}...")
        text_result = block_storage.get_block(text_block_uuid)
        
        if text_result:
            header, data, metadata = text_result
            logger.info(f"Retrieved text block: {data.decode('utf-8')}")
            logger.info(f"Block metadata: {json.dumps(metadata, indent=2)}")
        else:
            logger.warning("Failed to retrieve text block")
        
        # Retrieve embedding block
        logger.info(f"Retrieving embedding block {embedding_block_uuid}...")
        embedding_result = block_storage.get_block(embedding_block_uuid)
        
        if embedding_result:
            header, data, metadata = embedding_result
            logger.info(f"Retrieved embedding block with {metadata.get('count', 0)} embeddings")
            logger.info(f"Block metadata: {json.dumps(metadata, indent=2)}")
        else:
            logger.warning("Failed to retrieve embedding block")
        
        # Validate block chain integrity
        logger.info("Validating block chain integrity...")
        is_valid = block_storage.validate_integrity()
        logger.info(f"Block chain integrity: {'Valid' if is_valid else 'Invalid'}")
        
        # Validate signatures
        logger.info("Validating block signatures...")
        signature_results = block_storage.validate_all_signatures()
        logger.info(f"Signature validation results: {json.dumps(signature_results, indent=2)}")
        
        # Demonstrate streaming parser
        logger.info("Demonstrating streaming block parser...")
        streaming_parser = S3StreamingBlockParser(
            s3_client=block_storage.s3_client,
            bucket_name=bucket_name,
            prefix=f"{prefix}data/",
            chunk_size=8192  # 8KB chunks
        )
        
        parsed_headers = streaming_parser.parse_prefix(max_objects=10)
        logger.info(f"Parsed {len(parsed_headers)} blocks using streaming parser")
        
        parser_stats = streaming_parser.get_stats()
        logger.info(f"Parser stats: {json.dumps(parser_stats, indent=2)}")
        
        # Delete a block (optional - commented out to preserve blocks)
        # logger.info(f"Deleting knowledge graph block {kg_block_uuid}...")
        # deleted = block_storage.delete_block(kg_block_uuid)
        # logger.info(f"Block deletion: {'Successful' if deleted else 'Failed'}")
        
    except Exception as e:
        logger.error(f"Error in AWS S3 block storage demo: {e}", exc_info=True)


if __name__ == "__main__":
    main()
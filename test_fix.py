#!/usr/bin/env python3
"""
Test script to verify the fix for the HotBufferLayer initialization error.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_fix")

def test_enhanced_self_governing_maif():
    """Test the EnhancedSelfGoverningMAIF initialization."""
    try:
        from maif.lifecycle_management_enhanced import EnhancedSelfGoverningMAIF
        
        # Create a temporary MAIF path
        maif_path = Path("./test_maif.bin")
        
        logger.info("Importing EnhancedSelfGoverningMAIF succeeded")
        
        # Initialize the class
        gov_maif = EnhancedSelfGoverningMAIF(str(maif_path))
        
        logger.info("Successfully created EnhancedSelfGoverningMAIF instance")
        return True
    except Exception as e:
        logger.error(f"Error initializing EnhancedSelfGoverningMAIF: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error args: {e.args}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_enhanced_self_governing_maif()
    sys.exit(0 if success else 1)
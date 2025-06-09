"""
Integration utilities for MAIF format conversion and processing.
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ConversionResult:
    """Result of a conversion operation."""
    success: bool
    message: str = ""
    error: str = ""
    output_path: str = ""
    manifest_path: str = ""
    
    def __getitem__(self, key):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

class MAIFConverter:
    """MAIF format converter."""
    
    def __init__(self):
        pass
    
    def convert_to_maif(self, input_path: str, output_path: str, manifest_path: str, input_format: str) -> Dict[str, Any]:
        """Convert various formats to MAIF."""
        try:
            from .core import MAIFEncoder
            encoder = MAIFEncoder(agent_id="format_converter")
            
            if input_format == "json":
                with open(input_path, 'r') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                encoder.add_text_block(content, metadata={"source_format": "json"})
            elif input_format == "xml":
                # Use the processor's XML conversion method
                processor = EnhancedMAIFProcessor()
                return processor.convert_xml_to_maif(input_path, output_path, manifest_path)
            else:
                with open(input_path, 'r') as f:
                    content = f.read()
                encoder.add_text_block(content, metadata={"source_format": input_format})
            
            encoder.build_maif(output_path, manifest_path)
            return {"success": True, "message": f"{input_format} converted to MAIF successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}

class EnhancedMAIFProcessor:
    """Enhanced MAIF processor for format conversion and integration."""
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv', 'txt', 'zip', 'tar']
    
    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        mime_mapping = {
            "application/json": "json",
            "text/xml": "xml",
            "application/xml": "xml",
            "text/csv": "csv",
            "text/plain": "txt",
            "application/zip": "zip",
            "application/x-tar": "tar"
        }
        return mime_mapping.get(mime_type, "unknown")
    
    def convert_to_maif(self, input_path: str, output_path: str, manifest_path: str, input_format: str) -> ConversionResult:
        """Convert various formats to MAIF."""
        try:
            from .core import MAIFEncoder
            encoder = MAIFEncoder(agent_id="format_converter")
            
            if input_format == "json":
                with open(input_path, 'r') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2)
                encoder.add_text_block(content, metadata={"source_format": "json"})
            elif input_format == "xml":
                return self.convert_xml_to_maif(input_path, output_path, manifest_path)
            else:
                with open(input_path, 'r') as f:
                    content = f.read()
                encoder.add_text_block(content, metadata={"source_format": input_format})
            
            encoder.build_maif(output_path, manifest_path)
            return ConversionResult(
                success=True,
                message=f"{input_format} converted to MAIF successfully",
                output_path=output_path,
                manifest_path=manifest_path
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))
    
    def convert_xml_to_maif(self, xml_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert XML file to MAIF format."""
        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Convert XML to text representation
            xml_content = ET.tostring(root, encoding='unicode')
            
            # Create MAIF using core encoder
            from .core import MAIFEncoder
            encoder = MAIFEncoder(agent_id="xml_converter")
            encoder.add_text_block(xml_content, metadata={"source_format": "xml"})
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                message="XML converted to MAIF successfully",
                output_path=output_path,
                manifest_path=manifest_path
            )
        except Exception as e:
            return ConversionResult(success=False, error=str(e))


class MAIFPluginManager:
    """Plugin manager for MAIF extensions."""
    
    def __init__(self):
        self.plugins = []
        self.hooks = {
            "pre_conversion": [],
            "post_conversion": [],
            "pre_validation": [],
            "post_validation": []
        }
    
    def register_plugin(self, plugin):
        """Register a plugin."""
        self.plugins.append(plugin)
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all plugins for a specific hook."""
        if hook_name in self.hooks:
            for hook_func in self.hooks[hook_name]:
                hook_func(*args, **kwargs)
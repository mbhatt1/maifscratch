"""
Enhanced integration functionality for MAIF files.
"""

import json
import xml.etree.ElementTree as ET
import csv
import os
import zipfile
import tarfile
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from .core import MAIFEncoder, MAIFDecoder


@dataclass
class ConversionResult:
    """Result of a MAIF conversion operation."""
    success: bool = False
    output_path: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class EnhancedMAIFProcessor:
    """Enhanced processor for MAIF file operations and conversions."""
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv', 'txt', 'yaml']
        self.mime_type_mapping = {
            'application/json': 'json',
            'text/json': 'json',
            'application/xml': 'xml',
            'text/xml': 'xml',
            'text/csv': 'csv',
            'application/csv': 'csv',
            'text/plain': 'txt',
            'application/x-yaml': 'yaml',
            'text/yaml': 'yaml'
        }
    
    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        return self.mime_type_mapping.get(mime_type, 'unknown')
    
    def convert_json_to_maif(self, json_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert JSON file to MAIF format."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            encoder = MAIFEncoder(agent_id="json_converter")
            
            # Convert JSON data to MAIF blocks
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        encoder.add_text_block(value, metadata={"source": "json", "key": key})
                    else:
                        encoder.add_text_block(json.dumps(value), metadata={"source": "json", "key": key, "type": "json_object"})
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        encoder.add_text_block(item, metadata={"source": "json", "index": i})
                    else:
                        encoder.add_text_block(json.dumps(item), metadata={"source": "json", "index": i, "type": "json_object"})
            else:
                encoder.add_text_block(json.dumps(data), metadata={"source": "json", "type": "json_root"})
            
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"format": "json", "blocks_converted": len(encoder.blocks)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"JSON conversion failed: {str(e)}"]
            )
    
    def convert_xml_to_maif(self, xml_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert XML file to MAIF format."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            encoder = MAIFEncoder(agent_id="xml_converter")
            
            def process_element(element, path=""):
                current_path = f"{path}/{element.tag}" if path else element.tag
                
                # Add element text if present
                if element.text and element.text.strip():
                    encoder.add_text_block(
                        element.text.strip(),
                        metadata={"source": "xml", "path": current_path, "tag": element.tag}
                    )
                
                # Process attributes
                for attr_name, attr_value in element.attrib.items():
                    encoder.add_text_block(
                        attr_value,
                        metadata={"source": "xml", "path": current_path, "attribute": attr_name}
                    )
                
                # Process child elements
                for child in element:
                    process_element(child, current_path)
            
            process_element(root)
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"format": "xml", "blocks_converted": len(encoder.blocks)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"XML conversion failed: {str(e)}"]
            )
    
    def convert_csv_to_maif(self, csv_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert CSV file to MAIF format."""
        try:
            encoder = MAIFEncoder(agent_id="csv_converter")
            
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                headers = reader.fieldnames
                
                # Add headers as metadata block
                encoder.add_text_block(
                    json.dumps(headers),
                    metadata={"source": "csv", "type": "headers"}
                )
                
                # Add each row
                for row_num, row in enumerate(reader):
                    for column, value in row.items():
                        if value:  # Only add non-empty values
                            encoder.add_text_block(
                                value,
                                metadata={"source": "csv", "row": row_num, "column": column}
                            )
            
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"format": "csv", "blocks_converted": len(encoder.blocks)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"CSV conversion failed: {str(e)}"]
            )
    
    def convert_text_to_maif(self, text_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert text file to MAIF format."""
        try:
            with open(text_path, 'r') as f:
                content = f.read()
            
            encoder = MAIFEncoder(agent_id="text_converter")
            
            # Split into paragraphs or lines
            paragraphs = content.split('\n\n')
            if len(paragraphs) == 1:
                # No paragraph breaks, split by lines
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        encoder.add_text_block(
                            line.strip(),
                            metadata={"source": "text", "line": i}
                        )
            else:
                # Use paragraphs
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        encoder.add_text_block(
                            paragraph.strip(),
                            metadata={"source": "text", "paragraph": i}
                        )
            
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"format": "text", "blocks_converted": len(encoder.blocks)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"Text conversion failed: {str(e)}"]
            )
    
    def extract_and_convert_archive(self, archive_path: str, output_dir: str) -> ConversionResult:
        """Extract archive and convert contents to MAIF."""
        try:
            extracted_files = []
            
            # Determine archive type and extract
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                    extracted_files = zip_ref.namelist()
            elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(output_dir)
                    extracted_files = tar_ref.getnames()
            else:
                return ConversionResult(
                    success=False,
                    warnings=["Unsupported archive format"]
                )
            
            # Convert extracted files
            converted_files = []
            warnings = []
            
            for file_path in extracted_files:
                full_path = os.path.join(output_dir, file_path)
                if os.path.isfile(full_path):
                    try:
                        # Determine format and convert
                        ext = os.path.splitext(file_path)[1].lower()
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        
                        maif_path = os.path.join(output_dir, f"{base_name}.maif")
                        manifest_path = os.path.join(output_dir, f"{base_name}_manifest.json")
                        
                        if ext == '.json':
                            result = self.convert_json_to_maif(full_path, maif_path, manifest_path)
                        elif ext == '.xml':
                            result = self.convert_xml_to_maif(full_path, maif_path, manifest_path)
                        elif ext == '.csv':
                            result = self.convert_csv_to_maif(full_path, maif_path, manifest_path)
                        elif ext == '.txt':
                            result = self.convert_text_to_maif(full_path, maif_path, manifest_path)
                        else:
                            continue  # Skip unsupported files
                        
                        if result.success:
                            converted_files.append(maif_path)
                        else:
                            warnings.extend(result.warnings)
                            
                    except Exception as e:
                        warnings.append(f"Failed to convert {file_path}: {str(e)}")
            
            return ConversionResult(
                success=len(converted_files) > 0,
                output_path=output_dir,
                warnings=warnings,
                metadata={"converted_files": converted_files, "total_extracted": len(extracted_files)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"Archive extraction failed: {str(e)}"]
            )
    
    def batch_convert_directory(self, input_dir: str, output_dir: str) -> ConversionResult:
        """Batch convert all supported files in a directory."""
        try:
            converted_files = []
            warnings = []
            
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    ext = os.path.splitext(file)[1].lower()
                    base_name = os.path.splitext(file)[0]
                    
                    # Create relative path structure in output
                    rel_path = os.path.relpath(root, input_dir)
                    if rel_path == '.':
                        output_subdir = output_dir
                    else:
                        output_subdir = os.path.join(output_dir, rel_path)
                        os.makedirs(output_subdir, exist_ok=True)
                    
                    maif_path = os.path.join(output_subdir, f"{base_name}.maif")
                    manifest_path = os.path.join(output_subdir, f"{base_name}_manifest.json")
                    
                    try:
                        if ext == '.json':
                            result = self.convert_json_to_maif(file_path, maif_path, manifest_path)
                        elif ext == '.xml':
                            result = self.convert_xml_to_maif(file_path, maif_path, manifest_path)
                        elif ext == '.csv':
                            result = self.convert_csv_to_maif(file_path, maif_path, manifest_path)
                        elif ext == '.txt':
                            result = self.convert_text_to_maif(file_path, maif_path, manifest_path)
                        else:
                            continue  # Skip unsupported files
                        
                        if result.success:
                            converted_files.append(maif_path)
                        else:
                            warnings.extend(result.warnings)
                            
                    except Exception as e:
                        warnings.append(f"Failed to convert {file_path}: {str(e)}")
            
            return ConversionResult(
                success=len(converted_files) > 0,
                output_path=output_dir,
                warnings=warnings,
                metadata={"converted_files": converted_files, "total_converted": len(converted_files)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"Batch conversion failed: {str(e)}"]
            )


class MAIFConverter:
    """Legacy converter class for backward compatibility."""
    
    def __init__(self):
        self.processor = EnhancedMAIFProcessor()
        self.supported_formats = ['json', 'xml', 'csv', 'txt', 'yaml']  # Add missing attribute
    
    def convert_to_maif(self, input_path: str, output_path: str, manifest_path: str, format_type: str = None, input_format: str = None) -> ConversionResult:
        """Convert file to MAIF format."""
        # Use input_format if provided, otherwise use format_type
        actual_format = input_format or format_type
        
        if actual_format is None:
            # Auto-detect format from extension
            ext = os.path.splitext(input_path)[1].lower()
            format_map = {'.json': 'json', '.xml': 'xml', '.csv': 'csv', '.txt': 'txt'}
            actual_format = format_map.get(ext, 'txt')
        
        if actual_format == 'json':
            return self.processor.convert_json_to_maif(input_path, output_path, manifest_path)
        elif actual_format == 'xml':
            return self.processor.convert_xml_to_maif(input_path, output_path, manifest_path)
        elif actual_format == 'csv':
            return self.processor.convert_csv_to_maif(input_path, output_path, manifest_path)
        elif actual_format == 'txt':
            return self.processor.convert_text_to_maif(input_path, output_path, manifest_path)
        else:
            return ConversionResult(
                success=False,
                warnings=[f"Unsupported format: {actual_format}"]
            )
    
    def export_from_maif(self, maif_path: str, output_path: str, manifest_path: str, output_format: str) -> ConversionResult:
        """Export MAIF file to another format."""
        try:
            decoder = MAIFDecoder(maif_path, manifest_path)
            
            if output_format == 'json':
                # Export to JSON
                data = {
                    "metadata": decoder.manifest,
                    "text_blocks": decoder.get_text_blocks(),
                    "blocks": [block.to_dict() for block in decoder.blocks]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return ConversionResult(
                    success=True,
                    output_path=output_path,
                    metadata={"format": "json", "blocks_exported": len(decoder.blocks)}
                )
            else:
                return ConversionResult(
                    success=False,
                    warnings=[f"Unsupported export format: {output_format}"]
                )
                
        except Exception as e:
            return ConversionResult(
                success=False,
                warnings=[f"Export failed: {str(e)}"]
            )


class MAIFPluginManager:
    """Plugin manager for MAIF extensions."""
    
    def __init__(self):
        self.plugins = []  # Change to list for test compatibility
        self._plugin_dict = {}  # Keep dict for internal use
        self.hooks = {
            'pre_conversion': [],
            'post_conversion': [],
            'pre_validation': [],
            'post_validation': []
        }
    
    def register_plugin(self, name: str, plugin_class):
        """Register a plugin."""
        self._plugin_dict[name] = plugin_class
        self.plugins.append(name)  # Add to list for test compatibility
    
    def register_hook(self, hook_name: str, callback):
        """Register a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def execute_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all callbacks for a hook."""
        results = []
        for callback in self.hooks.get(hook_name, []):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results
    
    def get_plugin(self, name: str):
        """Get a registered plugin."""
        return self._plugin_dict.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugin_dict.keys())
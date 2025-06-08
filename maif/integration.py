"""
Integration and compatibility layer for MAIF with external formats and systems.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pathlib import Path
import mimetypes
from dataclasses import dataclass
import base64
import zipfile
import tarfile

@dataclass
class ConversionResult:
    """Result of format conversion."""
    success: bool
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

class MAIFConverter:
    """Convert between MAIF and other formats."""
    
    def __init__(self):
        self.supported_formats = {
            'json': self._convert_from_json,
            'xml': self._convert_from_xml,
            'zip': self._convert_from_zip,
            'tar': self._convert_from_tar,
            'csv': self._convert_from_csv,
            'txt': self._convert_from_text,
            'md': self._convert_from_markdown,
            'pdf': self._convert_from_pdf,
            'docx': self._convert_from_docx
        }
        
        self.export_formats = {
            'json': self._export_to_json,
            'xml': self._export_to_xml,
            'zip': self._export_to_zip,
            'csv': self._export_to_csv,
            'html': self._export_to_html
        }
    
    def convert_to_maif(self, input_path: str, output_path: str, 
                       format_hint: str = None) -> ConversionResult:
        """Convert external format to MAIF."""
        try:
            from .core import MAIFEncoder
            
            input_file = Path(input_path)
            if not input_file.exists():
                return ConversionResult(
                    success=False,
                    errors=[f"Input file not found: {input_path}"]
                )
            
            # Detect format
            if format_hint:
                file_format = format_hint.lower()
            else:
                file_format = input_file.suffix.lower().lstrip('.')
                if not file_format:
                    mime_type, _ = mimetypes.guess_type(input_path)
                    file_format = self._mime_to_format(mime_type)
            
            if file_format not in self.supported_formats:
                return ConversionResult(
                    success=False,
                    errors=[f"Unsupported format: {file_format}"]
                )
            
            # Convert
            encoder = MAIFEncoder(agent_id="maif_converter")
            converter_func = self.supported_formats[file_format]
            
            result = converter_func(input_path, encoder)
            if not result.success:
                return result
            
            # Build MAIF
            manifest_path = f"{output_path}.manifest.json"
            encoder.build_maif(output_path, manifest_path)
            
            result.output_path = output_path
            result.metadata = {
                'source_format': file_format,
                'source_file': str(input_file),
                'manifest_path': manifest_path
            }
            
            return result
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Conversion error: {str(e)}"]
            )
    
    def export_from_maif(self, maif_path: str, output_path: str,
                        target_format: str) -> ConversionResult:
        """Export MAIF to external format."""
        try:
            from .core import MAIFParser
            
            if target_format.lower() not in self.export_formats:
                return ConversionResult(
                    success=False,
                    errors=[f"Unsupported export format: {target_format}"]
                )
            
            # Parse MAIF
            manifest_path = f"{maif_path}.manifest.json"
            parser = MAIFParser(maif_path, manifest_path)
            
            # Export
            export_func = self.export_formats[target_format.lower()]
            result = export_func(parser, output_path)
            
            return result
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Export error: {str(e)}"]
            )
    
    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        mime_map = {
            'application/json': 'json',
            'application/xml': 'xml',
            'text/xml': 'xml',
            'application/zip': 'zip',
            'application/x-tar': 'tar',
            'text/csv': 'csv',
            'text/plain': 'txt',
            'text/markdown': 'md',
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
        }
        return mime_map.get(mime_type, 'unknown')
    
    def _convert_from_json(self, input_path: str, encoder) -> ConversionResult:
        """Convert JSON to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add JSON as structured data
            json_str = json.dumps(data, indent=2)
            hash_val = encoder.add_text_block(json_str)
            
            # Add metadata
            encoder.add_metadata_block({
                'source_type': 'json',
                'content_type': 'application/json',
                'structure': self._analyze_json_structure(data)
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"JSON conversion error: {str(e)}"]
            )
    
    def _convert_from_xml(self, input_path: str, encoder) -> ConversionResult:
        """Convert XML to MAIF."""
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # Convert to string
            xml_str = ET.tostring(root, encoding='unicode')
            hash_val = encoder.add_text_block(xml_str)
            
            # Add metadata
            encoder.add_metadata_block({
                'source_type': 'xml',
                'content_type': 'application/xml',
                'root_element': root.tag,
                'namespace': root.attrib.get('xmlns', '')
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"XML conversion error: {str(e)}"]
            )
    
    def _convert_from_zip(self, input_path: str, encoder) -> ConversionResult:
        """Convert ZIP archive to MAIF."""
        try:
            warnings = []
            
            with zipfile.ZipFile(input_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                for file_name in file_list:
                    try:
                        with zip_file.open(file_name) as f:
                            content = f.read()
                        
                        # Try to decode as text
                        try:
                            text_content = content.decode('utf-8')
                            hash_val = encoder.add_text_block(text_content)
                            content_type = 'text'
                        except UnicodeDecodeError:
                            hash_val = encoder.add_binary_block(content, "binary_data")
                            content_type = 'binary'
                        
                        # Add file metadata
                        encoder.add_metadata_block({
                            'source_file': file_name,
                            'content_type': content_type,
                            'size': len(content)
                        })
                        
                    except Exception as e:
                        warnings.append(f"Could not process {file_name}: {str(e)}")
            
            return ConversionResult(success=True, warnings=warnings)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"ZIP conversion error: {str(e)}"]
            )
    
    def _convert_from_tar(self, input_path: str, encoder) -> ConversionResult:
        """Convert TAR archive to MAIF."""
        try:
            warnings = []
            
            with tarfile.open(input_path, 'r') as tar_file:
                for member in tar_file.getmembers():
                    if member.isfile():
                        try:
                            f = tar_file.extractfile(member)
                            if f:
                                content = f.read()
                                
                                # Try to decode as text
                                try:
                                    text_content = content.decode('utf-8')
                                    hash_val = encoder.add_text_block(text_content)
                                    content_type = 'text'
                                except UnicodeDecodeError:
                                    hash_val = encoder.add_binary_block(content, "binary_data")
                                    content_type = 'binary'
                                
                                # Add file metadata
                                encoder.add_metadata_block({
                                    'source_file': member.name,
                                    'content_type': content_type,
                                    'size': member.size,
                                    'mode': oct(member.mode),
                                    'mtime': member.mtime
                                })
                        
                        except Exception as e:
                            warnings.append(f"Could not process {member.name}: {str(e)}")
            
            return ConversionResult(success=True, warnings=warnings)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"TAR conversion error: {str(e)}"]
            )
    
    def _convert_from_csv(self, input_path: str, encoder) -> ConversionResult:
        """Convert CSV to MAIF."""
        try:
            import csv
            
            with open(input_path, 'r', encoding='utf-8') as f:
                # Read entire CSV as text
                csv_content = f.read()
                hash_val = encoder.add_text_block(csv_content)
                
                # Parse for metadata
                f.seek(0)
                reader = csv.reader(f)
                rows = list(reader)
                
                encoder.add_metadata_block({
                    'source_type': 'csv',
                    'content_type': 'text/csv',
                    'row_count': len(rows),
                    'column_count': len(rows[0]) if rows else 0,
                    'headers': rows[0] if rows else []
                })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"CSV conversion error: {str(e)}"]
            )
    
    def _convert_from_text(self, input_path: str, encoder) -> ConversionResult:
        """Convert plain text to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hash_val = encoder.add_text_block(content)
            
            encoder.add_metadata_block({
                'source_type': 'text',
                'content_type': 'text/plain',
                'character_count': len(content),
                'line_count': content.count('\n') + 1
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Text conversion error: {str(e)}"]
            )
    
    def _convert_from_markdown(self, input_path: str, encoder) -> ConversionResult:
        """Convert Markdown to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hash_val = encoder.add_text_block(content)
            
            # Analyze markdown structure
            headers = []
            for line in content.split('\n'):
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    text = line.lstrip('#').strip()
                    headers.append({'level': level, 'text': text})
            
            encoder.add_metadata_block({
                'source_type': 'markdown',
                'content_type': 'text/markdown',
                'character_count': len(content),
                'line_count': content.count('\n') + 1,
                'headers': headers
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Markdown conversion error: {str(e)}"]
            )
    
    def _convert_from_pdf(self, input_path: str, encoder) -> ConversionResult:
        """Convert PDF to MAIF (requires PyPDF2 or similar)."""
        try:
            # This would require a PDF library like PyPDF2
            # For now, just read as binary
            with open(input_path, 'rb') as f:
                content = f.read()
            
            hash_val = encoder.add_binary_block(content, "pdf_data")
            
            encoder.add_metadata_block({
                'source_type': 'pdf',
                'content_type': 'application/pdf',
                'size': len(content)
            })
            
            return ConversionResult(
                success=True,
                warnings=["PDF text extraction not implemented - stored as binary"]
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"PDF conversion error: {str(e)}"]
            )
    
    def _convert_from_docx(self, input_path: str, encoder) -> ConversionResult:
        """Convert DOCX to MAIF (requires python-docx)."""
        try:
            # This would require python-docx library
            # For now, just read as binary
            with open(input_path, 'rb') as f:
                content = f.read()
            
            hash_val = encoder.add_binary_block(content, "docx_data")
            
            encoder.add_metadata_block({
                'source_type': 'docx',
                'content_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'size': len(content)
            })
            
            return ConversionResult(
                success=True,
                warnings=["DOCX text extraction not implemented - stored as binary"]
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"DOCX conversion error: {str(e)}"]
            )
    
    def _export_to_json(self, parser, output_path: str) -> ConversionResult:
        """Export MAIF to JSON."""
        try:
            content = parser.extract_content()
            metadata = parser.get_metadata()
            
            export_data = {
                'metadata': metadata,
                'content': {
                    'texts': content['texts'],
                    'embeddings': content['embeddings']
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return ConversionResult(success=True, output_path=output_path)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"JSON export error: {str(e)}"]
            )
    
    def _export_to_xml(self, parser, output_path: str) -> ConversionResult:
        """Export MAIF to XML."""
        try:
            content = parser.extract_content()
            metadata = parser.get_metadata()
            
            root = ET.Element("maif")
            
            # Add metadata
            meta_elem = ET.SubElement(root, "metadata")
            for key, value in metadata.items():
                elem = ET.SubElement(meta_elem, key)
                elem.text = str(value)
            
            # Add content
            content_elem = ET.SubElement(root, "content")
            
            # Add texts
            texts_elem = ET.SubElement(content_elem, "texts")
            for i, text in enumerate(content['texts']):
                text_elem = ET.SubElement(texts_elem, "text", id=str(i))
                text_elem.text = text
            
            # Write XML
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            return ConversionResult(success=True, output_path=output_path)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"XML export error: {str(e)}"]
            )
    
    def _export_to_zip(self, parser, output_path: str) -> ConversionResult:
        """Export MAIF to ZIP archive."""
        try:
            content = parser.extract_content()
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add text files
                for i, text in enumerate(content['texts']):
                    zip_file.writestr(f"text_{i}.txt", text)
                
                # Add embeddings as JSON
                if content['embeddings']:
                    embeddings_json = json.dumps(content['embeddings'], indent=2)
                    zip_file.writestr("embeddings.json", embeddings_json)
                
                # Add metadata
                metadata = parser.get_metadata()
                metadata_json = json.dumps(metadata, indent=2, default=str)
                zip_file.writestr("metadata.json", metadata_json)
            
            return ConversionResult(success=True, output_path=output_path)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"ZIP export error: {str(e)}"]
            )
    
    def _export_to_csv(self, parser, output_path: str) -> ConversionResult:
        """Export MAIF to CSV (embeddings only)."""
        try:
            import csv
            
            content = parser.extract_content()
            
            if not content['embeddings']:
                return ConversionResult(
                    success=False,
                    errors=["No embeddings found to export to CSV"]
                )
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                embedding_dim = len(content['embeddings'][0])
                header = [f'dim_{i}' for i in range(embedding_dim)]
                writer.writerow(header)
                
                # Write embeddings
                writer.writerows(content['embeddings'])
            
            return ConversionResult(success=True, output_path=output_path)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"CSV export error: {str(e)}"]
            )
    
    def _export_to_html(self, parser, output_path: str) -> ConversionResult:
        """Export MAIF to HTML."""
        try:
            content = parser.extract_content()
            metadata = parser.get_metadata()
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MAIF Export</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metadata {{ background: #f5f5f5; padding: 20px; margin-bottom: 20px; }}
        .text-block {{ margin-bottom: 20px; padding: 15px; border-left: 3px solid #007bff; }}
        .embeddings {{ background: #f8f9fa; padding: 15px; }}
    </style>
</head>
<body>
    <h1>MAIF Content Export</h1>
    
    <div class="metadata">
        <h2>Metadata</h2>
        <ul>
"""
            
            for key, value in metadata.items():
                html_content += f"            <li><strong>{key}:</strong> {value}</li>\n"
            
            html_content += """        </ul>
    </div>
    
    <div class="content">
        <h2>Text Content</h2>
"""
            
            for i, text in enumerate(content['texts']):
                html_content += f"""        <div class="text-block">
            <h3>Text Block {i+1}</h3>
            <pre>{text}</pre>
        </div>
"""
            
            if content['embeddings']:
                html_content += f"""        
        <div class="embeddings">
            <h3>Embeddings</h3>
            <p>Found {len(content['embeddings'])} embedding vectors of dimension {len(content['embeddings'][0])}</p>
        </div>
"""
            
            html_content += """    </div>
</body>
</html>"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return ConversionResult(success=True, output_path=output_path)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"HTML export error: {str(e)}"]
            )
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure for metadata."""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'key_count': len(data)
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': list(set(type(item).__name__ for item in data))
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:100]  # First 100 chars
            }

class MAIFPluginManager:
    """Plugin manager for extending MAIF functionality."""
    
    def __init__(self):
        self.plugins = {}
        self.hooks = {
            'pre_encode': [],
            'post_encode': [],
            'pre_decode': [],
            'post_decode': [],
            'pre_validate': [],
            'post_validate': []
        }
    
    def register_plugin(self, name: str, plugin_class):
        """Register a new plugin."""
        self.plugins[name] = plugin_class()
    
    def register_hook(self, hook_name: str, callback):
        """Register a hook callback."""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
    
    def execute_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all callbacks for a hook."""
        results = []
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Hook {hook_name} error: {e}")
        return results
    
    def get_plugin(self, name: str):
        """Get a registered plugin."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self.plugins.keys())

# Example plugin interface
class MAIFPlugin:
    """Base class for MAIF plugins."""
    
    def __init__(self):
        self.name = "base_plugin"
        self.version = "1.0.0"
    
    def initialize(self, config: Dict[str, Any] = None):
        """Initialize the plugin."""
        pass
    
    def process_block(self, block_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Process a block during encoding/decoding."""
        return block_data
    
    def validate_block(self, block_data: bytes, metadata: Dict[str, Any]) -> List[str]:
        """Validate a block and return any issues."""
        return []
    
    def cleanup(self):
        """Cleanup plugin resources."""
        pass

# Global instances
converter = MAIFConverter()
plugin_manager = MAIFPluginManager()
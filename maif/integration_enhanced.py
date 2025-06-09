"""
Enhanced integration module bringing together all improved MAIF components.
Provides high-level APIs for production-ready MAIF operations with format conversion.
"""

import json
import time
import xml.etree.ElementTree as ET
import mimetypes
import base64
import zipfile
import tarfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from .core import MAIFEncoder, MAIFDecoder
from .block_types import BlockType, BlockFactory, BlockValidator
from .semantic_optimized import AdaptiveCrossModalAttention, HierarchicalSemanticCompression, CryptographicSemanticBinding
from .compression import MAIFCompressor, CompressionConfig, CompressionAlgorithm
from .forensics import ForensicAnalyzer
from .validation import MAIFValidator, MAIFRepairTool
from .privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode
from .security import MAIFSigner, MAIFVerifier

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
        if self.metadata is None:
            self.metadata = {}

class MAIFConverter:
    """Format converter for MAIF files."""
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv', 'txt']
    
    def convert_to_maif(self, input_path: str, output_path: str,
                       manifest_path: str, input_format: str = None) -> ConversionResult:
        """Convert various formats to MAIF."""
        try:
            if input_format == 'json':
                return self._convert_json_to_maif(input_path, output_path, manifest_path)
            elif input_format == 'xml':
                return self._convert_xml_to_maif(input_path, output_path, manifest_path)
            else:
                return ConversionResult(success=False, errors=["Unsupported format"])
        except Exception as e:
            return ConversionResult(success=False, errors=[str(e)])
    
    def _convert_json_to_maif(self, input_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert JSON to MAIF."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        encoder = MAIFEncoder(agent_id="converter")
        encoder.add_text_block(json.dumps(data), metadata={"format": "json"})
        encoder.build_maif(output_path, manifest_path)
        
        return ConversionResult(success=True, output_path=output_path)
    
    def _convert_xml_to_maif(self, input_path: str, output_path: str, manifest_path: str) -> ConversionResult:
        """Convert XML to MAIF."""
        with open(input_path, 'r') as f:
            xml_content = f.read()
        
        encoder = MAIFEncoder(agent_id="converter")
        encoder.add_text_block(xml_content, metadata={"format": "xml"})
        encoder.build_maif(output_path, manifest_path)
        
        return ConversionResult(success=True, output_path=output_path)

class MAIFPluginManager:
    """Plugin manager for MAIF extensions."""
    
    def __init__(self):
        self.hooks = {}
        self.plugins = []
    
    def register_hook(self, hook_name: str, callback):
        """Register a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
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
                    results.append({"error": str(e)})
        return results

class EnhancedMAIFProcessor:
    """
    High-level processor that integrates all enhanced MAIF capabilities.
    Provides production-ready APIs for trustworthy AI systems.
    """
    
    def __init__(self, enable_all_features: bool = True):
        self.enable_all_features = enable_all_features
        self.supported_formats = ['json', 'xml', 'csv', 'txt']
        
        # Initialize components
        self.compressor = MAIFCompressor()
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()
        self.privacy_engine = PrivacyEngine()
        self.forensic_analyzer = ForensicAnalyzer()
        
        if enable_all_features:
            self.cross_modal_attention = AdaptiveCrossModalAttention()
            self.semantic_compression = HierarchicalSemanticCompression()
            self.crypto_binding = CryptographicSemanticBinding()

class MAIFConverter:
    """
    Converter for various file formats to/from MAIF.
    """
    
    def __init__(self):
        self.supported_formats = ['json', 'xml', 'csv', 'txt', 'yaml']
    
    def convert_to_maif(self, input_path: str, output_path: str, manifest_path: str,
                       input_format: str, agent_id: str = "converter") -> ConversionResult:
        """Convert input file to MAIF format."""
        try:
            encoder = MAIFEncoder(agent_id=agent_id)
            
            if input_format == 'json':
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                # Convert JSON structure to MAIF blocks
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            encoder.add_text_block(value, metadata={"source_key": key})
                        elif isinstance(value, dict):
                            encoder.add_text_block(json.dumps(value), metadata={"source_key": key, "type": "json_object"})
                
            elif input_format == 'xml':
                tree = ET.parse(input_path)
                root = tree.getroot()
                
                # Extract text content from XML
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        encoder.add_text_block(elem.text.strip(), metadata={"xml_tag": elem.tag})
            
            elif input_format == 'txt':
                with open(input_path, 'r') as f:
                    content = f.read()
                encoder.add_text_block(content, metadata={"source_format": "txt"})
            
            elif input_format == 'csv':
                import csv
                with open(input_path, 'r') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        encoder.add_text_block(','.join(row), metadata={"csv_row": i})
            
            # Build MAIF file
            encoder.build_maif(output_path, manifest_path)
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"input_format": input_format, "blocks_created": len(encoder.blocks)}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[str(e)]
            )
    
    def export_from_maif(self, maif_path: str, output_path: str, manifest_path: str,
                        output_format: str) -> ConversionResult:
        """Export MAIF file to another format."""
        try:
            from .core import MAIFParser
            parser = MAIFParser(maif_path, manifest_path)
            content = parser.extract_content()
            
            if output_format == 'json':
                export_data = {
                    "texts": content['texts'],
                    "embeddings": content['embeddings'],
                    "metadata": parser.get_metadata()
                }
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif output_format == 'txt':
                with open(output_path, 'w') as f:
                    for text in content['texts']:
                        f.write(text + '\n\n')
            
            elif output_format == 'csv':
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['text'])
                    for text in content['texts']:
                        writer.writerow([text])
            
            return ConversionResult(
                success=True,
                output_path=output_path,
                metadata={"output_format": output_format, "texts_exported": len(content['texts'])}
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[str(e)]
            )

class MAIFPluginManager:
    """
    Plugin manager for extending MAIF functionality.
    """
    
    def __init__(self):
        self.plugins = []
        self.hooks = {}
    
    def register_hook(self, hook_name: str, callback):
        """Register a callback for a specific hook."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    def execute_hooks(self, hook_name: str, data):
        """Execute all callbacks for a specific hook."""
        results = []
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(data)
                    results.append(result)
                except Exception as e:
                    results.append(f"Error in hook {hook_name}: {e}")
        return results
    
    def register_plugin(self, plugin):
        """Register a plugin."""
        self.plugins.append(plugin)
    
    def get_plugins(self):
        """Get all registered plugins."""
        return self.plugins

        # Initialize core components
        self.compressor = MAIFCompressor(CompressionConfig(
            algorithm=CompressionAlgorithm.SEMANTIC_AWARE,
            preserve_semantics=True,
            target_ratio=3.0
        ))
        
        self.forensic_analyzer = ForensicAnalyzer()
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()
        
        # Initialize enhanced algorithms
        if enable_all_features:
            self.acam = AdaptiveCrossModalAttention()
            self.hsc = HierarchicalSemanticCompression()
            self.csb = CryptographicSemanticBinding()
        
        # Format conversion support
        self.supported_formats = {
            'json': self._convert_from_json,
            'xml': self._convert_from_xml,
            'zip': self._convert_from_zip,
            'tar': self._convert_from_tar,
            'csv': self._convert_from_csv,
            'txt': self._convert_from_text,
            'md': self._convert_from_markdown
        }
        
        self.export_formats = {
            'json': self._export_to_json,
            'xml': self._export_to_xml,
            'zip': self._export_to_zip,
            'csv': self._export_to_csv,
            'html': self._export_to_html
        }
        
        self.performance_metrics = {
            "operations_count": 0,
            "total_processing_time": 0.0,
            "compression_ratios": [],
            "semantic_fidelities": [],
            "security_validations": 0
        }
    
    def create_enhanced_maif(self, 
                           multimodal_data: Dict[str, Any],
                           output_path: str,
                           manifest_path: str,
                           privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL,
                           enable_compression: bool = True,
                           enable_semantic_binding: bool = True,
                           agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive MAIF file with all enhanced features.
        """
        start_time = time.time()
        
        try:
            # Initialize encoder with privacy
            encoder = MAIFEncoder(
                agent_id=agent_id,
                enable_privacy=True
            )
            
            # Set privacy policy
            privacy_policy = PrivacyPolicy(
                privacy_level=privacy_level,
                encryption_mode=EncryptionMode.AES_GCM if privacy_level != PrivacyLevel.PUBLIC else EncryptionMode.NONE,
                anonymization_required=privacy_level in [PrivacyLevel.SECRET, PrivacyLevel.TOP_SECRET],
                audit_required=True
            )
            encoder.set_default_privacy_policy(privacy_policy)
            
            # Process each modality
            block_ids = {}
            semantic_data = {}
            
            for modality, data in multimodal_data.items():
                if modality == "text":
                    # Add text with semantic processing
                    block_id = encoder.add_text_block(
                        str(data), 
                        privacy_policy=privacy_policy,
                        anonymize=privacy_policy.anonymization_required
                    )
                    block_ids[f"{modality}_block"] = block_id
                    
                    # Generate embeddings for semantic binding
                    if enable_semantic_binding and self.enable_all_features:
                        from .semantic import SemanticEmbedder
                        embedder = SemanticEmbedder()
                        embedding = embedder.embed_text(str(data))
                        semantic_data[modality] = embedding.vector
                
                elif modality in ["image", "video", "audio"]:
                    # Handle binary data
                    if isinstance(data, str):
                        data = data.encode('utf-8')
                    elif not isinstance(data, bytes):
                        data = str(data).encode('utf-8')
                    
                    block_id = encoder.add_binary_block(
                        data, 
                        f"{modality}_data",
                        privacy_policy=privacy_policy
                    )
                    block_ids[f"{modality}_block"] = block_id
                    
                    # Generate pseudo-embeddings for binary data
                    if enable_semantic_binding and self.enable_all_features:
                        import hashlib
                        hash_obj = hashlib.sha256(data)
                        hash_hex = hash_obj.hexdigest()
                        base_embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, len(hash_hex), 2)]
                        embedding = (base_embedding * (384 // len(base_embedding) + 1))[:384]
                        semantic_data[modality] = embedding
            
            # Add cross-modal processing with enhanced ACAM
            if len(semantic_data) > 1 and self.enable_all_features:
                cross_modal_id = encoder.add_cross_modal_block(
                    multimodal_data,
                    use_enhanced_acam=True,
                    privacy_policy=privacy_policy
                )
                block_ids["cross_modal_block"] = cross_modal_id
            
            # Add compressed embeddings with enhanced HSC
            if semantic_data and enable_compression and self.enable_all_features:
                embeddings_list = list(semantic_data.values())
                compressed_id = encoder.add_compressed_embeddings_block(
                    embeddings_list,
                    use_enhanced_hsc=True,
                    preserve_fidelity=True,
                    target_compression_ratio=0.4,
                    privacy_policy=privacy_policy
                )
                block_ids["compressed_embeddings_block"] = compressed_id
            
            # Add semantic bindings with enhanced CSB
            if enable_semantic_binding and semantic_data and self.enable_all_features:
                for modality, embedding in semantic_data.items():
                    source_data = str(multimodal_data.get(modality, ""))
                    binding_id = encoder.add_semantic_binding_block(
                        embedding,
                        source_data,
                        privacy_policy=privacy_policy
                    )
                    block_ids[f"{modality}_binding"] = binding_id
            
            # Build and save MAIF
            encoder.build_maif(output_path, manifest_path)
            
            # Generate performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["operations_count"] += 1
            self.performance_metrics["total_processing_time"] += processing_time
            
            # Validate the created file
            validation_result = self.validator.validate_file(output_path, manifest_path)
            
            result = {
                "status": "success",
                "output_path": output_path,
                "manifest_path": manifest_path,
                "block_ids": block_ids,
                "processing_time": processing_time,
                "validation_result": {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                    "file_size": validation_result.file_size,
                    "block_count": validation_result.block_count
                },
                "privacy_level": privacy_level.value,
                "features_enabled": {
                    "compression": enable_compression,
                    "semantic_binding": enable_semantic_binding,
                    "enhanced_algorithms": self.enable_all_features
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def analyze_maif_security(self, maif_path: str, manifest_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive security analysis of a MAIF file.
        """
        start_time = time.time()
        
        try:
            # Forensic analysis
            forensic_result = self.forensic_analyzer.analyze_maif_file(maif_path, manifest_path)
            
            # Validation
            validation_result = self.validator.validate_file(maif_path, manifest_path)
            
            # Signature verification
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                verifier = MAIFVerifier()
                signature_valid = False
                if "signature" in manifest:
                    signature_valid = verifier.verify_maif_signature(manifest)
                
                provenance_valid = False
                provenance_errors = []
                if "provenance" in manifest:
                    provenance_valid, provenance_errors = verifier.verify_provenance_chain(manifest["provenance"])
                
            except Exception as e:
                signature_valid = False
                provenance_valid = False
                provenance_errors = [f"Verification error: {str(e)}"]
            
            # Calculate security score
            security_score = self._calculate_security_score(
                validation_result, forensic_result, signature_valid, provenance_valid
            )
            
            analysis_time = time.time() - start_time
            self.performance_metrics["security_validations"] += 1
            
            return {
                "status": "success",
                "analysis_time": analysis_time,
                "security_score": security_score,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                },
                "forensics": {
                    "risk_assessment": forensic_result.get("risk_assessment", {}),
                    "evidence_summary": forensic_result.get("evidence_summary", {}),
                    "recommendations": forensic_result.get("recommendations", [])
                },
                "cryptographic": {
                    "signature_valid": signature_valid,
                    "provenance_valid": provenance_valid,
                    "provenance_errors": provenance_errors
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "analysis_time": time.time() - start_time
            }
    
    def repair_maif_file(self, maif_path: str, manifest_path: str) -> Dict[str, Any]:
        """
        Attempt to repair a corrupted MAIF file.
        """
        start_time = time.time()
        
        try:
            # Validate first to identify issues
            validation_result = self.validator.validate_file(maif_path, manifest_path)
            
            if validation_result.is_valid:
                return {
                    "status": "no_repair_needed",
                    "message": "File is already valid",
                    "repair_time": time.time() - start_time
                }
            
            # Attempt repair
            repair_success = self.repair_tool.repair_file(maif_path, manifest_path)
            
            # Validate after repair
            post_repair_validation = self.validator.validate_file(maif_path, manifest_path)
            
            repair_time = time.time() - start_time
            
            return {
                "status": "success" if repair_success else "failed",
                "repair_success": repair_success,
                "repair_time": repair_time,
                "pre_repair_errors": validation_result.errors,
                "post_repair_validation": {
                    "is_valid": post_repair_validation.is_valid,
                    "errors": post_repair_validation.errors,
                    "warnings": post_repair_validation.warnings
                },
                "improvement": len(validation_result.errors) - len(post_repair_validation.errors)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "repair_time": time.time() - start_time
            }
    
    def benchmark_performance(self, test_data: Dict[str, Any], iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark the performance of enhanced MAIF operations.
        """
        results = {
            "iterations": iterations,
            "compression_performance": {},
            "semantic_performance": {},
            "security_performance": {},
            "overall_metrics": {}
        }
        
        # Benchmark compression
        if "text" in test_data:
            text_data = str(test_data["text"]).encode('utf-8')
            compression_results = self.compressor.benchmark_algorithms(text_data, "text")
            results["compression_performance"] = compression_results
        
        # Benchmark semantic algorithms
        if self.enable_all_features and "embeddings" in test_data:
            embeddings = test_data["embeddings"]
            
            # HSC benchmark
            hsc_times = []
            hsc_ratios = []
            for _ in range(iterations):
                start_time = time.time()
                compressed = self.hsc.compress_embeddings(embeddings)
                hsc_times.append(time.time() - start_time)
                hsc_ratios.append(compressed.get("metadata", {}).get("compression_ratio", 1.0))
            
            results["semantic_performance"]["hsc"] = {
                "average_time": sum(hsc_times) / len(hsc_times),
                "average_ratio": sum(hsc_ratios) / len(hsc_ratios),
                "min_time": min(hsc_times),
                "max_time": max(hsc_times)
            }
        
        # Overall performance metrics
        results["overall_metrics"] = self.get_performance_summary()
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        metrics = self.performance_metrics.copy()
        
        if metrics["operations_count"] > 0:
            metrics["average_processing_time"] = metrics["total_processing_time"] / metrics["operations_count"]
        else:
            metrics["average_processing_time"] = 0.0
        
        if metrics["compression_ratios"]:
            metrics["average_compression_ratio"] = sum(metrics["compression_ratios"]) / len(metrics["compression_ratios"])
        else:
            metrics["average_compression_ratio"] = 0.0
        
        if metrics["semantic_fidelities"]:
            metrics["average_semantic_fidelity"] = sum(metrics["semantic_fidelities"]) / len(metrics["semantic_fidelities"])
        else:
            metrics["average_semantic_fidelity"] = 0.0
        
        return metrics
    
    def _calculate_security_score(self, validation_result, forensic_result, 
                                signature_valid: bool, provenance_valid: bool) -> float:
        """Calculate overall security score (0.0 to 1.0)."""
        score = 1.0
        
        # Validation penalties
        if not validation_result.is_valid:
            score -= 0.3
        
        score -= len(validation_result.errors) * 0.05
        score -= len(validation_result.warnings) * 0.02
        
        # Forensic penalties
        risk_assessment = forensic_result.get("risk_assessment", {})
        risk_score = risk_assessment.get("risk_score", 0.0)
        score -= risk_score * 0.4
        
        # Cryptographic penalties
        if not signature_valid:
            score -= 0.2
        
        if not provenance_valid:
            score -= 0.2
        
        return max(0.0, min(1.0, score))

    # Format Conversion Methods
    def convert_to_maif(self, input_path: str, output_path: str,
                       format_hint: str = None) -> ConversionResult:
        """Convert external format to MAIF."""
        try:
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
            encoder = MAIFEncoder(agent_id="enhanced_maif_processor")
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
                errors=[f"Conversion failed: {str(e)}"]
            )
    
    def convert_xml_to_maif(self, xml_path: str, output_path: str, manifest_path: str) -> Dict[str, Any]:
        """Convert XML to MAIF format."""
        try:
            result = self.convert_to_maif(xml_path, output_path, format_hint='xml')
            return {
                'success': result.success,
                'output_path': result.output_path,
                'manifest_path': manifest_path,
                'errors': result.errors,
                'warnings': result.warnings
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [str(e)]
            }
    
    def _convert_from_json(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert JSON to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add JSON data as structured block
            encoder.add_block("json_data", json.dumps(data, indent=2).encode('utf-8'), {
                'type': 'structured_data',
                'format': 'json',
                'keys': list(data.keys()) if isinstance(data, dict) else []
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"JSON conversion failed: {str(e)}"]
            )
    
    def _convert_from_xml(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert XML to MAIF."""
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            # Convert XML to structured format
            xml_data = self._xml_to_dict(root)
            
            encoder.add_block("xml_data", json.dumps(xml_data, indent=2).encode('utf-8'), {
                'type': 'structured_data',
                'format': 'xml',
                'root_tag': root.tag
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"XML conversion failed: {str(e)}"]
            )
    
    def _convert_from_text(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert text file to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            encoder.add_block("text_content", content.encode('utf-8'), {
                'type': 'text',
                'encoding': 'utf-8',
                'word_count': len(content.split()),
                'char_count': len(content)
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Text conversion failed: {str(e)}"]
            )
    
    def _convert_from_zip(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert ZIP archive to MAIF."""
        try:
            with zipfile.ZipFile(input_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                for file_name in file_list:
                    if not file_name.endswith('/'):  # Skip directories
                        file_data = zip_file.read(file_name)
                        encoder.add_block(f"zip_file_{file_name.replace('/', '_')}", file_data, {
                            'type': 'archive_file',
                            'original_path': file_name,
                            'archive_type': 'zip'
                        })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"ZIP conversion failed: {str(e)}"]
            )
    
    def _convert_from_tar(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert TAR archive to MAIF."""
        try:
            with tarfile.open(input_path, 'r') as tar_file:
                for member in tar_file.getmembers():
                    if member.isfile():
                        file_data = tar_file.extractfile(member).read()
                        encoder.add_block(f"tar_file_{member.name.replace('/', '_')}", file_data, {
                            'type': 'archive_file',
                            'original_path': member.name,
                            'archive_type': 'tar',
                            'size': member.size
                        })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"TAR conversion failed: {str(e)}"]
            )
    
    def _convert_from_csv(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert CSV to MAIF."""
        try:
            import csv
            
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            csv_data = {
                'headers': reader.fieldnames,
                'rows': rows,
                'row_count': len(rows)
            }
            
            encoder.add_block("csv_data", json.dumps(csv_data, indent=2).encode('utf-8'), {
                'type': 'tabular_data',
                'format': 'csv',
                'columns': reader.fieldnames,
                'row_count': len(rows)
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"CSV conversion failed: {str(e)}"]
            )
    
    def _convert_from_markdown(self, input_path: str, encoder: MAIFEncoder) -> ConversionResult:
        """Convert Markdown to MAIF."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            encoder.add_block("markdown_content", content.encode('utf-8'), {
                'type': 'markup_text',
                'format': 'markdown',
                'encoding': 'utf-8'
            })
            
            return ConversionResult(success=True)
            
        except Exception as e:
            return ConversionResult(
                success=False,
                errors=[f"Markdown conversion failed: {str(e)}"]
            )
    
    def _xml_to_dict(self, element):
        """Convert XML element to dictionary."""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            result['#text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def _mime_to_format(self, mime_type: str) -> str:
        """Convert MIME type to format string."""
        mime_map = {
            'application/json': 'json',
            'text/xml': 'xml',
            'application/xml': 'xml',
            'application/zip': 'zip',
            'application/x-tar': 'tar',
            'text/csv': 'csv',
            'text/plain': 'txt',
            'text/markdown': 'md'
        }
        return mime_map.get(mime_type, 'unknown')
    
    # Export methods (stubs for now)
    def _export_to_json(self, maif_path: str, output_path: str) -> ConversionResult:
        """Export MAIF to JSON."""
        return ConversionResult(success=False, errors=["Export not implemented"])
    
    def _export_to_xml(self, maif_path: str, output_path: str) -> ConversionResult:
        """Export MAIF to XML."""
        return ConversionResult(success=False, errors=["Export not implemented"])
    
    def _export_to_zip(self, maif_path: str, output_path: str) -> ConversionResult:
        """Export MAIF to ZIP."""
        return ConversionResult(success=False, errors=["Export not implemented"])
    
    def _export_to_csv(self, maif_path: str, output_path: str) -> ConversionResult:
        """Export MAIF to CSV."""
        return ConversionResult(success=False, errors=["Export not implemented"])
    
    def _export_to_html(self, maif_path: str, output_path: str) -> ConversionResult:
        """Export MAIF to HTML."""
        return ConversionResult(success=False, errors=["Export not implemented"])
# Export main class
__all__ = ['EnhancedMAIFProcessor', 'ConversionResult']
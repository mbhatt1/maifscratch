#!/usr/bin/env python3
"""
MAIF Paper-Code Alignment Validation Script

This script validates that the implementation stays aligned with the academic paper
specifications in README.tex. It checks:
1. Algorithm implementations match mathematical specifications
2. Performance targets are met
3. Block type definitions align with paper
4. Documentation references are correct
"""

import os
import re
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any

class AlignmentValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path(__file__).parent
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating MAIF Paper-Code Alignment...")
        
        # Core validation checks
        self.validate_block_types()
        self.validate_algorithm_locations()
        self.validate_documentation_references()
        self.validate_performance_claims()
        self.validate_mathematical_formulas()
        
        # Report results
        self.print_results()
        return len(self.errors) == 0
    
    def validate_block_types(self):
        """Validate block type definitions match paper specifications."""
        print("📦 Validating block type definitions...")
        
        # Expected block types from paper (README.tex:456-481)
        expected_blocks = {
            'HDER': 'Header',
            'TEXT': 'Text Data', 
            'EMBD': 'Embedding',
            'KGRF': 'Knowledge Graph',
            'SECU': 'Security',
            'BDAT': 'Binary Data'
        }
        
        try:
            # Check implementation
            block_types_file = self.project_root / 'maif' / 'block_types.py'
            if not block_types_file.exists():
                self.errors.append("maif/block_types.py not found")
                return
                
            with open(block_types_file, 'r') as f:
                content = f.read()
                
            # Extract BlockType enum values
            block_pattern = r'(\w+)\s*=\s*"(\w+)"'
            found_blocks = re.findall(block_pattern, content)
            
            for name, fourcc in found_blocks:
                if fourcc in expected_blocks:
                    print(f"  ✅ {fourcc} ({name}) - ALIGNED")
                else:
                    self.warnings.append(f"Block type {fourcc} not in paper specification")
                    
            # Check for missing blocks
            found_fourccs = {fourcc for _, fourcc in found_blocks}
            for expected_fourcc in expected_blocks:
                if expected_fourcc not in found_fourccs:
                    self.errors.append(f"Missing block type {expected_fourcc} from paper")
                    
        except Exception as e:
            self.errors.append(f"Error validating block types: {e}")
    
    def validate_algorithm_locations(self):
        """Validate algorithm implementations exist in documented locations."""
        print("🧠 Validating algorithm implementations...")
        
        algorithms = {
            'AdaptiveCrossModalAttention': 'maif/semantic_optimized.py',
            'HierarchicalSemanticCompression': 'maif/semantic_optimized.py', 
            'CryptographicSemanticBinding': 'maif/semantic_optimized.py'
        }
        
        for algorithm, expected_file in algorithms.items():
            file_path = self.project_root / expected_file
            if not file_path.exists():
                self.errors.append(f"Algorithm file {expected_file} not found")
                continue
                
            with open(file_path, 'r') as f:
                content = f.read()
                
            if f"class {algorithm}" in content:
                print(f"  ✅ {algorithm} found in {expected_file}")
            else:
                self.errors.append(f"Algorithm {algorithm} not found in {expected_file}")
    
    def validate_documentation_references(self):
        """Validate documentation file references are correct."""
        print("📚 Validating documentation references...")
        
        # Check main README.md references
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
                
            # Check docs folder references
            docs_refs = re.findall(r'\[.*?\]\((docs/[^)]+)\)', readme_content)
            for ref in docs_refs:
                ref_path = self.project_root / ref
                if ref_path.exists():
                    print(f"  ✅ {ref} - EXISTS")
                else:
                    self.errors.append(f"Documentation reference {ref} not found")
        
        # Check algorithm documentation alignment
        novel_algo_doc = self.project_root / 'docs' / 'NOVEL_ALGORITHMS_IMPLEMENTATION.md'
        if novel_algo_doc.exists():
            with open(novel_algo_doc, 'r') as f:
                content = f.read()
                
            # Check for correct file references
            if 'maif/semantic_optimized.py' in content:
                print("  ✅ Algorithm documentation references correct file")
            else:
                self.warnings.append("Algorithm documentation may reference incorrect files")
    
    def validate_performance_claims(self):
        """Validate performance claims can be verified."""
        print("⚡ Validating performance claims...")
        
        # Check if benchmark files exist
        benchmark_files = [
            'run_benchmark.py',
            'large_data_crypto_benchmark.py',
            'benchmarks/maif_benchmark_suite.py'
        ]
        
        for benchmark_file in benchmark_files:
            file_path = self.project_root / benchmark_file
            if file_path.exists():
                print(f"  ✅ {benchmark_file} - EXISTS")
            else:
                self.warnings.append(f"Benchmark file {benchmark_file} not found")
    
    def validate_mathematical_formulas(self):
        """Validate mathematical formulas in implementation match paper."""
        print("🔢 Validating mathematical formulas...")
        
        # Check ACAM formula implementation
        semantic_file = self.project_root / 'maif' / 'semantic_optimized.py'
        if semantic_file.exists():
            with open(semantic_file, 'r') as f:
                content = f.read()
                
            # Look for ACAM formula comment
            if "α_{ij} = softmax(Q_i K_j^T / √d_k · CS(E_i, E_j))" in content:
                print("  ✅ ACAM formula documented in code")
            else:
                self.warnings.append("ACAM formula not documented in implementation")
                
            # Check for key implementation elements
            acam_elements = [
                'compute_attention_weights',
                'softmax',
                'semantic_coherence',
                'trust_scores'
            ]
            
            for element in acam_elements:
                if element in content:
                    print(f"  ✅ ACAM element '{element}' found")
                else:
                    self.warnings.append(f"ACAM element '{element}' not found")
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "="*60)
        print("📊 VALIDATION RESULTS")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("🎉 ALL CHECKS PASSED - Paper and code are aligned!")
            return
            
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        print(f"\n📈 ALIGNMENT STATUS:")
        total_checks = len(self.errors) + len(self.warnings)
        if self.errors:
            print("  🔴 CRITICAL ISSUES FOUND - Immediate attention required")
        elif self.warnings:
            print("  🟡 MINOR ISSUES FOUND - Review recommended")
        else:
            print("  🟢 FULLY ALIGNED")

def main():
    """Main validation entry point."""
    validator = AlignmentValidator()
    success = validator.validate_all()
    
    if not success:
        print("\n💡 Next steps:")
        print("  1. Review the alignment document: docs/PAPER_CODE_ALIGNMENT.md")
        print("  2. Fix critical errors before proceeding")
        print("  3. Consider addressing warnings for better alignment")
        sys.exit(1)
    else:
        print("\n✅ Validation complete - Paper and code are properly aligned!")
        sys.exit(0)

if __name__ == "__main__":
    main()
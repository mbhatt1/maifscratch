"""
Command-line interface for MAIF tools.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional
def create_privacy_maif():
    """CLI command to create MAIF files with privacy controls."""
    parser = argparse.ArgumentParser(description='Create a privacy-enabled MAIF file')
    parser.add_argument('output', help='Output MAIF file path')
    parser.add_argument('--manifest', help='Output manifest file path')
    parser.add_argument('--text', action='append', help='Add text content')
    parser.add_argument('--file', action='append', help='Add file content')
    parser.add_argument('--agent-id', help='Agent identifier')
    parser.add_argument('--privacy-level', choices=['public', 'internal', 'confidential', 'secret', 'top_secret'],
                       default='internal', help='Default privacy level')
    parser.add_argument('--encryption', choices=['none', 'aes_gcm', 'chacha20_poly1305'],
                       default='aes_gcm', help='Encryption mode')
    parser.add_argument('--anonymize', action='store_true', help='Enable automatic anonymization')
    parser.add_argument('--retention-days', type=int, help='Data retention period in days')
    parser.add_argument('--access-rule', action='append', nargs=3, metavar=('SUBJECT', 'RESOURCE', 'PERMISSIONS'),
                       help='Add access rule: subject resource permissions')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFEncoder
        from .privacy import PrivacyPolicy, PrivacyLevel, EncryptionMode, AccessRule
        
        # Create privacy policy
        privacy_level = PrivacyLevel(args.privacy_level)
        encryption_mode = EncryptionMode(args.encryption)
        
        policy = PrivacyPolicy(
            privacy_level=privacy_level,
            encryption_mode=encryption_mode,
            anonymization_required=args.anonymize,
            retention_period=args.retention_days,
            audit_required=True
        )
        
        # Initialize encoder with privacy
        encoder = MAIFEncoder(agent_id=args.agent_id, enable_privacy=True)
        encoder.set_default_privacy_policy(policy)
        
        # Add access rules
        if args.access_rule:
            for subject, resource, permissions in args.access_rule:
                perms = permissions.split(',')
                encoder.add_access_rule(subject, resource, perms)
                print(f"Added access rule: {subject} -> {resource} ({permissions})")
        
        # Add text content
        if args.text:
            for text in args.text:
                hash_val = encoder.add_text_block(text, anonymize=args.anonymize)
                print(f"Added encrypted text block: {hash_val[:16]}...")
        
        # Add file content
        if args.file:
            for file_path in args.file:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.txt', '.md', '.json']:
                    hash_val = encoder.add_text_block(data.decode('utf-8'), anonymize=args.anonymize)
                    block_type = "text_data"
                else:
                    hash_val = encoder.add_binary_block(data, "binary_data")
                    block_type = "binary_data"
                
                print(f"Added encrypted {block_type} from {file_path}: {hash_val[:16]}...")
        
        # Build MAIF
        manifest_path = args.manifest or f"{args.output}.manifest.json"
        encoder.save(args.output, manifest_path)
        
        # Generate privacy report
        privacy_report = encoder.get_privacy_report()
        print(f"\n✓ Privacy-enabled MAIF created: {args.output}")
        print(f"✓ Manifest: {manifest_path}")
        print(f"✓ Privacy level: {args.privacy_level}")
        print(f"✓ Encryption: {args.encryption}")
        print(f"✓ Encrypted blocks: {privacy_report.get('encrypted_blocks', 0)}")
        
    except Exception as e:
        print(f"Error creating privacy MAIF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def access_privacy_maif():
    """CLI command to access MAIF files with privacy controls."""
    parser = argparse.ArgumentParser(description='Access privacy-enabled MAIF file')
    parser.add_argument('maif_file', help='MAIF file to access')
    parser.add_argument('--manifest', help='Manifest file path')
    parser.add_argument('--agent-id', required=True, help='Requesting agent identifier')
    parser.add_argument('--list-blocks', action='store_true', help='List accessible blocks')
    parser.add_argument('--get-text', action='store_true', help='Extract accessible text')
    parser.add_argument('--privacy-report', action='store_true', help='Show privacy summary')
    parser.add_argument('--check-access', nargs=2, metavar=('BLOCK_ID', 'PERMISSION'),
                       help='Check access to specific block')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFDecoder
        from .privacy import PrivacyEngine
        
        # Create privacy engine (in real scenario, this would be shared/persistent)
        privacy_engine = PrivacyEngine()
        
        manifest_path = args.manifest or f"{args.maif_file}.manifest.json"
        decoder = MAIFDecoder(args.maif_file, manifest_path, 
                             privacy_engine=privacy_engine, 
                             requesting_agent=args.agent_id)
        
        print(f"Accessing MAIF as agent: {args.agent_id}")
        
        if args.list_blocks:
            accessible_blocks = decoder.get_accessible_blocks("read")
            print(f"\nAccessible blocks ({len(accessible_blocks)}):")
            for block in accessible_blocks:
                privacy_info = ""
                if block.metadata and 'privacy_policy' in block.metadata:
                    policy = block.metadata['privacy_policy']
                    privacy_info = f" [Privacy: {policy.get('privacy_level', 'unknown')}, " \
                                 f"Encryption: {policy.get('encryption_mode', 'unknown')}]"
                print(f"  {block.block_id}: {block.block_type}{privacy_info}")
        
        if args.get_text:
            texts = decoder.get_text_blocks()
            print(f"\nAccessible text blocks ({len(texts)}):")
            for i, text in enumerate(texts, 1):
                print(f"  {i}. {text[:100]}{'...' if len(text) > 100 else ''}")
        
        if args.privacy_report:
            report = decoder.get_privacy_summary()
            print(f"\nPrivacy Summary:")
            print(json.dumps(report, indent=2))
        
        if args.check_access:
            block_id, permission = args.check_access
            has_access = decoder.check_block_access(block_id, permission)
            print(f"\nAccess check: {args.agent_id} -> {block_id} ({permission})")
            print(f"Result: {'✓ Granted' if has_access else '✗ Denied'}")
        
    except Exception as e:
        print(f"Error accessing privacy MAIF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def manage_privacy():
    """CLI command to manage privacy settings."""
    parser = argparse.ArgumentParser(description='Manage MAIF privacy settings')
    parser.add_argument('command', choices=['anonymize', 'encrypt', 'access-rules', 'report'],
                       help='Privacy management command')
    parser.add_argument('--maif-file', help='MAIF file to manage')
    parser.add_argument('--manifest', help='Manifest file path')
    parser.add_argument('--agent-id', help='Agent identifier')
    parser.add_argument('--block-id', help='Specific block ID')
    parser.add_argument('--subject', help='Access rule subject')
    parser.add_argument('--resource', help='Access rule resource')
    parser.add_argument('--permissions', help='Access rule permissions (comma-separated)')
    parser.add_argument('--context', help='Anonymization context')
    
    args = parser.parse_args()
    
    try:
        from .privacy import PrivacyEngine, DifferentialPrivacy
        
        if args.command == 'anonymize':
            if not args.maif_file:
                print("Error: --maif-file required for anonymize command")
                sys.exit(1)
            
            privacy_engine = PrivacyEngine()
            
            # Example anonymization
            test_text = "John Smith at john.smith@company.com called 555-123-4567"
            anonymized = privacy_engine.anonymize_data(test_text, args.context or "default")
            
            print(f"Original: {test_text}")
            print(f"Anonymized: {anonymized}")
            
        elif args.command == 'encrypt':
            print("Encryption is automatically applied based on privacy policies.")
            print("Use create-privacy-maif with --encryption option.")
            
        elif args.command == 'access-rules':
            print("Access rules management:")
            print("Use create-privacy-maif with --access-rule option to add rules.")
            print("Use access-privacy-maif with --check-access to verify access.")
            
        elif args.command == 'report':
            if not args.maif_file:
                print("Error: --maif-file required for report command")
                sys.exit(1)
            
            from .core import MAIFDecoder
            privacy_engine = PrivacyEngine()
            
            manifest_path = args.manifest or f"{args.maif_file}.manifest.json"
            decoder = MAIFDecoder(args.maif_file, manifest_path, 
                                 privacy_engine=privacy_engine,
                                 requesting_agent=args.agent_id or "admin")
            
            report = decoder.get_privacy_summary()
            print("Privacy Report:")
            print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"Error managing privacy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_maif():
    """CLI command to create MAIF files."""
    parser = argparse.ArgumentParser(description='Create a new MAIF file')
    parser.add_argument('output', help='Output MAIF file path')
    parser.add_argument('--manifest', help='Output manifest file path')
    parser.add_argument('--text', action='append', help='Add text content')
    parser.add_argument('--file', action='append', help='Add file content')
    parser.add_argument('--agent-id', help='Agent identifier')
    parser.add_argument('--compress', action='store_true', help='Enable compression')
    parser.add_argument('--sign', action='store_true', help='Sign the MAIF file')
    parser.add_argument('--key', help='Private key file for signing')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFEncoder
        from .security import MAIFSigner
        
        # Initialize encoder
        encoder = MAIFEncoder(agent_id=args.agent_id)
        signer = MAIFSigner(private_key_path=args.key, agent_id=args.agent_id) if args.sign else None
        
        # Add text content
        if args.text:
            for text in args.text:
                hash_val = encoder.add_text_block(text)
                if signer:
                    signer.add_provenance_entry("add_text", hash_val)
                print(f"Added text block: {hash_val[:16]}...")
        
        # Add file content
        if args.file:
            for file_path in args.file:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.txt', '.md', '.json']:
                    hash_val = encoder.add_text_block(data.decode('utf-8'))
                    block_type = "text_data"
                else:
                    hash_val = encoder.add_binary_block(data, "binary_data")
                    block_type = "binary_data"
                
                if signer:
                    signer.add_provenance_entry(f"add_{block_type}", hash_val)
                print(f"Added {block_type} from {file_path}: {hash_val[:16]}...")
        
        # Build MAIF
        manifest_path = args.manifest or f"{args.output}.manifest.json"
        encoder.build_maif(args.output, manifest_path)
        
        # Sign if requested
        if signer:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            signed_manifest = signer.sign_maif_manifest(manifest)
            
            with open(manifest_path, 'w') as f:
                json.dump(signed_manifest, f, indent=2)
            
            print(f"✓ Signed MAIF created: {args.output}")
        else:
            print(f"✓ MAIF created: {args.output}")
        
        print(f"✓ Manifest: {manifest_path}")
        
    except Exception as e:
        print(f"Error creating MAIF: {e}")
        sys.exit(1)

def verify_maif():
    """CLI command to verify MAIF files."""
    parser = argparse.ArgumentParser(description='Verify a MAIF file')
    parser.add_argument('maif_file', help='MAIF file to verify')
    parser.add_argument('--manifest', help='Manifest file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFParser
        from .security import MAIFVerifier
        from .validation import MAIFValidator, MAIFRepairTool
        
        manifest_path = args.manifest or f"{args.maif_file}.manifest.json"
        
        if not os.path.exists(args.maif_file):
            print(f"Error: MAIF file not found: {args.maif_file}")
            sys.exit(1)
        
        if not os.path.exists(manifest_path):
            print(f"Error: Manifest file not found: {manifest_path}")
            sys.exit(1)
        
        # Perform validation
        print("Validating MAIF file...")
        validator = MAIFValidator()
        report = validator.validate_file(args.maif_file, manifest_path)
        
        # Display results
        if report.is_valid:
            print("✓ MAIF file is valid")
        else:
            print("✗ MAIF file has issues")
        
        # Show statistics
        stats = report.statistics
        print(f"\nStatistics:")
        print(f"  Total blocks: {stats.get('total_blocks', 0)}")
        print(f"  File size: {stats.get('total_size', 0):,} bytes")
        print(f"  Integrity checks: {stats.get('integrity_checks', 0)}")
        print(f"  Signature checks: {stats.get('signature_checks', 0)}")
        
        # Show issues
        if report.issues:
            print(f"\nIssues found: {len(report.issues)}")
            
            for severity in ['critical', 'error', 'warning', 'info']:
                issues = [i for i in report.issues if i.severity.value == severity]
                if issues:
                    print(f"\n{severity.upper()} ({len(issues)}):")
                    for issue in issues:
                        print(f"  - {issue.message}")
                        if args.verbose and issue.suggested_fix:
                            print(f"    Fix: {issue.suggested_fix}")
        
        # Attempt repair if requested
        if args.repair and not report.is_valid:
            print("\nAttempting repairs...")
            repair_tool = MAIFRepairTool()
            if repair_tool.repair_file(args.maif_file, manifest_path):
                print("✓ Repairs completed")
                for log_entry in repair_tool.repair_log:
                    print(f"  {log_entry}")
            else:
                print("✗ Could not repair all issues")
        
        # Exit with appropriate code
        if report.is_valid:
            sys.exit(0)
        else:
            critical_errors = len([i for i in report.issues if i.severity.value == 'critical'])
            sys.exit(1 if critical_errors > 0 else 2)
        
    except Exception as e:
        print(f"Error verifying MAIF: {e}")
        sys.exit(1)

def analyze_maif():
    """CLI command to analyze MAIF files."""
    parser = argparse.ArgumentParser(description='Analyze a MAIF file')
    parser.add_argument('maif_file', help='MAIF file to analyze')
    parser.add_argument('--manifest', help='Manifest file path')
    parser.add_argument('--output', '-o', help='Output report file')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    parser.add_argument('--forensic', action='store_true', help='Perform forensic analysis')
    parser.add_argument('--timeline', action='store_true', help='Show version timeline')
    parser.add_argument('--agents', action='store_true', help='Show agent activity')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFParser
        from .security import MAIFVerifier
        from .forensics import ForensicAnalyzer
        
        manifest_path = args.manifest or f"{args.maif_file}.manifest.json"
        
        # Parse MAIF
        parser_obj = MAIFParser(args.maif_file, manifest_path)
        
        # Basic analysis
        metadata = parser_obj.get_metadata()
        content = parser_obj.extract_content()
        
        if args.format == 'json':
            result = {
                "metadata": metadata,
                "content_summary": {
                    "text_blocks": len(content['texts']),
                    "embeddings": len(content['embeddings'])
                }
            }
        else:
            result = []
            result.append("MAIF Analysis Report")
            result.append("=" * 50)
            result.append(f"File: {args.maif_file}")
            result.append(f"Version: {metadata.get('version', 'unknown')}")
            result.append(f"Created: {metadata.get('created', 'unknown')}")
            result.append(f"Blocks: {metadata.get('block_count', 0)}")
            result.append(f"Text blocks: {len(content['texts'])}")
            result.append(f"Embeddings: {len(content['embeddings'])}")
        
        # Forensic analysis
        if args.forensic:
            verifier = MAIFVerifier()
            analyzer = ForensicAnalyzer()
            forensic_report = analyzer.analyze_maif(parser_obj, verifier)
            
            if args.format == 'json':
                result["forensic_analysis"] = forensic_report.to_dict()
            else:
                result.append("\nForensic Analysis")
                result.append("-" * 20)
                result.append(f"Status: {forensic_report.integrity_status}")
                result.append(f"Events analyzed: {forensic_report.events_analyzed}")
                result.append(f"Evidence found: {len(forensic_report.evidence)}")
                
                if forensic_report.evidence:
                    result.append("\nEvidence:")
                    for evidence in forensic_report.evidence[:5]:  # Show first 5
                        result.append(f"  - {evidence.severity.upper()}: {evidence.description}")
        
        # Timeline analysis
        if args.timeline and hasattr(parser_obj.decoder, 'version_history'):
            timeline = parser_obj.decoder.get_version_timeline()
            
            if args.format == 'json':
                result["timeline"] = [v.to_dict() for v in timeline]
            else:
                result.append(f"\nVersion Timeline ({len(timeline)} events)")
                result.append("-" * 30)
                for event in timeline[-10:]:  # Show last 10 events
                    result.append(f"  {event.datetime_str}: {event.operation} by {event.agent_id}")
        
        # Agent activity
        if args.agents and hasattr(parser_obj.decoder, 'version_history'):
            agent_stats = {}
            for version in parser_obj.decoder.version_history:
                agent_id = version.agent_id
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {"operations": 0, "last_activity": version.timestamp}
                agent_stats[agent_id]["operations"] += 1
                agent_stats[agent_id]["last_activity"] = max(
                    agent_stats[agent_id]["last_activity"], 
                    version.timestamp
                )
            
            if args.format == 'json':
                result["agent_activity"] = agent_stats
            else:
                result.append(f"\nAgent Activity ({len(agent_stats)} agents)")
                result.append("-" * 25)
                for agent_id, stats in agent_stats.items():
                    result.append(f"  {agent_id}: {stats['operations']} operations")
        
        # Output results
        if args.format == 'json':
            output_text = json.dumps(result, indent=2)
        else:
            output_text = '\n'.join(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"Analysis saved to: {args.output}")
        else:
            print(output_text)
        
    except Exception as e:
        print(f"Error analyzing MAIF: {e}")
        sys.exit(1)

def extract_content():
    """CLI command to extract content from MAIF files."""
    parser = argparse.ArgumentParser(description='Extract content from MAIF file')
    parser.add_argument('maif_file', help='MAIF file to extract from')
    parser.add_argument('--manifest', help='Manifest file path')
    parser.add_argument('--output-dir', '-o', help='Output directory', default='.')
    parser.add_argument('--type', choices=['text', 'embeddings', 'all'], default='all', help='Content type to extract')
    parser.add_argument('--format', choices=['json', 'txt', 'csv'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    try:
        from .core import MAIFParser
        import csv
        
        manifest_path = args.manifest or f"{args.maif_file}.manifest.json"
        parser_obj = MAIFParser(args.maif_file, manifest_path)
        content = parser_obj.extract_content()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Extract text content
        if args.type in ['text', 'all'] and content['texts']:
            if args.format == 'json':
                with open(output_dir / 'texts.json', 'w') as f:
                    json.dump(content['texts'], f, indent=2)
            elif args.format == 'txt':
                for i, text in enumerate(content['texts']):
                    with open(output_dir / f'text_{i}.txt', 'w') as f:
                        f.write(text)
            print(f"Extracted {len(content['texts'])} text blocks")
        
        # Extract embeddings
        if args.type in ['embeddings', 'all'] and content['embeddings']:
            if args.format == 'json':
                with open(output_dir / 'embeddings.json', 'w') as f:
                    json.dump(content['embeddings'], f, indent=2)
            elif args.format == 'csv':
                with open(output_dir / 'embeddings.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'dim_{i}' for i in range(len(content['embeddings'][0]))])
                    writer.writerows(content['embeddings'])
            print(f"Extracted {len(content['embeddings'])} embeddings")
        
        print(f"Content extracted to: {output_dir}")
        
    except Exception as e:
        print(f"Error extracting content: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='MAIF file format tools')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add subcommands
    subparsers.add_parser('create', help='Create a new MAIF file')
    subparsers.add_parser('verify', help='Verify a MAIF file')
    subparsers.add_parser('analyze', help='Analyze a MAIF file')
    subparsers.add_parser('extract', help='Extract content from MAIF file')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_maif()
    elif args.command == 'verify':
        verify_maif()
    elif args.command == 'analyze':
        analyze_maif()
    elif args.command == 'extract':
        extract_content()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
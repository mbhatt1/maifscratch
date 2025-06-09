"""
Command-line interface for MAIF tools.
"""

import click
import json
import sys
import os
from pathlib import Path
from typing import Optional

@click.command()
@click.option('--input', 'input_file', help='Input file path')
@click.option('--output', required=True, help='Output MAIF file path')
@click.option('--manifest', help='Output manifest file path')
@click.option('--text', multiple=True, help='Add text content')
@click.option('--file', 'files', multiple=True, help='Add file content')
@click.option('--agent-id', help='Agent identifier')
@click.option('--privacy-level', type=click.Choice(['public', 'internal', 'confidential', 'secret', 'top_secret']),
              default='internal', help='Default privacy level')
@click.option('--encryption', type=click.Choice(['none', 'aes_gcm', 'chacha20_poly1305']),
              default='aes_gcm', help='Encryption mode')
@click.option('--anonymize', is_flag=True, help='Enable automatic anonymization')
@click.option('--retention-days', type=int, help='Data retention period in days')
@click.option('--access-rule', multiple=True, nargs=3, help='Add access rule: subject resource permissions')
def create_privacy_maif(input_file, output, manifest, text, files, agent_id, privacy_level, 
                       encryption, anonymize, retention_days, access_rule):
    """CLI command to create MAIF files with privacy controls."""
    try:
        from .core import MAIFEncoder
        from .privacy import PrivacyPolicy, PrivacyLevel, EncryptionMode, AccessRule
        
        # Create privacy policy
        privacy_level_enum = PrivacyLevel(privacy_level)
        encryption_mode = EncryptionMode(encryption)
        
        policy = PrivacyPolicy(
            privacy_level=privacy_level_enum,
            encryption_mode=encryption_mode,
            anonymization_required=anonymize,
            retention_period=retention_days,
            audit_required=True
        )
        
        # Initialize encoder with privacy
        encoder = MAIFEncoder(agent_id=agent_id, enable_privacy=True)
        encoder.set_default_privacy_policy(policy)
        
        # Add access rules
        if access_rule:
            for subject, resource, permissions in access_rule:
                perms = permissions.split(',')
                encoder.add_access_rule(subject, resource, perms)
                click.echo(f"Added access rule: {subject} -> {resource} ({permissions})")
        
        # Add input file content if provided
        if input_file and os.path.exists(input_file):
            with open(input_file, 'r') as f:
                content = f.read()
            hash_val = encoder.add_text_block(content, anonymize=anonymize)
            click.echo(f"Added encrypted text block from {input_file}: {hash_val[:16]}...")
        
        # Add text content
        if text:
            for text_content in text:
                hash_val = encoder.add_text_block(text_content, anonymize=anonymize)
                click.echo(f"Added encrypted text block: {hash_val[:16]}...")
        
        # Add file content
        if files:
            for file_path in files:
                if not os.path.exists(file_path):
                    click.echo(f"Warning: File not found: {file_path}")
                    continue
                
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.txt', '.md', '.json']:
                    hash_val = encoder.add_text_block(data.decode('utf-8'), anonymize=anonymize)
                    block_type = "text_data"
                else:
                    hash_val = encoder.add_binary_block(data, "binary_data")
                    block_type = "binary_data"
                
                click.echo(f"Added encrypted {block_type} from {file_path}: {hash_val[:16]}...")
        
        # Build MAIF
        manifest_path = manifest or f"{output}.manifest.json"
        encoder.save(output, manifest_path)
        
        # Generate privacy report
        privacy_report = encoder.get_privacy_report()
        click.echo(f"\n✓ Privacy-enabled MAIF created: {output}")
        click.echo(f"✓ Manifest: {manifest_path}")
        click.echo(f"✓ Privacy level: {privacy_level}")
        click.echo(f"✓ Encryption: {encryption}")
        click.echo(f"✓ Encrypted blocks: {privacy_report.get('encrypted_blocks', 0)}")
        
    except Exception as e:
        click.echo(f"Error creating privacy MAIF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@click.command()
@click.option('--maif-file', required=True, help='MAIF file to access')
@click.option('--manifest', help='Manifest file path')
@click.option('--user-id', help='Requesting user identifier')
@click.option('--agent-id', help='Requesting agent identifier')
@click.option('--permission', help='Permission to check')
@click.option('--list-blocks', is_flag=True, help='List accessible blocks')
@click.option('--get-text', is_flag=True, help='Extract accessible text')
@click.option('--privacy-report', is_flag=True, help='Show privacy summary')
@click.option('--check-access', nargs=2, help='Check access to specific block')
def access_privacy_maif(maif_file, manifest, user_id, agent_id, permission, list_blocks, 
                       get_text, privacy_report, check_access):
    """CLI command to access MAIF files with privacy controls."""
    try:
        from .core import MAIFDecoder
        from .privacy import PrivacyEngine
        
        # Create privacy engine (in real scenario, this would be shared/persistent)
        privacy_engine = PrivacyEngine()
        
        manifest_path = manifest or f"{maif_file}.manifest.json"
        requesting_agent = agent_id or user_id or "default_user"
        
        decoder = MAIFDecoder(maif_file, manifest_path, 
                             privacy_engine=privacy_engine, 
                             requesting_agent=requesting_agent)
        
        click.echo(f"Accessing MAIF as agent: {requesting_agent}")
        
        if list_blocks:
            accessible_blocks = decoder.get_accessible_blocks("read")
            click.echo(f"\nAccessible blocks ({len(accessible_blocks)}):")
            for block in accessible_blocks:
                privacy_info = ""
                if block.metadata and 'privacy_policy' in block.metadata:
                    policy = block.metadata['privacy_policy']
                    privacy_info = f" [Privacy: {policy.get('privacy_level', 'unknown')}, " \
                                 f"Encryption: {policy.get('encryption_mode', 'unknown')}]"
                click.echo(f"  {block.block_id}: {block.block_type}{privacy_info}")
        
        if get_text:
            texts = decoder.get_text_blocks()
            click.echo(f"\nAccessible text blocks ({len(texts)}):")
            for i, text in enumerate(texts, 1):
                click.echo(f"  {i}. {text[:100]}{'...' if len(text) > 100 else ''}")
        
        if privacy_report:
            report = decoder.get_privacy_summary()
            click.echo(f"\nPrivacy Summary:")
            click.echo(json.dumps(report, indent=2))
        
        if check_access:
            block_id, perm = check_access
            has_access = decoder.check_block_access(block_id, perm)
            click.echo(f"\nAccess check: {requesting_agent} -> {block_id} ({perm})")
            click.echo(f"Result: {'✓ Granted' if has_access else '✗ Denied'}")
        
    except Exception as e:
        click.echo(f"Error accessing privacy MAIF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@click.command()
@click.argument('command', type=click.Choice(['anonymize', 'encrypt', 'access-rules', 'report']))
@click.option('--maif-file', help='MAIF file to manage')
@click.option('--manifest', help='Manifest file path')
@click.option('--agent-id', help='Agent identifier')
@click.option('--block-id', help='Specific block ID')
@click.option('--subject', help='Access rule subject')
@click.option('--resource', help='Access rule resource')
@click.option('--permissions', help='Access rule permissions (comma-separated)')
@click.option('--context', help='Anonymization context')
def manage_privacy(command, maif_file, manifest, agent_id, block_id, subject, resource, permissions, context):
    """CLI command to manage privacy settings."""
    try:
        from .privacy import PrivacyEngine, DifferentialPrivacy
        
        if command == 'anonymize':
            if not maif_file:
                click.echo("Error: --maif-file required for anonymize command")
                sys.exit(1)
            
            privacy_engine = PrivacyEngine()
            
            # Example anonymization
            test_text = "John Smith at john.smith@company.com called 555-123-4567"
            anonymized = privacy_engine.anonymize_data(test_text, context or "default")
            
            click.echo(f"Original: {test_text}")
            click.echo(f"Anonymized: {anonymized}")
            
        elif command == 'encrypt':
            click.echo("Encryption is automatically applied based on privacy policies.")
            click.echo("Use create-privacy-maif with --encryption option.")
            
        elif command == 'access-rules':
            click.echo("Access rules management:")
            click.echo("Use create-privacy-maif with --access-rule option to add rules.")
            click.echo("Use access-privacy-maif with --check-access to verify access.")
            
        elif command == 'report':
            if not maif_file:
                click.echo("Error: --maif-file required for report command")
                sys.exit(1)
            
            from .core import MAIFDecoder
            privacy_engine = PrivacyEngine()
            
            manifest_path = manifest or f"{maif_file}.manifest.json"
            decoder = MAIFDecoder(maif_file, manifest_path, 
                                 privacy_engine=privacy_engine,
                                 requesting_agent=agent_id or "admin")
            
            report = decoder.get_privacy_summary()
            click.echo("Privacy Report:")
            click.echo(json.dumps(report, indent=2))
        
    except Exception as e:
        click.echo(f"Error managing privacy: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

@click.command()
@click.option('--input', 'input_file', help='Input file path')
@click.option('--output', required=True, help='Output MAIF file path')
@click.option('--manifest', help='Output manifest file path')
@click.option('--text', multiple=True, help='Add text content')
@click.option('--file', 'files', multiple=True, help='Add file content')
@click.option('--agent-id', help='Agent identifier')
@click.option('--compress', is_flag=True, help='Enable compression')
@click.option('--sign', is_flag=True, help='Sign the MAIF file')
@click.option('--key', help='Private key file for signing')
def create_maif(input_file, output, manifest, text, files, agent_id, compress, sign, key):
    """CLI command to create MAIF files."""
    try:
        from .core import MAIFEncoder
        from .security import MAIFSigner
        
        # Initialize encoder
        encoder = MAIFEncoder(agent_id=agent_id)
        signer = MAIFSigner(private_key_path=key, agent_id=agent_id) if sign else None
        
        # Add input file content if provided
        if input_file and os.path.exists(input_file):
            with open(input_file, 'r') as f:
                content = f.read()
            hash_val = encoder.add_text_block(content)
            if signer:
                signer.add_provenance_entry("add_text", hash_val)
            click.echo(f"Added text block from {input_file}: {hash_val[:16]}...")
        
        # Add text content
        if text:
            for text_content in text:
                hash_val = encoder.add_text_block(text_content)
                if signer:
                    signer.add_provenance_entry("add_text", hash_val)
                click.echo(f"Added text block: {hash_val[:16]}...")
        
        # Add file content
        if files:
            for file_path in files:
                if not os.path.exists(file_path):
                    click.echo(f"Warning: File not found: {file_path}")
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
                click.echo(f"Added {block_type} from {file_path}: {hash_val[:16]}...")
        
        # Build MAIF
        manifest_path = manifest or f"{output}.manifest.json"
        encoder.build_maif(output, manifest_path)
        
        # Sign if requested
        if signer:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            signed_manifest = signer.sign_maif_manifest(manifest_data)
            
            with open(manifest_path, 'w') as f:
                json.dump(signed_manifest, f, indent=2)
            
            click.echo(f"✓ Signed MAIF created: {output}")
        else:
            click.echo(f"✓ MAIF created: {output}")
        
        click.echo(f"✓ Manifest: {manifest_path}")
        
    except Exception as e:
        click.echo(f"Error creating MAIF: {e}")
        sys.exit(1)

@click.command()
@click.option('--input', 'maif_file', required=True, help='MAIF file to verify')
@click.option('--manifest', help='Manifest file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--repair', is_flag=True, help='Attempt to repair issues')
def verify_maif(maif_file, manifest, verbose, repair):
    """CLI command to verify MAIF files."""
    try:
        from .core import MAIFParser
        from .security import MAIFVerifier
        from .validation import MAIFValidator, MAIFRepairTool
        
        manifest_path = manifest or f"{maif_file}.manifest.json"
        
        if not os.path.exists(maif_file):
            click.echo(f"Error: MAIF file not found: {maif_file}")
            sys.exit(1)
        
        if not os.path.exists(manifest_path):
            click.echo(f"Error: Manifest file not found: {manifest_path}")
            sys.exit(1)
        
        # Perform validation
        click.echo("Validating MAIF file...")
        validator = MAIFValidator()
        report = validator.validate_file(maif_file, manifest_path)
        
        # Display results
        if report.is_valid:
            click.echo("✓ MAIF file is valid")
        else:
            click.echo("✗ MAIF file has issues")
        
        # Show statistics
        stats = report.statistics
        click.echo(f"\nStatistics:")
        click.echo(f"  Total blocks: {stats.get('total_blocks', 0)}")
        click.echo(f"  File size: {stats.get('total_size', 0):,} bytes")
        click.echo(f"  Integrity checks: {stats.get('integrity_checks', 0)}")
        click.echo(f"  Signature checks: {stats.get('signature_checks', 0)}")
        
        # Show issues
        if report.issues:
            click.echo(f"\nIssues found: {len(report.issues)}")
            
            for severity in ['critical', 'error', 'warning', 'info']:
                issues = [i for i in report.issues if i.severity.value == severity]
                if issues:
                    click.echo(f"\n{severity.upper()} ({len(issues)}):")
                    for issue in issues:
                        click.echo(f"  - {issue.message}")
                        if verbose and issue.suggested_fix:
                            click.echo(f"    Fix: {issue.suggested_fix}")
        
        # Attempt repair if requested
        if repair and not report.is_valid:
            click.echo("\nAttempting repairs...")
            repair_tool = MAIFRepairTool()
            if repair_tool.repair_file(maif_file, manifest_path):
                click.echo("✓ Repairs completed")
                for log_entry in repair_tool.repair_log:
                    click.echo(f"  {log_entry}")
            else:
                click.echo("✗ Could not repair all issues")
        
        # Exit with appropriate code
        if report.is_valid:
            sys.exit(0)
        else:
            critical_errors = len([i for i in report.issues if i.severity.value == 'critical'])
            sys.exit(1 if critical_errors > 0 else 2)
        
    except Exception as e:
        click.echo(f"Error verifying MAIF: {e}")
        sys.exit(1)

@click.command()
@click.option('--input', 'maif_file', required=True, help='MAIF file to analyze')
@click.option('--manifest', help='Manifest file path')
@click.option('--output', '-o', help='Output report file')
@click.option('--format', type=click.Choice(['json', 'text']), default='text', help='Output format')
@click.option('--forensic', is_flag=True, help='Perform forensic analysis')
@click.option('--timeline', is_flag=True, help='Show version timeline')
@click.option('--agents', is_flag=True, help='Show agent activity')
def analyze_maif(maif_file, manifest, output, format, forensic, timeline, agents):
    """CLI command to analyze MAIF files."""
    try:
        from .core import MAIFParser
        from .security import MAIFVerifier
        from .forensics import ForensicAnalyzer
        
        manifest_path = manifest or f"{maif_file}.manifest.json"
        
        # Parse MAIF
        parser_obj = MAIFParser(maif_file, manifest_path)
        
        # Basic analysis
        metadata = parser_obj.get_metadata()
        content = parser_obj.extract_content()
        
        if format == 'json':
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
            result.append(f"File: {maif_file}")
            result.append(f"Version: {metadata.get('version', 'unknown')}")
            result.append(f"Created: {metadata.get('created', 'unknown')}")
            result.append(f"Blocks: {metadata.get('block_count', 0)}")
            result.append(f"Text blocks: {len(content['texts'])}")
            result.append(f"Embeddings: {len(content['embeddings'])}")
        
        # Forensic analysis
        if forensic:
            verifier = MAIFVerifier()
            analyzer = ForensicAnalyzer()
            forensic_report = analyzer.analyze_maif(parser_obj, verifier)
            
            if format == 'json':
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
        if timeline and hasattr(parser_obj.decoder, 'version_history'):
            timeline_data = parser_obj.decoder.get_version_timeline()
            
            if format == 'json':
                result["timeline"] = [v.to_dict() for v in timeline_data]
            else:
                result.append(f"\nVersion Timeline ({len(timeline_data)} events)")
                result.append("-" * 30)
                for event in timeline_data[-10:]:  # Show last 10 events
                    result.append(f"  {event.datetime_str}: {event.operation} by {event.agent_id}")
        
        # Agent activity
        if agents and hasattr(parser_obj.decoder, 'version_history'):
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
            
            if format == 'json':
                result["agent_activity"] = agent_stats
            else:
                result.append(f"\nAgent Activity ({len(agent_stats)} agents)")
                result.append("-" * 25)
                for agent_id, stats in agent_stats.items():
                    result.append(f"  {agent_id}: {stats['operations']} operations")
        
        # Output results
        if format == 'json':
            output_text = json.dumps(result, indent=2)
        else:
            output_text = '\n'.join(result)
        
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
            click.echo(f"Analysis saved to: {output}")
        else:
            click.echo(output_text)
        
    except Exception as e:
        click.echo(f"Error analyzing MAIF: {e}")
        sys.exit(1)

@click.command()
@click.option('--input', 'maif_file', required=True, help='MAIF file to extract from')
@click.option('--manifest', help='Manifest file path')
@click.option('--output-dir', '-o', help='Output directory', default='.')
@click.option('--type', type=click.Choice(['text', 'embeddings', 'all']), default='all', help='Content type to extract')
@click.option('--format', type=click.Choice(['json', 'txt', 'csv']), default='json', help='Output format')
def extract_content(maif_file, manifest, output_dir, type, format):
    """CLI command to extract content from MAIF files."""
    try:
        from .core import MAIFParser
        import csv
        
        manifest_path = manifest or f"{maif_file}.manifest.json"
        parser_obj = MAIFParser(maif_file, manifest_path)
        content = parser_obj.extract_content()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract text content
        if type in ['text', 'all'] and content['texts']:
            if format == 'json':
                with open(output_path / 'texts.json', 'w') as f:
                    json.dump(content['texts'], f, indent=2)
            elif format == 'txt':
                for i, text in enumerate(content['texts']):
                    with open(output_path / f'text_{i}.txt', 'w') as f:
                        f.write(text)
            click.echo(f"Extracted {len(content['texts'])} text blocks")
        
        # Extract embeddings
        if type in ['embeddings', 'all'] and content['embeddings']:
            if format == 'json':
                with open(output_path / 'embeddings.json', 'w') as f:
                    json.dump(content['embeddings'], f, indent=2)
            elif format == 'csv':
                with open(output_path / 'embeddings.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f'dim_{i}' for i in range(len(content['embeddings'][0]))])
                    writer.writerows(content['embeddings'])
            click.echo(f"Extracted {len(content['embeddings'])} embeddings")
        
        click.echo(f"Content extracted to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error extracting content: {e}")
        sys.exit(1)

@click.group()
def main():
    """MAIF file format tools."""
    pass

# Add commands to the main group
main.add_command(create_privacy_maif, name='create-privacy-maif')
main.add_command(access_privacy_maif, name='access-privacy-maif')
main.add_command(manage_privacy, name='manage-privacy')
main.add_command(create_maif, name='create-maif')
main.add_command(verify_maif, name='verify-maif')
main.add_command(analyze_maif, name='analyze-maif')
main.add_command(extract_content, name='extract-content')

if __name__ == '__main__':
    main()
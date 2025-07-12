"""
AWS Macie Privacy Integration Demo
==================================

This example demonstrates how to use the MAIF privacy module with AWS Macie
for automated discovery, classification, and protection of sensitive data.
"""

import json
import time
import os
from maif.aws_macie_privacy import AWSMaciePrivacy, DataSensitivity
from maif.privacy import PrivacyLevel, EncryptionMode
from maif.core import MAIFClient, ArtifactConfig, EncodingType


def demonstrate_aws_macie_privacy():
    """Demonstrate AWS Macie privacy integration features."""
    
    print("=== AWS Macie Privacy Integration Demo ===\n")
    
    # Initialize the AWS Macie privacy engine
    print("1. Initializing AWS Macie Privacy Engine...")
    privacy_engine = AWSMaciePrivacy(
        region_name="us-east-1",
        enable_auto_classification=True
    )
    print("   ✓ Privacy engine initialized with Macie integration")
    print("   ✓ Custom data identifiers created for MAIF patterns\n")
    
    # Demonstrate local data classification
    print("2. Testing local data classification...")
    
    test_data_samples = [
        (b"User email: john.doe@example.com", "sample1"),
        (b"SSN: 123-45-6789", "sample2"),
        (b"Just some regular text data", "sample3"),
        (b"Credit card: 1234-5678-9012-3456", "sample4")
    ]
    
    for data, resource_id in test_data_samples:
        privacy_level = privacy_engine.classify_data(data, resource_id)
        sensitivity = privacy_engine.sensitivity_mappings.get(resource_id, DataSensitivity.LOW)
        print(f"   - Resource '{resource_id}': Privacy Level = {privacy_level.value}, Sensitivity = {sensitivity.value}")
    
    print()
    
    # Demonstrate privacy policy application
    print("3. Applying privacy policies based on classification...")
    
    for data, resource_id in test_data_samples[:2]:  # Process first two samples
        protected_data, policy = privacy_engine.apply_privacy_policy_from_macie(data, resource_id)
        
        print(f"\n   Resource: {resource_id}")
        print(f"   - Original size: {len(data)} bytes")
        print(f"   - Protected size: {len(protected_data)} bytes")
        print(f"   - Privacy level: {policy.privacy_level.value}")
        print(f"   - Encryption mode: {policy.encryption_mode.value}")
        print(f"   - Anonymization required: {policy.anonymization_required}")
        print(f"   - Retention period: {policy.retention_period} days")
    
    print()
    
    # Demonstrate S3 bucket scanning (simulation)
    print("4. Simulating S3 bucket scan with Macie...")
    print("   Note: Actual S3 scanning requires a real S3 bucket with data")
    
    # Show what would happen with a real bucket
    bucket_name = "my-maif-data-bucket"
    print(f"   Example command: privacy_engine.scan_s3_bucket('{bucket_name}')")
    print("   This would:")
    print("   - Create a Macie classification job")
    print("   - Scan all objects in the bucket")
    print("   - Identify sensitive data using built-in and custom identifiers")
    print("   - Generate findings for each sensitive data discovery\n")
    
    # Demonstrate compliance report generation
    print("5. Generating compliance report...")
    
    # Add some mock findings for demonstration
    from maif.aws_macie_privacy import MacieFinding
    mock_findings = [
        MacieFinding(
            finding_id="finding-001",
            severity="HIGH",
            type="SensitiveData:S3Object/Personal",
            resource_arn="arn:aws:s3:::my-bucket/sensitive-doc.txt",
            created_at=time.time(),
            sensitive_data_identifications=[{"type": "EMAIL_ADDRESS", "count": 5}],
            confidence=0.95,
            count=5,
            category="CLASSIFICATION",
            description="Email addresses detected"
        ),
        MacieFinding(
            finding_id="finding-002",
            severity="CRITICAL",
            type="SensitiveData:S3Object/Financial",
            resource_arn="arn:aws:s3:::my-bucket/financial-data.csv",
            created_at=time.time(),
            sensitive_data_identifications=[{"type": "CREDIT_CARD", "count": 100}],
            confidence=0.99,
            count=100,
            category="CLASSIFICATION",
            description="Credit card numbers detected"
        )
    ]
    
    # Add mock findings to the engine
    privacy_engine.macie_findings = mock_findings
    
    # Process findings
    policies = privacy_engine.process_macie_findings()
    
    print("   Processed Macie findings and created privacy policies:")
    for resource_id, policy in policies.items():
        print(f"   - {resource_id}: {policy.privacy_level.value} (retention: {policy.retention_period} days)")
    
    # Generate report
    report = privacy_engine.generate_compliance_report()
    
    print(f"\n   Compliance Report:")
    print(f"   - Compliance Score: {report['compliance_score']}/100")
    print(f"   - Total Findings: {report['total_findings']}")
    print(f"   - Severity Breakdown:")
    for severity, count in report['severity_breakdown'].items():
        if count > 0:
            print(f"     • {severity}: {count}")
    print(f"   - Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"     • [{rec['priority']}] {rec['action']}")
    
    print()
    
    # Demonstrate privacy insights
    print("6. Getting privacy insights...")
    insights = privacy_engine.get_privacy_insights()
    
    print(f"   Privacy Management Overview:")
    print(f"   - Encryption keys managed: {insights['encryption_keys_managed']}")
    print(f"   - Active privacy policies: {insights['privacy_policies_active']}")
    print(f"   - Access rules defined: {insights['access_rules_defined']}")
    
    print(f"\n   Sensitivity Classifications:")
    for sensitivity, count in insights['sensitivity_classifications'].items():
        if count > 0:
            print(f"   - {sensitivity}: {count} resources")
    
    print(f"\n   Macie Integration Status:")
    macie_status = insights['macie_integration']
    print(f"   - Findings processed: {macie_status['findings_processed']}")
    print(f"   - Sensitive data instances: {macie_status['sensitive_data_instances']}")
    print(f"   - Custom identifiers: {macie_status['custom_identifiers']}")
    
    print()
    
    # Demonstrate integration with MAIF client
    print("7. Testing integration with MAIF client...")
    
    # Create MAIF client with Macie-based privacy engine
    client = MAIFClient(
        storage_path="./demo_macie_storage"
    )
    
    # Set custom privacy engine
    client.privacy_engine = privacy_engine
    
    # Create an artifact with sensitive data
    config = ArtifactConfig(
        name="customer-data-model",
        artifact_type="model",
        encoding=EncodingType.PYTORCH,
        metadata={
            "description": "Model trained on customer data",
            "contains_pii": True,
            "data_types": ["email", "name", "address"]
        }
    )
    
    # Simulate model data with embedded sensitive information
    model_data = b"MODEL_DATA_WITH_EMAIL:john.doe@example.com_AND_MORE_DATA"
    
    # Classify the data
    privacy_level = privacy_engine.classify_data(model_data, "customer-model")
    print(f"   Model data classified as: {privacy_level.value}")
    
    # Apply privacy protection
    protected_data, policy = privacy_engine.apply_privacy_policy_from_macie(
        model_data,
        "customer-model"
    )
    
    print(f"   Privacy protection applied:")
    print(f"   - Encryption: {policy.encryption_mode.value}")
    print(f"   - Retention: {policy.retention_period} days")
    print(f"   - Geographic restrictions: {policy.geographic_restrictions}")
    
    print("\n=== Demo completed successfully! ===")
    
    # Show example of continuous monitoring setup
    print("\n8. Continuous Monitoring Setup Example:")
    print("   To enable continuous monitoring of S3 buckets:")
    print("   ```python")
    print("   privacy_engine.enable_continuous_monitoring(")
    print("       buckets=['my-data-bucket', 'my-models-bucket'],")
    print("       scan_interval=3600  # Scan every hour")
    print("   )")
    print("   ```")
    
    # Show Macie metrics
    print("\n9. Macie Operation Metrics:")
    metrics = privacy_engine.macie_client.get_metrics()
    print(f"   - Total operations: {metrics['total_operations']}")
    print(f"   - Success rate: {metrics['success_rate']:.1f}%")
    print(f"   - Average latency: {metrics['average_latency']:.3f}s")
    print(f"   - Findings discovered: {metrics['findings_discovered']}")
    print(f"   - Sensitive data found: {metrics['sensitive_data_found']}")


def demonstrate_advanced_privacy_scenarios():
    """Demonstrate advanced privacy scenarios with Macie."""
    
    print("\n=== Advanced Privacy Scenarios ===\n")
    
    privacy_engine = AWSMaciePrivacy(region_name="us-east-1")
    
    # Scenario 1: Multi-level data protection
    print("1. Multi-level Data Protection:")
    
    data_samples = {
        "public_data": b"This is public information about our products",
        "internal_data": b"Internal project code: PROJ-2024-001",
        "confidential_data": b"Customer ID: CUST-12345, Revenue: $1M",
        "secret_data": b"API_KEY=sk-1234567890abcdef"
    }
    
    for data_type, data in data_samples.items():
        # Classify and protect
        privacy_level = privacy_engine.classify_data(data, data_type)
        protected_data, policy = privacy_engine.apply_privacy_policy_from_macie(data, data_type)
        
        print(f"\n   {data_type}:")
        print(f"   - Classification: {privacy_level.value}")
        print(f"   - Encrypted: {'Yes' if len(protected_data) > len(data) else 'No'}")
        print(f"   - Retention: {policy.retention_period} days")
        print(f"   - Audit required: {policy.audit_required}")
    
    # Scenario 2: Batch privacy assessment
    print("\n\n2. Batch Privacy Assessment:")
    
    # Simulate processing multiple files
    files_to_assess = [
        ("model_v1.pkl", b"PYTORCH_MODEL_DATA"),
        ("config.json", b'{"api_key": "secret123"}'),
        ("readme.txt", b"Installation instructions"),
        ("users.csv", b"name,email\njohn,john@example.com")
    ]
    
    high_risk_files = []
    for filename, content in files_to_assess:
        privacy_level = privacy_engine.classify_data(content, filename)
        if privacy_level.value in ["high", "top_secret"]:
            high_risk_files.append(filename)
    
    print(f"   Assessed {len(files_to_assess)} files")
    print(f"   High-risk files requiring immediate protection: {high_risk_files}")
    
    # Scenario 3: Privacy policy enforcement
    print("\n3. Privacy Policy Enforcement:")
    
    # Create a strict policy for financial data
    from maif.privacy import PrivacyPolicy
    financial_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.TOP_SECRET,
        encryption_mode=EncryptionMode.AES_GCM,
        retention_period=90,  # 90 days max retention
        anonymization_required=True,
        audit_required=True,
        geographic_restrictions=["US", "EU"],
        purpose_limitation=["analytics", "compliance"]
    )
    
    privacy_engine.privacy_policies["financial_data"] = financial_policy
    
    print("   Created strict policy for financial data:")
    print(f"   - Retention: {financial_policy.retention_period} days")
    print(f"   - Geographic restrictions: {financial_policy.geographic_restrictions}")
    print(f"   - Allowed purposes: {financial_policy.purpose_limitation}")
    
    print("\n=== Advanced scenarios completed ===")


def show_macie_setup_instructions():
    """Show instructions for setting up AWS Macie."""
    
    print("\n=== AWS Macie Setup Instructions ===\n")
    
    print("To use AWS Macie with MAIF, you need:")
    print("\n1. Enable Macie in your AWS account:")
    print("   aws macie2 enable-macie")
    
    print("\n2. Required IAM permissions:")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "macie2:EnableMacie",
                    "macie2:CreateClassificationJob",
                    "macie2:CreateCustomDataIdentifier",
                    "macie2:GetFindings",
                    "macie2:ListFindings",
                    "macie2:DescribeClassificationJob",
                    "macie2:GetMacieSession",
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": "*"
            }
        ]
    }
    print(json.dumps(policy, indent=2))
    
    print("\n3. Create S3 buckets for Macie to scan:")
    print("   aws s3 mb s3://my-maif-data-bucket")
    print("   aws s3 mb s3://my-maif-models-bucket")
    
    print("\n4. Configure Macie to scan your buckets:")
    print("   - Use the AWS Console or CLI to create classification jobs")
    print("   - Enable automated sensitive data discovery")
    print("   - Set up EventBridge rules for real-time alerts")
    
    print("\n5. Best practices:")
    print("   - Use custom data identifiers for domain-specific patterns")
    print("   - Implement automated remediation workflows")
    print("   - Regular review of Macie findings and policies")
    print("   - Integration with AWS Security Hub for centralized monitoring")


if __name__ == "__main__":
    # Check if AWS credentials are configured
    import boto3
    
    try:
        # Try to create a Macie client to verify credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}\n")
        
        # Run the main demo
        demonstrate_aws_macie_privacy()
        
        # Show advanced scenarios
        demonstrate_advanced_privacy_scenarios()
        
        # Show setup instructions
        show_macie_setup_instructions()
        
    except Exception as e:
        print("ERROR: AWS credentials not configured or insufficient permissions.")
        print("\nTo run this demo, you need:")
        print("1. AWS credentials configured (aws configure)")
        print("2. Macie enabled in your AWS account")
        print("3. Appropriate IAM permissions for Macie")
        
        # Show setup instructions even on error
        show_macie_setup_instructions()
"""
AWS CloudWatch Compliance Logging Demo
======================================

This example demonstrates how to use the MAIF compliance logging module with 
AWS CloudWatch Logs for centralized logging, monitoring, and alerting.
"""

import json
import time
import os
from datetime import datetime, timedelta
from maif.aws_cloudwatch_compliance import AWSCloudWatchComplianceLogger, LogLevel, LogCategory
from maif.core import MAIFClient, ArtifactConfig, EncodingType


def demonstrate_cloudwatch_compliance_logging():
    """Demonstrate CloudWatch compliance logging features."""
    
    print("=== AWS CloudWatch Compliance Logging Demo ===\n")
    
    # Initialize the CloudWatch compliance logger
    print("1. Initializing CloudWatch Compliance Logger...")
    compliance_logger = AWSCloudWatchComplianceLogger(
        db_path="./demo_compliance.db",
        region_name="us-east-1",
        log_group_name="maif-compliance-demo",
        retention_days=30,
        enable_metrics=True,
        batch_size=10,
        batch_interval=2.0  # Send batches every 2 seconds for demo
    )
    print("   ✓ Compliance logger initialized with CloudWatch backend")
    print(f"   ✓ Log group: {compliance_logger.log_group_name}")
    print(f"   ✓ Retention: {compliance_logger.retention_days} days")
    print(f"   ✓ Metrics enabled: {compliance_logger.enable_metrics}\n")
    
    # Demonstrate various compliance events
    print("2. Logging compliance events...")
    
    # Log access event
    entry_id = compliance_logger.log(
        level=LogLevel.INFO,
        category=LogCategory.ACCESS,
        user_id="demo-user-001",
        action="read_artifact",
        resource_id="artifact-12345",
        details={
            "ip_address": "192.168.1.100",
            "user_agent": "MAIF-Client/1.0",
            "success": True
        }
    )
    print(f"   ✓ Logged access event: {entry_id}")
    
    # Log security event
    entry_id = compliance_logger.log(
        level=LogLevel.WARNING,
        category=LogCategory.SECURITY,
        user_id="demo-user-002",
        action="unauthorized_access_attempt",
        resource_id="artifact-67890",
        details={
            "reason": "insufficient_permissions",
            "required_permission": "write",
            "user_permissions": ["read"]
        }
    )
    print(f"   ✓ Logged security warning: {entry_id}")
    
    # Log data modification
    entry_id = compliance_logger.log(
        level=LogLevel.INFO,
        category=LogCategory.DATA,
        user_id="demo-user-001",
        action="update",
        resource_id="model-abc123",
        details={
            "changes": {
                "version": {"old": "1.0", "new": "1.1"},
                "size": {"old": 1048576, "new": 2097152}
            },
            "timestamp": time.time()
        }
    )
    print(f"   ✓ Logged data modification: {entry_id}")
    
    # Log critical security event
    entry_id = compliance_logger.log(
        level=LogLevel.CRITICAL,
        category=LogCategory.SECURITY,
        user_id="unknown",
        action="data_breach_attempt",
        resource_id="sensitive-data-001",
        details={
            "attack_type": "sql_injection",
            "source_ip": "10.0.0.1",
            "blocked": True,
            "alert_sent": True
        }
    )
    print(f"   ✓ Logged critical security event: {entry_id}")
    
    # Log admin action
    entry_id = compliance_logger.log(
        level=LogLevel.INFO,
        category=LogCategory.ADMIN,
        user_id="admin-001",
        action="grant_permission",
        resource_id="user-003",
        details={
            "permission": "admin",
            "scope": "global",
            "expiry": (datetime.utcnow() + timedelta(days=90)).isoformat()
        }
    )
    print(f"   ✓ Logged admin action: {entry_id}")
    
    # Wait for batch processing
    print("\n   Waiting for batch processing...")
    time.sleep(3)
    
    print()
    
    # Demonstrate integration with MAIF client
    print("3. Testing integration with MAIF client...")
    
    # Create MAIF client with compliance logging
    client = MAIFClient(
        storage_path="./demo_cloudwatch_storage"
    )
    
    # Log artifact creation
    config = ArtifactConfig(
        name="compliance-test-model",
        artifact_type="model",
        encoding=EncodingType.PYTORCH,
        metadata={
            "description": "Model for compliance testing",
            "compliance_required": True,
            "data_classification": "confidential"
        }
    )
    
    # Simulate model data
    model_data = b"PYTORCH_MODEL_DATA_FOR_COMPLIANCE_TEST"
    
    # Log the creation event
    compliance_logger.log(
        level=LogLevel.INFO,
        category=LogCategory.DATA,
        user_id="demo-user-001",
        action="create_artifact",
        resource_id=f"model-{int(time.time())}",
        details={
            "artifact_type": config.artifact_type,
            "encoding": config.encoding.value,
            "size": len(model_data),
            "metadata": config.metadata
        }
    )
    print("   ✓ Logged artifact creation to CloudWatch")
    
    # Create dashboard
    print("\n4. Creating CloudWatch dashboard...")
    try:
        dashboard_info = compliance_logger.create_compliance_dashboard()
        print(f"   ✓ Created dashboard: {dashboard_info['dashboard_name']}")
        print(f"   ✓ Dashboard URL: {dashboard_info['dashboard_url']}")
    except Exception as e:
        print(f"   ! Dashboard creation failed (may already exist): {e}")
    
    # Query logs
    print("\n5. Querying compliance logs...")
    
    # Query recent security events
    security_events = compliance_logger.query(
        categories=[LogCategory.SECURITY],
        limit=10
    )
    print(f"   ✓ Found {len(security_events)} security events")
    
    for event in security_events[:3]:  # Show first 3
        print(f"     - {event.action} by {event.user_id} at {datetime.fromtimestamp(event.timestamp).isoformat()}")
    
    # Get compliance summary
    print("\n6. Compliance summary:")
    summary = compliance_logger.get_compliance_summary()
    
    print(f"   Total events: {summary.get('total_entries', 0)}")
    print(f"   CloudWatch metrics:")
    if 'cloudwatch' in summary:
        cw_metrics = summary['cloudwatch']
        print(f"   - Events sent: {cw_metrics['events_sent']}")
        print(f"   - Batches sent: {cw_metrics['batches_sent']}")
        print(f"   - Success rate: {cw_metrics['success_rate']:.1f}%")
        print(f"   - Log streams: {cw_metrics['streams']}")
    
    # Get CloudWatch client metrics
    print("\n7. CloudWatch client metrics:")
    cw_metrics = compliance_logger.cloudwatch_client.get_metrics()
    print(f"   - Total operations: {cw_metrics['total_operations']}")
    print(f"   - Successful operations: {cw_metrics['successful_operations']}")
    print(f"   - Failed operations: {cw_metrics['failed_operations']}")
    print(f"   - Average latency: {cw_metrics['average_latency']:.3f}s")
    print(f"   - Throttling events: {cw_metrics['throttling_count']}")
    
    # Close logger
    compliance_logger.close()
    print("\n=== Demo completed successfully! ===")


def demonstrate_advanced_compliance_scenarios():
    """Demonstrate advanced compliance scenarios with CloudWatch."""
    
    print("\n=== Advanced Compliance Scenarios ===\n")
    
    compliance_logger = AWSCloudWatchComplianceLogger(
        db_path=":memory:",  # In-memory for demo
        region_name="us-east-1",
        log_group_name="maif-compliance-advanced",
        retention_days=7,
        enable_metrics=True
    )
    
    # Scenario 1: Compliance audit trail
    print("1. Creating tamper-evident audit trail:")
    
    # Simulate a series of related actions
    transaction_id = f"txn-{int(time.time())}"
    
    actions = [
        ("begin_transaction", "Transaction started"),
        ("validate_input", "Input validation passed"),
        ("authorize_action", "Authorization granted"),
        ("execute_operation", "Operation completed"),
        ("commit_transaction", "Transaction committed")
    ]
    
    for action, description in actions:
        compliance_logger.log(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            user_id="system",
            action=action,
            resource_id=transaction_id,
            details={
                "description": description,
                "step": actions.index((action, description)) + 1,
                "total_steps": len(actions)
            }
        )
        time.sleep(0.1)  # Small delay to show progression
    
    print(f"   ✓ Created audit trail for transaction {transaction_id}")
    
    # Scenario 2: Real-time alerting setup
    print("\n2. Setting up real-time alerts:")
    
    # Note: This requires an SNS topic to be created
    print("   Example alert configuration:")
    print("   - Critical events: Alert immediately")
    print("   - Security violations: Alert after 5 events in 5 minutes")
    print("   - Access denied: Alert after 10 events in 5 minutes")
    
    # Show how to enable alerts (requires SNS topic)
    print("\n   To enable real-time alerts:")
    print("   ```python")
    print("   sns_topic_arn = 'arn:aws:sns:us-east-1:123456789012:compliance-alerts'")
    print("   alarms = compliance_logger.enable_real_time_alerts(")
    print("       sns_topic_arn=sns_topic_arn,")
    print("       critical_threshold=1,")
    print("       security_threshold=5")
    print("   )")
    print("   ```")
    
    # Scenario 3: Compliance reporting
    print("\n3. Generating compliance reports:")
    
    # Log various compliance-relevant events
    compliance_events = [
        (LogLevel.INFO, LogCategory.ACCESS, "user-001", "login", "system", {"method": "password"}),
        (LogLevel.INFO, LogCategory.DATA, "user-001", "export", "dataset-001", {"format": "csv", "rows": 1000}),
        (LogLevel.WARNING, LogCategory.SECURITY, "user-002", "failed_login", "system", {"attempts": 3}),
        (LogLevel.INFO, LogCategory.ADMIN, "admin-001", "config_change", "security_settings", {"mfa": "enabled"}),
        (LogLevel.ERROR, LogCategory.SYSTEM, "system", "backup_failed", "backup-001", {"reason": "disk_full"})
    ]
    
    for level, category, user_id, action, resource_id, details in compliance_events:
        compliance_logger.log(level, category, user_id, action, resource_id, details)
    
    # Generate report
    report = compliance_logger.verify_integrity()
    print(f"   ✓ Log integrity verified: {report['is_valid']}")
    print(f"   ✓ Total entries: {report['total_entries']}")
    print(f"   ✓ Valid entries: {report['valid_entries']}")
    
    # Scenario 4: Export for long-term retention
    print("\n4. Exporting logs for long-term retention:")
    print("   Example S3 export:")
    print("   ```python")
    print("   export_info = compliance_logger.export_to_s3(")
    print("       s3_bucket='my-compliance-archive',")
    print("       s3_prefix='maif/compliance/2024/'")
    print("   )")
    print("   ```")
    print("   This exports logs to S3 for cost-effective long-term storage")
    
    compliance_logger.close()
    print("\n=== Advanced scenarios completed ===")


def show_cloudwatch_queries():
    """Show useful CloudWatch Insights queries for compliance analysis."""
    
    print("\n=== Useful CloudWatch Insights Queries ===\n")
    
    queries = {
        "Security Events Summary": """
fields @timestamp, level_name, user_id, action, resource_id
| filter category = "security"
| stats count() by level_name
""",
        
        "Failed Access Attempts": """
fields @timestamp, user_id, resource_id, details.reason
| filter action = "unauthorized_access_attempt" or action = "access_denied"
| sort @timestamp desc
| limit 100
""",
        
        "User Activity Timeline": """
fields @timestamp, user_id, action, category
| filter user_id = "specific-user-id"
| sort @timestamp asc
""",
        
        "Data Modifications": """
fields @timestamp, user_id, action, resource_id, details.changes
| filter category = "data" and (action = "update" or action = "delete")
| sort @timestamp desc
""",
        
        "Critical Events Alert": """
fields @timestamp, user_id, action, resource_id, details
| filter level >= 50
| sort @timestamp desc
| limit 50
""",
        
        "Compliance Score Calculation": """
fields category, level
| stats count() by category, level
| sort category asc, level desc
"""
    }
    
    for query_name, query in queries.items():
        print(f"{query_name}:")
        print(f"```")
        print(query.strip())
        print(f"```\n")
    
    print("Use these queries in CloudWatch Insights to analyze compliance logs.")


def show_setup_instructions():
    """Show instructions for setting up CloudWatch compliance logging."""
    
    print("\n=== CloudWatch Compliance Setup Instructions ===\n")
    
    print("1. Required IAM permissions:")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:PutRetentionPolicy",
                    "logs:PutMetricFilter",
                    "logs:CreateExportTask",
                    "logs:DescribeLogGroups",
                    "logs:DescribeLogStreams",
                    "cloudwatch:PutMetricAlarm",
                    "cloudwatch:PutDashboard"
                ],
                "Resource": "*"
            }
        ]
    }
    print(json.dumps(policy, indent=2))
    
    print("\n2. Best practices:")
    print("   - Use appropriate retention periods (compliance may require 7+ years)")
    print("   - Enable encryption at rest using KMS")
    print("   - Set up cross-region replication for disaster recovery")
    print("   - Use metric filters for real-time monitoring")
    print("   - Export to S3 for cost-effective long-term storage")
    
    print("\n3. Cost optimization:")
    print("   - Use log group expiration for non-critical logs")
    print("   - Compress logs before sending (automatic with SDK)")
    print("   - Use S3 for logs older than 30 days")
    print("   - Set up lifecycle policies for archived logs")
    
    print("\n4. Integration with other AWS services:")
    print("   - AWS Security Hub: Centralized security findings")
    print("   - AWS Config: Track configuration changes")
    print("   - Amazon GuardDuty: Threat detection")
    print("   - AWS CloudTrail: API activity logging")


if __name__ == "__main__":
    # Check if AWS credentials are configured
    import boto3
    
    try:
        # Try to create a CloudWatch client to verify credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}\n")
        
        # Run the main demo
        demonstrate_cloudwatch_compliance_logging()
        
        # Show advanced scenarios
        demonstrate_advanced_compliance_scenarios()
        
        # Show useful queries
        show_cloudwatch_queries()
        
        # Show setup instructions
        show_setup_instructions()
        
    except Exception as e:
        print("ERROR: AWS credentials not configured or insufficient permissions.")
        print("\nTo run this demo, you need:")
        print("1. AWS credentials configured (aws configure)")
        print("2. CloudWatch Logs permissions")
        
        # Show setup instructions even on error
        show_setup_instructions()
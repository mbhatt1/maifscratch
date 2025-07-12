"""
AWS Macie Integration for MAIF Privacy
======================================

Integrates MAIF's privacy module with AWS Macie for automated discovery,
classification, and protection of sensitive data.
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict

import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

from .privacy import (
    PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode,
    AccessRule
)

# Configure logger
logger = logging.getLogger(__name__)


class MacieError(Exception):
    """Base exception for Macie errors."""
    pass


class MacieConnectionError(MacieError):
    """Exception for Macie connection errors."""
    pass


class MacieThrottlingError(MacieError):
    """Exception for Macie throttling errors."""
    pass


class MacieValidationError(MacieError):
    """Exception for Macie validation errors."""
    pass


class DataSensitivity(Enum):
    """Data sensitivity levels based on Macie findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MacieFinding:
    """Represents a Macie finding."""
    finding_id: str
    severity: str
    type: str
    resource_arn: str
    created_at: float
    sensitive_data_identifications: List[Dict[str, Any]]
    confidence: float
    count: int
    category: str
    description: str


class MacieClient:
    """Client for AWS Macie service with production-ready features."""
    
    # Set of known transient errors that can be retried
    RETRYABLE_ERRORS = {
        'ThrottlingException', 
        'Throttling', 
        'RequestLimitExceeded',
        'TooManyRequestsException',
        'ServiceUnavailable',
        'InternalServerError',
        'InternalFailure',
        'ServiceFailure',
        'LimitExceededException'
    }
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                 max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0):
        """
        Initialize Macie client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            
        Raises:
            MacieConnectionError: If unable to initialize the Macie client
        """
        # Validate inputs
        if not region_name:
            raise MacieValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise MacieValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise MacieValidationError("base_delay and max_delay must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing Macie client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize Macie client
            self.macie_client = session.client('macie2')
            
            # Initialize S3 client for data access
            self.s3_client = session.client('s3')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "throttling_count": 0,
                "findings_discovered": 0,
                "sensitive_data_found": 0
            }
            
            logger.info("Macie client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Macie client: {e}", exc_info=True)
            raise MacieConnectionError(f"Failed to initialize Macie client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            MacieError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing Macie {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"Macie {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track throttling
                    if error_code == 'ThrottlingException' or error_code == 'Throttling':
                        self.metrics["throttling_count"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Macie {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"Macie {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise MacieThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"Macie {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise MacieValidationError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"Macie {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise MacieError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Macie {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    logger.error(f"Macie {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise MacieConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                logger.error(f"Unexpected error in Macie {operation_name}: {str(e)}", exc_info=True)
                raise MacieError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise MacieError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise MacieError(f"Operation failed after {retries} retries")
    
    def enable_macie(self) -> Dict[str, Any]:
        """
        Enable Macie for the account.
        
        Returns:
            Response from Macie
        """
        logger.info("Enabling Macie for the account")
        
        def enable_operation():
            return self.macie_client.enable_macie()
        
        return self._execute_with_retry("enable_macie", enable_operation)
    
    def create_classification_job(self, bucket_name: str, job_name: str,
                                 custom_data_identifiers: Optional[List[str]] = None,
                                 managed_data_identifiers: Optional[List[str]] = None) -> str:
        """
        Create a classification job to scan S3 bucket for sensitive data.
        
        Args:
            bucket_name: S3 bucket name to scan
            job_name: Name for the classification job
            custom_data_identifiers: List of custom data identifier IDs
            managed_data_identifiers: List of managed data identifier types
            
        Returns:
            Job ID
        """
        logger.info(f"Creating classification job {job_name} for bucket {bucket_name}")
        
        # Prepare job configuration
        job_config = {
            'name': job_name,
            'description': f'MAIF privacy scan for {bucket_name}',
            's3JobDefinition': {
                'bucketDefinitions': [
                    {
                        'accountId': boto3.client('sts').get_caller_identity()['Account'],
                        'buckets': [bucket_name]
                    }
                ]
            },
            'jobType': 'ONE_TIME'
        }
        
        # Add custom data identifiers if provided
        if custom_data_identifiers:
            job_config['customDataIdentifierIds'] = custom_data_identifiers
        
        # Add managed data identifiers if provided
        if managed_data_identifiers:
            job_config['managedDataIdentifierSelector'] = 'INCLUDE'
            job_config['managedDataIdentifierIds'] = managed_data_identifiers
        else:
            # Use all managed data identifiers by default
            job_config['managedDataIdentifierSelector'] = 'ALL'
        
        def create_job_operation():
            response = self.macie_client.create_classification_job(**job_config)
            return response['jobId']
        
        job_id = self._execute_with_retry("create_classification_job", create_job_operation)
        logger.info(f"Created classification job with ID: {job_id}")
        return job_id
    
    def get_findings(self, finding_criteria: Optional[Dict[str, Any]] = None,
                    max_results: int = 100) -> List[MacieFinding]:
        """
        Get Macie findings.
        
        Args:
            finding_criteria: Criteria to filter findings
            max_results: Maximum number of findings to return
            
        Returns:
            List of Macie findings
        """
        logger.info("Getting Macie findings")
        
        # Default criteria if none provided
        if finding_criteria is None:
            finding_criteria = {
                'criterion': {
                    'severity.description': {
                        'gte': 1  # Get all severity levels
                    }
                }
            }
        
        def get_findings_operation():
            # Get finding IDs first
            response = self.macie_client.list_findings(
                findingCriteria=finding_criteria,
                maxResults=max_results
            )
            
            finding_ids = response.get('findingIds', [])
            
            if not finding_ids:
                return []
            
            # Get detailed findings
            findings_response = self.macie_client.get_findings(
                findingIds=finding_ids
            )
            
            findings = []
            for finding_data in findings_response.get('findings', []):
                # Parse finding
                finding = MacieFinding(
                    finding_id=finding_data['id'],
                    severity=finding_data['severity']['description'],
                    type=finding_data['type'],
                    resource_arn=finding_data['resourcesAffected']['s3Object']['arn'],
                    created_at=finding_data['createdAt'].timestamp(),
                    sensitive_data_identifications=finding_data.get('classificationDetails', {}).get('result', {}).get('sensitiveData', []),
                    confidence=finding_data.get('confidence', 0),
                    count=finding_data.get('count', 0),
                    category=finding_data.get('category', 'UNKNOWN'),
                    description=finding_data.get('description', '')
                )
                findings.append(finding)
                
                # Update metrics
                self.metrics["findings_discovered"] += 1
                if finding.sensitive_data_identifications:
                    self.metrics["sensitive_data_found"] += 1
            
            return findings
        
        findings = self._execute_with_retry("get_findings", get_findings_operation)
        logger.info(f"Retrieved {len(findings)} findings")
        return findings
    
    def create_custom_data_identifier(self, name: str, regex: str,
                                    description: str, keywords: Optional[List[str]] = None,
                                    ignore_words: Optional[List[str]] = None) -> str:
        """
        Create a custom data identifier for specific sensitive data patterns.
        
        Args:
            name: Name of the custom data identifier
            regex: Regular expression pattern
            description: Description of what this identifies
            keywords: Keywords that must be nearby
            ignore_words: Words to ignore
            
        Returns:
            Custom data identifier ID
        """
        logger.info(f"Creating custom data identifier: {name}")
        
        identifier_config = {
            'name': name,
            'description': description,
            'regex': regex
        }
        
        if keywords:
            identifier_config['keywords'] = keywords
        
        if ignore_words:
            identifier_config['ignoreWords'] = ignore_words
        
        def create_identifier_operation():
            response = self.macie_client.create_custom_data_identifier(**identifier_config)
            return response['customDataIdentifierId']
        
        identifier_id = self._execute_with_retry("create_custom_data_identifier", create_identifier_operation)
        logger.info(f"Created custom data identifier with ID: {identifier_id}")
        return identifier_id
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Macie operations.
        
        Returns:
            Metrics for Macie operations
        """
        # Calculate average latency
        if self.metrics["operation_latencies"]:
            avg_latency = sum(self.metrics["operation_latencies"]) / len(self.metrics["operation_latencies"])
        else:
            avg_latency = 0
        
        # Calculate success rate
        total_ops = self.metrics["total_operations"]
        if total_ops > 0:
            success_rate = (self.metrics["successful_operations"] / total_ops) * 100
        else:
            success_rate = 0
        
        # Return metrics
        return {
            "total_operations": self.metrics["total_operations"],
            "successful_operations": self.metrics["successful_operations"],
            "failed_operations": self.metrics["failed_operations"],
            "retried_operations": self.metrics["retried_operations"],
            "success_rate": success_rate,
            "average_latency": avg_latency,
            "throttling_count": self.metrics["throttling_count"],
            "findings_discovered": self.metrics["findings_discovered"],
            "sensitive_data_found": self.metrics["sensitive_data_found"]
        }


class AWSMaciePrivacy(PrivacyEngine):
    """Privacy engine that uses AWS Macie for data discovery and classification."""
    
    def __init__(self, region_name: str = "us-east-1", enable_auto_classification: bool = True):
        """
        Initialize AWS Macie privacy engine.
        
        Args:
            region_name: AWS region name
            enable_auto_classification: Enable automatic data classification
        """
        super().__init__()
        
        # Initialize Macie client
        self.macie_client = MacieClient(region_name=region_name)
        
        # Configuration
        self.enable_auto_classification = enable_auto_classification
        
        # Macie-specific tracking
        self.classification_jobs: Dict[str, str] = {}  # bucket -> job_id
        self.sensitivity_mappings: Dict[str, DataSensitivity] = {}  # resource -> sensitivity
        self.macie_findings: List[MacieFinding] = []
        
        # Mapping of Macie severity to privacy levels
        self.severity_to_privacy_level = {
            "LOW": PrivacyLevel.LOW,
            "MEDIUM": PrivacyLevel.MEDIUM,
            "HIGH": PrivacyLevel.HIGH,
            "CRITICAL": PrivacyLevel.TOP_SECRET
        }
        
        # Custom data identifiers for MAIF-specific patterns
        self.custom_identifiers = {}
        self._initialize_custom_identifiers()
    
    def _initialize_custom_identifiers(self):
        """Initialize custom data identifiers for MAIF-specific patterns."""
        try:
            # Create identifier for MAIF artifact IDs
            self.custom_identifiers['maif_artifact_id'] = self.macie_client.create_custom_data_identifier(
                name="MAIF-Artifact-ID",
                regex=r"maif-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
                description="MAIF artifact identifiers",
                keywords=["artifact", "maif", "model", "data"]
            )
            
            # Create identifier for MAIF encryption keys
            self.custom_identifiers['maif_encryption_key'] = self.macie_client.create_custom_data_identifier(
                name="MAIF-Encryption-Key",
                regex=r"-----BEGIN (RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (RSA |EC )?PRIVATE KEY-----",
                description="MAIF encryption keys",
                keywords=["encryption", "key", "private", "secret"]
            )
            
            logger.info("Initialized custom data identifiers for MAIF patterns")
            
        except Exception as e:
            logger.warning(f"Failed to create custom data identifiers: {e}")
    
    def classify_data(self, data: bytes, resource_id: str) -> PrivacyLevel:
        """
        Classify data sensitivity using local analysis.
        
        Args:
            data: Data to classify
            resource_id: Resource identifier
            
        Returns:
            Privacy level based on content analysis
        """
        # Perform quick local classification
        data_str = data.decode('utf-8', errors='ignore').lower()
        
        # Check for highly sensitive patterns
        if any(pattern in data_str for pattern in ['ssn', 'social security', 'credit card', 'password']):
            self.sensitivity_mappings[resource_id] = DataSensitivity.CRITICAL
            return PrivacyLevel.TOP_SECRET
        
        # Check for medium sensitivity patterns
        if any(pattern in data_str for pattern in ['email', 'phone', 'address', 'name']):
            self.sensitivity_mappings[resource_id] = DataSensitivity.HIGH
            return PrivacyLevel.HIGH
        
        # Check for low sensitivity patterns
        if any(pattern in data_str for pattern in ['date', 'time', 'id', 'number']):
            self.sensitivity_mappings[resource_id] = DataSensitivity.MEDIUM
            return PrivacyLevel.MEDIUM
        
        # Default to low sensitivity
        self.sensitivity_mappings[resource_id] = DataSensitivity.LOW
        return PrivacyLevel.LOW
    
    def scan_s3_bucket(self, bucket_name: str, wait_for_completion: bool = False) -> str:
        """
        Scan an S3 bucket for sensitive data using Macie.
        
        Args:
            bucket_name: S3 bucket name to scan
            wait_for_completion: Wait for scan to complete
            
        Returns:
            Classification job ID
        """
        logger.info(f"Starting Macie scan for bucket: {bucket_name}")
        
        # Create classification job
        job_id = self.macie_client.create_classification_job(
            bucket_name=bucket_name,
            job_name=f"maif-privacy-scan-{int(time.time())}",
            custom_data_identifiers=list(self.custom_identifiers.values())
        )
        
        self.classification_jobs[bucket_name] = job_id
        
        if wait_for_completion:
            # Poll for job completion
            while True:
                try:
                    response = self.macie_client.macie_client.describe_classification_job(jobId=job_id)
                    status = response['jobStatus']
                    
                    if status in ['COMPLETE', 'CANCELLED', 'USER_PAUSED']:
                        logger.info(f"Classification job {job_id} completed with status: {status}")
                        break
                    
                    time.sleep(5)  # Wait 5 seconds before next check
                    
                except Exception as e:
                    logger.error(f"Error checking job status: {e}")
                    break
        
        return job_id
    
    def process_macie_findings(self, max_findings: int = 100) -> Dict[str, PrivacyPolicy]:
        """
        Process Macie findings and create privacy policies.
        
        Args:
            max_findings: Maximum number of findings to process
            
        Returns:
            Dictionary mapping resource IDs to privacy policies
        """
        logger.info("Processing Macie findings")
        
        # Get latest findings
        findings = self.macie_client.get_findings(max_results=max_findings)
        self.macie_findings.extend(findings)
        
        # Process each finding
        policies = {}
        
        for finding in findings:
            # Determine privacy level based on severity
            severity = finding.severity.upper()
            privacy_level = self.severity_to_privacy_level.get(severity, PrivacyLevel.MEDIUM)
            
            # Determine encryption mode based on sensitivity
            if severity in ["HIGH", "CRITICAL"]:
                encryption_mode = EncryptionMode.AES_GCM
            else:
                encryption_mode = EncryptionMode.AES_GCM  # Always encrypt
            
            # Create privacy policy
            policy = PrivacyPolicy(
                privacy_level=privacy_level,
                encryption_mode=encryption_mode,
                retention_period=90 if severity == "CRITICAL" else 365,  # Shorter retention for sensitive data
                anonymization_required=severity in ["HIGH", "CRITICAL"],
                audit_required=True,
                geographic_restrictions=["US"] if severity == "CRITICAL" else [],
                purpose_limitation=["authorized_use_only"] if severity in ["HIGH", "CRITICAL"] else []
            )
            
            # Extract resource ID from ARN
            resource_id = finding.resource_arn.split('/')[-1]
            policies[resource_id] = policy
            self.privacy_policies[resource_id] = policy
            
            logger.info(f"Created privacy policy for {resource_id}: {privacy_level.value}")
        
        return policies
    
    def apply_privacy_policy_from_macie(self, data: bytes, resource_id: str) -> Tuple[bytes, PrivacyPolicy]:
        """
        Apply privacy policy based on Macie findings.
        
        Args:
            data: Data to protect
            resource_id: Resource identifier
            
        Returns:
            Protected data and applied policy
        """
        # Check if we have a policy from Macie findings
        if resource_id in self.privacy_policies:
            policy = self.privacy_policies[resource_id]
        else:
            # Perform local classification
            privacy_level = self.classify_data(data, resource_id)
            policy = PrivacyPolicy(
                privacy_level=privacy_level,
                encryption_mode=EncryptionMode.AES_GCM,
                retention_period=365,
                anonymization_required=privacy_level.value in ["high", "top_secret"],
                audit_required=True
            )
            self.privacy_policies[resource_id] = policy
        
        # Apply encryption based on policy
        if policy.encryption_mode != EncryptionMode.NONE:
            encrypted_data, metadata = self.encrypt_data(
                data,
                resource_id,
                encryption_mode=policy.encryption_mode
            )
            
            # Add policy metadata
            metadata['privacy_policy'] = asdict(policy)
            metadata['macie_classification'] = self.sensitivity_mappings.get(resource_id, DataSensitivity.LOW).value
            
            return encrypted_data, policy
        
        return data, policy
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate a compliance report based on Macie findings.
        
        Returns:
            Compliance report with statistics and recommendations
        """
        logger.info("Generating compliance report")
        
        # Analyze findings
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        data_types_found = {}
        affected_resources = set()
        
        for finding in self.macie_findings:
            severity_counts[finding.severity.upper()] += 1
            affected_resources.add(finding.resource_arn)
            
            # Count data types
            for identification in finding.sensitive_data_identifications:
                data_type = identification.get('type', 'UNKNOWN')
                data_types_found[data_type] = data_types_found.get(data_type, 0) + 1
        
        # Generate recommendations
        recommendations = []
        
        if severity_counts["CRITICAL"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "action": "Immediately encrypt or remove critical sensitive data",
                "affected_count": severity_counts["CRITICAL"]
            })
        
        if severity_counts["HIGH"] > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "Review and apply strong encryption to high-sensitivity data",
                "affected_count": severity_counts["HIGH"]
            })
        
        # Calculate compliance score (0-100)
        total_findings = sum(severity_counts.values())
        if total_findings > 0:
            # Weighted score: CRITICAL=-40, HIGH=-20, MEDIUM=-10, LOW=-5
            penalty = (
                severity_counts["CRITICAL"] * 40 +
                severity_counts["HIGH"] * 20 +
                severity_counts["MEDIUM"] * 10 +
                severity_counts["LOW"] * 5
            )
            compliance_score = max(0, 100 - penalty)
        else:
            compliance_score = 100
        
        report = {
            "report_timestamp": time.time(),
            "compliance_score": compliance_score,
            "total_findings": total_findings,
            "severity_breakdown": severity_counts,
            "sensitive_data_types": data_types_found,
            "affected_resources_count": len(affected_resources),
            "recommendations": recommendations,
            "privacy_policies_created": len(self.privacy_policies),
            "classification_jobs_run": len(self.classification_jobs),
            "metrics": self.macie_client.get_metrics()
        }
        
        return report
    
    def enable_continuous_monitoring(self, buckets: List[str], scan_interval: int = 3600):
        """
        Enable continuous monitoring of S3 buckets.
        
        Args:
            buckets: List of S3 bucket names to monitor
            scan_interval: Interval between scans in seconds
        """
        logger.info(f"Enabling continuous monitoring for {len(buckets)} buckets")
        
        # This would typically be implemented with a scheduled Lambda function
        # For demo purposes, we'll just create the initial jobs
        for bucket in buckets:
            try:
                job_id = self.scan_s3_bucket(bucket, wait_for_completion=False)
                logger.info(f"Started monitoring job {job_id} for bucket {bucket}")
            except Exception as e:
                logger.error(f"Failed to start monitoring for bucket {bucket}: {e}")
    
    def get_privacy_insights(self) -> Dict[str, Any]:
        """
        Get privacy insights combining local and Macie data.
        
        Returns:
            Privacy insights and statistics
        """
        # Get Macie metrics
        macie_metrics = self.macie_client.get_metrics()
        
        # Combine with local privacy engine metrics
        insights = {
            "encryption_keys_managed": len(self.encryption_keys),
            "privacy_policies_active": len(self.privacy_policies),
            "access_rules_defined": len(self.access_rules),
            "sensitivity_classifications": {
                sensitivity.value: count 
                for sensitivity, count in self._count_sensitivities().items()
            },
            "macie_integration": {
                "findings_processed": macie_metrics["findings_discovered"],
                "sensitive_data_instances": macie_metrics["sensitive_data_found"],
                "classification_jobs": len(self.classification_jobs),
                "custom_identifiers": len(self.custom_identifiers)
            },
            "recommendations": self._generate_privacy_recommendations()
        }
        
        return insights
    
    def _count_sensitivities(self) -> Dict[DataSensitivity, int]:
        """Count resources by sensitivity level."""
        counts = {sensitivity: 0 for sensitivity in DataSensitivity}
        for sensitivity in self.sensitivity_mappings.values():
            counts[sensitivity] += 1
        return counts
    
    def _generate_privacy_recommendations(self) -> List[Dict[str, str]]:
        """Generate privacy recommendations based on current state."""
        recommendations = []
        
        # Check for unencrypted sensitive data
        unencrypted_sensitive = sum(
            1 for resource_id, sensitivity in self.sensitivity_mappings.items()
            if sensitivity in [DataSensitivity.HIGH, DataSensitivity.CRITICAL]
            and resource_id not in self.encryption_keys
        )
        
        if unencrypted_sensitive > 0:
            recommendations.append({
                "priority": "HIGH",
                "recommendation": f"Encrypt {unencrypted_sensitive} sensitive resources",
                "impact": "Reduces data breach risk"
            })
        
        # Check for missing privacy policies
        resources_without_policies = len(self.sensitivity_mappings) - len(self.privacy_policies)
        if resources_without_policies > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "recommendation": f"Define privacy policies for {resources_without_policies} resources",
                "impact": "Ensures consistent data handling"
            })
        
        return recommendations


# Export all public classes and functions
__all__ = [
    'MacieError',
    'MacieConnectionError',
    'MacieThrottlingError',
    'MacieValidationError',
    'MacieClient',
    'MacieFinding',
    'DataSensitivity',
    'AWSMaciePrivacy'
]
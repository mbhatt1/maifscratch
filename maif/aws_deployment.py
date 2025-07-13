"""
AWS Deployment Tools for MAIF Agents
====================================

Provides CloudFormation templates, CDK constructs, and deployment utilities
for deploying MAIF agents to AWS Lambda, ECS, and other compute services.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import boto3
from dataclasses import dataclass, field

# Import centralized credential and config management
from .aws_config import get_aws_config, AWSConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for MAIF agent deployment."""
    agent_name: str
    agent_class: str
    runtime: str = "python3.9"
    memory_mb: int = 512
    timeout_seconds: int = 300
    environment_vars: Dict[str, str] = field(default_factory=dict)
    vpc_config: Optional[Dict[str, Any]] = None
    iam_role_arn: Optional[str] = None
    s3_bucket: Optional[str] = None
    ecr_repository: Optional[str] = None
    ecs_cluster: Optional[str] = None
    enable_xray: bool = True
    enable_cloudwatch: bool = True
    aws_region: str = "us-east-1"


class CloudFormationGenerator:
    """Generates CloudFormation templates for MAIF agent deployment."""
    
    @staticmethod
    def generate_lambda_template(config: DeploymentConfig) -> Dict[str, Any]:
        """Generate CloudFormation template for Lambda deployment."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"MAIF Agent Lambda Deployment - {config.agent_name}",
            "Parameters": {
                "S3Bucket": {
                    "Type": "String",
                    "Description": "S3 bucket containing the Lambda deployment package",
                    "Default": config.s3_bucket or "maif-agent-deployments"
                },
                "S3Key": {
                    "Type": "String",
                    "Description": "S3 key for the Lambda deployment package",
                    "Default": f"{config.agent_name}/lambda.zip"
                }
            },
            "Resources": {
                "MAIFAgentRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [{
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": "sts:AssumeRole"
                            }]
                        },
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                        ],
                        "Policies": [{
                            "PolicyName": "MAIFAgentPolicy",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:GetObject",
                                            "s3:PutObject",
                                            "s3:ListBucket"
                                        ],
                                        "Resource": ["*"]
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "kms:Decrypt",
                                            "kms:Encrypt",
                                            "kms:GenerateDataKey"
                                        ],
                                        "Resource": ["*"]
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "bedrock:InvokeModel",
                                            "bedrock:InvokeModelWithResponseStream"
                                        ],
                                        "Resource": ["*"]
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "logs:CreateLogGroup",
                                            "logs:CreateLogStream",
                                            "logs:PutLogEvents"
                                        ],
                                        "Resource": ["*"]
                                    },
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "xray:PutTraceSegments",
                                            "xray:PutTelemetryRecords"
                                        ],
                                        "Resource": ["*"]
                                    }
                                ]
                            }
                        }]
                    }
                },
                "MAIFAgentLambda": {
                    "Type": "AWS::Lambda::Function",
                    "Properties": {
                        "FunctionName": config.agent_name,
                        "Runtime": config.runtime,
                        "Handler": f"{config.agent_name}_handler.handler",
                        "Code": {
                            "S3Bucket": {"Ref": "S3Bucket"},
                            "S3Key": {"Ref": "S3Key"}
                        },
                        "MemorySize": config.memory_mb,
                        "Timeout": config.timeout_seconds,
                        "Role": {"Fn::GetAtt": ["MAIFAgentRole", "Arn"]},
                        "Environment": {
                            "Variables": {
                                **config.environment_vars,
                                "MAIF_AGENT_CLASS": config.agent_class,
                                "MAIF_USE_AWS": "true",
                                "AWS_XRAY_TRACING_NAME": config.agent_name
                            }
                        },
                        "TracingConfig": {
                            "Mode": "Active" if config.enable_xray else "PassThrough"
                        }
                    }
                },
                "MAIFAgentLogGroup": {
                    "Type": "AWS::Logs::LogGroup",
                    "Properties": {
                        "LogGroupName": f"/aws/lambda/{config.agent_name}",
                        "RetentionInDays": 30
                    }
                }
            },
            "Outputs": {
                "FunctionArn": {
                    "Description": "ARN of the MAIF Agent Lambda function",
                    "Value": {"Fn::GetAtt": ["MAIFAgentLambda", "Arn"]}
                },
                "FunctionName": {
                    "Description": "Name of the MAIF Agent Lambda function",
                    "Value": {"Ref": "MAIFAgentLambda"}
                }
            }
        }
        
        # Add VPC configuration if specified
        if config.vpc_config:
            template["Resources"]["MAIFAgentLambda"]["Properties"]["VpcConfig"] = config.vpc_config
        
        return template
    
    @staticmethod
    def generate_ecs_template(config: DeploymentConfig) -> Dict[str, Any]:
        """Generate CloudFormation template for ECS deployment."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"MAIF Agent ECS Deployment - {config.agent_name}",
            "Parameters": {
                "ClusterName": {
                    "Type": "String",
                    "Description": "ECS Cluster name",
                    "Default": config.ecs_cluster or "maif-agents"
                },
                "ImageUri": {
                    "Type": "String",
                    "Description": "Docker image URI for the MAIF agent"
                }
            },
            "Resources": {
                "MAIFAgentTaskRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [{
                                "Effect": "Allow",
                                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                                "Action": "sts:AssumeRole"
                            }]
                        },
                        "Policies": [{
                            "PolicyName": "MAIFAgentTaskPolicy",
                            "PolicyDocument": {
                                "Version": "2012-10-17",
                                "Statement": [
                                    {
                                        "Effect": "Allow",
                                        "Action": [
                                            "s3:*",
                                            "kms:*",
                                            "bedrock:*",
                                            "logs:*",
                                            "xray:*"
                                        ],
                                        "Resource": ["*"]
                                    }
                                ]
                            }
                        }]
                    }
                },
                "MAIFAgentTaskDefinition": {
                    "Type": "AWS::ECS::TaskDefinition",
                    "Properties": {
                        "Family": config.agent_name,
                        "RequiresCompatibilities": ["FARGATE"],
                        "NetworkMode": "awsvpc",
                        "Cpu": "512",
                        "Memory": str(config.memory_mb),
                        "TaskRoleArn": {"Ref": "MAIFAgentTaskRole"},
                        "ExecutionRoleArn": {"Ref": "MAIFAgentTaskRole"},
                        "ContainerDefinitions": [{
                            "Name": config.agent_name,
                            "Image": {"Ref": "ImageUri"},
                            "Essential": True,
                            "Environment": [
                                {"Name": k, "Value": v} 
                                for k, v in {
                                    **config.environment_vars,
                                    "MAIF_AGENT_CLASS": config.agent_class,
                                    "MAIF_USE_AWS": "true"
                                }.items()
                            ],
                            "LogConfiguration": {
                                "LogDriver": "awslogs",
                                "Options": {
                                    "awslogs-group": f"/ecs/{config.agent_name}",
                                    "awslogs-region": config.aws_region,
                                    "awslogs-stream-prefix": "ecs"
                                }
                            }
                        }]
                    }
                },
                "MAIFAgentService": {
                    "Type": "AWS::ECS::Service",
                    "Properties": {
                        "ServiceName": config.agent_name,
                        "Cluster": {"Ref": "ClusterName"},
                        "TaskDefinition": {"Ref": "MAIFAgentTaskDefinition"},
                        "DesiredCount": 1,
                        "LaunchType": "FARGATE",
                        "NetworkConfiguration": {
                            "AwsvpcConfiguration": {
                                "AssignPublicIp": "ENABLED",
                                "Subnets": [],  # Must be provided
                                "SecurityGroups": []  # Must be provided
                            }
                        }
                    }
                }
            },
            "Outputs": {
                "ServiceArn": {
                    "Description": "ARN of the MAIF Agent ECS Service",
                    "Value": {"Ref": "MAIFAgentService"}
                }
            }
        }
        
        return template


class LambdaPackager:
    """Creates Lambda deployment packages for MAIF agents."""
    
    @staticmethod
    def create_handler_file(agent_class: str, agent_module: str) -> str:
        """Generate Lambda handler code."""
        handler_code = f'''
import os
import json
import asyncio
import logging
from {agent_module} import {agent_class}
from maif_sdk.aws_backend import AWSConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """AWS Lambda handler for MAIF agent."""
    
    # Configure AWS
    aws_config = AWSConfig(
        region_name=os.environ.get('AWS_REGION', 'us-east-1'),
        s3_bucket=os.environ.get('MAIF_S3_BUCKET', 'maif-agent-data'),
        use_encryption=True,
        use_compliance_logging=True
    )
    
    # Create agent
    agent = {agent_class}(
        agent_id=context.request_id,
        workspace_path="/tmp/agent_workspace",
        use_aws=True,
        aws_config=aws_config
    )
    
    # Run agent asynchronously
    loop = asyncio.get_event_loop()
    
    try:
        # Initialize and run
        loop.run_until_complete(agent.initialize())
        
        # Process event
        if event.get('action') == 'perceive':
            result = loop.run_until_complete(
                agent.perceive(event['data'], event.get('type', 'text'))
            )
        elif event.get('action') == 'reason':
            result = loop.run_until_complete(
                agent.reason(event.get('context', []))
            )
        else:
            # Default action
            result = loop.run_until_complete(agent.run())
        
        # Dump state before shutdown
        dump_path = agent.dump_complete_state()
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'agent_id': agent.agent_id,
                'state_dump': str(dump_path),
                'result': str(result) if result else None
            }})
        }}
        
    except Exception as e:
        logger.error(f"Agent execution failed: {{e}}")
        return {{
            'statusCode': 500,
            'body': json.dumps({{
                'error': str(e)
            }})
        }}
    finally:
        agent.shutdown()
'''
        return handler_code
    
    @staticmethod
    def create_deployment_package(
        agent_class: str,
        agent_module: str,
        output_path: Path,
        include_paths: List[Path] = None
    ) -> Path:
        """Create a Lambda deployment package."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create handler file
        handler_path = output_path / f"{agent_class.lower()}_handler.py"
        handler_path.write_text(
            LambdaPackager.create_handler_file(agent_class, agent_module)
        )
        
        # Create requirements.txt
        requirements = [
            "boto3",
            "aws-xray-sdk",
            "numpy",
            "aiofiles"
        ]
        
        requirements_path = output_path / "requirements.txt"
        requirements_path.write_text("\n".join(requirements))
        
        # Copy MAIF modules
        maif_path = Path(__file__).parent
        shutil.copytree(maif_path, output_path / "maif", dirs_exist_ok=True)
        
        # Copy additional paths
        if include_paths:
            for path in include_paths:
                if path.is_file():
                    shutil.copy2(path, output_path)
                else:
                    shutil.copytree(path, output_path / path.name, dirs_exist_ok=True)
        
        # Create zip file
        zip_path = output_path.parent / f"{agent_class.lower()}_lambda.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(output_path)
                    zf.write(file_path, arcname)
        
        logger.info(f"Lambda package created: {zip_path}")
        return zip_path


class DockerfileGenerator:
    """Generates Dockerfiles for ECS/Fargate deployment."""
    
    @staticmethod
    def generate_dockerfile(agent_class: str, agent_module: str) -> str:
        """Generate Dockerfile for MAIF agent."""
        dockerfile = f'''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MAIF framework
COPY maif/ ./maif/
COPY maif_sdk/ ./maif_sdk/

# Copy agent code
COPY {agent_module.replace('.', '/')}.py .

# Create entrypoint script
RUN echo '#!/usr/bin/env python3\\n\\
import asyncio\\n\\
import os\\n\\
from {agent_module} import {agent_class}\\n\\
from maif_sdk.aws_backend import AWSConfig\\n\\
\\n\\
async def main():\\n\\
    aws_config = AWSConfig(\\n\\
        region_name=os.environ.get("AWS_REGION", "us-east-1"),\\n\\
        s3_bucket=os.environ.get("MAIF_S3_BUCKET", "maif-agent-data")\\n\\
    )\\n\\
    \\n\\
    agent = {agent_class}(\\n\\
        agent_id=os.environ.get("AGENT_ID", "ecs-agent"),\\n\\
        workspace_path="/data",\\n\\
        use_aws=True,\\n\\
        aws_config=aws_config\\n\\
    )\\n\\
    \\n\\
    await agent.initialize()\\n\\
    await agent.run()\\n\\
    agent.shutdown()\\n\\
\\n\\
if __name__ == "__main__":\\n\\
    asyncio.run(main())' > entrypoint.py

RUN chmod +x entrypoint.py

# Create data directory
RUN mkdir /data

# Set entrypoint
ENTRYPOINT ["python3", "entrypoint.py"]
'''
        return dockerfile


class CDKConstructs:
    """AWS CDK constructs for MAIF agent deployment."""
    
    @staticmethod
    def generate_cdk_stack(config: DeploymentConfig) -> str:
        """Generate CDK stack code for agent deployment."""
        cdk_code = f'''
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
    aws_ecs as ecs,
    aws_ecr as ecr,
    Duration,
    RemovalPolicy
)
from constructs import Construct


class MAIFAgentStack(Stack):
    """CDK Stack for deploying MAIF agents."""
    
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        
        # S3 bucket for agent data
        agent_bucket = s3.Bucket(
            self, "AgentDataBucket",
            bucket_name="{config.agent_name}-data",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # IAM role for Lambda
        lambda_role = iam.Role(
            self, "AgentLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ]
        )
        
        # Add permissions
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey",
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords"
            ],
            resources=["*"]
        ))
        
        # Lambda function
        agent_lambda = lambda_.Function(
            self, "AgentLambda",
            function_name="{config.agent_name}",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="{config.agent_name}_handler.handler",
            code=lambda_.Code.from_asset("./lambda_package.zip"),
            memory_size={config.memory_mb},
            timeout=Duration.seconds({config.timeout_seconds}),
            role=lambda_role,
            environment={{
                "MAIF_AGENT_CLASS": "{config.agent_class}",
                "MAIF_USE_AWS": "true",
                "MAIF_S3_BUCKET": agent_bucket.bucket_name,
                **{json.dumps(config.environment_vars)}
            }},
            tracing=lambda_.Tracing.ACTIVE if {config.enable_xray} else lambda_.Tracing.DISABLED
        )
        
        # Log group
        logs.LogGroup(
            self, "AgentLogGroup",
            log_group_name=f"/aws/lambda/{{{config.agent_name}}}",
            retention=logs.RetentionDays.THIRTY_DAYS,
            removal_policy=RemovalPolicy.DESTROY
        )
        
        # ECS resources (optional)
        if "{config.ecs_cluster}":
            # ECS cluster
            cluster = ecs.Cluster(
                self, "AgentCluster",
                cluster_name="{config.ecs_cluster}"
            )
            
            # Task definition
            task_definition = ecs.FargateTaskDefinition(
                self, "AgentTaskDef",
                family="{config.agent_name}",
                cpu=512,
                memory_limit_mib={config.memory_mb}
            )
            
            # Container
            container = task_definition.add_container(
                "AgentContainer",
                image=ecs.ContainerImage.from_registry("{config.ecr_repository}"),
                environment={{
                    "MAIF_AGENT_CLASS": "{config.agent_class}",
                    "MAIF_USE_AWS": "true",
                    "MAIF_S3_BUCKET": agent_bucket.bucket_name,
                    **{json.dumps(config.environment_vars)}
                }},
                logging=ecs.LogDrivers.aws_logs(
                    stream_prefix="{config.agent_name}",
                    log_retention=logs.RetentionDays.THIRTY_DAYS
                )
            )
            
            # ECS service
            ecs.FargateService(
                self, "AgentService",
                cluster=cluster,
                task_definition=task_definition,
                desired_count=1
            )
'''
        return cdk_code


class DeploymentManager:
    """Manages deployment of MAIF agents to AWS."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.cf_generator = CloudFormationGenerator()
        self.lambda_packager = LambdaPackager()
        self.docker_generator = DockerfileGenerator()
        
    def deploy_to_lambda(self, agent_module: str, output_dir: Path = None):
        """Deploy agent to AWS Lambda."""
        if not output_dir:
            output_dir = Path(f"./deployments/{self.config.agent_name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate CloudFormation template
        cf_template = self.cf_generator.generate_lambda_template(self.config)
        cf_path = output_dir / "cloudformation.yaml"
        
        with open(cf_path, 'w') as f:
            yaml.dump(cf_template, f, default_flow_style=False)
        
        # Create Lambda package
        package_dir = output_dir / "lambda_package"
        zip_path = self.lambda_packager.create_deployment_package(
            self.config.agent_class,
            agent_module,
            package_dir
        )
        
        # Upload to S3 if bucket specified
        if self.config.s3_bucket:
            s3_client = self.aws_config.get_client('s3')
            s3_key = f"{self.config.agent_name}/lambda.zip"

            s3_client.upload_file(
                str(zip_path),
                self.config.s3_bucket,
                s3_key
            )

            logger.info(f"Lambda package uploaded to s3://{self.config.s3_bucket}/{s3_key}")

        # Deploy CloudFormation stack
        cf_client = self.aws_config.get_client('cloudformation')
        
        try:
            cf_client.create_stack(
                StackName=f"{self.config.agent_name}-lambda-stack",
                TemplateBody=open(cf_path).read(),
                Parameters=[
                    {"ParameterKey": "S3Bucket", "ParameterValue": self.config.s3_bucket},
                    {"ParameterKey": "S3Key", "ParameterValue": f"{self.config.agent_name}/lambda.zip"}
                ],
                Capabilities=['CAPABILITY_IAM']
            )
            logger.info(f"CloudFormation stack creation initiated: {self.config.agent_name}-lambda-stack")
        except Exception as e:
            logger.error(f"Failed to create CloudFormation stack: {e}")
            raise
        
        return {
            "cloudformation_template": cf_path,
            "lambda_package": zip_path,
            "stack_name": f"{self.config.agent_name}-lambda-stack"
        }
    
    def deploy_to_ecs(self, agent_module: str, output_dir: Path = None):
        """Deploy agent to AWS ECS/Fargate."""
        if not output_dir:
            output_dir = Path(f"./deployments/{self.config.agent_name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Dockerfile
        dockerfile = self.docker_generator.generate_dockerfile(
            self.config.agent_class,
            agent_module
        )
        
        dockerfile_path = output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        
        # Generate CloudFormation template
        cf_template = self.cf_generator.generate_ecs_template(self.config)
        cf_path = output_dir / "ecs_cloudformation.yaml"
        
        with open(cf_path, 'w') as f:
            yaml.dump(cf_template, f, default_flow_style=False)
        
        # Build and push Docker image if ECR repository specified
        if self.config.ecr_repository:
            # Build image
            os.system(f"docker build -t {self.config.agent_name} {output_dir}")
            
            # Tag and push to ECR
            ecr_client = self.aws_config.get_client('ecr')
            
            # Get login token
            token = ecr_client.get_authorization_token()
            
            # Push image
            image_uri = f"{self.config.ecr_repository}:{self.config.agent_name}"
            os.system(f"docker tag {self.config.agent_name} {image_uri}")
            os.system(f"docker push {image_uri}")
            
            logger.info(f"Docker image pushed to {image_uri}")
        
        return {
            "dockerfile": dockerfile_path,
            "cloudformation_template": cf_path,
            "image_uri": image_uri if self.config.ecr_repository else None
        }
    
    def generate_cdk_project(self, output_dir: Path = None):
        """Generate AWS CDK project for agent deployment."""
        if not output_dir:
            output_dir = Path(f"./deployments/{self.config.agent_name}_cdk")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate CDK stack
        cdk_stack = CDKConstructs.generate_cdk_stack(self.config)
        
        # Write stack file
        stack_path = output_dir / "maif_agent_stack.py"
        stack_path.write_text(cdk_stack)
        
        # Generate app.py
        app_code = f'''#!/usr/bin/env python3
import os
from aws_cdk import App
from maif_agent_stack import MAIFAgentStack

app = App()

MAIFAgentStack(
    app, "{self.config.agent_name}-stack",
    env={{
        'account': os.environ['CDK_DEFAULT_ACCOUNT'],
        'region': '{self.config.aws_region}'
    }}
)

app.synth()
'''
        
        app_path = output_dir / "app.py"
        app_path.write_text(app_code)
        os.chmod(app_path, 0o755)
        
        # Generate cdk.json
        cdk_json = {
            "app": "python3 app.py",
            "context": {
                "@aws-cdk/core:enableStackNameDuplicates": True,
                "@aws-cdk/core:stackRelativeExports": True
            }
        }
        
        cdk_json_path = output_dir / "cdk.json"
        with open(cdk_json_path, 'w') as f:
            json.dump(cdk_json, f, indent=2)
        
        # Generate requirements.txt
        requirements = [
            "aws-cdk-lib>=2.0.0",
            "constructs>=10.0.0"
        ]
        
        req_path = output_dir / "requirements.txt"
        req_path.write_text("\n".join(requirements))
        
        logger.info(f"CDK project generated at {output_dir}")
        
        return {
            "project_dir": output_dir,
            "stack_file": stack_path,
            "app_file": app_path
        }


# Convenience functions
def deploy_agent_to_lambda(
    agent_name: str,
    agent_class: str,
    agent_module: str,
    s3_bucket: str,
    **kwargs
) -> Dict[str, Any]:
    """Deploy a MAIF agent to AWS Lambda."""
    config = DeploymentConfig(
        agent_name=agent_name,
        agent_class=agent_class,
        s3_bucket=s3_bucket,
        **kwargs
    )
    
    manager = DeploymentManager(config)
    return manager.deploy_to_lambda(agent_module)


def deploy_agent_to_ecs(
    agent_name: str,
    agent_class: str,
    agent_module: str,
    ecr_repository: str,
    ecs_cluster: str,
    **kwargs
) -> Dict[str, Any]:
    """Deploy a MAIF agent to AWS ECS/Fargate."""
    config = DeploymentConfig(
        agent_name=agent_name,
        agent_class=agent_class,
        ecr_repository=ecr_repository,
        ecs_cluster=ecs_cluster,
        **kwargs
    )
    
    manager = DeploymentManager(config)
    return manager.deploy_to_ecs(agent_module)
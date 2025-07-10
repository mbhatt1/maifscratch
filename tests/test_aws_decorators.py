#!/usr/bin/env python3
"""
Tests for AWS decorators integration with MAIF agentic framework.
"""

import unittest
import asyncio
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import MAIF components
from maif.agentic_framework import MAIFAgent, AgentState
from maif.aws_decorators import (
    maif_agent, aws_agent, aws_bedrock, aws_s3, aws_kms, aws_lambda, aws_dynamodb, aws_step_functions,
    enhance_perception_with_bedrock, enhance_reasoning_with_bedrock, enhance_execution_with_aws,
    enhance_with_step_functions,
    AWSEnhancedPerceptionSystem, AWSEnhancedReasoningSystem, AWSExecutionSystem, StepFunctionsWorkflowSystem
)


class TestAWSDecorators(unittest.TestCase):
    """Test AWS decorator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test workspace
        self.test_workspace = Path("./test_workspace")
        self.test_workspace.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        # In a real test, you might want to remove the test workspace
        # But for safety, we'll leave it in place
        pass
    
    @patch('boto3.Session')
    def test_aws_bedrock_decorator(self, mock_session):
        """Test AWS Bedrock decorator."""
        # Set up mock
        mock_bedrock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_bedrock_client
        
        # Define test class
        @maif_agent(workspace="./test_workspace")
        class TestAgent:
            @aws_bedrock()
            def generate_text(self, prompt, bedrock=None):
                # Use the injected bedrock client
                return bedrock.generate_text_block(prompt)
        
        # Create agent
        agent = TestAgent()
        
        # Call method with decorator
        agent.generate_text("Test prompt")
        
        # Verify bedrock integration was created and used
        self.assertTrue(hasattr(agent, '_bedrock_client'))
        self.assertTrue(hasattr(agent, '_bedrock_integration'))
    
    @patch('boto3.Session')
    def test_aws_s3_decorator(self, mock_session):
        """Test AWS S3 decorator."""
        # Set up mock
        mock_s3_client = MagicMock()
        mock_session.return_value.client.return_value = mock_s3_client
        
        # Define test class
        @maif_agent(workspace="./test_workspace")
        class TestAgent:
            @aws_s3()
            def store_data(self, data, bucket, key, s3_client=None):
                # Use the injected s3 client
                s3_client.put_object(Bucket=bucket, Key=key, Body=data)
                return f"s3://{bucket}/{key}"
        
        # Create agent
        agent = TestAgent()
        
        # Call method with decorator
        result = agent.store_data("test data", "test-bucket", "test-key")
        
        # Verify s3 client was created and used
        self.assertTrue(hasattr(agent, '_s3_client'))
        mock_s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-key", Body="test data"
        )
        self.assertEqual(result, "s3://test-bucket/test-key")
    
    @patch('maif.aws_decorators.BedrockClient')
    def test_enhance_perception_decorator(self, mock_bedrock_client):
        """Test enhance perception decorator."""
        # Define test class
        @maif_agent(workspace="./test_workspace")
        @enhance_perception_with_bedrock()
        class TestAgent:
            pass
        
        # Create agent
        agent = TestAgent()
        
        # Verify perception system was enhanced
        self.assertIsInstance(agent.perception, AWSEnhancedPerceptionSystem)
    
    @patch('maif.aws_decorators.BedrockClient')
    def test_enhance_reasoning_decorator(self, mock_bedrock_client):
        """Test enhance reasoning decorator."""
        # Define test class
        @maif_agent(workspace="./test_workspace")
        @enhance_reasoning_with_bedrock()
        class TestAgent:
            pass
        
        # Create agent
        agent = TestAgent()
        
        # Verify reasoning system was enhanced
        self.assertIsInstance(agent.reasoning, AWSEnhancedReasoningSystem)
    
    @patch('boto3.Session')
    def test_enhance_execution_decorator(self, mock_session):
        """Test enhance execution decorator."""
        # Set up mocks
        mock_session.return_value.client.return_value = MagicMock()
        mock_session.return_value.resource.return_value = MagicMock()
        
        # Define test class
        @maif_agent(workspace="./test_workspace")
        @enhance_execution_with_aws()
        class TestAgent:
            pass
        
        # Create agent
        agent = TestAgent()
        
        # Verify execution system was enhanced
        self.assertIsInstance(agent.execution, AWSExecutionSystem)
        
        # Verify AWS-specific executors were added
        self.assertIn("invoke_lambda", agent.execution.executors)
        self.assertIn("store_in_s3", agent.execution.executors)
        self.assertIn("query_dynamodb", agent.execution.executors)
        self.assertIn("generate_image", agent.execution.executors)
    
    @patch('boto3.Session')
    @patch('maif.aws_decorators.BedrockClient')
    def test_aws_agent_decorator(self, mock_bedrock_client, mock_session):
        """Test AWS agent decorator."""
        # Set up mocks
        mock_session.return_value.client.return_value = MagicMock()
        mock_session.return_value.resource.return_value = MagicMock()
        
        # Define test class
        @aws_agent(workspace="./test_workspace")
        class TestAgent:
            pass
        
        # Create agent
        agent = TestAgent()
        
        # Verify all systems were enhanced
        self.assertIsInstance(agent.perception, AWSEnhancedPerceptionSystem)
        self.assertIsInstance(agent.reasoning, AWSEnhancedReasoningSystem)
        self.assertIsInstance(agent.execution, AWSExecutionSystem)


class TestAWSAgentIntegration(unittest.TestCase):
    """Test AWS agent integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip tests if AWS credentials are not available
        if not os.environ.get('AWS_ACCESS_KEY_ID'):
            self.skipTest("AWS credentials not available")
        
        # Create test workspace
        self.test_workspace = Path("./test_workspace")
        self.test_workspace.mkdir(exist_ok=True)
    
    @unittest.skip("Requires AWS credentials")
    def test_bedrock_text_generation(self):
        """Test Bedrock text generation."""
        # Define test agent
        @maif_agent(workspace="./test_workspace")
        class TestAgent:
            @aws_bedrock()
            def generate_text(self, prompt, bedrock=None):
                return bedrock.generate_text_block(prompt)
        
        # Create agent
        agent = TestAgent()
        
        # Generate text
        result = agent.generate_text("Hello, world!")
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)
    
    @unittest.skip("Requires AWS credentials")
    def test_s3_integration(self):
        """Test S3 integration."""
        # Define test agent
        @maif_agent(workspace="./test_workspace")
        class TestAgent:
            @aws_s3()
            def store_data(self, data, bucket, key, s3_client=None):
                s3_client.put_object(Bucket=bucket, Key=key, Body=data)
                return f"s3://{bucket}/{key}"
            
            @aws_s3()
            def get_data(self, bucket, key, s3_client=None):
                response = s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read().decode('utf-8')
        
        # Create agent
        agent = TestAgent()
        
        # Test bucket and key
        bucket = "your-test-bucket"
        key = "test/data.txt"
        data = "Hello, S3!"
        
        # Store data
        s3_uri = agent.store_data(data, bucket, key)
        
        # Get data
        retrieved_data = agent.get_data(bucket, key)
        
        # Verify data
        self.assertEqual(retrieved_data, data)
    
    @unittest.skip("Requires AWS credentials")
    def test_step_functions_integration(self):
        """Test Step Functions integration."""
        # Define test agent
        @maif_agent(workspace="./test_workspace")
        @enhance_with_step_functions()
        class TestAgent:
            pass
        
        # Create agent
        agent = TestAgent()
        
        # Register workflow
        agent.workflow.register_workflow(
            "test_workflow",
            "arn:aws:states:us-east-1:123456789012:stateMachine:TestWorkflow",
            "Test workflow"
        )
        
        # Verify workflow was registered
        self.assertIn("test_workflow", agent.workflow.workflows)
        self.assertEqual(
            "arn:aws:states:us-east-1:123456789012:stateMachine:TestWorkflow",
            agent.workflow.workflows["test_workflow"]["state_machine_arn"]
        )


if __name__ == '__main__':
    unittest.main()
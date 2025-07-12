"""
AWS API Gateway Integration for MAIF
====================================

Provides HTTP endpoint capabilities for MAIF agents via API Gateway and Lambda.
"""

import json
import base64
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    path: str
    method: str
    handler: Callable
    auth_required: bool = True
    rate_limit: int = 100
    cors_enabled: bool = True
    timeout_seconds: int = 29
    

class APIGatewayIntegration:
    """Integrates MAIF agents with AWS API Gateway."""
    
    def __init__(
        self,
        api_name: str,
        stage_name: str = "prod",
        region: Optional[str] = None
    ):
        self.api_name = api_name
        self.stage_name = stage_name
        self.region = region
        
        # AWS clients
        self.apigateway = boto3.client('apigateway', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        
        # Endpoints registry
        self.endpoints: List[APIEndpoint] = []
        self.api_id: Optional[str] = None
        self.rest_api_url: Optional[str] = None
        
    def add_endpoint(
        self,
        path: str,
        method: str,
        handler: Callable,
        **kwargs
    ):
        """Register an API endpoint."""
        endpoint = APIEndpoint(
            path=path,
            method=method.upper(),
            handler=handler,
            **kwargs
        )
        self.endpoints.append(endpoint)
        logger.info(f"Registered endpoint: {method} {path}")
        
    def create_lambda_handler(self, agent_func: Callable) -> Callable:
        """Create Lambda handler for agent function."""
        def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
            try:
                # Parse request
                body = event.get('body', {})
                if isinstance(body, str):
                    body = json.loads(body)
                    
                headers = event.get('headers', {})
                query_params = event.get('queryStringParameters', {})
                path_params = event.get('pathParameters', {})
                
                # Prepare request context
                request_context = {
                    'headers': headers,
                    'query_params': query_params,
                    'path_params': path_params,
                    'request_id': context.request_id,
                    'source_ip': event.get('requestContext', {}).get('identity', {}).get('sourceIp')
                }
                
                # Call agent function
                result = agent_func(body, request_context)
                
                # Format response
                response = {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'X-Request-Id': context.request_id
                    },
                    'body': json.dumps(result)
                }
                
                # Add CORS headers if enabled
                if hasattr(agent_func, '_cors_enabled') and agent_func._cors_enabled:
                    response['headers'].update({
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization'
                    })
                    
                return response
                
            except Exception as e:
                logger.error(f"Lambda handler error: {e}")
                return {
                    'statusCode': 500,
                    'headers': {
                        'Content-Type': 'application/json',
                        'X-Request-Id': context.request_id
                    },
                    'body': json.dumps({
                        'error': str(e),
                        'request_id': context.request_id
                    })
                }
                
        return lambda_handler
        
    def create_api(self) -> str:
        """Create API Gateway REST API."""
        try:
            # Create REST API
            response = self.apigateway.create_rest_api(
                name=self.api_name,
                description=f"MAIF API Gateway for {self.api_name}",
                endpointConfiguration={'types': ['REGIONAL']}
            )
            
            self.api_id = response['id']
            logger.info(f"Created API Gateway: {self.api_id}")
            
            # Get root resource
            resources = self.apigateway.get_resources(restApiId=self.api_id)
            root_resource_id = resources['items'][0]['id']
            
            # Create resources and methods for each endpoint
            for endpoint in self.endpoints:
                self._create_api_resource(endpoint, root_resource_id)
                
            # Deploy API
            self.deploy_api()
            
            return self.api_id
            
        except ClientError as e:
            logger.error(f"Failed to create API: {e}")
            raise
            
    def _create_api_resource(self, endpoint: APIEndpoint, parent_id: str):
        """Create API Gateway resource and method."""
        try:
            # Create resource path
            path_parts = endpoint.path.strip('/').split('/')
            current_parent = parent_id
            
            for part in path_parts:
                # Check if resource exists
                resources = self.apigateway.get_resources(
                    restApiId=self.api_id,
                    limit=500
                )
                
                existing = None
                for resource in resources['items']:
                    if resource.get('pathPart') == part and resource.get('parentId') == current_parent:
                        existing = resource
                        break
                        
                if existing:
                    current_parent = existing['id']
                else:
                    # Create resource
                    response = self.apigateway.create_resource(
                        restApiId=self.api_id,
                        parentId=current_parent,
                        pathPart=part
                    )
                    current_parent = response['id']
                    
            # Create method
            self.apigateway.put_method(
                restApiId=self.api_id,
                resourceId=current_parent,
                httpMethod=endpoint.method,
                authorizationType='AWS_IAM' if endpoint.auth_required else 'NONE',
                requestParameters={
                    'method.request.header.Authorization': endpoint.auth_required
                }
            )
            
            # Create Lambda integration
            lambda_arn = f"arn:aws:lambda:{self.region}:{{account_id}}:function:{self.api_name}-{endpoint.path.replace('/', '-')}"
            
            self.apigateway.put_integration(
                restApiId=self.api_id,
                resourceId=current_parent,
                httpMethod=endpoint.method,
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations",
                timeoutInMillis=endpoint.timeout_seconds * 1000
            )
            
            # Add method response
            self.apigateway.put_method_response(
                restApiId=self.api_id,
                resourceId=current_parent,
                httpMethod=endpoint.method,
                statusCode='200',
                responseModels={'application/json': 'Empty'}
            )
            
            # Add integration response
            self.apigateway.put_integration_response(
                restApiId=self.api_id,
                resourceId=current_parent,
                httpMethod=endpoint.method,
                statusCode='200',
                responseTemplates={'application/json': ''}
            )
            
            # Add CORS if enabled
            if endpoint.cors_enabled:
                self._enable_cors(current_parent)
                
            logger.info(f"Created API resource: {endpoint.method} {endpoint.path}")
            
        except ClientError as e:
            logger.error(f"Failed to create API resource: {e}")
            raise
            
    def _enable_cors(self, resource_id: str):
        """Enable CORS for a resource."""
        try:
            # Add OPTIONS method
            self.apigateway.put_method(
                restApiId=self.api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                authorizationType='NONE'
            )
            
            # Add mock integration for OPTIONS
            self.apigateway.put_integration(
                restApiId=self.api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                type='MOCK',
                requestTemplates={'application/json': '{"statusCode": 200}'}
            )
            
            # Add OPTIONS method response
            self.apigateway.put_method_response(
                restApiId=self.api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                statusCode='200',
                responseParameters={
                    'method.response.header.Access-Control-Allow-Headers': True,
                    'method.response.header.Access-Control-Allow-Methods': True,
                    'method.response.header.Access-Control-Allow-Origin': True
                }
            )
            
            # Add OPTIONS integration response
            self.apigateway.put_integration_response(
                restApiId=self.api_id,
                resourceId=resource_id,
                httpMethod='OPTIONS',
                statusCode='200',
                responseParameters={
                    'method.response.header.Access-Control-Allow-Headers': "'Content-Type,Authorization'",
                    'method.response.header.Access-Control-Allow-Methods': "'GET,POST,PUT,DELETE,OPTIONS'",
                    'method.response.header.Access-Control-Allow-Origin': "'*'"
                }
            )
            
        except ClientError as e:
            logger.error(f"Failed to enable CORS: {e}")
            
    def deploy_api(self):
        """Deploy API to stage."""
        try:
            deployment = self.apigateway.create_deployment(
                restApiId=self.api_id,
                stageName=self.stage_name,
                description=f"Deployment at {datetime.utcnow().isoformat()}"
            )
            
            self.rest_api_url = f"https://{self.api_id}.execute-api.{self.region}.amazonaws.com/{self.stage_name}"
            logger.info(f"API deployed to: {self.rest_api_url}")
            
            # Set stage variables
            self.apigateway.update_stage(
                restApiId=self.api_id,
                stageName=self.stage_name,
                patchOperations=[
                    {
                        'op': 'replace',
                        'path': '/throttle/rateLimit',
                        'value': '1000'
                    },
                    {
                        'op': 'replace',
                        'path': '/throttle/burstLimit',
                        'value': '2000'
                    }
                ]
            )
            
        except ClientError as e:
            logger.error(f"Failed to deploy API: {e}")
            raise
            
    def generate_sdk(self, sdk_type: str = "javascript") -> bytes:
        """Generate client SDK for the API."""
        try:
            response = self.apigateway.get_sdk(
                restApiId=self.api_id,
                stageName=self.stage_name,
                sdkType=sdk_type
            )
            
            return response['body'].read()
            
        except ClientError as e:
            logger.error(f"Failed to generate SDK: {e}")
            raise


def api_endpoint(
    path: str,
    method: str = "POST",
    auth_required: bool = True,
    rate_limit: int = 100,
    cors_enabled: bool = True
):
    """Decorator to expose agent function as API endpoint."""
    def decorator(func):
        # Add metadata
        func._api_path = path
        func._api_method = method
        func._auth_required = auth_required
        func._rate_limit = rate_limit
        func._cors_enabled = cors_enabled
        
        return func
    return decorator


class APIGatewayHandler:
    """Handler for API Gateway Lambda integration."""
    
    def __init__(self, agent):
        self.agent = agent
        self.endpoints = self._discover_endpoints()
        
    def _discover_endpoints(self) -> Dict[str, Callable]:
        """Discover API endpoints from agent methods."""
        endpoints = {}
        
        for attr_name in dir(self.agent):
            attr = getattr(self.agent, attr_name)
            if callable(attr) and hasattr(attr, '_api_path'):
                key = f"{attr._api_method}:{attr._api_path}"
                endpoints[key] = attr
                
        return endpoints
        
    def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Handle API Gateway request."""
        # Extract method and path
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        # Find matching endpoint
        key = f"{method}:{path}"
        handler = self.endpoints.get(key)
        
        if not handler:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': 'Endpoint not found'})
            }
            
        # Check authentication if required
        if handler._auth_required:
            auth_header = event.get('headers', {}).get('Authorization')
            if not auth_header:
                return {
                    'statusCode': 401,
                    'body': json.dumps({'error': 'Authorization required'})
                }
                
        # Parse request body
        body = event.get('body', '{}')
        if event.get('isBase64Encoded'):
            body = base64.b64decode(body).decode('utf-8')
            
        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid JSON'})
            }
            
        # Call handler
        try:
            result = handler(request_data)
            
            response = {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(result)
            }
            
            # Add CORS headers
            if handler._cors_enabled:
                response['headers'].update({
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization'
                })
                
            return response
            
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e),
                    'request_id': context.request_id
                })
            }


# Usage example:
"""
from maif.agentic_framework import MAIFAgent

class MyAPIAgent(MAIFAgent):
    
    @api_endpoint("/analyze", method="POST", auth_required=True)
    def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Process data
        result = self.reasoning_system.process(data)
        return {"analysis": result}
        
    @api_endpoint("/status", method="GET", auth_required=False)
    def get_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "agent_id": self.agent_id,
            "uptime": self.get_uptime()
        }

# Deploy to API Gateway
agent = MyAPIAgent()
api_handler = APIGatewayHandler(agent)

# Lambda handler
def lambda_handler(event, context):
    return api_handler.handle_request(event, context)

# Or use integration class
api = APIGatewayIntegration("my-maif-api")
api.add_endpoint("/analyze", "POST", agent.analyze_data)
api.add_endpoint("/status", "GET", agent.get_status)
api.create_api()
"""
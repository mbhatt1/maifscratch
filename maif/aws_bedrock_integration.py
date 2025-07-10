"""
AWS Bedrock Integration for MAIF
================================

Provides integration with AWS Bedrock for AI model inference, embedding generation,
and semantic operations.
"""

import json
import time
import hashlib
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
import boto3
from botocore.exceptions import ClientError
import numpy as np


class BedrockModelProvider:
    """Enum-like class for Bedrock model providers."""
    AMAZON = "amazon"
    ANTHROPIC = "anthropic"
    AI21 = "ai21"
    COHERE = "cohere"
    STABILITY = "stability"


class BedrockModelType:
    """Enum-like class for Bedrock model types."""
    TEXT = "text"
    EMBEDDING = "embedding"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class BedrockClient:
    """Client for AWS Bedrock service."""
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None):
        """
        Initialize Bedrock client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
        """
        # Initialize AWS session
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.bedrock_client = session.client('bedrock-runtime')
        
        # Cache for model info
        self.model_info_cache: Dict[str, Dict[str, Any]] = {}
        
        # Request history for auditing
        self.request_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Bedrock models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Use bedrock client to list models
            # Note: In the actual AWS SDK, this would be a different client (bedrock, not bedrock-runtime)
            bedrock_client = boto3.client('bedrock')
            response = bedrock_client.list_foundation_models()
            return response.get('modelSummaries', [])
        except ClientError as e:
            print(f"Error listing Bedrock models: {e}")
            return []
    
    def _record_request(self, model_id: str, request_type: str, input_data: Any, output_data: Any):
        """Record request for auditing."""
        # Create request record
        request_record = {
            "timestamp": time.time(),
            "model_id": model_id,
            "request_type": request_type,
            "input_hash": hashlib.sha256(str(input_data).encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(str(output_data).encode()).hexdigest()[:16]
        }
        
        # Add to history
        self.request_history.append(request_record)
        
        # Limit history size
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
    
    def invoke_text_model(self, model_id: str, prompt: str, 
                         max_tokens: int = 1000, 
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> Optional[str]:
        """
        Invoke a text generation model.
        
        Args:
            model_id: Bedrock model ID
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text if successful, None otherwise
        """
        try:
            # Prepare request body based on model provider
            if "anthropic" in model_id.lower():
                # Anthropic Claude models
                request_body = {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            elif "amazon.titan" in model_id.lower():
                # Amazon Titan models
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": top_p
                    }
                }
            elif "ai21" in model_id.lower():
                # AI21 Jurassic models
                request_body = {
                    "prompt": prompt,
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p
                }
            elif "cohere" in model_id.lower():
                # Cohere models
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "p": top_p
                }
            else:
                # Default format
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response based on model provider
            response_body = json.loads(response['body'].read())
            
            if "anthropic" in model_id.lower():
                result = response_body.get('completion', '')
            elif "amazon.titan" in model_id.lower():
                result = response_body.get('results', [{}])[0].get('outputText', '')
            elif "ai21" in model_id.lower():
                result = response_body.get('completions', [{}])[0].get('data', {}).get('text', '')
            elif "cohere" in model_id.lower():
                result = response_body.get('generations', [{}])[0].get('text', '')
            else:
                result = str(response_body)
            
            # Record request
            self._record_request(model_id, "text_generation", prompt, result)
            
            return result
        except ClientError as e:
            print(f"Error invoking Bedrock model: {e}")
            return None
    
    def generate_embeddings(self, model_id: str, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for texts.
        
        Args:
            model_id: Bedrock embedding model ID
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors if successful, None otherwise
        """
        try:
            # Prepare request body based on model provider
            if "cohere" in model_id.lower():
                # Cohere embedding models
                request_body = {
                    "texts": texts,
                    "input_type": "search_document"
                }
            elif "amazon.titan" in model_id.lower():
                # Amazon Titan embedding models
                request_body = {
                    "inputText": texts[0] if len(texts) == 1 else texts
                }
            else:
                # Default format
                request_body = {
                    "input": texts
                }
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response based on model provider
            response_body = json.loads(response['body'].read())
            
            if "cohere" in model_id.lower():
                embeddings = response_body.get('embeddings', [])
            elif "amazon.titan" in model_id.lower():
                if isinstance(response_body.get('embedding', []), list):
                    embeddings = [response_body.get('embedding', [])]
                else:
                    embeddings = response_body.get('embeddings', [])
            else:
                embeddings = response_body.get('embeddings', [])
            
            # Record request
            self._record_request(model_id, "embedding_generation", texts, "embeddings")
            
            return embeddings
        except ClientError as e:
            print(f"Error generating embeddings: {e}")
            return None
    
    def generate_image(self, model_id: str, prompt: str, 
                      width: int = 1024, height: int = 1024,
                      num_images: int = 1) -> Optional[List[bytes]]:
        """
        Generate images using Bedrock image models.
        
        Args:
            model_id: Bedrock image model ID
            prompt: Text prompt
            width: Image width
            height: Image height
            num_images: Number of images to generate
            
        Returns:
            List of image data (bytes) if successful, None otherwise
        """
        try:
            # Prepare request body based on model provider
            if "stability" in model_id.lower():
                # Stability AI models
                request_body = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "steps": 30,
                    "width": width,
                    "height": height,
                    "seed": int(time.time()) % 2147483647,
                    "samples": num_images
                }
            else:
                # Default format
                request_body = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_images": num_images
                }
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response based on model provider
            response_body = json.loads(response['body'].read())
            
            if "stability" in model_id.lower():
                images = []
                for artifact in response_body.get('artifacts', []):
                    if artifact.get('finishReason') == 'SUCCESS':
                        image_data = base64.b64decode(artifact.get('base64', ''))
                        images.append(image_data)
            else:
                images = []
                for image_b64 in response_body.get('images', []):
                    image_data = base64.b64decode(image_b64)
                    images.append(image_data)
            
            # Record request
            self._record_request(model_id, "image_generation", prompt, f"{len(images)} images")
            
            return images
        except ClientError as e:
            print(f"Error generating images: {e}")
            return None


class MAIFBedrockIntegration:
    """Integration between MAIF and AWS Bedrock."""
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize MAIF Bedrock integration.
        
        Args:
            bedrock_client: Bedrock client
        """
        self.bedrock_client = bedrock_client
        
        # Default models
        self.default_text_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.default_embedding_model = "amazon.titan-embed-text-v1"
        self.default_image_model = "stability.stable-diffusion-xl-v1"
        
        # Embedding cache
        self.embedding_cache: Dict[str, List[float]] = {}
    
    def set_default_models(self, text_model: Optional[str] = None, 
                          embedding_model: Optional[str] = None,
                          image_model: Optional[str] = None):
        """Set default models for different operations."""
        if text_model:
            self.default_text_model = text_model
        if embedding_model:
            self.default_embedding_model = embedding_model
        if image_model:
            self.default_image_model = image_model
    
    def generate_text_block(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text using Bedrock and format as MAIF block.
        
        Args:
            prompt: Text prompt
            metadata: Optional metadata
            
        Returns:
            Dictionary with text and metadata
        """
        # Generate text
        text = self.bedrock_client.invoke_text_model(
            self.default_text_model,
            prompt
        )
        
        if not text:
            text = "Error generating text"
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "source": "bedrock",
            "model": self.default_text_model,
            "prompt": prompt,
            "timestamp": time.time()
        })
        
        # Generate embeddings
        embeddings = self.bedrock_client.generate_embeddings(
            self.default_embedding_model,
            [text]
        )
        
        if embeddings and len(embeddings) > 0:
            metadata["embedding"] = embeddings[0]
        
        return {
            "text": text,
            "metadata": metadata
        }
    
    def generate_image_block(self, prompt: str, metadata: Optional[Dict[str, Any]] = None,
                            width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Generate image using Bedrock and format as MAIF block.
        
        Args:
            prompt: Text prompt
            metadata: Optional metadata
            width: Image width
            height: Image height
            
        Returns:
            Dictionary with image data and metadata
        """
        # Generate image
        images = self.bedrock_client.generate_image(
            self.default_image_model,
            prompt,
            width=width,
            height=height,
            num_images=1
        )
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "source": "bedrock",
            "model": self.default_image_model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "timestamp": time.time()
        })
        
        # Generate text embedding for the prompt
        embeddings = self.bedrock_client.generate_embeddings(
            self.default_embedding_model,
            [prompt]
        )
        
        if embeddings and len(embeddings) > 0:
            metadata["prompt_embedding"] = embeddings[0]
        
        return {
            "image_data": images[0] if images and len(images) > 0 else None,
            "metadata": metadata
        }
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embeddings for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector if successful, None otherwise
        """
        # Check cache
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate embedding
        embeddings = self.bedrock_client.generate_embeddings(
            self.default_embedding_model,
            [text]
        )
        
        if embeddings and len(embeddings) > 0:
            # Cache embedding
            self.embedding_cache[text_hash] = embeddings[0]
            return embeddings[0]
        
        return None
    
    def semantic_search(self, query: str, texts: List[str]) -> List[Tuple[int, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            texts: List of texts to search
            
        Returns:
            List of (index, score) tuples sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        
        # Generate embeddings for texts
        text_embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            if embedding:
                text_embeddings.append(embedding)
            else:
                text_embeddings.append([0.0] * len(query_embedding))
        
        # Calculate cosine similarity
        query_embedding_np = np.array(query_embedding)
        results = []
        for i, embedding in enumerate(text_embeddings):
            embedding_np = np.array(embedding)
            similarity = np.dot(query_embedding_np, embedding_np) / (
                np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding_np)
            )
            results.append((i, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def analyze_image(self, image_data: bytes, prompt: str = "Describe this image in detail.") -> str:
        """
        Analyze image using multimodal model.
        
        Args:
            image_data: Image data
            prompt: Text prompt
            
        Returns:
            Analysis text
        """
        try:
            # Encode image as base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare request for Claude 3 Sonnet Vision
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            # Invoke model
            response = self.bedrock_client.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            result = response_body.get('content', [{}])[0].get('text', '')
            
            # Record request
            self.bedrock_client._record_request(
                "anthropic.claude-3-sonnet-20240229-v1:0", 
                "image_analysis", 
                prompt, 
                result
            )
            
            return result
        except ClientError as e:
            print(f"Error analyzing image: {e}")
            return "Error analyzing image"


# Helper functions for easy integration
def create_bedrock_integration(region_name: str = "us-east-1", 
                             profile_name: Optional[str] = None) -> MAIFBedrockIntegration:
    """
    Create MAIF Bedrock integration.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        
    Returns:
        MAIFBedrockIntegration
    """
    bedrock_client = BedrockClient(region_name=region_name, profile_name=profile_name)
    return MAIFBedrockIntegration(bedrock_client)
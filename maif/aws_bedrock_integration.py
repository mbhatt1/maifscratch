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
import logging
import datetime
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError
import numpy as np

# Import centralized credential and config management
from .aws_config import get_aws_config, AWSConfig


# Configure logger
logger = logging.getLogger(__name__)


class BedrockError(Exception):
    """Base exception for Bedrock integration errors."""
    pass


class BedrockConnectionError(BedrockError):
    """Exception for Bedrock connection errors."""
    pass


class BedrockThrottlingError(BedrockError):
    """Exception for Bedrock throttling errors."""
    pass


class BedrockValidationError(BedrockError):
    """Exception for Bedrock validation errors."""
    pass


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
    
    # Set of known transient errors that can be retried
    RETRYABLE_ERRORS = {
        'ThrottlingException', 
        'Throttling', 
        'RequestLimitExceeded',
        'TooManyRequestsException',
        'ServiceUnavailable',
        'InternalServerError',
        'InternalFailure',
        'ServiceFailure'
    }
    
    def __init__(self, region_name: Optional[str] = None, profile_name: Optional[str] = None,
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0,
                aws_config: Optional[AWSConfig] = None):
        """
        Initialize Bedrock client.
        
        Args:
            region_name: AWS region name (deprecated, use aws_config)
            profile_name: AWS profile name (deprecated, use aws_config)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            aws_config: Centralized AWS configuration (preferred)
        
        Raises:
            BedrockConnectionError: If unable to initialize the Bedrock client
        """
        # Validate inputs
        if max_retries < 0:
            raise BedrockValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise BedrockValidationError("base_delay and max_delay must be positive")
        
        # Use provided config or get global config
        self.aws_config = aws_config or get_aws_config()
        
        # Handle deprecated parameters for backward compatibility
        if region_name or profile_name:
            logger.warning("region_name and profile_name parameters are deprecated. Use aws_config instead.")
            if not aws_config:
                # Create a temporary config if using deprecated params
                from .aws_credentials import configure_aws_credentials
                cred_manager = configure_aws_credentials(
                    profile_name=profile_name,
                    region_name=region_name
                )
                self.aws_config = AWSConfig(credential_manager=cred_manager)
        
        # Initialize AWS clients using centralized config
        try:
            logger.info(f"Initializing Bedrock client in region {self.aws_config.credential_manager.region_name}")
            
            # Get bedrock-specific configuration
            bedrock_runtime_config = self.aws_config.get_service_config('bedrock-runtime')
            bedrock_config = self.aws_config.get_service_config('bedrock')
            
            # Initialize both bedrock and bedrock-runtime clients
            self.bedrock_client = self.aws_config.get_client('bedrock-runtime', config=bedrock_runtime_config)
            self.bedrock_management_client = self.aws_config.get_client('bedrock', config=bedrock_config)
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Cache for model info
            self.model_info_cache: Dict[str, Dict[str, Any]] = {}
            self.model_cache_expiry = time.time() + 3600  # Cache expires after 1 hour
            
            # Request history for auditing
            self.request_history: List[Dict[str, Any]] = []
            self.max_history_size = 1000
            
            # Metrics for monitoring
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "retried_requests": 0,
                "request_latencies": []
            }
            
            # Thread safety
            self._lock = threading.RLock()
            
            logger.info("Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}", exc_info=True)
            raise BedrockConnectionError(f"Failed to initialize Bedrock client: {e}")
    
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
            BedrockError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        with self._lock:
            self.metrics["total_requests"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                with self._lock:
                    self.metrics["successful_requests"] += 1
                    latency = time.time() - start_time
                    self.metrics["request_latencies"].append(latency)
                    
                    # Trim latencies list if it gets too large
                    if len(self.metrics["request_latencies"]) > 1000:
                        self.metrics["request_latencies"] = self.metrics["request_latencies"][-1000:]
                
                logger.debug(f"{operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_requests"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"{operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    with self._lock:
                        self.metrics["failed_requests"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"{operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise BedrockThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    else:
                        logger.error(
                            f"{operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise BedrockError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_requests"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"{operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    with self._lock:
                        self.metrics["failed_requests"] += 1
                    logger.error(f"{operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise BedrockConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                with self._lock:
                    self.metrics["failed_requests"] += 1
                logger.error(f"Unexpected error in {operation_name}: {str(e)}", exc_info=True)
                raise BedrockError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise BedrockError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise BedrockError(f"Operation failed after {retries} retries")
    
    def _record_request(self, model_id: str, request_type: str, input_data: Any, output_data: Any):
        """
        Record request for auditing.
        
        Args:
            model_id: Bedrock model ID
            request_type: Type of request
            input_data: Input data
            output_data: Output data
        """
        # Create request record with more detailed information
        request_record = {
            "timestamp": time.time(),
            "iso_time": datetime.datetime.now().isoformat(),
            "model_id": model_id,
            "request_type": request_type,
            "input_hash": hashlib.sha256(str(input_data).encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(str(output_data).encode()).hexdigest()[:16],
            "success": True
        }
        
        # Add to history with thread safety
        with self._lock:
            self.request_history.append(request_record)
            
            # Limit history size
            if len(self.request_history) > self.max_history_size:
                self.request_history = self.request_history[-self.max_history_size:]
        
        logger.debug(f"Recorded {request_type} request for model {model_id}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Bedrock models.
        
        Returns:
            List of model information dictionaries
            
        Raises:
            BedrockError: If an error occurs while listing models
        """
        # Check if cache is valid
        with self._lock:
            if self.model_info_cache and time.time() < self.model_cache_expiry:
                logger.debug("Returning cached model list")
                return list(self.model_info_cache.values())
        
        logger.info("Fetching available Bedrock models")
        
        # Use the management client to list models
        def list_models_operation():
            response = self.bedrock_management_client.list_foundation_models()
            return response.get('modelSummaries', [])
        
        models = self._execute_with_retry("list_models", list_models_operation)
        
        # Update cache
        with self._lock:
            self.model_info_cache = {model['modelId']: model for model in models}
            self.model_cache_expiry = time.time() + 3600  # Cache expires after 1 hour
        
        logger.info(f"Found {len(models)} available Bedrock models")
        return models
    
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
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during model invocation
        """
        # Validate inputs
        if not model_id:
            raise BedrockValidationError("model_id cannot be empty")
        
        if not prompt:
            raise BedrockValidationError("prompt cannot be empty")
        
        if max_tokens <= 0:
            raise BedrockValidationError("max_tokens must be positive")
        
        if temperature < 0 or temperature > 1:
            raise BedrockValidationError("temperature must be between 0 and 1")
        
        if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
            raise BedrockValidationError("top_p must be between 0 and 1 (exclusive)")
        
        logger.info(f"Invoking text model {model_id}")
        logger.debug(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
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
        
        # Define the invoke operation
        def invoke_operation():
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
            
            return result
        
        # Execute with retry
        result = self._execute_with_retry("invoke_text_model", invoke_operation)
        
        # Record request
        self._record_request(model_id, "text_generation", prompt, result)
        
        logger.info(f"Successfully generated text with model {model_id}")
        logger.debug(f"Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        
        return result
    
    def generate_embeddings(self, model_id: str, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for texts.
        
        Args:
            model_id: Bedrock embedding model ID
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors if successful, None otherwise
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during embedding generation
        """
        # Validate inputs
        if not model_id:
            raise BedrockValidationError("model_id cannot be empty")
        
        if not texts:
            raise BedrockValidationError("texts cannot be empty")
        
        if not isinstance(texts, list):
            raise BedrockValidationError("texts must be a list")
        
        # Validate each text
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise BedrockValidationError(f"texts[{i}] must be a string")
        
        logger.info(f"Generating embeddings with model {model_id} for {len(texts)} texts")
        
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
        
        # Define the embedding operation
        def embedding_operation():
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
            
            return embeddings
        
        # Execute with retry
        embeddings = self._execute_with_retry("generate_embeddings", embedding_operation)
        
        # Record request
        self._record_request(model_id, "embedding_generation", texts, "embeddings")
        
        logger.info(f"Successfully generated embeddings for {len(texts)} texts")
        
        return embeddings
    
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
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during image generation
        """
        # Validate inputs
        if not model_id:
            raise BedrockValidationError("model_id cannot be empty")
        
        if not prompt:
            raise BedrockValidationError("prompt cannot be empty")
        
        if width <= 0 or height <= 0:
            raise BedrockValidationError("width and height must be positive")
        
        if num_images <= 0:
            raise BedrockValidationError("num_images must be positive")
        
        logger.info(f"Generating {num_images} images with model {model_id}")
        logger.debug(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
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
        
        # Define the image generation operation
        def image_generation_operation():
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
            
            return images
        
        # Execute with retry
        images = self._execute_with_retry("generate_image", image_generation_operation)
        
        # Record request
        self._record_request(model_id, "image_generation", prompt, f"{len(images)} images")
        
        logger.info(f"Successfully generated {len(images)} images")
        
        return images


class MAIFBedrockIntegration:
    """Integration between MAIF and AWS Bedrock."""
    
    def __init__(self, bedrock_client: BedrockClient, 
                embedding_cache_ttl: int = 3600,
                embedding_cache_max_size: int = 1000):
        """
        Initialize MAIF Bedrock integration.
        
        Args:
            bedrock_client: Bedrock client
            embedding_cache_ttl: Time-to-live for embedding cache entries (seconds)
            embedding_cache_max_size: Maximum number of entries in embedding cache
            
        Raises:
            BedrockValidationError: If input validation fails
        """
        # Validate inputs
        if not bedrock_client:
            raise BedrockValidationError("bedrock_client cannot be None")
        
        if embedding_cache_ttl <= 0:
            raise BedrockValidationError("embedding_cache_ttl must be positive")
        
        if embedding_cache_max_size <= 0:
            raise BedrockValidationError("embedding_cache_max_size must be positive")
        
        logger.info("Initializing MAIF Bedrock integration")
        
        self.bedrock_client = bedrock_client
        
        # Default models
        self.default_text_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.default_embedding_model = "amazon.titan-embed-text-v1"
        self.default_image_model = "stability.stable-diffusion-xl-v1"
        self.default_multimodal_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        # Embedding cache with TTL
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache_ttl = embedding_cache_ttl
        self.embedding_cache_max_size = embedding_cache_max_size
        
        # Metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("MAIF Bedrock integration initialized successfully")
    
    def set_default_models(self, text_model: Optional[str] = None, 
                          embedding_model: Optional[str] = None,
                          image_model: Optional[str] = None,
                          multimodal_model: Optional[str] = None):
        """
        Set default models for different operations.
        
        Args:
            text_model: Default text generation model
            embedding_model: Default embedding model
            image_model: Default image generation model
            multimodal_model: Default multimodal model
        """
        if text_model:
            logger.info(f"Setting default text model to {text_model}")
            self.default_text_model = text_model
        
        if embedding_model:
            logger.info(f"Setting default embedding model to {embedding_model}")
            self.default_embedding_model = embedding_model
        
        if image_model:
            logger.info(f"Setting default image model to {image_model}")
            self.default_image_model = image_model
            
        if multimodal_model:
            logger.info(f"Setting default multimodal model to {multimodal_model}")
            self.default_multimodal_model = multimodal_model
    
    def _clean_embedding_cache(self):
        """Clean expired entries from embedding cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, value in self.embedding_cache.items()
                if current_time > value.get("expiry", 0)
            ]
            
            for key in expired_keys:
                del self.embedding_cache[key]
                self.metrics["cache_evictions"] += 1
            
            # If cache is still too large, remove oldest entries
            if len(self.embedding_cache) > self.embedding_cache_max_size:
                # Sort by access time (oldest first)
                sorted_items = sorted(
                    self.embedding_cache.items(),
                    key=lambda x: x[1].get("last_access", 0)
                )
                
                # Remove oldest entries
                num_to_remove = len(self.embedding_cache) - self.embedding_cache_max_size
                for key, _ in sorted_items[:num_to_remove]:
                    del self.embedding_cache[key]
                    self.metrics["cache_evictions"] += 1
            
            logger.debug(f"Cleaned embedding cache: {len(expired_keys)} expired entries removed")
    
    def generate_text_block(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate text using Bedrock and format as MAIF block.
        
        Args:
            prompt: Text prompt
            metadata: Optional metadata
            
        Returns:
            Dictionary with text and metadata
            
        Raises:
            BedrockError: If an error occurs during text generation
        """
        # Validate inputs
        if not prompt:
            raise BedrockValidationError("prompt cannot be empty")
        
        logger.info("Generating text block")
        
        # Generate text
        try:
            text = self.bedrock_client.invoke_text_model(
                self.default_text_model,
                prompt
            )
            
            if text is None:
                raise BedrockError("Text generation returned None")
                
        except BedrockError as e:
            logger.error(f"Failed to generate text: {e}")
            text = f"Error generating text: {str(e)}"
        
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
        try:
            if text and isinstance(text, str):
                embeddings = self.bedrock_client.generate_embeddings(
                    self.default_embedding_model,
                    [text]
                )
                
                if embeddings and len(embeddings) > 0 and embeddings[0]:
                    metadata["embedding"] = embeddings[0]
        except BedrockError as e:
            logger.warning(f"Failed to generate embeddings for text block: {e}")
        
        logger.info("Text block generated successfully")
        
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
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during image generation
        """
        # Validate inputs
        if not prompt:
            raise BedrockValidationError("prompt cannot be empty")
        
        if width <= 0 or height <= 0:
            raise BedrockValidationError("width and height must be positive")
        
        logger.info(f"Generating image block ({width}x{height})")
        
        # Generate image
        try:
            images = self.bedrock_client.generate_image(
                self.default_image_model,
                prompt,
                width=width,
                height=height,
                num_images=1
            )
            
            if not images or len(images) == 0:
                raise BedrockError("No images generated")
            
            image_data = images[0]
            
            if not image_data:
                raise BedrockError("Generated image data is empty")
                
        except BedrockError as e:
            logger.error(f"Failed to generate image: {e}")
            # Re-raise the exception for proper error handling
            raise
        
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
        try:
            embeddings = self.bedrock_client.generate_embeddings(
                self.default_embedding_model,
                [prompt]
            )
            
            if embeddings and len(embeddings) > 0:
                metadata["prompt_embedding"] = embeddings[0]
        except BedrockError as e:
            logger.warning(f"Failed to generate embeddings for image prompt: {e}")
        
        logger.info("Image block generated successfully")
        
        return {
            "image_data": image_data,
            "metadata": metadata
        }
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embeddings for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector if successful, None otherwise
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during embedding generation
        """
        # Validate inputs
        if not text:
            raise BedrockValidationError("text cannot be empty")
        
        # Check cache
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Clean cache periodically
        self._clean_embedding_cache()
        
        # Check if in cache and not expired
        if text_hash in self.embedding_cache:
            cache_entry = self.embedding_cache[text_hash]
            current_time = time.time()
            
            if current_time <= cache_entry.get("expiry", 0):
                # Update last access time
                self.embedding_cache[text_hash]["last_access"] = current_time
                self.metrics["cache_hits"] += 1
                logger.debug(f"Embedding cache hit for text hash {text_hash[:8]}")
                return cache_entry.get("embedding")
        
        # Cache miss or expired
        self.metrics["cache_misses"] += 1
        logger.debug(f"Embedding cache miss for text hash {text_hash[:8]}")
        
        # Generate embedding
        try:
            embeddings = self.bedrock_client.generate_embeddings(
                self.default_embedding_model,
                [text]
            )
            
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                
                # Cache embedding with TTL
                self.embedding_cache[text_hash] = {
                    "embedding": embedding,
                    "expiry": time.time() + self.embedding_cache_ttl,
                    "last_access": time.time()
                }
                
                logger.debug(f"Cached embedding for text hash {text_hash[:8]}")
                return embedding
            else:
                logger.warning("Received empty embeddings from Bedrock")
                return None
                
        except BedrockError as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def analyze_image(self, image_data: bytes, prompt: str = "Describe this image in detail.",
                     model_id: Optional[str] = None) -> str:
        """
        Analyze image using multimodal model.
        
        Args:
            image_data: Image data
            prompt: Text prompt
            model_id: Optional model ID (uses default_multimodal_model if not provided)
            
        Returns:
            Analysis text
            
        Raises:
            BedrockValidationError: If input validation fails
            BedrockError: If an error occurs during image analysis
        """
        # Validate inputs
        if not image_data:
            raise BedrockValidationError("image_data cannot be empty")
        
        # Use provided model_id or default
        model_id = model_id or self.default_multimodal_model
        
        logger.info(f"Analyzing image with model {model_id}")
        logger.debug(f"Prompt: {prompt}")
        
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
            
            # Define the analysis operation
            def analyze_operation():
                response = self.bedrock_client.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                result = response_body.get('content', [{}])[0].get('text', '')
                return result
            
            # Execute with retry
            result = self.bedrock_client._execute_with_retry("analyze_image", analyze_operation)
            
            # Record request
            self.bedrock_client._record_request(
                model_id,
                "image_analysis",
                prompt,
                result
            )
            
            logger.info("Image analysis completed successfully")
            
            return result
            
        except BedrockError as e:
            logger.error(f"Error analyzing image: {e}", exc_info=True)
            return f"Error analyzing image: {str(e)}"

# Helper functions for easy integration
def create_bedrock_integration(region_name: str = "us-east-1", 
                             profile_name: Optional[str] = None,
                             max_retries: int = 3,
                             embedding_cache_ttl: int = 3600,
                             embedding_cache_max_size: int = 1000) -> MAIFBedrockIntegration:
    """
    Create MAIF Bedrock integration.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        max_retries: Maximum number of retries for transient errors
        embedding_cache_ttl: Time-to-live for embedding cache entries (seconds)
        embedding_cache_max_size: Maximum number of entries in embedding cache
        
    Returns:
        MAIFBedrockIntegration
        
    Raises:
        BedrockConnectionError: If unable to initialize the Bedrock client
    """
    logger.info(f"Creating Bedrock integration in region {region_name}")
    
    try:
        bedrock_client = BedrockClient(
            region_name=region_name, 
            profile_name=profile_name,
            max_retries=max_retries
        )
        
        integration = MAIFBedrockIntegration(
            bedrock_client=bedrock_client,
            embedding_cache_ttl=embedding_cache_ttl,
            embedding_cache_max_size=embedding_cache_max_size
        )
        
        logger.info("Bedrock integration created successfully")
        return integration
        
    except BedrockError as e:
        logger.error(f"Failed to create Bedrock integration: {e}", exc_info=True)
        raise
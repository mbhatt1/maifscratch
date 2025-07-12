"""
AWS Distributed Integration for MAIF
===================================

Integrates MAIF's distributed module with AWS services for enhanced scalability and reliability.
"""

import json
import time
import logging
import hashlib
import uuid
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from maif.distributed import (
    DistributedCoordinator, ShardManager, DistributedLock,
    VectorClock, GCounter, LWWRegister, ORSet, CRDTOperation
)
from maif.aws_dynamodb_integration import DynamoDBClient
from maif.aws_lambda_integration import LambdaClient
from maif.aws_stepfunctions_integration import StepFunctionsClient
from maif.aws_s3_integration import S3Client

# Configure logger
logger = logging.getLogger(__name__)


class AWSDistributedError(Exception):
    """Base exception for AWS distributed integration errors."""
    pass


class AWSVectorClock(VectorClock):
    """Vector clock implementation with DynamoDB persistence."""
    
    def __init__(self, node_id: str, dynamodb_client: DynamoDBClient, table_name: str):
        """
        Initialize AWS Vector Clock.
        
        Args:
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for vector clock storage
        """
        super().__init__(node_id)
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self._load_from_dynamodb()
    
    def _load_from_dynamodb(self):
        """Load vector clock from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                table_name=self.table_name,
                key={'node_id': {'S': self.node_id}},
                consistent_read=True
            )
            
            if 'Item' in response:
                clock_data = json.loads(response['Item'].get('clock', {}).get('S', '{}'))
                self.clock = defaultdict(int, {k: int(v) for k, v in clock_data.items()})
                logger.debug(f"Loaded vector clock for {self.node_id} from DynamoDB")
            else:
                logger.debug(f"No existing vector clock found for {self.node_id}, using default")
        except Exception as e:
            logger.warning(f"Error loading vector clock from DynamoDB: {e}")
    
    def _save_to_dynamodb(self):
        """Save vector clock to DynamoDB."""
        try:
            self.dynamodb_client.put_item(
                table_name=self.table_name,
                item={
                    'node_id': {'S': self.node_id},
                    'clock': {'S': json.dumps(dict(self.clock))},
                    'updated_at': {'S': datetime.utcnow().isoformat()}
                }
            )
            logger.debug(f"Saved vector clock for {self.node_id} to DynamoDB")
        except Exception as e:
            logger.error(f"Error saving vector clock to DynamoDB: {e}")
    
    def increment(self):
        """Increment this node's clock and persist to DynamoDB."""
        super().increment()
        self._save_to_dynamodb()
    
    def update(self, other_clock: Dict[str, int]):
        """Update with another vector clock and persist to DynamoDB."""
        super().update(other_clock)
        self._save_to_dynamodb()


class AWSGCounter(GCounter):
    """Grow-only counter CRDT with DynamoDB persistence."""
    
    def __init__(self, node_id: str, dynamodb_client: DynamoDBClient, 
                table_name: str, counter_id: str):
        """
        Initialize AWS G-Counter.
        
        Args:
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for counter storage
            counter_id: Unique identifier for this counter
        """
        super().__init__(node_id)
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.counter_id = counter_id
        self._load_from_dynamodb()
    
    def _load_from_dynamodb(self):
        """Load counter from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                table_name=self.table_name,
                key={
                    'counter_id': {'S': self.counter_id},
                    'node_id': {'S': self.node_id}
                },
                consistent_read=True
            )
            
            if 'Item' in response:
                self.counts[self.node_id] = int(response['Item'].get('count', {}).get('N', '0'))
                logger.debug(f"Loaded counter {self.counter_id} for {self.node_id} from DynamoDB")
            else:
                logger.debug(f"No existing counter found for {self.counter_id}:{self.node_id}, using default")
        except Exception as e:
            logger.warning(f"Error loading counter from DynamoDB: {e}")
    
    def _save_to_dynamodb(self):
        """Save counter to DynamoDB."""
        try:
            self.dynamodb_client.put_item(
                table_name=self.table_name,
                item={
                    'counter_id': {'S': self.counter_id},
                    'node_id': {'S': self.node_id},
                    'count': {'N': str(self.counts[self.node_id])},
                    'updated_at': {'S': datetime.utcnow().isoformat()}
                }
            )
            logger.debug(f"Saved counter {self.counter_id} for {self.node_id} to DynamoDB")
        except Exception as e:
            logger.error(f"Error saving counter to DynamoDB: {e}")
    
    def increment(self, value: int = 1):
        """Increment counter and persist to DynamoDB."""
        super().increment(value)
        self._save_to_dynamodb()
    
    def merge(self, other: 'GCounter'):
        """Merge with another counter and persist to DynamoDB."""
        super().merge(other)
        self._save_to_dynamodb()
    
    def load_all_nodes(self):
        """Load counter values for all nodes from DynamoDB."""
        try:
            # Query for all nodes with this counter_id
            response = self.dynamodb_client.query(
                table_name=self.table_name,
                key_condition_expression="counter_id = :cid",
                expression_attribute_values={
                    ":cid": {"S": self.counter_id}
                }
            )
            
            # Update counts from all nodes
            for item in response.get('Items', []):
                node_id = item.get('node_id', {}).get('S')
                count = int(item.get('count', {}).get('N', '0'))
                if node_id and node_id != self.node_id:
                    self.counts[node_id] = max(self.counts[node_id], count)
            
            logger.debug(f"Loaded all node counts for counter {self.counter_id}")
        except Exception as e:
            logger.warning(f"Error loading all node counts from DynamoDB: {e}")


class AWSLWWRegister(LWWRegister):
    """Last-Write-Wins Register CRDT with DynamoDB persistence."""
    
    def __init__(self, node_id: str, dynamodb_client: DynamoDBClient, 
                table_name: str, register_id: str):
        """
        Initialize AWS LWW-Register.
        
        Args:
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for register storage
            register_id: Unique identifier for this register
        """
        super().__init__(node_id)
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.register_id = register_id
        self._load_from_dynamodb()
    
    def _load_from_dynamodb(self):
        """Load register from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                table_name=self.table_name,
                key={'register_id': {'S': self.register_id}},
                consistent_read=True
            )
            
            if 'Item' in response:
                item = response['Item']
                self.value = json.loads(item.get('value', {}).get('S', 'null'))
                self.timestamp = float(item.get('timestamp', {}).get('N', '0'))
                self.writer = item.get('writer', {}).get('S', "")
                logger.debug(f"Loaded register {self.register_id} from DynamoDB")
            else:
                logger.debug(f"No existing register found for {self.register_id}, using default")
        except Exception as e:
            logger.warning(f"Error loading register from DynamoDB: {e}")
    
    def _save_to_dynamodb(self):
        """Save register to DynamoDB."""
        try:
            self.dynamodb_client.put_item(
                table_name=self.table_name,
                item={
                    'register_id': {'S': self.register_id},
                    'value': {'S': json.dumps(self.value)},
                    'timestamp': {'N': str(self.timestamp)},
                    'writer': {'S': self.writer},
                    'updated_at': {'S': datetime.utcnow().isoformat()}
                }
            )
            logger.debug(f"Saved register {self.register_id} to DynamoDB")
        except Exception as e:
            logger.error(f"Error saving register to DynamoDB: {e}")
    
    def set(self, value: Any):
        """Set value with current timestamp and persist to DynamoDB."""
        super().set(value)
        self._save_to_dynamodb()
    
    def merge(self, other: 'LWWRegister'):
        """Merge with another register and persist to DynamoDB if changed."""
        old_value = self.value
        old_timestamp = self.timestamp
        
        super().merge(other)
        
        if self.value != old_value or self.timestamp != old_timestamp:
            self._save_to_dynamodb()


class AWSORSet(ORSet):
    """Observed-Remove Set CRDT with DynamoDB persistence."""
    
    def __init__(self, node_id: str, dynamodb_client: DynamoDBClient, 
                table_name: str, set_id: str):
        """
        Initialize AWS OR-Set.
        
        Args:
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for set storage
            set_id: Unique identifier for this set
        """
        super().__init__(node_id)
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.set_id = set_id
        self._load_from_dynamodb()
    
    def _load_from_dynamodb(self):
        """Load set from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                table_name=self.table_name,
                key={'set_id': {'S': self.set_id}},
                consistent_read=True
            )
            
            if 'Item' in response:
                item = response['Item']
                elements_data = json.loads(item.get('elements', {}).get('S', '{}'))
                tombstones_data = json.loads(item.get('tombstones', {}).get('S', '[]'))
                
                # Convert elements data to proper format
                self.elements = defaultdict(set)
                for elem, tags in elements_data.items():
                    self.elements[elem] = set(tags)
                
                # Convert tombstones data to proper format
                self.tombstones = set()
                for elem_tag in tombstones_data:
                    if len(elem_tag) == 2:
                        self.tombstones.add(tuple(elem_tag))
                
                logger.debug(f"Loaded OR-Set {self.set_id} from DynamoDB")
            else:
                logger.debug(f"No existing OR-Set found for {self.set_id}, using default")
        except Exception as e:
            logger.warning(f"Error loading OR-Set from DynamoDB: {e}")
    
    def _save_to_dynamodb(self):
        """Save set to DynamoDB."""
        try:
            # Convert elements to serializable format
            elements_data = {}
            for elem, tags in self.elements.items():
                elements_data[str(elem)] = list(tags)
            
            # Convert tombstones to serializable format
            tombstones_data = [list(t) for t in self.tombstones]
            
            self.dynamodb_client.put_item(
                table_name=self.table_name,
                item={
                    'set_id': {'S': self.set_id},
                    'elements': {'S': json.dumps(elements_data)},
                    'tombstones': {'S': json.dumps(tombstones_data)},
                    'updated_at': {'S': datetime.utcnow().isoformat()}
                }
            )
            logger.debug(f"Saved OR-Set {self.set_id} to DynamoDB")
        except Exception as e:
            logger.error(f"Error saving OR-Set to DynamoDB: {e}")
    
    def add(self, element: Any):
        """Add element to set and persist to DynamoDB."""
        super().add(element)
        self._save_to_dynamodb()
    
    def remove(self, element: Any):
        """Remove element from set and persist to DynamoDB."""
        super().remove(element)
        self._save_to_dynamodb()
    
    def merge(self, other: 'ORSet'):
        """Merge with another set and persist to DynamoDB."""
        old_elements = self.elements.copy()
        old_tombstones = self.tombstones.copy()
        
        super().merge(other)
        
        if self.elements != old_elements or self.tombstones != old_tombstones:
            self._save_to_dynamodb()


class AWSDistributedLock:
    """Distributed lock implementation using DynamoDB."""
    
    def __init__(self, lock_id: str, node_id: str, 
                dynamodb_client: DynamoDBClient, table_name: str,
                ttl_seconds: int = 60):
        """
        Initialize AWS Distributed Lock.
        
        Args:
            lock_id: Lock identifier
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for lock storage
            ttl_seconds: Time-to-live for lock in seconds
        """
        self.lock_id = lock_id
        self.node_id = node_id
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.is_locked = False
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire the distributed lock.
        
        Args:
            timeout: Maximum time to wait for lock acquisition
            
        Returns:
            True if lock acquired, False otherwise
        """
        with self._lock:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    # Try to acquire lock with conditional write
                    expiry_time = int(time.time() + self.ttl_seconds)
                    
                    self.dynamodb_client.put_item(
                        table_name=self.table_name,
                        item={
                            'lock_id': {'S': self.lock_id},
                            'node_id': {'S': self.node_id},
                            'acquired_at': {'N': str(int(time.time()))},
                            'ttl': {'N': str(expiry_time)}
                        },
                        condition_expression="attribute_not_exists(lock_id) OR ttl < :now",
                        expression_attribute_values={
                            ":now": {"N": str(int(time.time()))}
                        }
                    )
                    
                    # Lock acquired
                    self.is_locked = True
                    logger.info(f"Acquired distributed lock {self.lock_id}")
                    return True
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                        # Lock is held by someone else, wait and retry
                        logger.debug(f"Lock {self.lock_id} is held by another node, waiting...")
                        time.sleep(1)
                    else:
                        # Unexpected error
                        logger.error(f"Error acquiring lock {self.lock_id}: {e}")
                        return False
            
            # Timeout
            logger.warning(f"Timeout acquiring lock {self.lock_id}")
            return False
    
    def release(self):
        """Release the distributed lock."""
        with self._lock:
            if self.is_locked:
                try:
                    # Delete the lock item
                    self.dynamodb_client.delete_item(
                        table_name=self.table_name,
                        key={
                            'lock_id': {'S': self.lock_id}
                        },
                        condition_expression="node_id = :nid",
                        expression_attribute_values={
                            ":nid": {"S": self.node_id}
                        }
                    )
                    
                    self.is_locked = False
                    logger.info(f"Released distributed lock {self.lock_id}")
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                        # Lock is held by someone else
                        logger.warning(f"Cannot release lock {self.lock_id} - not the owner")
                    else:
                        # Unexpected error
                        logger.error(f"Error releasing lock {self.lock_id}: {e}")
    
    def refresh(self):
        """Refresh the lock TTL."""
        with self._lock:
            if self.is_locked:
                try:
                    # Update the TTL
                    expiry_time = int(time.time() + self.ttl_seconds)
                    
                    self.dynamodb_client.update_item(
                        table_name=self.table_name,
                        key={
                            'lock_id': {'S': self.lock_id}
                        },
                        update_expression="SET ttl = :ttl",
                        condition_expression="node_id = :nid",
                        expression_attribute_values={
                            ":ttl": {"N": str(expiry_time)},
                            ":nid": {"S": self.node_id}
                        }
                    )
                    
                    logger.debug(f"Refreshed distributed lock {self.lock_id}")
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                        # Lock is held by someone else
                        logger.warning(f"Cannot refresh lock {self.lock_id} - not the owner")
                        self.is_locked = False
                    else:
                        # Unexpected error
                        logger.error(f"Error refreshing lock {self.lock_id}: {e}")


class AWSShardManager:
    """Manages MAIF sharding across nodes using DynamoDB."""
    
    def __init__(self, node_id: str, dynamodb_client: DynamoDBClient, 
                table_name: str, num_shards: int = 16):
        """
        Initialize AWS Shard Manager.
        
        Args:
            node_id: Node identifier
            dynamodb_client: DynamoDB client
            table_name: DynamoDB table name for shard mapping
            num_shards: Number of shards
        """
        self.node_id = node_id
        self.dynamodb_client = dynamodb_client
        self.table_name = table_name
        self.num_shards = num_shards
        self.shard_map = {}  # shard_id -> node_id
        self.local_shards = set()
        self.replication_factor = 3
        self._lock = threading.Lock()
        self._load_shard_map()
    
    def _load_shard_map(self):
        """Load shard map from DynamoDB."""
        try:
            # Scan the shard mapping table
            response = self.dynamodb_client.scan(
                table_name=self.table_name,
                projection_expression="shard_id, node_id"
            )
            
            with self._lock:
                self.shard_map = {}
                self.local_shards = set()
                
                for item in response.get('Items', []):
                    shard_id = int(item.get('shard_id', {}).get('N', '0'))
                    node_id = item.get('node_id', {}).get('S', '')
                    
                    if shard_id >= 0 and node_id:
                        self.shard_map[shard_id] = node_id
                        
                        if node_id == self.node_id:
                            self.local_shards.add(shard_id)
            
            logger.info(f"Loaded shard map from DynamoDB: {len(self.shard_map)} mappings")
        except Exception as e:
            logger.error(f"Error loading shard map from DynamoDB: {e}")
    
    def get_shard_id(self, key: str) -> int:
        """Get shard ID for a key using consistent hashing."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_shards
    
    def get_responsible_nodes(self, shard_id: int) -> List[str]:
        """Get nodes responsible for a shard (including replicas)."""
        nodes = []
        
        # Primary node
        primary = self.shard_map.get(shard_id)
        if primary:
            nodes.append(primary)
        
        # Replica nodes (next nodes in ring)
        all_nodes = sorted(set(self.shard_map.values()))
        if primary in all_nodes:
            primary_idx = all_nodes.index(primary)
            for i in range(1, self.replication_factor):
                if len(all_nodes) > 1:  # Only if we have multiple nodes
                    replica_idx = (primary_idx + i) % len(all_nodes)
                    nodes.append(all_nodes[replica_idx])
        
        return nodes
    
    def rebalance_shards(self, active_nodes: List[str]):
        """Rebalance shards across active nodes."""
        if not active_nodes:
            return
        
        with self._lock:
            # Clear current assignments
            self.shard_map.clear()
            self.local_shards.clear()
            
            # Assign shards round-robin
            for shard_id in range(self.num_shards):
                node_idx = shard_id % len(active_nodes)
                assigned_node = active_nodes[node_idx]
                self.shard_map[shard_id] = assigned_node
                
                # Update DynamoDB
                try:
                    self.dynamodb_client.put_item(
                        table_name=self.table_name,
                        item={
                            'shard_id': {'N': str(shard_id)},
                            'node_id': {'S': assigned_node},
                            'updated_at': {'S': datetime.utcnow().isoformat()}
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating shard mapping in DynamoDB: {e}")
                
                if assigned_node == self.node_id:
                    self.local_shards.add(shard_id)
            
            # Check replicas
            for shard_id in range(self.num_shards):
                responsible_nodes = self.get_responsible_nodes(shard_id)
                if self.node_id in responsible_nodes:
                    self.local_shards.add(shard_id)
    
    def is_responsible_for(self, key: str) -> bool:
        """Check if this node is responsible for a key."""
        shard_id = self.get_shard_id(key)
        return shard_id in self.local_shards


class AWSDistributedCoordinator:
    """Coordinates distributed MAIF operations using AWS services."""
    
    def __init__(self, node_id: str, maif_path: str, 
                 region_name: str = "us-east-1",
                 dynamodb_table_prefix: str = "maif_distributed_",
                 s3_bucket: Optional[str] = None,
                 lambda_function_prefix: Optional[str] = None,
                 step_functions_state_machine: Optional[str] = None):
        """
        Initialize AWS Distributed Coordinator.
        
        Args:
            node_id: Node identifier
            maif_path: Path to MAIF file
            region_name: AWS region name
            dynamodb_table_prefix: Prefix for DynamoDB tables
            s3_bucket: S3 bucket for block storage
            lambda_function_prefix: Prefix for Lambda functions
            step_functions_state_machine: State machine ARN for workflow orchestration
        """
        self.node_id = node_id
        self.maif_path = Path(maif_path)
        self.region_name = region_name
        
        # Table names
        self.vector_clock_table = f"{dynamodb_table_prefix}vector_clocks"
        self.counter_table = f"{dynamodb_table_prefix}counters"
        self.register_table = f"{dynamodb_table_prefix}registers"
        self.set_table = f"{dynamodb_table_prefix}sets"
        self.lock_table = f"{dynamodb_table_prefix}locks"
        self.shard_table = f"{dynamodb_table_prefix}shards"
        
        # AWS clients
        self.dynamodb_client = DynamoDBClient(region_name=region_name)
        self.s3_client = S3Client(region_name=region_name)
        self.lambda_client = LambdaClient(region_name=region_name)
        self.sfn_client = StepFunctionsClient(region_name=region_name)
        
        # S3 bucket
        self.s3_bucket = s3_bucket
        
        # Lambda function prefix
        self.lambda_function_prefix = lambda_function_prefix
        
        # Step Functions state machine
        self.step_functions_state_machine = step_functions_state_machine
        
        # Components
        self.shard_manager = AWSShardManager(node_id, self.dynamodb_client, self.shard_table)
        self.vector_clock = AWSVectorClock(node_id, self.dynamodb_client, self.vector_clock_table)
        self.locks = {}  # lock_id -> AWSDistributedLock
        
        # CRDT state
        self.block_counter = AWSGCounter(node_id, self.dynamodb_client, self.counter_table, "block_counter")
        self.metadata_registers = {}  # key -> AWSLWWRegister
        self.active_nodes = AWSORSet(node_id, self.dynamodb_client, self.set_table, "active_nodes")
        
        # Initialize MAIF
        from .core import MAIFEncoder, MAIFDecoder
        
        self.manifest_path = self.maif_path.with_suffix('.json')
        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path), str(self.manifest_path))
            self.encoder = MAIFEncoder(existing_maif_path=str(self.maif_path),
                                      existing_manifest_path=str(self.manifest_path))
        else:
            self.encoder = MAIFEncoder()
            self.decoder = None
        
        # Start coordinator
        self.start()
    
    def start(self):
        """Start the distributed coordinator."""
        # Add this node to active nodes
        self.active_nodes.add(self.node_id)
        
        # Load block counter from all nodes
        self.block_counter.load_all_nodes()
        
        # Initial shard rebalance
        self.shard_manager.rebalance_shards(list(self.active_nodes.values()))
        
        logger.info(f"AWS Distributed Coordinator started for node {self.node_id}")
    
    def acquire_lock(self, resource_id: str, timeout: float = 30.0) -> bool:
        """
        Acquire a distributed lock.
        
        Args:
            resource_id: Resource identifier
            timeout: Maximum time to wait for lock acquisition
            
        Returns:
            True if lock acquired, False otherwise
        """
        lock_id = f"maif:{resource_id}"
        
        if lock_id not in self.locks:
            self.locks[lock_id] = AWSDistributedLock(
                lock_id, self.node_id, self.dynamodb_client, self.lock_table
            )
        
        return self.locks[lock_id].acquire(timeout)
    
    def release_lock(self, resource_id: str):
        """
        Release a distributed lock.
        
        Args:
            resource_id: Resource identifier
        """
        lock_id = f"maif:{resource_id}"
        
        if lock_id in self.locks:
            self.locks[lock_id].release()
    
    def add_block_distributed(self, block_data: bytes, block_type: str,
                             metadata: Optional[Dict] = None) -> str:
        """
        Add a block with distributed coordination.
        
        Args:
            block_data: Block data
            block_type: Block type
            metadata: Block metadata
            
        Returns:
            Block ID
        """
        # Generate block ID
        block_id = hashlib.sha256(block_data).hexdigest()
        
        # Check if responsible for this block
        if not self.shard_manager.is_responsible_for(block_id):
            # If S3 bucket is configured, store block there
            if self.s3_bucket:
                # Store in S3
                s3_key = f"blocks/{block_id}"
                
                try:
                    # Upload block data to S3
                    self.s3_client.put_object(
                        bucket_name=self.s3_bucket,
                        key=s3_key,
                        data=block_data,
                        metadata={
                            'block_type': block_type,
                            'node_id': self.node_id,
                            'timestamp': str(time.time())
                        }
                    )
                    
                    # Store metadata if provided
                    if metadata:
                        metadata_key = f"metadata/{block_id}"
                        self.s3_client.put_object(
                            bucket_name=self.s3_bucket,
                            key=metadata_key,
                            data=json.dumps(metadata).encode(),
                            metadata={
                                'block_id': block_id,
                                'node_id': self.node_id,
                                'timestamp': str(time.time())
                            }
                        )
                    
                    logger.info(f"Stored block {block_id} in S3 bucket {self.s3_bucket}")
                    
                except Exception as e:
                    logger.error(f"Error storing block in S3: {e}")
            
            # If Lambda function is configured, invoke it to process the block
            if self.lambda_function_prefix:
                try:
                    # Invoke Lambda function to process block
                    function_name = f"{self.lambda_function_prefix}_process_block"
                    
                    payload = {
                        'block_id': block_id,
                        'block_type': block_type,
                        's3_bucket': self.s3_bucket,
                        's3_key': f"blocks/{block_id}" if self.s3_bucket else None,
                        'node_id': self.node_id,
                        'timestamp': time.time()
                    }
                    
                    if metadata:
                        payload['metadata'] = metadata
                    
                    self.lambda_client.invoke_function(
                        function_name=function_name,
                        payload=json.dumps(payload),
                        invocation_type='Event'  # Asynchronous invocation
                    )
                    
                    logger.info(f"Invoked Lambda function {function_name} to process block {block_id}")
                    
                except Exception as e:
                    logger.error(f"Error invoking Lambda function: {e}")
            
            return block_id
        
        # Add locally
        self.encoder.add_binary_block(block_data, block_type, metadata)
        
        # Update CRDT counters
        self.block_counter.increment()
        self.vector_clock.increment()
        
        # If S3 bucket is configured, also store a copy there for redundancy
        if self.s3_bucket:
            try:
                s3_key = f"blocks/{block_id}"
                
                self.s3_client.put_object(
                    bucket_name=self.s3_bucket,
                    key=s3_key,
                    data=block_data,
                    metadata={
                        'block_type': block_type,
                        'node_id': self.node_id,
                        'timestamp': str(time.time())
                    }
                )
                
                if metadata:
                    metadata_key = f"metadata/{block_id}"
                    self.s3_client.put_object(
                        bucket_name=self.s3_bucket,
                        key=metadata_key,
                        data=json.dumps(metadata).encode(),
                        metadata={
                            'block_id': block_id,
                            'node_id': self.node_id,
                            'timestamp': str(time.time())
                        }
                    )
                
                logger.debug(f"Stored redundant copy of block {block_id} in S3")
                
            except Exception as e:
                logger.warning(f"Error storing redundant copy in S3: {e}")
        
        return block_id
    
    def get_block_distributed(self, block_id: str) -> Optional[bytes]:
        """
        Get a block with distributed coordination.
        
        Args:
            block_id: Block ID
            
        Returns:
            Block data or None if not found
        """
        # Check if we have the block locally
        if self.decoder:
            try:
                block_data = self.decoder.get_binary_block(block_id)
                if block_data:
                    logger.debug(f"Retrieved block {block_id} from local storage")
                    return block_data
            except Exception as e:
                logger.warning(f"Error retrieving block {block_id} locally: {e}")
        
        # If not found locally and we have S3 bucket, try S3
        if self.s3_bucket:
            try:
                s3_key = f"blocks/{block_id}"
                
                response = self.s3_client.get_object(
                    bucket_name=self.s3_bucket,
                    key=s3_key
                )
                
                if response and 'Body' in response:
                    block_data = response['Body'].read()
                    logger.info(f"Retrieved block {block_id} from S3")
                    
                    # Store locally for future use
                    if block_data:
                        block_type = response.get('Metadata', {}).get('block_type', 'binary')
                        
                        # Get metadata if available
                        metadata = None
                        try:
                            metadata_key = f"metadata/{block_id}"
                            metadata_response = self.s3_client.get_object(
                                bucket_name=self.s3_bucket,
                                key=metadata_key
                            )
                            
                            if metadata_response and 'Body' in metadata_response:
                                metadata = json.loads(metadata_response['Body'].read().decode())
                        except:
                            pass
                        
                        # Store locally
                        self.encoder.add_binary_block(block_data, block_type, metadata)
                        
                    return block_data
            except Exception as e:
                logger.warning(f"Error retrieving block {block_id} from S3: {e}")
        
        # If still not found and Lambda function is configured, try Lambda
        if self.lambda_function_prefix:
            try:
                function_name = f"{self.lambda_function_prefix}_get_block"
                
                payload = {
                    'block_id': block_id,
                    'node_id': self.node_id
                }
                
                response = self.lambda_client.invoke_function(
                    function_name=function_name,
                    payload=json.dumps(payload),
                    invocation_type='RequestResponse'  # Synchronous invocation
                )
                
                if response and 'Payload' in response:
                    result = json.loads(response['Payload'].read().decode())
                    
                    if result.get('found', False) and 'data' in result:
                        import base64
                        block_data = base64.b64decode(result['data'])
                        
                        # Store locally for future use
                        if block_data:
                            block_type = result.get('block_type', 'binary')
                            metadata = result.get('metadata')
                            
                            self.encoder.add_binary_block(block_data, block_type, metadata)
                        
                        logger.info(f"Retrieved block {block_id} via Lambda")
                        return block_data
            except Exception as e:
                logger.warning(f"Error retrieving block {block_id} via Lambda: {e}")
        
        logger.warning(f"Block {block_id} not found in any storage")
        return None
    
    def start_distributed_workflow(self, workflow_input: Dict[str, Any]) -> Optional[str]:
        """
        Start a distributed workflow using Step Functions.
        
        Args:
            workflow_input: Input for the workflow
            
        Returns:
            Execution ARN if successful, None otherwise
        """
        if not self.step_functions_state_machine:
            logger.warning("No Step Functions state machine configured")
            return None
        
        try:
            # Start execution
            response = self.sfn_client.start_execution(
                state_machine_arn=self.step_functions_state_machine,
                input=json.dumps(workflow_input),
                name=f"maif-workflow-{uuid.uuid4()}"
            )
            
            execution_arn = response.get('executionArn')
            logger.info(f"Started distributed workflow: {execution_arn}")
            return execution_arn
            
        except Exception as e:
            logger.error(f"Error starting distributed workflow: {e}")
            return None
    
    def check_workflow_status(self, execution_arn: str) -> Dict[str, Any]:
        """
        Check the status of a distributed workflow.
        
        Args:
            execution_arn: Execution ARN
            
        Returns:
            Workflow status information
        """
        try:
            response = self.sfn_client.describe_execution(execution_arn)
            
            status = {
                'status': response.get('status'),
                'started_at': response.get('startDate'),
                'execution_arn': execution_arn
            }
            
            if response.get('status') in ('SUCCEEDED', 'FAILED'):
                status['stopped_at'] = response.get('stopDate')
                
                if 'output' in response:
                    status['output'] = json.loads(response.get('output', '{}'))
                
                if 'error' in response:
                    status['error'] = response.get('error')
                    status['cause'] = response.get('cause')
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking workflow status: {e}")
            return {'status': 'UNKNOWN', 'error': str(e)}
    
    def create_required_tables(self):
        """Create required DynamoDB tables if they don't exist."""
        tables_to_create = [
            {
                'name': self.vector_clock_table,
                'key_schema': [
                    {'AttributeName': 'node_id', 'KeyType': 'HASH'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'node_id', 'AttributeType': 'S'}
                ]
            },
            {
                'name': self.counter_table,
                'key_schema': [
                    {'AttributeName': 'counter_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'node_id', 'KeyType': 'RANGE'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'counter_id', 'AttributeType': 'S'},
                    {'AttributeName': 'node_id', 'AttributeType': 'S'}
                ]
            },
            {
                'name': self.register_table,
                'key_schema': [
                    {'AttributeName': 'register_id', 'KeyType': 'HASH'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'register_id', 'AttributeType': 'S'}
                ]
            },
            {
                'name': self.set_table,
                'key_schema': [
                    {'AttributeName': 'set_id', 'KeyType': 'HASH'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'set_id', 'AttributeType': 'S'}
                ]
            },
            {
                'name': self.lock_table,
                'key_schema': [
                    {'AttributeName': 'lock_id', 'KeyType': 'HASH'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'lock_id', 'AttributeType': 'S'}
                ]
            },
            {
                'name': self.shard_table,
                'key_schema': [
                    {'AttributeName': 'shard_id', 'KeyType': 'HASH'}
                ],
                'attribute_definitions': [
                    {'AttributeName': 'shard_id', 'AttributeType': 'S'}
                ]
            }
        ]
        
        for table_info in tables_to_create:
            try:
                # Check if table exists
                existing_tables = self.dynamodb_client.list_tables()
                
                if table_info['name'] not in existing_tables:
                    # Create table
                    self.dynamodb_client.create_table(
                        table_name=table_info['name'],
                        key_schema=table_info['key_schema'],
                        attribute_definitions=table_info['attribute_definitions'],
                        provisioned_throughput={
                            'ReadCapacityUnits': 5,
                            'WriteCapacityUnits': 5
                        }
                    )
                    
                    logger.info(f"Created DynamoDB table: {table_info['name']}")
                else:
                    logger.info(f"DynamoDB table already exists: {table_info['name']}")
                    
            except Exception as e:
                logger.error(f"Error creating DynamoDB table {table_info['name']}: {e}")
    
    def create_s3_bucket_if_needed(self):
        """Create S3 bucket if it doesn't exist."""
        if not self.s3_bucket:
            return
        
        try:
            # Check if bucket exists
            buckets = self.s3_client.list_buckets()
            
            if self.s3_bucket not in [b['Name'] for b in buckets.get('Buckets', [])]:
                # Create bucket
                self.s3_client.create_bucket(
                    bucket_name=self.s3_bucket,
                    region=self.region_name
                )
                
                logger.info(f"Created S3 bucket: {self.s3_bucket}")
            else:
                logger.info(f"S3 bucket already exists: {self.s3_bucket}")
                
        except Exception as e:
            logger.error(f"Error creating S3 bucket {self.s3_bucket}: {e}")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get cluster statistics.
        
        Returns:
            Cluster statistics
        """
        return {
            'node_id': self.node_id,
            'active_nodes': list(self.active_nodes.values()),
            'total_blocks': self.block_counter.value(),
            'local_shards': list(self.shard_manager.local_shards),
            'vector_clock': self.vector_clock.to_dict(),
            'aws_region': self.region_name,
            's3_bucket': self.s3_bucket,
            'lambda_function_prefix': self.lambda_function_prefix,
            'step_functions_state_machine': self.step_functions_state_machine
        }


# Export classes
__all__ = [
    'AWSDistributedCoordinator',
    'AWSShardManager',
    'AWSDistributedLock',
    'AWSVectorClock',
    'AWSGCounter',
    'AWSLWWRegister',
    'AWSORSet'
]
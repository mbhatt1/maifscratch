"""
Test PyTorch model storage and complete recovery using MAIF.
Demonstrates full model lifecycle: save -> store in MAIF -> recover -> verify.
"""

import pytest
import tempfile
import os
import json
import time
import random
from maif.core import MAIFEncoder, MAIFDecoder


class TestPyTorchModelRecovery:
    """Test complete PyTorch model storage and recovery with MAIF."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Set random seed for reproducible tests
        random.seed(42)
        
        # Create temporary files
        self.temp_maif = tempfile.NamedTemporaryFile(suffix='.maif', delete=False)
        self.temp_manifest = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        self.maif_path = self.temp_maif.name
        self.manifest_path = self.temp_manifest.name
        self.temp_maif.close()
        self.temp_manifest.close()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.maif_path)
            os.unlink(self.manifest_path)
        except:
            pass
    
    def create_neural_network_model(self):
        """Create a complete neural network model with all components."""
        
        # 1. Network Architecture
        architecture = {
            'model_name': 'SimpleClassifier',
            'framework': 'pytorch',
            'version': '1.0',
            'layers': [
                {
                    'name': 'fc1',
                    'type': 'linear',
                    'input_size': 784,  # 28x28 MNIST
                    'output_size': 128,
                    'activation': 'relu',
                    'dropout': 0.2
                },
                {
                    'name': 'fc2', 
                    'type': 'linear',
                    'input_size': 128,
                    'output_size': 64,
                    'activation': 'relu',
                    'dropout': 0.3
                },
                {
                    'name': 'fc3',
                    'type': 'linear', 
                    'input_size': 64,
                    'output_size': 10,
                    'activation': 'softmax'
                }
            ],
            'loss_function': 'cross_entropy',
            'optimizer': {
                'type': 'adam',
                'learning_rate': 0.001,
                'beta1': 0.9,
                'beta2': 0.999,
                'weight_decay': 1e-4
            }
        }
        
        # 2. Model Weights (simulate trained weights)
        weights = {}
        total_params = 0
        
        for layer in architecture['layers']:
            layer_name = layer['name']
            input_size = layer['input_size']
            output_size = layer['output_size']
            
            # Generate weights with Xavier initialization
            limit = (6.0 / (input_size + output_size)) ** 0.5
            layer_weights = [
                [random.uniform(-limit, limit) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            layer_biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]
            
            weights[layer_name] = {
                'weight': layer_weights,
                'bias': layer_biases,
                'shape': [output_size, input_size]
            }
            
            total_params += (input_size * output_size) + output_size
        
        # 3. Training Configuration
        training_config = {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate_schedule': 'cosine_annealing',
            'data_augmentation': True,
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.001
            },
            'regularization': {
                'l2_weight_decay': 1e-4,
                'dropout_rates': [0.2, 0.3, 0.0]
            }
        }
        
        # 4. Training History
        training_history = []
        for epoch in range(25):  # Simulate 25 epochs of training
            train_loss = 2.3 - (epoch * 0.08) + random.uniform(-0.1, 0.1)
            train_acc = 0.1 + (epoch * 0.03) + random.uniform(-0.02, 0.02)
            val_loss = train_loss + random.uniform(0, 0.15)
            val_acc = train_acc - random.uniform(0, 0.03)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': round(max(0.1, train_loss), 4),
                'train_accuracy': round(min(0.99, max(0.1, train_acc)), 4),
                'val_loss': round(max(0.1, val_loss), 4),
                'val_accuracy': round(min(0.99, max(0.1, val_acc)), 4),
                'learning_rate': 0.001 * (0.95 ** epoch),  # Decay
                'timestamp': time.time() + epoch * 3600  # Simulate hourly training
            })
        
        # 5. Model Metadata
        model_metadata = {
            'created_at': time.time(),
            'total_parameters': total_params,
            'model_size_mb': (total_params * 4) / (1024 * 1024),  # Assume float32
            'dataset': {
                'name': 'MNIST',
                'num_classes': 10,
                'input_shape': [1, 28, 28],
                'train_samples': 60000,
                'test_samples': 10000
            },
            'performance': {
                'best_train_accuracy': max(h['train_accuracy'] for h in training_history),
                'best_val_accuracy': max(h['val_accuracy'] for h in training_history),
                'final_train_loss': training_history[-1]['train_loss'],
                'final_val_loss': training_history[-1]['val_loss']
            },
            'hardware': {
                'device': 'cuda:0',
                'gpu_memory_used': '2.1GB',
                'training_time_hours': 2.5
            }
        }
        
        # 6. Optimizer State (simulate)
        optimizer_state = {
            'state_dict': {
                'param_groups': [
                    {
                        'lr': 0.001,
                        'betas': [0.9, 0.999],
                        'eps': 1e-08,
                        'weight_decay': 1e-4
                    }
                ]
            },
            'momentum_buffers': {
                layer_name: {
                    'weight_momentum': [[random.uniform(-0.01, 0.01) for _ in range(layer['input_size'])] 
                                      for _ in range(layer['output_size'])],
                    'bias_momentum': [random.uniform(-0.01, 0.01) for _ in range(layer['output_size'])]
                }
                for layer_name, layer in zip([l['name'] for l in architecture['layers']], architecture['layers'])
            }
        }
        
        return {
            'architecture': architecture,
            'weights': weights,
            'training_config': training_config,
            'training_history': training_history,
            'model_metadata': model_metadata,
            'optimizer_state': optimizer_state
        }
    
    def test_complete_model_storage_and_recovery(self):
        """Test storing and recovering a complete PyTorch model."""
        
        print("\n🔥 Testing Complete PyTorch Model Storage & Recovery")
        
        # Step 1: Create complete model
        print("\n📦 Step 1: Creating complete neural network model...")
        model_data = self.create_neural_network_model()
        
        # Verify model components
        assert 'architecture' in model_data
        assert 'weights' in model_data
        assert 'training_config' in model_data
        assert 'training_history' in model_data
        assert 'model_metadata' in model_data
        assert 'optimizer_state' in model_data
        
        print(f"  ✓ Model: {model_data['architecture']['model_name']}")
        print(f"  ✓ Layers: {len(model_data['architecture']['layers'])}")
        print(f"  ✓ Parameters: {model_data['model_metadata']['total_parameters']:,}")
        print(f"  ✓ Training epochs: {len(model_data['training_history'])}")
        
        # Step 2: Store model in MAIF
        print("\n💾 Step 2: Storing model components in MAIF...")
        encoder = MAIFEncoder()
        
        stored_blocks = {}
        
        # Store architecture
        arch_json = json.dumps(model_data['architecture'], indent=2).encode('utf-8')
        stored_blocks['architecture'] = encoder.add_binary_block(
            arch_json, 'model_architecture',
            {'type': 'neural_network_architecture', 'framework': 'pytorch'}
        )
        
        # Store weights
        weights_json = json.dumps(model_data['weights'], indent=2).encode('utf-8')
        stored_blocks['weights'] = encoder.add_binary_block(
            weights_json, 'model_weights',
            {'type': 'neural_network_weights', 'total_params': model_data['model_metadata']['total_parameters']}
        )
        
        # Store training config
        config_json = json.dumps(model_data['training_config'], indent=2).encode('utf-8')
        stored_blocks['training_config'] = encoder.add_binary_block(
            config_json, 'training_config',
            {'type': 'training_configuration'}
        )
        
        # Store training history
        history_json = json.dumps(model_data['training_history'], indent=2).encode('utf-8')
        stored_blocks['training_history'] = encoder.add_binary_block(
            history_json, 'training_history',
            {'type': 'training_metrics', 'epochs': len(model_data['training_history'])}
        )
        
        # Store model metadata
        metadata_json = json.dumps(model_data['model_metadata'], indent=2).encode('utf-8')
        stored_blocks['model_metadata'] = encoder.add_binary_block(
            metadata_json, 'model_metadata',
            {'type': 'model_information'}
        )
        
        # Store optimizer state
        optimizer_json = json.dumps(model_data['optimizer_state'], indent=2).encode('utf-8')
        stored_blocks['optimizer_state'] = encoder.add_binary_block(
            optimizer_json, 'optimizer_state',
            {'type': 'optimizer_checkpoint'}
        )
        
        # Save MAIF file
        encoder.save(self.maif_path, self.manifest_path)
        
        print(f"  ✓ Stored {len(stored_blocks)} model components")
        for component, block_id in stored_blocks.items():
            print(f"    • {component}: {block_id[:16]}...")
        
        # Step 3: Verify file integrity
        print("\n🔍 Step 3: Verifying file integrity...")
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        integrity_ok = decoder.verify_integrity()
        assert integrity_ok, "File integrity check failed"
        print(f"  ✓ Integrity check: PASSED")
        print(f"  ✓ Total blocks: {len(decoder.blocks)}")
        
        # Step 4: Recover complete model
        print("\n🔄 Step 4: Recovering complete model from MAIF...")
        
        recovered_model = {}
        component_mapping = {
            'model_architecture': 'architecture',
            'model_weights': 'weights', 
            'training_config': 'training_config',
            'training_history': 'training_history',
            'model_metadata': 'model_metadata',
            'optimizer_state': 'optimizer_state'
        }
        
        for block in decoder.blocks:
            block_type = block.block_type
            if block_type in component_mapping:
                component_name = component_mapping[block_type]
                
                # Extract and parse data
                data = decoder._extract_block_data(block)
                assert data is not None, f"Failed to extract data for {component_name}"
                
                parsed_data = json.loads(data.decode('utf-8'))
                recovered_model[component_name] = parsed_data
                
                print(f"  ✓ Recovered {component_name}")
        
        # Step 5: Verify complete recovery
        print("\n✅ Step 5: Verifying complete model recovery...")
        
        # Check all components recovered
        expected_components = set(model_data.keys())
        recovered_components = set(recovered_model.keys())
        assert expected_components == recovered_components, f"Missing components: {expected_components - recovered_components}"
        
        # Verify architecture
        orig_arch = model_data['architecture']
        recovered_arch = recovered_model['architecture']
        assert orig_arch['model_name'] == recovered_arch['model_name']
        assert len(orig_arch['layers']) == len(recovered_arch['layers'])
        assert orig_arch['optimizer']['learning_rate'] == recovered_arch['optimizer']['learning_rate']
        print(f"  ✓ Architecture: {recovered_arch['model_name']} with {len(recovered_arch['layers'])} layers")
        
        # Verify weights
        orig_weights = model_data['weights']
        recovered_weights = recovered_model['weights']
        assert set(orig_weights.keys()) == set(recovered_weights.keys())
        
        for layer_name in orig_weights:
            orig_layer = orig_weights[layer_name]
            recovered_layer = recovered_weights[layer_name]
            assert orig_layer['shape'] == recovered_layer['shape']
            assert len(orig_layer['weight']) == len(recovered_layer['weight'])
            assert len(orig_layer['bias']) == len(recovered_layer['bias'])
        print(f"  ✓ Weights: All {len(recovered_weights)} layers recovered")
        
        # Verify training history
        orig_history = model_data['training_history']
        recovered_history = recovered_model['training_history']
        assert len(orig_history) == len(recovered_history)
        assert orig_history[0]['epoch'] == recovered_history[0]['epoch']
        assert orig_history[-1]['train_accuracy'] == recovered_history[-1]['train_accuracy']
        print(f"  ✓ Training history: {len(recovered_history)} epochs")
        
        # Verify metadata
        orig_metadata = model_data['model_metadata']
        recovered_metadata = recovered_model['model_metadata']
        assert orig_metadata['total_parameters'] == recovered_metadata['total_parameters']
        assert orig_metadata['dataset']['name'] == recovered_metadata['dataset']['name']
        print(f"  ✓ Metadata: {recovered_metadata['total_parameters']:,} parameters")
        
        # Verify optimizer state
        orig_optimizer = model_data['optimizer_state']
        recovered_optimizer = recovered_model['optimizer_state']
        assert orig_optimizer['state_dict']['param_groups'][0]['lr'] == recovered_optimizer['state_dict']['param_groups'][0]['lr']
        print(f"  ✓ Optimizer state: Learning rate {recovered_optimizer['state_dict']['param_groups'][0]['lr']}")
        
        print(f"\n🎉 Complete model recovery successful!")
        print(f"  • All {len(expected_components)} components recovered")
        print(f"  • Model ready for inference or continued training")
        print(f"  • File size: {os.path.getsize(self.maif_path):,} bytes")
        
        # Test passes - all assertions completed successfully
        assert True
    
    def test_model_versioning_and_comparison(self):
        """Test storing multiple model versions and comparing them."""
        
        print("\n🔄 Testing Model Versioning & Comparison")
        
        # Create encoder
        encoder = MAIFEncoder()
        
        # Create and store model v1.0
        print("\n📦 Creating model v1.0...")
        model_v1 = self.create_neural_network_model()
        model_v1['architecture']['version'] = '1.0'
        
        v1_json = json.dumps(model_v1, indent=2).encode('utf-8')
        v1_block_id = encoder.add_binary_block(
            v1_json, 'complete_model',
            {'type': 'complete_pytorch_model', 'version': '1.0'}
        )
        
        # Create improved model v2.0
        print("📦 Creating improved model v2.0...")
        model_v2 = self.create_neural_network_model()
        model_v2['architecture']['version'] = '2.0'
        model_v2['architecture']['layers'].append({
            'name': 'fc4',
            'type': 'linear',
            'input_size': 10,
            'output_size': 10,
            'activation': 'softmax'
        })
        # Improve performance metrics
        for epoch_data in model_v2['training_history']:
            epoch_data['train_accuracy'] += 0.05
            epoch_data['val_accuracy'] += 0.04
        
        v2_json = json.dumps(model_v2, indent=2).encode('utf-8')
        v2_block_id = encoder._add_block(
            'complete_model', v2_json,
            {'type': 'complete_pytorch_model', 'version': '2.0', 'parent_version': '1.0'},
            update_block_id=v1_block_id
        )
        
        # Save and verify
        encoder.save(self.maif_path, self.manifest_path)
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        # Verify both versions stored
        model_blocks = [b for b in decoder.blocks if b.block_type == 'complete_model']
        assert len(model_blocks) == 2, f"Expected 2 model versions, got {len(model_blocks)}"
        
        # Extract and compare versions
        versions = {}
        for block in model_blocks:
            data = decoder._extract_block_data(block)
            model_data = json.loads(data.decode('utf-8'))
            version = model_data['architecture']['version']
            versions[version] = model_data
        
        assert '1.0' in versions and '2.0' in versions
        
        # Compare models
        v1_layers = len(versions['1.0']['architecture']['layers'])
        v2_layers = len(versions['2.0']['architecture']['layers'])
        assert v2_layers > v1_layers, "v2.0 should have more layers"
        
        v1_acc = max(h['train_accuracy'] for h in versions['1.0']['training_history'])
        v2_acc = max(h['train_accuracy'] for h in versions['2.0']['training_history'])
        assert v2_acc > v1_acc, "v2.0 should have better accuracy"
        
        print(f"  ✓ Model v1.0: {v1_layers} layers, {v1_acc:.4f} accuracy")
        print(f"  ✓ Model v2.0: {v2_layers} layers, {v2_acc:.4f} accuracy")
        print(f"  ✓ Version comparison successful")
    
    def test_model_component_extraction(self):
        """Test extracting specific model components."""
        
        print("\n🎯 Testing Selective Model Component Extraction")
        
        # Store complete model
        model_data = self.create_neural_network_model()
        encoder = MAIFEncoder()
        
        # Store each component separately
        components = {
            'architecture': ('model_architecture', model_data['architecture']),
            'weights': ('model_weights', model_data['weights']),
            'history': ('training_history', model_data['training_history'])
        }
        
        for comp_name, (block_type, comp_data) in components.items():
            comp_json = json.dumps(comp_data, indent=2).encode('utf-8')
            encoder.add_binary_block(comp_json, block_type, {'component': comp_name})
        
        encoder.save(self.maif_path, self.manifest_path)
        
        # Test selective extraction
        decoder = MAIFDecoder(self.maif_path, self.manifest_path)
        
        # Extract only architecture
        arch_block = next(b for b in decoder.blocks if b.block_type == 'model_architecture')
        arch_data = decoder._extract_block_data(arch_block)
        architecture = json.loads(arch_data.decode('utf-8'))
        
        assert architecture['model_name'] == model_data['architecture']['model_name']
        assert len(architecture['layers']) == len(model_data['architecture']['layers'])
        
        # Extract only weights
        weights_block = next(b for b in decoder.blocks if b.block_type == 'model_weights')
        weights_data = decoder._extract_block_data(weights_block)
        weights = json.loads(weights_data.decode('utf-8'))
        
        assert set(weights.keys()) == set(model_data['weights'].keys())
        
        print(f"  ✓ Selective extraction: architecture and weights")
        print(f"  ✓ Architecture: {architecture['model_name']}")
        print(f"  ✓ Weights: {len(weights)} layers")


if __name__ == "__main__":
    # Run the test directly
    test_instance = TestPyTorchModelRecovery()
    test_instance.setup_method()
    
    try:
        test_instance.test_complete_model_storage_and_recovery()
        test_instance.test_model_versioning_and_comparison()
        test_instance.test_model_component_extraction()
        print("\n🎉 All PyTorch model recovery tests passed!")
    finally:
        test_instance.teardown_method()
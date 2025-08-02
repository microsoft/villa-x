#!/usr/bin/env python3
"""
Test script for villa-X Actor Model implementation.

This script tests various components of the Actor Model to ensure
everything works correctly before training and upload.
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig, setup_actor_model_for_training
        print("✅ Actor model imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation with different configurations."""
    print("🧪 Testing model creation...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        
        # Test basic configuration
        config = ActorModelConfig(
            vlm_model_name="microsoft/molmo-7B-D-0924",
            action_dim=7,
            action_horizon=10,
            use_latent_actions=True,
        )
        
        model = ActorModel(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully with {param_count:,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 5
        visual_features = torch.randn(batch_size, seq_len, config.vlm_hidden_size)
        # Latent actions typically have one less timestep (for action sequences)
        latent_actions = torch.randn(batch_size, seq_len, config.latent_action_dim)
        
        model.eval()
        with torch.no_grad():
            outputs = model(visual_features, latent_actions)
            print(f"✅ Forward pass successful: {outputs.actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_setup_function():
    """Test the setup function."""
    print("🧪 Testing setup function...")
    
    try:
        from lam.actor_model import setup_actor_model_for_training
        
        model = setup_actor_model_for_training(
            vlm_model_name="microsoft/molmo-7B-D-0924",
            action_dim=7,
            action_horizon=10,
        )
        
        print(f"✅ Setup function successful")
        return True
        
    except Exception as e:
        print(f"❌ Setup function error: {e}")
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation."""
    print("🧪 Testing loss computation...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        
        config = ActorModelConfig()
        model = ActorModel(config)
        
        batch_size = 2
        predicted_actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        target_actions = torch.randn(batch_size, config.action_horizon, config.action_dim)
        
        loss = model.compute_loss(predicted_actions, target_actions)
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        # Test with mask
        mask = torch.ones(batch_size, config.action_horizon)
        mask[0, -2:] = 0  # Mask last 2 timesteps for first batch
        
        masked_loss = model.compute_loss(predicted_actions, target_actions, mask)
        print(f"✅ Masked loss computation successful: {masked_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation error: {e}")
        traceback.print_exc()
        return False

def test_hub_integration():
    """Test Hugging Face Hub integration."""
    print("🧪 Testing Hub integration...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        from huggingface_hub import PyTorchModelHubMixin
        
        # Check if ActorModel has Hub mixin
        assert issubclass(ActorModel, PyTorchModelHubMixin)
        print("✅ PyTorchModelHubMixin integration confirmed")
        
        # Test model has hub methods
        config = ActorModelConfig()
        model = ActorModel(config)
        
        assert hasattr(model, 'push_to_hub')
        assert hasattr(model, 'from_pretrained')
        print("✅ Hub methods available")
        
        return True
        
    except Exception as e:
        print(f"❌ Hub integration error: {e}")
        # This might fail if huggingface_hub is not installed
        if "huggingface_hub" in str(e):
            print("ℹ️ Hugging Face Hub not available - this is optional")
            return True
        traceback.print_exc()
        return False

def test_config_variations():
    """Test different configuration variations."""
    print("🧪 Testing configuration variations...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        
        # Test Molmo config
        molmo_config = ActorModelConfig(
            vlm_model_name="microsoft/molmo-7B-D-0924",
            action_dim=7,
            action_horizon=10,
        )
        molmo_model = ActorModel(molmo_config)
        print("✅ Molmo configuration works")
        
        # Test InternVL config  
        internvl_config = ActorModelConfig(
            vlm_model_name="OpenGVLab/InternVL2-4B",
            action_dim=6,
            action_horizon=15,
            use_latent_actions=False,
        )
        internvl_model = ActorModel(internvl_config)
        print("✅ InternVL configuration works")
        
        # Test custom config
        custom_config = ActorModelConfig(
            vlm_model_name="custom/vlm-model",
            vlm_hidden_size=1024,
            action_dim=12,
            action_horizon=5,
            hidden_size=256,
            num_layers=2,
        )
        custom_model = ActorModel(custom_config)
        print("✅ Custom configuration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration variation error: {e}")
        traceback.print_exc()
        return False

def test_prediction_interface():
    """Test the prediction interface."""
    print("🧪 Testing prediction interface...")
    
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        
        config = ActorModelConfig()
        model = ActorModel(config)
        
        # Test prediction without latent actions
        visual_features = torch.randn(1, 5, config.vlm_hidden_size)
        actions = model.predict_actions(visual_features)
        
        expected_shape = (1, config.action_horizon, config.action_dim)
        assert actions.shape == expected_shape, f"Expected {expected_shape}, got {actions.shape}"
        print("✅ Prediction without latent actions works")
        
        # Test prediction with latent actions
        latent_actions = torch.randn(1, 5, config.latent_action_dim)  # Same length as visual features
        actions_with_latent = model.predict_actions(visual_features, latent_actions)
        
        assert actions_with_latent.shape == expected_shape
        print("✅ Prediction with latent actions works")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction interface error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Running villa-X Actor Model Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Setup Function", test_setup_function),
        ("Loss Computation", test_loss_computation),
        ("Hub Integration", test_hub_integration),
        ("Configuration Variations", test_config_variations),
        ("Prediction Interface", test_prediction_interface),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Actor Model is ready for training and Hub upload.")
    else:
        print(f"⚠️ {failed} test(s) failed. Please fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

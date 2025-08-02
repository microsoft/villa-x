#!/usr/bin/env python3
"""
Additional test for sequence length handling in Actor Model.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lam.actor_model import ActorModel, ActorModelConfig

def test_sequence_length_handling():
    """Test that the model handles different sequence lengths correctly."""
    print("🧪 Testing sequence length handling...")
    
    config = ActorModelConfig()
    model = ActorModel(config)
    model.eval()
    
    batch_size = 1
    
    # Test case 1: Latent actions shorter than visual features
    visual_features = torch.randn(batch_size, 8, config.vlm_hidden_size)
    latent_actions = torch.randn(batch_size, 5, config.latent_action_dim)
    
    with torch.no_grad():
        outputs = model(visual_features, latent_actions)
        print(f"✅ Shorter latent actions: visual {visual_features.shape[1]}, latent {latent_actions.shape[1]} -> actions {outputs.actions.shape}")
    
    # Test case 2: Latent actions same length as visual features
    visual_features = torch.randn(batch_size, 6, config.vlm_hidden_size)
    latent_actions = torch.randn(batch_size, 6, config.latent_action_dim)
    
    with torch.no_grad():
        outputs = model(visual_features, latent_actions)
        print(f"✅ Same length: visual {visual_features.shape[1]}, latent {latent_actions.shape[1]} -> actions {outputs.actions.shape}")
    
    # Test case 3: Latent actions longer than visual features
    visual_features = torch.randn(batch_size, 4, config.vlm_hidden_size)
    latent_actions = torch.randn(batch_size, 7, config.latent_action_dim)
    
    with torch.no_grad():
        outputs = model(visual_features, latent_actions)
        print(f"✅ Longer latent actions: visual {visual_features.shape[1]}, latent {latent_actions.shape[1]} -> actions {outputs.actions.shape}")
    
    # Test case 4: No latent actions
    visual_features = torch.randn(batch_size, 5, config.vlm_hidden_size)
    
    with torch.no_grad():
        outputs = model(visual_features, latent_actions=None)
        print(f"✅ No latent actions: visual {visual_features.shape[1]} -> actions {outputs.actions.shape}")
    
    print("🎉 All sequence length tests passed!")

def test_realistic_sequence_from_lam():
    """Test with sequence lengths that would come from LAM."""
    print("🧪 Testing realistic LAM sequence scenario...")
    
    config = ActorModelConfig()
    model = ActorModel(config)
    model.eval()
    
    # Realistic scenario: 8 video frames -> 7 latent actions
    batch_size = 1
    video_frames = 8
    visual_features = torch.randn(batch_size, video_frames, config.vlm_hidden_size)
    latent_actions = torch.randn(batch_size, video_frames - 1, config.latent_action_dim)  # N-1 actions
    
    with torch.no_grad():
        outputs = model(visual_features, latent_actions)
        print(f"✅ LAM scenario: {video_frames} frames, {video_frames-1} latent actions -> {outputs.actions.shape}")
    
    # Multiple sequences of different lengths (batch processing)
    visual_batch = torch.randn(3, 6, config.vlm_hidden_size)
    latent_batch = torch.randn(3, 6, config.latent_action_dim)  # Same length for simplicity
    
    with torch.no_grad():
        outputs = model(visual_batch, latent_batch)
        print(f"✅ Batch processing: {outputs.actions.shape}")
    
    print("🎉 Realistic LAM scenario tests passed!")

if __name__ == "__main__":
    print("🚀 Running Additional Actor Model Tests")
    print("=" * 45)
    
    test_sequence_length_handling()
    print()
    test_realistic_sequence_from_lam()
    
    print("\n✅ All additional tests completed successfully!")

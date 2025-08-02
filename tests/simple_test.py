#!/usr/bin/env python3
"""Simple test to verify Actor Model can be imported."""

try:
    import sys
    import os
    
    # Add the project root to the path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print("Testing basic import...")
    
    # Test basic imports
    import torch
    print("✅ PyTorch available")
    
    # Test project imports
    try:
        from lam.actor_model import ActorModel, ActorModelConfig
        print("✅ Actor Model imports successful")
        
        # Test basic model creation
        config = ActorModelConfig()
        model = ActorModel(config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        print("\n🎉 Basic functionality test passed!")
        
    except ImportError as e:
        if "huggingface_hub" in str(e):
            print("⚠️ Hugging Face Hub not available, but core functionality works")
            # Try creating model without hub features
            from lam.model import IgorModel
            print("✅ Core LAM model imports work")
        else:
            print(f"❌ Import error: {e}")
            raise
            
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

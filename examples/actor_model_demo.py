#!/usr/bin/env python3
"""
Example usage of villa-X Actor Model with different VLMs.

This script demonstrates how to:
1. Setup the Actor Model with different VLMs (Molmo, InternVL)
2. Extract visual features from the VLM
3. Integrate with the Latent Action Model (LAM)
4. Predict robot actions
5. Load/save models with Hugging Face Hub
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple

from lam.actor_model import ActorModel, ActorModelConfig, setup_actor_model_for_training
from lam.model import IgorModel


class VLMFeatureExtractor:
    """Base class for extracting features from Vision-Language Models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the VLM model and processor."""
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def extract_features(self, images: List[Image.Image], text: Optional[str] = None) -> torch.Tensor:
        """Extract visual features from images."""
        raise NotImplementedError("Subclasses must implement extract_features()")


class MolmoFeatureExtractor(VLMFeatureExtractor):
    """Feature extractor for Molmo vision-language model."""
    
    def load_model(self):
        """Load Molmo model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            print(f"✅ Loaded Molmo model: {self.model_name}")
            
        except ImportError:
            print("❌ Transformers library not found. Please install: pip install transformers")
            raise
        except Exception as e:
            print(f"❌ Error loading Molmo model: {e}")
            print("Note: Molmo may require specific versions or access permissions.")
            raise
    
    def extract_features(self, images: List[Image.Image], text: Optional[str] = None) -> torch.Tensor:
        """Extract visual features from images using Molmo."""
        if self.model is None:
            self.load_model()
        
        # Default instruction for robot action prediction
        if text is None:
            text = "Describe what the robot should do next based on this image."
        
        # Process inputs
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Extract features (using hidden states from last layer)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Average pool over sequence length to get fixed-size representation
            visual_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
            
        return visual_features


class InternVLFeatureExtractor(VLMFeatureExtractor):
    """Feature extractor for InternVL vision-language model."""
    
    def load_model(self):
        """Load InternVL model and processor."""
        try:
            from transformers import AutoProcessor, AutoModel
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            print(f"✅ Loaded InternVL model: {self.model_name}")
            
        except ImportError:
            print("❌ Transformers library not found. Please install: pip install transformers")
            raise
        except Exception as e:
            print(f"❌ Error loading InternVL model: {e}")
            raise
    
    def extract_features(self, images: List[Image.Image], text: Optional[str] = None) -> torch.Tensor:
        """Extract visual features from images using InternVL."""
        if self.model is None:
            self.load_model()
        
        if text is None:
            text = "What should the robot do next?"
        
        # Process inputs
        inputs = self.processor(
            images=images,
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the pooled output or last hidden state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                visual_features = outputs.pooler_output
            else:
                visual_features = outputs.last_hidden_state.mean(dim=1)
        
        return visual_features


class ActorModelDemo:
    """Demonstration of the Actor Model with different configurations."""
    
    def __init__(self, vlm_name: str = "microsoft/molmo-7B-D-0924"):
        self.vlm_name = vlm_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize VLM feature extractor
        if "molmo" in vlm_name.lower():
            self.vlm_extractor = MolmoFeatureExtractor(vlm_name)
        elif "internvl" in vlm_name.lower():
            self.vlm_extractor = InternVLFeatureExtractor(vlm_name)
        else:
            print(f"⚠️ Unknown VLM: {vlm_name}. Using Molmo as fallback.")
            self.vlm_extractor = MolmoFeatureExtractor("microsoft/molmo-7B-D-0924")
        
        # Initialize Actor Model
        self.actor_model = None
        self.lam_model = None
    
    def setup_actor_model(
        self, 
        use_latent_actions: bool = True,
        action_dim: int = 7,
        action_horizon: int = 10
    ):
        """Setup the Actor Model."""
        config = ActorModelConfig(
            vlm_model_name=self.vlm_name,
            use_latent_actions=use_latent_actions,
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        
        self.actor_model = ActorModel(config).to(self.device)
        print(f"✅ Actor Model initialized with {sum(p.numel() for p in self.actor_model.parameters()):,} parameters")
    
    def load_lam_model(self, lam_path: str):
        """Load the Latent Action Model."""
        try:
            self.lam_model = IgorModel.from_pretrained(lam_path).to(self.device)
            print(f"✅ LAM model loaded from: {lam_path}")
        except Exception as e:
            print(f"❌ Error loading LAM model: {e}")
            print("Continuing without LAM model (latent actions will be disabled)")
    
    def predict_actions_from_images(
        self, 
        images: List[Image.Image],
        instruction: Optional[str] = None,
        use_latent_actions: bool = True,
    ) -> torch.Tensor:
        """Predict robot actions from a sequence of images."""
        if self.actor_model is None:
            raise ValueError("Actor model not initialized. Call setup_actor_model() first.")
        
        # Extract visual features
        print("🔍 Extracting visual features from VLM...")
        visual_features = self.vlm_extractor.extract_features(images, instruction)
        
        # Add sequence dimension if needed
        if len(visual_features.shape) == 1:
            visual_features = visual_features.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
        elif len(visual_features.shape) == 2:
            visual_features = visual_features.unsqueeze(1)  # [batch, 1, features]
        
        # Extract latent actions if LAM model is available
        latent_actions = None
        if use_latent_actions and self.lam_model is not None:
            print("🧠 Extracting latent actions from LAM...")
            # Convert images to video tensor for LAM
            video_tensor = self.images_to_video_tensor(images)
            latent_actions = self.extract_latent_actions(video_tensor)
        
        # Predict actions
        print("🤖 Predicting robot actions...")
        self.actor_model.eval()
        with torch.no_grad():
            actions = self.actor_model.predict_actions(visual_features, latent_actions)
        
        return actions
    
    def images_to_video_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL images to video tensor for LAM."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Convert images to tensor
        video_frames = []
        for img in images:
            frame = transform(img.convert('RGB'))
            video_frames.append(frame)
        
        # Stack into video tensor [T, C, H, W]
        video_tensor = torch.stack(video_frames)
        
        # Add batch dimension [1, T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        
        return video_tensor
    
    def extract_latent_actions(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Extract latent actions using LAM."""
        if self.lam_model is None:
            return None
        
        latent_actions = self.lam_model.idm(video_tensor)
        
        # Convert list of tensors to single tensor
        if isinstance(latent_actions, list):
            latent_actions = torch.cat([la.squeeze(1) for la in latent_actions], dim=0)
            latent_actions = latent_actions.view(1, -1, 512)  # [1, seq_len, latent_dim]
        
        return latent_actions
    
    def demonstrate_inference(self):
        """Demonstrate end-to-end inference."""
        print("🚀 Starting Actor Model demonstration...")
        
        # Create dummy images for demonstration
        dummy_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue'),
        ]
        
        # Setup models
        self.setup_actor_model()
        
        # Note: LAM model would be loaded from actual checkpoint
        # self.load_lam_model("path/to/lam/model")
        
        # Predict actions
        try:
            actions = self.predict_actions_from_images(
                images=dummy_images,
                instruction="Pick up the red object and place it in the container.",
                use_latent_actions=False,  # Set to True if LAM is loaded
            )
            
            print(f"✅ Predicted actions shape: {actions.shape}")
            print(f"Actions (first 3 timesteps):")
            for i in range(min(3, actions.shape[1])):
                action = actions[0, i].cpu().numpy()
                print(f"  Step {i+1}: {action}")
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
    
    def save_and_load_demo(self, repo_id: str = "test/villa-x-actor-demo"):
        """Demonstrate saving and loading with Hugging Face Hub."""
        if self.actor_model is None:
            self.setup_actor_model()
        
        print(f"💾 Demonstrating save/load with repo: {repo_id}")
        
        try:
            # Save to hub (would require authentication)
            # self.actor_model.push_to_hub(repo_id)
            # print(f"✅ Model saved to: https://huggingface.co/{repo_id}")
            
            # Load from hub
            # loaded_model = ActorModel.from_pretrained(repo_id)
            # print("✅ Model loaded successfully from Hub")
            
            print("ℹ️ Hub operations commented out (require authentication)")
            
        except Exception as e:
            print(f"❌ Error with Hub operations: {e}")


def main():
    """Main demonstration function."""
    print("🏠 villa-X Actor Model Demo")
    print("=" * 50)
    
    # Test with different VLMs
    vlm_models = [
        "microsoft/molmo-7B-D-0924",
        # "OpenGVLab/InternVL2-4B",  # Uncomment to test InternVL
    ]
    
    for vlm_name in vlm_models:
        print(f"\n🔧 Testing with VLM: {vlm_name}")
        print("-" * 40)
        
        try:
            demo = ActorModelDemo(vlm_name)
            demo.demonstrate_inference()
            # demo.save_and_load_demo()
            
        except Exception as e:
            print(f"❌ Error with {vlm_name}: {e}")
            continue
    
    print("\n✨ Demo completed!")
    print("\nNext steps:")
    print("1. Implement your dataset loading in train_actor_model.py")
    print("2. Train the model with real robotic data")
    print("3. Upload trained models to Hugging Face Hub")
    print("4. Share the models with the community! 🤗")


if __name__ == "__main__":
    main()

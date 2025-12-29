"""
Early Fusion Strategy (Feature-level fusion)
Survey Section 2.3: "projects information into the space that LLM can understand"

Combines features from all modalities BEFORE processing.
Advantages: Rich inter-modal interactions
Disadvantages: Computationally expensive
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Project features to common embedding space"""
    
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.projection(x)


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism (Survey: cross-attention)"""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (num_modalities, batch, embed_dim)
        Returns:
            fused: (batch, embed_dim)
        """
        # Self-attention across modalities
        attended, attention_weights = self.attention(
            features, features, features
        )
        
        # Residual connection
        attended = self.norm(attended + features)
        
        # Average across modalities
        fused = attended.mean(dim=0)
        
        return fused


class EarlyFusion:
    """
    Early Fusion: Combine features at the earliest stage
    
    Survey Reference:
    - Section 2.3: "token-level fusion, processing features into tokens"
    - "MLP-based interface to bridge the modality gap"
    
    Pipeline:
    1. Extract features from each modality
    2. Project to common embedding space (MLP)
    3. Concatenate or fuse features (attention/add)
    4. Process fused features together
    """
    
    def __init__(
        self, 
        visual_dim: int = 768,
        audio_dim: int = 768, 
        video_dim: int = 512,
        fusion_dim: int = 512,
        fusion_method: str = "attention",  # "concat", "add", "attention"
        device: str = "cpu"
    ):
        self.fusion_method = fusion_method
        self.fusion_dim = fusion_dim
        self.device = device
        
        # Feature projections (Survey: "MLP-based interface")
        self.visual_proj = FeatureProjection(visual_dim, fusion_dim).to(device)
        self.audio_proj = FeatureProjection(audio_dim, fusion_dim).to(device)
        self.video_proj = FeatureProjection(video_dim, fusion_dim).to(device)
        
        # Attention mechanism for weighted fusion
        if fusion_method == "attention":
            self.attention_fusion = AttentionFusion(
                embed_dim=fusion_dim,
                num_heads=8
            ).to(device)
        
        # Final projection for concatenated features
        if fusion_method == "concat":
            self.final_proj = nn.Linear(
                fusion_dim * 3, 
                fusion_dim
            ).to(device)
        
        print(f"[EarlyFusion] Initialized with {fusion_method} fusion")
        print(f"  Visual: {visual_dim} → {fusion_dim}")
        print(f"  Audio:  {audio_dim} → {fusion_dim}")
        print(f"  Video:  {video_dim} → {fusion_dim}")
        
        # Set to eval mode by default
        self.eval()
    
    def train(self):
        """Set modules to training mode"""
        self.visual_proj.train()
        self.audio_proj.train()
        self.video_proj.train()
        if self.fusion_method == "attention":
            self.attention_fusion.train()
        if self.fusion_method == "concat":
            self.final_proj.train()
    
    def eval(self):
        """Set modules to evaluation mode"""
        self.visual_proj.eval()
        self.audio_proj.eval()
        self.video_proj.eval()
        if self.fusion_method == "attention":
            self.attention_fusion.eval()
        if self.fusion_method == "concat":
            self.final_proj.eval()
    
    def fuse(
        self, 
        visual_features: np.ndarray,
        audio_features: np.ndarray,
        video_features: np.ndarray,
        mask: Optional[Dict[str, bool]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuse multimodal features at feature level
        
        Args:
            visual_features: Visual features (batch, visual_dim) or (visual_dim,)
            audio_features: Audio features (batch, audio_dim) or (audio_dim,)
            video_features: Video features (batch, video_dim) or (video_dim,)
            mask: Optional mask to disable certain modalities
            
        Returns:
            (fused_features, fusion_metadata)
        """
        
        # Ensure 2D (add batch dimension if needed)
        if visual_features.ndim == 1:
            visual_features = visual_features[np.newaxis, :]
        if audio_features.ndim == 1:
            audio_features = audio_features[np.newaxis, :]
        if video_features.ndim == 1:
            video_features = video_features[np.newaxis, :]
        
        # Convert to tensors
        visual_tensor = torch.FloatTensor(visual_features).to(self.device)
        audio_tensor = torch.FloatTensor(audio_features).to(self.device)
        video_tensor = torch.FloatTensor(video_features).to(self.device)
        
        with torch.no_grad():
            # Project to common space
            visual_proj = self.visual_proj(visual_tensor)
            audio_proj = self.audio_proj(audio_tensor)
            video_proj = self.video_proj(video_tensor)
            
            # Apply mask if provided
            if mask:
                if not mask.get('visual', True):
                    visual_proj = torch.zeros_like(visual_proj)
                if not mask.get('audio', True):
                    audio_proj = torch.zeros_like(audio_proj)
                if not mask.get('video', True):
                    video_proj = torch.zeros_like(video_proj)
            
            # Fuse based on method
            if self.fusion_method == "concat":
                fused, metadata = self._concat_fusion(visual_proj, audio_proj, video_proj)
            elif self.fusion_method == "add":
                fused, metadata = self._additive_fusion(visual_proj, audio_proj, video_proj)
            elif self.fusion_method == "attention":
                fused, metadata = self._attention_fusion(visual_proj, audio_proj, video_proj)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Convert back to numpy
        fused_np = fused.cpu().numpy()
        
        # Add general metadata
        metadata.update({
            "fusion_type": "early",
            "fusion_method": self.fusion_method,
            "output_shape": fused_np.shape
        })
        
        return fused_np, metadata
    
    def _concat_fusion(
        self, 
        visual: torch.Tensor, 
        audio: torch.Tensor, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Simple concatenation (Survey: "concatenated with text tokens")
        
        Advantage: Preserves all information
        Disadvantage: High dimensionality
        """
        # Concatenate along feature dimension
        concat = torch.cat([visual, audio, video], dim=-1)
        
        # Optional: Project back to fusion_dim
        if hasattr(self, 'final_proj'):
            fused = self.final_proj(concat)
        else:
            fused = concat
        
        metadata = {
            "method": "concatenation",
            "original_dims": {
                "visual": visual.shape[-1],
                "audio": audio.shape[-1],
                "video": video.shape[-1]
            }
        }
        
        return fused, metadata
    
    def _additive_fusion(
        self, 
        visual: torch.Tensor, 
        audio: torch.Tensor, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Element-wise addition with averaging
        
        Advantage: Simple, maintains dimensionality
        Disadvantage: Equal weighting (no learned importance)
        """
        # Average across modalities
        fused = (visual + audio + video) / 3.0
        
        metadata = {
            "method": "additive",
            "weights": {"visual": 1/3, "audio": 1/3, "video": 1/3}
        }
        
        return fused, metadata
    
    def _attention_fusion(
        self, 
        visual: torch.Tensor, 
        audio: torch.Tensor, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Attention-based fusion (Survey: "cross-attention layers")
        
        Advantage: Learns to weight different modalities
        Disadvantage: More parameters to train
        """
        # Stack modalities: (num_modalities, batch, dim)
        stacked = torch.stack([visual, audio, video], dim=0)
        
        # Apply attention fusion
        fused = self.attention_fusion(stacked)
        
        metadata = {
            "method": "attention",
            "num_heads": self.attention_fusion.attention.num_heads,
            "note": "Learned adaptive weighting across modalities"
        }
        
        return fused, metadata
    
    def get_fusion_stats(self, fused_features: np.ndarray) -> Dict:
        """Get statistics about fused features"""
        return {
            "shape": fused_features.shape,
            "mean": float(np.mean(fused_features)),
            "std": float(np.std(fused_features)),
            "min": float(np.min(fused_features)),
            "max": float(np.max(fused_features)),
            "fusion_method": self.fusion_method,
            "sparsity": float(np.mean(fused_features == 0))
        }
    
    def save_weights(self, path: str):
        """Save learnable weights"""
        state_dict = {
            'visual_proj': self.visual_proj.state_dict(),
            'audio_proj': self.audio_proj.state_dict(),
            'video_proj': self.video_proj.state_dict(),
        }
        
        if self.fusion_method == "attention":
            state_dict['attention_fusion'] = self.attention_fusion.state_dict()
        
        if self.fusion_method == "concat" and hasattr(self, 'final_proj'):
            state_dict['final_proj'] = self.final_proj.state_dict()
        
        torch.save(state_dict, path)
        print(f"[EarlyFusion] Weights saved to {path}")
    
    def load_weights(self, path: str):
        """Load pretrained weights"""
        state_dict = torch.load(path, map_location=self.device)
        
        self.visual_proj.load_state_dict(state_dict['visual_proj'])
        self.audio_proj.load_state_dict(state_dict['audio_proj'])
        self.video_proj.load_state_dict(state_dict['video_proj'])
        
        if 'attention_fusion' in state_dict:
            self.attention_fusion.load_state_dict(state_dict['attention_fusion'])
        
        if 'final_proj' in state_dict and hasattr(self, 'final_proj'):
            self.final_proj.load_state_dict(state_dict['final_proj'])
        
        print(f"[EarlyFusion] Weights loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Early Fusion Demo")
    print("=" * 60)
    
    # Test all fusion methods
    methods = ["concat", "add", "attention"]
    
    for method in methods:
        print(f"\n### Testing {method.upper()} fusion ###\n")
        
        # Initialize fusion module
        early_fusion = EarlyFusion(
            visual_dim=768,
            audio_dim=768,
            video_dim=512,
            fusion_dim=512,
            fusion_method=method,
            device="cpu"
        )
        
        # Mock features
        batch_size = 4
        visual_feat = np.random.randn(batch_size, 768)
        audio_feat = np.random.randn(batch_size, 768)
        video_feat = np.random.randn(batch_size, 512)
        
        # Fuse
        fused, metadata = early_fusion.fuse(visual_feat, audio_feat, video_feat)
        
        # Stats
        stats = early_fusion.get_fusion_stats(fused)
        
        print(f"Fusion Metadata:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")
        
        print(f"\nFusion Statistics:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        print("\n" + "-" * 60)
    
    # Test with modality masking
    print("\n### Testing with Modality Masking ###\n")
    
    early_fusion = EarlyFusion(fusion_method="attention")
    
    # Disable audio modality
    mask = {"visual": True, "audio": False, "video": True}
    
    fused_masked, meta_masked = early_fusion.fuse(
        np.random.randn(768),
        np.random.randn(768),
        np.random.randn(512),
        mask=mask
    )
    
    print(f"Masked Fusion (audio disabled):")
    print(f"  Shape: {fused_masked.shape}")
    print(f"  Method: {meta_masked['method']}")
    
    print("\n" + "=" * 60)
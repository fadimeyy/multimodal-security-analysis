"""
Hybrid Fusion Strategy (Context-aware fusion)
YOUR NOVEL CONTRIBUTION for the paper!

Adaptively combines early and late fusion based on:
- Scene context (indoor/outdoor, crowded/empty)
- Modality reliability (audio quality, lighting conditions)
- Temporal urgency (emergency vs routine monitoring)

Survey mentions: "CogVLM [75] plugs in a visual expert module"
We extend this idea to dynamic, context-aware fusion.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class FusionStrategy(Enum):
    """Dynamic fusion strategy selection"""
    EARLY = "early"          # Feature-level fusion
    LATE = "late"            # Decision-level fusion
    ADAPTIVE = "adaptive"    # Context-based selection


@dataclass
class SceneContext:
    """Context information for adaptive fusion"""
    scene_type: str          # indoor, outdoor, crowded, etc.
    lighting_quality: float  # 0-1, affects visual reliability
    audio_quality: float     # 0-1, affects audio reliability
    motion_level: float      # 0-1, affects temporal analysis
    urgency: str            # routine, alert, emergency


class HybridFusion:
    """
    Context-Aware Hybrid Fusion
    
    Key Innovation: Dynamically select fusion strategy based on:
    1. Scene context (lighting, audio quality)
    2. Modality reliability
    3. Task urgency
    
    Pipeline:
    1. Analyze scene context
    2. Assess modality reliability
    3. Select optimal fusion strategy
    4. Execute fusion
    5. Validate results
    """
    
    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.ADAPTIVE,
        reliability_threshold: float = 0.6
    ):
        self.default_strategy = default_strategy
        self.reliability_threshold = reliability_threshold
        
        # Strategy selection history (for learning)
        self.strategy_history = []
        self.performance_history = []
        
        # Learned weights for modalities (per context)
        self.context_weights = {
            "indoor_routine": {"visual": 0.5, "audio": 0.3, "video": 0.2},
            "indoor_emergency": {"visual": 0.4, "audio": 0.4, "video": 0.2},
            "outdoor_routine": {"visual": 0.6, "audio": 0.2, "video": 0.2},
            "outdoor_emergency": {"visual": 0.4, "audio": 0.5, "video": 0.1},
            "low_light": {"visual": 0.2, "audio": 0.6, "video": 0.2},
            "high_noise": {"visual": 0.6, "audio": 0.2, "video": 0.2}
        }
        
        print(f"[HybridFusion] Initialized with {default_strategy.value} strategy")
    
    def fuse(
        self,
        visual_features: Dict,
        audio_features: Dict,
        video_features: Dict,
        context: Optional[SceneContext] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Context-aware fusion
        
        Args:
            visual_features: Visual features with metadata
            audio_features: Audio features with metadata
            video_features: Video features with metadata
            context: Scene context information
            
        Returns:
            (fused_features, fusion_metadata)
        """
        
        # Step 1: Analyze context
        if context is None:
            context = self._infer_context(visual_features, audio_features, video_features)
        
        # Step 2: Assess modality reliability
        reliability = self._assess_reliability(
            visual_features, audio_features, video_features, context
        )
        
        # Step 3: Select fusion strategy
        strategy = self._select_strategy(context, reliability)
        
        # Step 4: Execute fusion
        fused_features, fusion_details = self._execute_fusion(
            visual_features, audio_features, video_features,
            strategy, reliability, context
        )
        
        # Step 5: Prepare metadata
        metadata = {
            "strategy": strategy.value,
            "context": {
                "scene_type": context.scene_type,
                "lighting_quality": context.lighting_quality,
                "audio_quality": context.audio_quality,
                "urgency": context.urgency
            },
            "reliability": reliability,
            "fusion_details": fusion_details
        }
        
        # Record for learning
        self.strategy_history.append(strategy)
        
        return fused_features, metadata
    
    def _infer_context(
        self,
        visual_feat: Dict,
        audio_feat: Dict,
        video_feat: Dict
    ) -> SceneContext:
        """Infer scene context from features"""
        
        # Simple heuristics (can be replaced with learned model)
        visual_quality = visual_feat.get('quality', 0.7)
        audio_quality = audio_feat.get('quality', 0.7)
        
        # Detect scene type
        objects = visual_feat.get('objects', [])
        if any(obj in ['building', 'room', 'furniture'] for obj in objects):
            scene_type = "indoor"
        else:
            scene_type = "outdoor"
        
        # Detect urgency
        audio_text = audio_feat.get('transcription', '').lower()
        if any(word in audio_text for word in ['help', 'emergency', 'yardım']):
            urgency = "emergency"
        elif visual_feat.get('threat_detected', False):
            urgency = "alert"
        else:
            urgency = "routine"
        
        # Motion level
        motion = video_feat.get('motion_detected', False)
        motion_level = 0.8 if motion else 0.3
        
        return SceneContext(
            scene_type=scene_type,
            lighting_quality=visual_quality,
            audio_quality=audio_quality,
            motion_level=motion_level,
            urgency=urgency
        )
    
    def _assess_reliability(
        self,
        visual_feat: Dict,
        audio_feat: Dict,
        video_feat: Dict,
        context: SceneContext
    ) -> Dict[str, float]:
        """
        Assess reliability of each modality given context
        KEY INNOVATION: Context-aware reliability assessment
        """
        
        reliability = {}
        
        # Visual reliability
        visual_rel = context.lighting_quality
        if context.scene_type == "outdoor" and visual_rel > 0.7:
            visual_rel += 0.1  # Outdoor usually better lit
        elif context.lighting_quality < 0.4:
            visual_rel *= 0.5  # Low light significantly affects visual
        reliability["visual"] = np.clip(visual_rel, 0, 1)
        
        # Audio reliability
        audio_rel = context.audio_quality
        if context.urgency == "emergency":
            audio_rel += 0.2  # Emergency sounds usually clear
        if context.motion_level > 0.7:
            audio_rel *= 0.8  # High motion can create noise
        reliability["audio"] = np.clip(audio_rel, 0, 1)
        
        # Video (temporal) reliability
        video_rel = (context.lighting_quality + context.audio_quality) / 2
        if context.motion_level > 0.5:
            video_rel += 0.2  # Motion benefits temporal analysis
        reliability["video"] = np.clip(video_rel, 0, 1)
        
        return reliability
    
    def _select_strategy(
        self,
        context: SceneContext,
        reliability: Dict[str, float]
    ) -> FusionStrategy:
        """
        Select optimal fusion strategy
        KEY INNOVATION: Context-aware strategy selection
        
        Rules:
        - Emergency + High reliability → Early fusion (rich interactions)
        - Routine + Mixed reliability → Late fusion (robust)
        - Low reliability modalities → Late fusion (isolate errors)
        """
        
        if self.default_strategy != FusionStrategy.ADAPTIVE:
            return self.default_strategy
        
        # Calculate average reliability
        avg_reliability = np.mean(list(reliability.values()))
        
        # Emergency situation
        if context.urgency == "emergency":
            if avg_reliability >= self.reliability_threshold:
                return FusionStrategy.EARLY  # Rich fusion for critical decisions
            else:
                return FusionStrategy.LATE   # Safer when reliability is low
        
        # Routine monitoring
        else:
            if avg_reliability >= 0.7:
                return FusionStrategy.EARLY  # Can afford rich fusion
            else:
                return FusionStrategy.LATE   # Play safe
    
    def _execute_fusion(
        self,
        visual_feat: Dict,
        audio_feat: Dict,
        video_feat: Dict,
        strategy: FusionStrategy,
        reliability: Dict[str, float],
        context: SceneContext
    ) -> Tuple[np.ndarray, Dict]:
        """Execute selected fusion strategy"""
        
        if strategy == FusionStrategy.EARLY:
            return self._early_fusion(
                visual_feat, audio_feat, video_feat, reliability
            )
        else:  # LATE
            return self._late_fusion(
                visual_feat, audio_feat, video_feat, reliability, context
            )
    
    def _early_fusion(
        self,
        visual_feat: Dict,
        audio_feat: Dict,
        video_feat: Dict,
        reliability: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Feature-level fusion with reliability weighting
        """
        
        # Extract feature vectors
        v_vec = np.array(visual_feat.get('features', np.random.randn(512)))
        a_vec = np.array(audio_feat.get('features', np.random.randn(512)))
        vid_vec = np.array(video_feat.get('features', np.random.randn(512)))
        
        # Ensure same dimension
        target_dim = 512
        v_vec = self._resize_features(v_vec, target_dim)
        a_vec = self._resize_features(a_vec, target_dim)
        vid_vec = self._resize_features(vid_vec, target_dim)
        
        # Reliability-weighted fusion
        w_v = reliability.get("visual", 0.33)
        w_a = reliability.get("audio", 0.33)
        w_vid = reliability.get("video", 0.33)
        
        # Normalize weights
        total_w = w_v + w_a + w_vid
        w_v, w_a, w_vid = w_v/total_w, w_a/total_w, w_vid/total_w
        
        # Weighted fusion
        fused = w_v * v_vec + w_a * a_vec + w_vid * vid_vec
        
        details = {
            "method": "early_fusion",
            "weights": {"visual": w_v, "audio": w_a, "video": w_vid}
        }
        
        return fused, details
    
    def _late_fusion(
        self,
        visual_feat: Dict,
        audio_feat: Dict,
        video_feat: Dict,
        reliability: Dict[str, float],
        context: SceneContext
    ) -> Tuple[np.ndarray, Dict]:
        """
        Decision-level fusion with context-aware weighting
        """
        
        # Get context-specific weights
        context_key = f"{context.scene_type}_{context.urgency}"
        weights = self.context_weights.get(
            context_key,
            {"visual": 0.4, "audio": 0.35, "video": 0.25}
        )
        
        # Adjust weights by reliability
        adj_weights = {
            k: v * reliability.get(k, 0.5) 
            for k, v in weights.items()
        }
        
        # Normalize
        total = sum(adj_weights.values())
        adj_weights = {k: v/total for k, v in adj_weights.items()}
        
        # Extract decisions (confidence scores)
        v_conf = visual_feat.get('confidence', 0.5)
        a_conf = audio_feat.get('confidence', 0.5)
        vid_conf = video_feat.get('confidence', 0.5)
        
        # Weighted decision
        final_confidence = (
            adj_weights["visual"] * v_conf +
            adj_weights["audio"] * a_conf +
            adj_weights["video"] * vid_conf
        )
        
        # Create fusion result (simplified)
        fused = np.array([final_confidence] * 512)  # Dummy feature vector
        
        details = {
            "method": "late_fusion",
            "context_key": context_key,
            "base_weights": weights,
            "reliability_adjusted_weights": adj_weights,
            "modality_confidences": {
                "visual": v_conf,
                "audio": a_conf,
                "video": vid_conf
            },
            "final_confidence": final_confidence
        }
        
        return fused, details
    
    def _resize_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize features to target dimension"""
        if len(features) == target_dim:
            return features
        elif len(features) > target_dim:
            return features[:target_dim]
        else:
            # Pad with zeros
            return np.pad(features, (0, target_dim - len(features)))
    
    def update_context_weights(
        self,
        context_key: str,
        modality: str,
        performance: float
    ):
        """
        Update context-specific weights based on feedback
        Online learning component
        """
        
        if context_key in self.context_weights:
            alpha = 0.1  # Learning rate
            current = self.context_weights[context_key].get(modality, 0.33)
            updated = (1 - alpha) * current + alpha * performance
            
            self.context_weights[context_key][modality] = updated
            
            # Renormalize
            total = sum(self.context_weights[context_key].values())
            self.context_weights[context_key] = {
                k: v/total 
                for k, v in self.context_weights[context_key].items()
            }
            
            print(f"[HybridFusion] Updated {context_key} weights: "
                  f"{self.context_weights[context_key]}")


# Example usage
if __name__ == "__main__":
    # Initialize hybrid fusion
    hybrid = HybridFusion(
        default_strategy=FusionStrategy.ADAPTIVE,
        reliability_threshold=0.6
    )
    
    # Mock features
    visual = {
        "features": np.random.randn(512),
        "quality": 0.8,
        "confidence": 0.85,
        "objects": ["person", "building"],
        "threat_detected": False
    }
    
    audio = {
        "features": np.random.randn(512),
        "quality": 0.7,
        "confidence": 0.75,
        "transcription": "normal conversation"
    }
    
    video = {
        "features": np.random.randn(512),
        "confidence": 0.70,
        "motion_detected": True
    }
    
    # Test 1: Routine scenario
    print("=== Test 1: Routine Monitoring ===")
    fused1, meta1 = hybrid.fuse(visual, audio, video)
    print(f"Strategy: {meta1['strategy']}")
    print(f"Context: {meta1['context']}")
    print(f"Reliability: {meta1['reliability']}")
    print()
    
    # Test 2: Emergency scenario
    print("=== Test 2: Emergency Scenario ===")
    audio["transcription"] = "help emergency"
    visual["threat_detected"] = True
    
    fused2, meta2 = hybrid.fuse(visual, audio, video)
    print(f"Strategy: {meta2['strategy']}")
    print(f"Context: {meta2['context']}")
    print(f"Fusion details: {meta2['fusion_details']}")
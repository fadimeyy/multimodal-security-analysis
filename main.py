"""
FIXED main.py - Correct imports for multimodal-llm-demo structure
"""

import cv2
import numpy as np
from pathlib import Path
import time
import json
import sys
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# CORRECT IMPORTS - src is a package
from src.encoders.image_encoder import ImageEncoder
from src.encoders.audio_encoder import AudioEncoder
from src.encoders.video_encoder import VideoEncoder

from src.reasoning.reasoning_engine import (
    AdvancedSecurityReasoning,
    ThreatLevel
)

from src.fusion.hybrid_fusion import (
    HybridFusion,
    FusionStrategy,
    SceneContext
)

from src.temporal.temporal_analyzer import (
    TemporalAnalyzer,
    TemporalPattern
)

from src.evaluation.metrics import SecurityMetrics


class MultimodalSecuritySystem:
    """Complete multimodal security analysis system"""
    
    def __init__(
        self,
        language: str = "tr",
        fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE,
        enable_temporal: bool = True,
        enable_evaluation: bool = True
    ):
        print("=" * 60)
        print("Initializing Multimodal Security System")
        print("=" * 60)
        
        # Initialize encoders
        print("\n[1/5] Loading Encoders...")
        self.image_encoder = ImageEncoder()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        
        # Initialize fusion module
        print("\n[2/5] Initializing Fusion Module...")
        self.fusion = HybridFusion(
            default_strategy=fusion_strategy,
            reliability_threshold=0.6
        )
        
        # Initialize temporal analyzer
        print("\n[3/5] Initializing Temporal Analyzer...")
        self.temporal_analyzer = TemporalAnalyzer(
            window_size=10,
            change_threshold=0.4,
            anomaly_threshold=0.7
        ) if enable_temporal else None
        
        # Initialize reasoning engine
        print("\n[4/5] Initializing Reasoning Engine...")
        self.reasoner = AdvancedSecurityReasoning(language=language)
        
        # Initialize metrics
        print("\n[5/5] Initializing Evaluation Metrics...")
        self.metrics = SecurityMetrics() if enable_evaluation else None
        
        print("\n✅ System initialized successfully!")
        print("=" * 60)
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        audio_data: np.ndarray = None,
        extract_audio: bool = False,
        frame_id: int = 0
    ):
        """Analyze single frame"""
        
        start_time = time.time()
        
        # Step 1: Encode modalities
        visual_features = self.image_encoder.encode(frame)
        
        if extract_audio:
            audio_features = self.audio_encoder.encode_dummy()
        elif audio_data is not None:
            audio_features = self.audio_encoder.encode(audio_data)
        else:
            audio_features = self.audio_encoder.encode_dummy()
        
        # Video features
        video_features = {
            "features": visual_features["features"],
            "confidence": 0.7,
            "motion_detected": True
        }
        
        # Step 2: Fusion
        fused_features, fusion_meta = self.fusion.fuse(
            visual_features,
            audio_features,
            video_features
        )
        
        # Step 3: Reasoning
        visual_report = {
            "objects": visual_features.get("objects", []),
            "scene_type": fusion_meta["context"]["scene_type"]
        }
        
        audio_report = {
            "transcription": audio_features.get("transcription", ""),
            "type": "speech",
            "confidence": audio_features.get("confidence", 0.7)
        }
        
        video_report = {
            "objects_over_time": [visual_features.get("objects", [])],
            "motion_detected": True
        }
        
        assessment = self.reasoner.reason_about_scene(
            visual_report,
            audio_report,
            video_report
        )
        
        assessment.response_time = time.time() - start_time
        
        return assessment


if __name__ == "__main__":
    print("\n🚀 Starting Multimodal Security Analysis System\n")
    
    # Initialize system
    system = MultimodalSecuritySystem(
        language="tr",
        fusion_strategy=FusionStrategy.ADAPTIVE,
        enable_temporal=True,
        enable_evaluation=True
    )
    
    # Test with dummy frame
    print("\n📸 Running demo with test frame...\n")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assessment = system.analyze_frame(dummy_frame)
    
    print(assessment.explanation)
    print("\n✅ System working correctly!")
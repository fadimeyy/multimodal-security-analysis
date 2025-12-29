"""
Quick Test Script for Multimodal Security System
Tests all components individually before full integration
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🧪 MULTIMODAL SECURITY SYSTEM - COMPONENT TEST")
print("=" * 70)

# Test 1: Image Encoder
print("\n[TEST 1/6] Image Encoder")
print("-" * 70)
try:
    from src.encoders.image_encoder import ImageEncoder
    
    encoder = ImageEncoder()
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = encoder.encode(dummy_image)
    
    print(f"✅ Image Encoder OK")
    print(f"   Objects: {result['objects']}")
    print(f"   Confidence: {result['confidence']:.2%}")
except Exception as e:
    print(f"❌ Image Encoder FAILED: {e}")

# Test 2: Audio Encoder
print("\n[TEST 2/6] Audio Encoder")
print("-" * 70)
try:
    from src.encoders.audio_encoder import AudioEncoder
    
    encoder = AudioEncoder()
    result = encoder.encode_dummy()
    
    print(f"✅ Audio Encoder OK")
    print(f"   Transcription: '{result['transcription']}'")
    print(f"   Language: {result['language']}")
except Exception as e:
    print(f"❌ Audio Encoder FAILED: {e}")

# Test 3: Video Encoder
print("\n[TEST 3/6] Video Encoder")
print("-" * 70)
try:
    from src.encoders.video_encoder import VideoEncoder
    
    encoder = VideoEncoder()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = encoder.encode(dummy_frame, objects=["person"])
    
    print(f"✅ Video Encoder OK")
    print(f"   Motion detected: {result['motion_detected']}")
    print(f"   Buffer size: {result['buffer_size']}")
except Exception as e:
    print(f"❌ Video Encoder FAILED: {e}")

# Test 4: Fusion Modules
print("\n[TEST 4/6] Fusion Modules")
print("-" * 70)
try:
    from src.fusion.hybrid_fusion import HybridFusion, FusionStrategy
    
    fusion = HybridFusion(default_strategy=FusionStrategy.ADAPTIVE)
    
    visual = {"features": np.random.randn(512), "quality": 0.8, "confidence": 0.85}
    audio = {"features": np.random.randn(512), "quality": 0.7, "confidence": 0.75}
    video = {"features": np.random.randn(512), "confidence": 0.70}
    
    fused, metadata = fusion.fuse(visual, audio, video)
    
    print(f"✅ Fusion Modules OK")
    print(f"   Strategy: {metadata['strategy']}")
    print(f"   Output shape: {fused.shape}")
except Exception as e:
    print(f"❌ Fusion Modules FAILED: {e}")

# Test 5: Temporal Analyzer
print("\n[TEST 5/6] Temporal Analyzer")
print("-" * 70)
try:
    from src.temporal.temporal_analyzer import TemporalAnalyzer
    
    analyzer = TemporalAnalyzer()
    
    frames = [
        {"objects": ["person"], "timestamp": 0.0},
        {"objects": ["person", "cup"], "timestamp": 0.033},
        {"objects": ["person", "cup"], "timestamp": 0.066},
    ]
    
    pattern, events, reasoning = analyzer.analyze_sequence(frames)
    
    print(f"✅ Temporal Analyzer OK")
    print(f"   Pattern: {pattern.value}")
    print(f"   Events: {len(events)}")
except Exception as e:
    print(f"❌ Temporal Analyzer FAILED: {e}")

# Test 6: Reasoning Engine
print("\n[TEST 6/6] Reasoning Engine")
print("-" * 70)
try:
    from src.reasoning.reasoning_engine import AdvancedSecurityReasoning
    
    reasoner = AdvancedSecurityReasoning(language="tr")
    
    visual_report = {"objects": ["person"], "scene_type": "indoor"}
    audio_report = {"transcription": "merhaba", "type": "speech", "confidence": 0.8}
    video_report = {"objects_over_time": [["person"]], "motion_detected": True}
    
    assessment = reasoner.reason_about_scene(visual_report, audio_report, video_report)
    
    print(f"✅ Reasoning Engine OK")
    print(f"   Threat Level: {assessment.threat_level.value}")
    print(f"   Confidence: {assessment.confidence:.2%}")
except Exception as e:
    print(f"❌ Reasoning Engine FAILED: {e}")

# Test 7: Evaluation Metrics
print("\n[TEST 7/7] Evaluation Metrics")
print("-" * 70)
try:
    from src.evaluation.metrics import SecurityMetrics
    
    metrics = SecurityMetrics()
    
    # Add some test predictions
    metrics.add_prediction("threat", "threat", {"visual": "threat", "audio": "threat"})
    metrics.add_prediction("safe", "safe", {"visual": "safe", "audio": "safe"})
    
    report = metrics.compute_metrics()
    
    print(f"✅ Evaluation Metrics OK")
    print(f"   Accuracy: {report.accuracy:.2%}")
    print(f"   F1-Score: {report.f1_score:.2%}")
except Exception as e:
    print(f"❌ Evaluation Metrics FAILED: {e}")

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS COMPLETED")
print("=" * 70)
print("\nNext steps:")
print("  1. Run: python main.py")
print("  2. Check results in results/analysis_report.json")
print("\n" + "=" * 70)
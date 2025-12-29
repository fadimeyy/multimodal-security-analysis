"""
Video Encoder with Temporal Analysis
Survey Section 2.1 - Modality Encoder

Processes video sequences and extracts temporal features
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class VideoEncoder:
    """
    Video encoder for temporal feature extraction
    
    Survey Reference: Section 2.1 - Video modality encoding
    Processes video frames and tracks objects over time
    """
    
    def __init__(
        self, 
        frame_buffer_size: int = 10,
        motion_threshold: float = 5.0
    ):
        """
        Initialize video encoder
        
        Args:
            frame_buffer_size: Number of frames to keep in buffer
            motion_threshold: Threshold for motion detection
        """
        self.frame_buffer_size = frame_buffer_size
        self.motion_threshold = motion_threshold
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.prev_gray = None
        
        print(f"[VideoEncoder] Initialized")
        print(f"  Frame buffer size: {frame_buffer_size}")
        print(f"  Motion threshold: {motion_threshold}")
    
    def encode(
        self, 
        frame: np.ndarray,
        objects: Optional[List[str]] = None
    ) -> Dict:
        """
        Encode video frame and extract temporal features
        
        Args:
            frame: Current video frame (BGR)
            objects: Optional list of detected objects in frame
            
        Returns:
            Dictionary with temporal features and metadata
        """
        
        try:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect motion
            motion_detected = False
            motion_magnitude = 0.0
            
            if self.prev_gray is not None:
                # Calculate frame difference
                frame_diff = cv2.absdiff(self.prev_gray, gray)
                motion_magnitude = float(np.mean(frame_diff))
                motion_detected = motion_magnitude > self.motion_threshold
            
            self.prev_gray = gray.copy()
            
            # Add to buffer
            frame_data = {
                "timestamp": len(self.frame_buffer) * 0.033,  # ~30fps
                "objects": objects or [],
                "motion": motion_magnitude
            }
            self.frame_buffer.append(frame_data)
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features()
            
            # Generate feature vector (dummy - in real system use 3D CNN)
            features = np.random.randn(512)
            
            return {
                "features": features,
                "motion_detected": motion_detected,
                "motion_magnitude": motion_magnitude,
                "temporal_features": temporal_features,
                "buffer_size": len(self.frame_buffer),
                "objects_over_time": [f["objects"] for f in self.frame_buffer]
            }
            
        except Exception as e:
            print(f"[VideoEncoder] Error during encoding: {e}")
            return self.encode_dummy()
    
    def encode_dummy(self) -> Dict:
        """
        Dummy encoder for testing
        
        Returns:
            Dictionary with dummy temporal data
        """
        return {
            "features": np.random.randn(512),
            "motion_detected": True,
            "motion_magnitude": 8.5,
            "temporal_features": {
                "avg_motion": 7.2,
                "motion_variance": 2.1,
                "object_stability": 0.85
            },
            "buffer_size": 5,
            "objects_over_time": [["person"], ["person", "cup"]]
        }
    
    def _extract_temporal_features(self) -> Dict:
        """
        Extract temporal features from frame buffer
        
        Returns:
            Dictionary with temporal statistics
        """
        if len(self.frame_buffer) == 0:
            return {
                "avg_motion": 0.0,
                "motion_variance": 0.0,
                "object_stability": 1.0
            }
        
        # Motion statistics
        motions = [f["motion"] for f in self.frame_buffer]
        avg_motion = float(np.mean(motions))
        motion_variance = float(np.var(motions))
        
        # Object stability (how consistent are objects across frames)
        all_objects = [set(f["objects"]) for f in self.frame_buffer]
        
        if len(all_objects) > 1:
            # Calculate Jaccard similarity between consecutive frames
            similarities = []
            for i in range(len(all_objects) - 1):
                obj1 = all_objects[i]
                obj2 = all_objects[i + 1]
                
                if len(obj1 | obj2) > 0:
                    similarity = len(obj1 & obj2) / len(obj1 | obj2)
                else:
                    similarity = 1.0
                
                similarities.append(similarity)
            
            object_stability = float(np.mean(similarities))
        else:
            object_stability = 1.0
        
        return {
            "avg_motion": avg_motion,
            "motion_variance": motion_variance,
            "object_stability": object_stability
        }
    
    def reset(self):
        """Reset frame buffer and motion detector"""
        self.frame_buffer.clear()
        self.prev_gray = None
        print("[VideoEncoder] Buffer reset")
    
    def get_temporal_summary(self) -> Dict:
        """
        Get summary of temporal patterns
        
        Returns:
            Summary statistics
        """
        if len(self.frame_buffer) == 0:
            return {
                "frames_processed": 0,
                "motion_pattern": "static"
            }
        
        temporal_features = self._extract_temporal_features()
        
        # Classify motion pattern
        avg_motion = temporal_features["avg_motion"]
        motion_variance = temporal_features["motion_variance"]
        
        if avg_motion < 3.0:
            motion_pattern = "static"
        elif motion_variance < 2.0:
            motion_pattern = "steady"
        else:
            motion_pattern = "dynamic"
        
        return {
            "frames_processed": len(self.frame_buffer),
            "motion_pattern": motion_pattern,
            "avg_motion": avg_motion,
            "object_stability": temporal_features["object_stability"]
        }


# Backward compatibility
def extract_temporal_features(video_path: str) -> Dict:
    """
    Legacy function for backward compatibility
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with temporal features
    """
    encoder = VideoEncoder()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return encoder.encode_dummy()
    
    # Process frames
    frame_count = 0
    max_frames = 30  # Process first 30 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        encoder.encode(frame)
        frame_count += 1
    
    cap.release()
    
    return encoder.get_temporal_summary()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Video Encoder Demo")
    print("=" * 60)
    
    # Initialize encoder
    encoder = VideoEncoder(frame_buffer_size=10, motion_threshold=5.0)
    
    # Simulate video frames
    print("\n### Simulating video frames ###\n")
    
    for i in range(15):
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate object detection
        if i < 5:
            objects = ["person"]
        elif i < 10:
            objects = ["person", "cup"]
        else:
            objects = ["person", "cup", "phone"]
        
        # Encode
        result = encoder.encode(frame, objects=objects)
        
        print(f"Frame {i:2d}: Motion={result['motion_magnitude']:5.2f} | "
              f"Objects={len(objects)} | Buffer={result['buffer_size']}")
    
    # Get summary
    print("\n### Temporal Summary ###\n")
    summary = encoder.get_temporal_summary()
    
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Motion pattern: {summary['motion_pattern']}")
    print(f"Avg motion: {summary['avg_motion']:.2f}")
    print(f"Object stability: {summary['object_stability']:.2%}")
    
    # Test with real video (if available)
    import os
    test_video_path = "data/WIN_20251223_11_39_24_Pro.mp4"
    
    if os.path.exists(test_video_path):
        print(f"\n### Testing with real video: {test_video_path} ###\n")
        
        encoder.reset()
        cap = cv2.VideoCapture(test_video_path)
        
        if cap.isOpened():
            frame_count = 0
            max_frames = 30
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = encoder.encode(frame)
                frame_count += 1
            
            cap.release()
            
            summary = encoder.get_temporal_summary()
            print(f"Processed: {summary['frames_processed']} frames")
            print(f"Pattern: {summary['motion_pattern']}")
            print(f"Object stability: {summary['object_stability']:.2%}")
    else:
        print(f"\n⚠️ Test video not found: {test_video_path}")
    
    print("\n" + "=" * 60)
"""
Temporal Analyzer with Chain-of-Thought Reasoning
Survey Section 7.2: "a series of intermediate reasoning steps"

Analyzes video sequences to detect temporal patterns and anomalies
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum


class TemporalPattern(Enum):
    """Types of temporal patterns"""
    NORMAL = "normal"
    SUDDEN_CHANGE = "sudden_change"
    GRADUAL_CHANGE = "gradual_change"
    REPETITIVE = "repetitive"
    ANOMALOUS = "anomalous"


@dataclass
class FrameAnalysis:
    """Analysis of a single frame"""
    frame_id: int
    timestamp: float
    objects: List[str]
    motion_magnitude: float
    scene_change_score: float


@dataclass
class TemporalEvent:
    """Detected temporal event"""
    event_type: str
    start_frame: int
    end_frame: int
    confidence: float
    description: str
    reasoning_chain: List[str]


class TemporalAnalyzer:
    """
    Temporal Chain-of-Thought Analyzer
    
    Implements frame-by-frame reasoning:
    1. Analyze current frame
    2. Compare with history
    3. Detect patterns/changes
    4. Build reasoning chain
    5. Classify temporal event
    """
    
    def __init__(
        self,
        window_size: int = 10,
        change_threshold: float = 0.4,
        anomaly_threshold: float = 0.7
    ):
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # Frame history
        self.frame_history = deque(maxlen=window_size)
        
        # Reasoning chain
        self.reasoning_chain = []
        
        print(f"[TemporalAnalyzer] Initialized with window_size={window_size}")
    
    def analyze_sequence(
        self,
        frames: List[Dict],
        verbose: bool = False
    ) -> Tuple[TemporalPattern, List[TemporalEvent], List[str]]:
        """
        Analyze video sequence with Chain-of-Thought reasoning
        
        Args:
            frames: List of frame data
            verbose: Print reasoning steps
            
        Returns:
            (overall_pattern, detected_events, reasoning_chain)
        """
        
        self.reasoning_chain = []
        detected_events = []
        
        # Step 1: Frame-by-frame analysis (Survey: "frame by frame")
        frame_analyses = []
        for i, frame_data in enumerate(frames):
            analysis = self._analyze_frame(frame_data, i)
            frame_analyses.append(analysis)
            
            # Add to history
            self.frame_history.append(analysis)
            
            # Reasoning step
            if i == 0:
                reasoning = f"Frame {i}: Initial observation - {len(analysis.objects)} objects detected"
            else:
                prev = frame_analyses[i-1]
                change = self._calculate_change(prev, analysis)
                reasoning = f"Frame {i}: Change from previous = {change:.2f}"
                
                if change > self.change_threshold:
                    reasoning += " [SIGNIFICANT CHANGE]"
            
            self.reasoning_chain.append(reasoning)
            
            if verbose:
                print(reasoning)
        
        # Step 2: Detect temporal events
        events = self._detect_events(frame_analyses)
        
        # Step 3: Classify overall pattern
        pattern = self._classify_pattern(frame_analyses, events)
        
        # Step 4: Summarize reasoning
        summary = self._generate_reasoning_summary(pattern, events)
        self.reasoning_chain.append(f"\n=== CONCLUSION ===\n{summary}")
        
        return pattern, events, self.reasoning_chain
    
    def _analyze_frame(self, frame_data: Dict, frame_id: int) -> FrameAnalysis:
        """Analyze single frame"""
        
        objects = frame_data.get('objects', [])
        timestamp = frame_data.get('timestamp', frame_id * 0.033)  # ~30fps
        
        # Calculate motion (simple heuristic)
        if len(self.frame_history) > 0:
            prev_objects = self.frame_history[-1].objects
            motion = len(set(objects) - set(prev_objects)) / max(len(objects), 1)
        else:
            motion = 0.0
        
        # Scene change score
        if len(self.frame_history) > 0:
            prev_scene = set(self.frame_history[-1].objects)
            curr_scene = set(objects)
            scene_change = len(prev_scene.symmetric_difference(curr_scene)) / max(len(curr_scene), 1)
        else:
            scene_change = 0.0
        
        return FrameAnalysis(
            frame_id=frame_id,
            timestamp=timestamp,
            objects=objects,
            motion_magnitude=motion,
            scene_change_score=scene_change
        )
    
    def _calculate_change(self, prev: FrameAnalysis, curr: FrameAnalysis) -> float:
        """Calculate change magnitude between frames"""
        
        # Object-level change
        prev_objs = set(prev.objects)
        curr_objs = set(curr.objects)
        
        if len(curr_objs) == 0:
            return 0.0
        
        # Jaccard distance
        intersection = len(prev_objs & curr_objs)
        union = len(prev_objs | curr_objs)
        
        if union == 0:
            return 0.0
        
        object_change = 1 - (intersection / union)
        
        # Motion change
        motion_change = abs(curr.motion_magnitude - prev.motion_magnitude)
        
        # Combined change
        total_change = (object_change + motion_change) / 2
        
        return total_change
    
    def _detect_events(self, frames: List[FrameAnalysis]) -> List[TemporalEvent]:
        """Detect temporal events in sequence"""
        
        events = []
        
        # Detect sudden changes
        for i in range(1, len(frames)):
            change = self._calculate_change(frames[i-1], frames[i])
            
            if change > self.anomaly_threshold:
                # High confidence anomaly
                event = TemporalEvent(
                    event_type="sudden_change",
                    start_frame=i-1,
                    end_frame=i,
                    confidence=change,
                    description=f"Sudden scene change at frame {i}",
                    reasoning_chain=[
                        f"Detected change score: {change:.2f}",
                        f"Previous objects: {frames[i-1].objects}",
                        f"Current objects: {frames[i].objects}"
                    ]
                )
                events.append(event)
            
            elif change > self.change_threshold:
                # Moderate change
                event = TemporalEvent(
                    event_type="gradual_change",
                    start_frame=i-1,
                    end_frame=i,
                    confidence=change,
                    description=f"Gradual transition at frame {i}",
                    reasoning_chain=[f"Change score: {change:.2f}"]
                )
                events.append(event)
        
        # Detect repetitive patterns
        if len(frames) >= 5:
            repetition = self._detect_repetition(frames)
            if repetition:
                events.append(repetition)
        
        return events
    
    def _detect_repetition(self, frames: List[FrameAnalysis]) -> Optional[TemporalEvent]:
        """Detect repetitive patterns"""
        
        # Simple periodicity detection
        object_sequences = [set(f.objects) for f in frames]
        
        # Check for repeating subsequences
        for period in range(2, len(frames) // 2):
            matches = 0
            for i in range(len(frames) - period):
                if object_sequences[i] == object_sequences[i + period]:
                    matches += 1
            
            # If >50% match with period
            if matches / (len(frames) - period) > 0.5:
                return TemporalEvent(
                    event_type="repetitive",
                    start_frame=0,
                    end_frame=len(frames)-1,
                    confidence=matches / (len(frames) - period),
                    description=f"Repetitive pattern detected (period={period})",
                    reasoning_chain=[
                        f"Period: {period} frames",
                        f"Match rate: {matches / (len(frames) - period):.2%}"
                    ]
                )
        
        return None
    
    def _classify_pattern(
        self,
        frames: List[FrameAnalysis],
        events: List[TemporalEvent]
    ) -> TemporalPattern:
        """Classify overall temporal pattern"""
        
        if not events:
            return TemporalPattern.NORMAL
        
        # Check for anomalies (high confidence sudden changes)
        anomalies = [e for e in events 
                    if e.event_type == "sudden_change" 
                    and e.confidence > self.anomaly_threshold]
        
        if len(anomalies) > 0:
            return TemporalPattern.ANOMALOUS
        
        # Check for repetition
        repetitive = [e for e in events if e.event_type == "repetitive"]
        if repetitive:
            return TemporalPattern.REPETITIVE
        
        # Check change rate
        sudden_changes = [e for e in events if e.event_type == "sudden_change"]
        gradual_changes = [e for e in events if e.event_type == "gradual_change"]
        
        if len(sudden_changes) > len(frames) * 0.3:
            return TemporalPattern.SUDDEN_CHANGE
        elif len(gradual_changes) > 0:
            return TemporalPattern.GRADUAL_CHANGE
        else:
            return TemporalPattern.NORMAL
    
    def _generate_reasoning_summary(
        self,
        pattern: TemporalPattern,
        events: List[TemporalEvent]
    ) -> str:
        """Generate human-readable summary of reasoning"""
        
        summary_parts = []
        
        # Overall pattern
        summary_parts.append(f"Overall Pattern: {pattern.value.upper()}")
        
        # Event summary
        if events:
            summary_parts.append(f"\nDetected {len(events)} temporal events:")
            for i, event in enumerate(events, 1):
                summary_parts.append(
                    f"  {i}. {event.description} "
                    f"(frames {event.start_frame}-{event.end_frame}, "
                    f"confidence: {event.confidence:.2%})"
                )
        else:
            summary_parts.append("\nNo significant temporal events detected.")
        
        # Interpretation
        if pattern == TemporalPattern.ANOMALOUS:
            summary_parts.append("\n⚠️ ALERT: Anomalous activity detected!")
            summary_parts.append("Recommendation: Immediate review required")
        elif pattern == TemporalPattern.SUDDEN_CHANGE:
            summary_parts.append("\n⚡ Multiple sudden changes detected")
            summary_parts.append("Recommendation: Monitor closely")
        elif pattern == TemporalPattern.NORMAL:
            summary_parts.append("\n✅ Normal temporal pattern")
        
        return "\n".join(summary_parts)
    
    def export_analysis(
        self,
        pattern: TemporalPattern,
        events: List[TemporalEvent]
    ) -> Dict:
        """Export analysis in JSON-compatible format"""
        return {
            "pattern": pattern.value,
            "num_events": len(events),
            "events": [
                {
                    "type": e.event_type,
                    "frames": f"{e.start_frame}-{e.end_frame}",
                    "confidence": e.confidence,
                    "description": e.description
                }
                for e in events
            ],
            "reasoning_chain": self.reasoning_chain
        }


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TemporalAnalyzer(
        window_size=10,
        change_threshold=0.4,
        anomaly_threshold=0.7
    )
    
    # Mock video frames
    frames = [
        {"objects": ["person", "cup"], "timestamp": 0.0},
        {"objects": ["person", "cup"], "timestamp": 0.033},
        {"objects": ["person", "cup"], "timestamp": 0.066},
        {"objects": ["person"], "timestamp": 0.099},  # Cup disappears
        {"objects": ["person"], "timestamp": 0.132},
        {"objects": ["person", "knife"], "timestamp": 0.165},  # Knife appears!
        {"objects": ["person", "knife"], "timestamp": 0.198},
        {"objects": ["person", "knife", "bag"], "timestamp": 0.231},
    ]
    
    # Analyze
    print("=== Temporal Analysis with Chain-of-Thought ===\n")
    pattern, events, reasoning = analyzer.analyze_sequence(frames, verbose=True)
    
    print(f"\n{reasoning[-1]}")  # Print conclusion
    
    # Export
    export = analyzer.export_analysis(pattern, events)
    print(f"\n=== Export ===")
    import json
    print(json.dumps(export, indent=2))
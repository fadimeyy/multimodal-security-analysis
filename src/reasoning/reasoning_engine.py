"""
Advanced Security Reasoning Engine
Inspired by Survey: "A Survey on Multimodal Large Language Models" (Section 7)
Implements: Multimodal CoT, Cross-Modal Verification, Temporal Reasoning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ThreatLevel(Enum):
    """Threat classification levels"""
    SAFE = "safe"
    UNCERTAIN = "uncertain"
    SUSPICIOUS = "suspicious"
    DANGER = "danger"

@dataclass
class ReasoningStep:
    """Single step in chain-of-thought reasoning"""
    timestamp: float
    observation: str
    modality: str
    confidence: float
    evidence: Dict

@dataclass
class ThreatAssessment:
    """Final threat assessment result"""
    threat_level: ThreatLevel
    confidence: float
    reasoning_chain: List[ReasoningStep]
    explanation: str
    recommendations: List[str]
    timestamp: float

class AdvancedSecurityReasoning:
    """
    Main reasoning engine implementing:
    - Multimodal Chain-of-Thought (M-CoT)
    - Cross-Modal Verification
    - Temporal Reasoning
    - Explainable AI
    """
    
    def __init__(self, language: str = "tr"):
        self.language = language
        self.reasoning_history = []
        self.temporal_window = []
        self.max_history = 10
        
        # Threat keywords for different modalities
        self.threat_keywords = {
            "visual": ["weapon", "knife", "gun", "fight", "violence", "fire"],
            "audio": ["help", "scream", "gunshot", "glass breaking", "alarm", 
                     "yardım", "çığlık", "cam kırılması", "silah sesi"],
            "temporal": ["running", "falling", "sudden movement", "crowd panic"]
        }
        
        # Confidence thresholds
        self.thresholds = {
            "high_confidence": 0.85,
            "medium_confidence": 0.60,
            "low_confidence": 0.40
        }
    
    def reason_about_scene(
        self, 
        visual_report: Dict, 
        audio_report: Dict, 
        video_report: Dict
    ) -> ThreatAssessment:
        """
        Main reasoning pipeline (Survey Section 7.2 - Multimodal CoT)
        
        Args:
            visual_report: Visual analysis results
            audio_report: Audio analysis results
            video_report: Video temporal analysis results
            
        Returns:
            ThreatAssessment with full reasoning chain
        """
        
        # Step 1: Initialize reasoning chain
        reasoning_chain = []
        current_time = np.random.random()  # Simulated timestamp
        
        # Step 2: Analyze each modality (Survey Section 2.3 - Token-level fusion)
        visual_step = self._analyze_visual(visual_report, current_time)
        reasoning_chain.append(visual_step)
        
        audio_step = self._analyze_audio(audio_report, current_time)
        reasoning_chain.append(audio_step)
        
        video_step = self._analyze_temporal(video_report, current_time)
        reasoning_chain.append(video_step)
        
        # Step 3: Cross-modal verification (Survey Section 6 - Hallucination mitigation)
        verification_result = self._cross_modal_verification(
            visual_step, audio_step, video_step
        )
        
        # Step 4: Temporal consistency check
        temporal_consistency = self._check_temporal_consistency(reasoning_chain)
        
        # Step 5: Calculate final threat level
        threat_level, confidence = self._calculate_threat_level(
            verification_result, temporal_consistency
        )
        
        # Step 6: Generate explanation (Survey Section 7.3 - LLM-Aided Reasoning)
        explanation = self._generate_explanation(
            reasoning_chain, threat_level, confidence
        )
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(threat_level)
        
        # Step 8: Update history
        self._update_history(reasoning_chain)
        
        return ThreatAssessment(
            threat_level=threat_level,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            explanation=explanation,
            recommendations=recommendations,
            timestamp=current_time
        )
    
    def _analyze_visual(self, visual_report: Dict, timestamp: float) -> ReasoningStep:
        """Analyze visual modality with threat detection"""
        
        objects = visual_report.get('objects', [])
        scene_type = visual_report.get('scene_type', 'unknown')
        
        # Detect threatening objects
        threat_objects = [obj for obj in objects 
                         if any(keyword in obj.lower() 
                               for keyword in self.threat_keywords['visual'])]
        
        # Calculate confidence
        confidence = min(1.0, len(objects) * 0.1 + 0.5)
        
        observation = f"Visual: Detected {len(objects)} objects"
        if threat_objects:
            observation += f" including concerning items: {', '.join(threat_objects)}"
            confidence = min(1.0, confidence + 0.3)
        
        return ReasoningStep(
            timestamp=timestamp,
            observation=observation,
            modality="visual",
            confidence=confidence,
            evidence={
                "objects": objects,
                "threat_objects": threat_objects,
                "scene_type": scene_type
            }
        )
    
    def _analyze_audio(self, audio_report: Dict, timestamp: float) -> ReasoningStep:
        """Analyze audio modality with threat detection"""
        
        transcription = audio_report.get('transcription', '').lower()
        audio_type = audio_report.get('type', 'unknown')
        
        # Detect threatening sounds/words
        threat_detected = any(keyword in transcription 
                            for keyword in self.threat_keywords['audio'])
        
        confidence = audio_report.get('confidence', 0.7)
        
        observation = f"Audio: '{transcription}'"
        if threat_detected:
            observation += " [⚠ THREAT KEYWORDS DETECTED]"
            confidence = min(1.0, confidence + 0.2)
        
        return ReasoningStep(
            timestamp=timestamp,
            observation=observation,
            modality="audio",
            confidence=confidence,
            evidence={
                "transcription": transcription,
                "audio_type": audio_type,
                "threat_detected": threat_detected
            }
        )
    
    def _analyze_temporal(self, video_report: Dict, timestamp: float) -> ReasoningStep:
        """Analyze temporal patterns in video"""
        
        objects_over_time = video_report.get('objects_over_time', [])
        motion_detected = video_report.get('motion_detected', False)
        
        # Detect unusual patterns
        if len(objects_over_time) > 0:
            # Check for sudden changes
            object_changes = len(set([tuple(frame) for frame in objects_over_time]))
            unusual_activity = object_changes > len(objects_over_time) * 0.5
        else:
            unusual_activity = False
        
        confidence = 0.6 if motion_detected else 0.4
        
        observation = f"Video: Tracking {len(objects_over_time)} frames"
        if unusual_activity:
            observation += " with unusual activity patterns"
            confidence += 0.2
        
        return ReasoningStep(
            timestamp=timestamp,
            observation=observation,
            modality="temporal",
            confidence=confidence,
            evidence={
                "frames": len(objects_over_time),
                "motion_detected": motion_detected,
                "unusual_activity": unusual_activity
            }
        )
    
    def _cross_modal_verification(
        self, 
        visual: ReasoningStep, 
        audio: ReasoningStep, 
        temporal: ReasoningStep
    ) -> Dict:
        """
        Cross-modal verification to reduce false alarms
        Inspired by Survey Section 6 - Hallucination mitigation
        """
        
        # Count threat indicators across modalities
        threat_indicators = 0
        total_confidence = 0
        
        if visual.evidence.get('threat_objects'):
            threat_indicators += 1
            total_confidence += visual.confidence
        
        if audio.evidence.get('threat_detected'):
            threat_indicators += 1
            total_confidence += audio.confidence
        
        if temporal.evidence.get('unusual_activity'):
            threat_indicators += 1
            total_confidence += temporal.confidence
        
        # Multi-modal agreement
        agreement_score = threat_indicators / 3.0
        avg_confidence = total_confidence / max(threat_indicators, 1)
        
        # Cross-modal consistency check
        if threat_indicators >= 2:
            verified = True
            verification_confidence = min(1.0, avg_confidence * agreement_score * 1.5)
        elif threat_indicators == 1:
            verified = False
            verification_confidence = avg_confidence * 0.5
        else:
            verified = False
            verification_confidence = 0.3
        
        return {
            "verified": verified,
            "confidence": verification_confidence,
            "agreement_score": agreement_score,
            "threat_indicators": threat_indicators
        }
    
    def _check_temporal_consistency(self, reasoning_chain: List[ReasoningStep]) -> float:
        """Check consistency with historical observations"""
        
        if len(self.reasoning_history) == 0:
            return 0.7  # Neutral score for first observation
        
        # Compare with recent history
        recent_history = self.reasoning_history[-3:]
        
        # Simple consistency metric
        current_threats = sum(1 for step in reasoning_chain 
                            if step.confidence > self.thresholds['medium_confidence'])
        
        historical_avg = np.mean([
            sum(1 for step in hist if step.confidence > self.thresholds['medium_confidence'])
            for hist in recent_history
        ])
        
        # Sudden spike in threats = potentially inconsistent
        if current_threats > historical_avg * 2:
            return 0.5  # Lower consistency score
        else:
            return 0.9  # High consistency
    
    def _calculate_threat_level(
        self, 
        verification: Dict, 
        temporal_consistency: float
    ) -> Tuple[ThreatLevel, float]:
        """Calculate final threat level and confidence"""
        
        base_confidence = verification['confidence']
        adjusted_confidence = base_confidence * temporal_consistency
        
        # Determine threat level
        if not verification['verified']:
            if adjusted_confidence < self.thresholds['low_confidence']:
                return ThreatLevel.SAFE, adjusted_confidence
            else:
                return ThreatLevel.UNCERTAIN, adjusted_confidence
        else:
            if adjusted_confidence >= self.thresholds['high_confidence']:
                return ThreatLevel.DANGER, adjusted_confidence
            elif adjusted_confidence >= self.thresholds['medium_confidence']:
                return ThreatLevel.SUSPICIOUS, adjusted_confidence
            else:
                return ThreatLevel.UNCERTAIN, adjusted_confidence
    
    def _generate_explanation(
        self, 
        reasoning_chain: List[ReasoningStep], 
        threat_level: ThreatLevel,
        confidence: float
    ) -> str:
        """
        Generate human-readable explanation
        Inspired by Survey Section 7.3 - LLM-Aided Visual Reasoning
        """
        
        if self.language == "tr":
            explanation = self._generate_turkish_explanation(
                reasoning_chain, threat_level, confidence
            )
        else:
            explanation = self._generate_english_explanation(
                reasoning_chain, threat_level, confidence
            )
        
        return explanation
    
    def _generate_turkish_explanation(
        self, 
        chain: List[ReasoningStep], 
        level: ThreatLevel,
        conf: float
    ) -> str:
        """Generate Turkish explanation"""
        
        parts = ["🔍 Güvenlik Analizi Sonucu:\n"]
        
        # Add modality observations
        for step in chain:
            parts.append(f"• {step.observation} (Güven: {step.confidence:.2f})")
        
        # Add assessment
        parts.append(f"\n📊 Genel Değerlendirme:")
        parts.append(f"Tehdit Seviyesi: {level.value.upper()}")
        parts.append(f"Güven Skoru: {conf:.2%}")
        
        # Add reasoning
        if level == ThreatLevel.DANGER:
            parts.append("\n⚠️ ACİL: Ciddi tehdit tespit edildi!")
            parts.append("Birden fazla modalite tehlike sinyali gösteriyor.")
        elif level == ThreatLevel.SUSPICIOUS:
            parts.append("\n⚡ DİKKAT: Şüpheli aktivite tespit edildi.")
            parts.append("Durumu yakından izleyin.")
        elif level == ThreatLevel.UNCERTAIN:
            parts.append("\n❓ BELİRSİZ: Net bir tehdit yok ama dikkatli olun.")
        else:
            parts.append("\n✅ GÜVENLİ: Normal aktivite, tehdit yok.")
        
        return "\n".join(parts)
    
    def _generate_english_explanation(
        self, 
        chain: List[ReasoningStep], 
        level: ThreatLevel,
        conf: float
    ) -> str:
        """Generate English explanation"""
        
        parts = ["🔍 Security Analysis Result:\n"]
        
        for step in chain:
            parts.append(f"• {step.observation} (Confidence: {step.confidence:.2f})")
        
        parts.append(f"\n📊 Overall Assessment:")
        parts.append(f"Threat Level: {level.value.upper()}")
        parts.append(f"Confidence Score: {conf:.2%}")
        
        if level == ThreatLevel.DANGER:
            parts.append("\n⚠️ URGENT: Serious threat detected!")
            parts.append("Multiple modalities indicate danger.")
        elif level == ThreatLevel.SUSPICIOUS:
            parts.append("\n⚡ ALERT: Suspicious activity detected.")
            parts.append("Monitor the situation closely.")
        elif level == ThreatLevel.UNCERTAIN:
            parts.append("\n❓ UNCLEAR: No clear threat but stay vigilant.")
        else:
            parts.append("\n✅ SAFE: Normal activity, no threats.")
        
        return "\n".join(parts)
    
    def _generate_recommendations(self, threat_level: ThreatLevel) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = {
            ThreatLevel.DANGER: [
                "Alert security personnel immediately",
                "Lock down affected areas",
                "Initiate emergency protocols",
                "Contact law enforcement if necessary"
            ],
            ThreatLevel.SUSPICIOUS: [
                "Increase monitoring in the area",
                "Deploy additional security personnel",
                "Review recent footage",
                "Prepare emergency response team"
            ],
            ThreatLevel.UNCERTAIN: [
                "Continue routine monitoring",
                "Log incident for analysis",
                "Review system calibration"
            ],
            ThreatLevel.SAFE: [
                "Maintain normal operations",
                "Continue periodic monitoring"
            ]
        }
        
        return recommendations.get(threat_level, [])
    
    def _update_history(self, reasoning_chain: List[ReasoningStep]):
        """Update reasoning history for temporal analysis"""
        self.reasoning_history.append(reasoning_chain)
        
        # Keep only recent history
        if len(self.reasoning_history) > self.max_history:
            self.reasoning_history.pop(0)
    
    def export_reasoning_chain(self, assessment: ThreatAssessment) -> Dict:
        """Export reasoning chain for analysis/debugging"""
        return {
            "threat_level": assessment.threat_level.value,
            "confidence": assessment.confidence,
            "timestamp": assessment.timestamp,
            "reasoning_steps": [
                {
                    "timestamp": step.timestamp,
                    "modality": step.modality,
                    "observation": step.observation,
                    "confidence": step.confidence,
                    "evidence": step.evidence
                }
                for step in assessment.reasoning_chain
            ],
            "explanation": assessment.explanation,
            "recommendations": assessment.recommendations
        }


# Example usage
if __name__ == "__main__":
    # Initialize reasoning engine
    reasoner = AdvancedSecurityReasoning(language="tr")
    
    # Mock reports
    visual = {
        "objects": ["person", "knife", "bag"],
        "scene_type": "indoor"
    }
    
    audio = {
        "transcription": "yardım edin",
        "type": "speech",
        "confidence": 0.85
    }
    
    video = {
        "objects_over_time": [["person"], ["person", "knife"]],
        "motion_detected": True
    }
    
    # Reason about scene
    assessment = reasoner.reason_about_scene(visual, audio, video)
    
    # Print results
    print(assessment.explanation)
    print("\n📋 Öneriler:")
    for rec in assessment.recommendations:
        print(f"  • {rec}")
    
    # Export for logging
    export_data = reasoner.export_reasoning_chain(assessment)
    print(f"\n📄 Export: {json.dumps(export_data, indent=2, ensure_ascii=False)}")
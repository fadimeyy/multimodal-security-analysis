"""
Late Fusion Strategy (Decision-level fusion)
Survey Section 2.3: Each modality processed independently

Process each modality separately, then combine decisions.
Advantages: Modular, interpretable, computationally efficient
Disadvantages: Limited inter-modal interactions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ModalityDecision:
    """Decision from a single modality"""
    modality: str
    confidence: float
    prediction: str
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def __repr__(self):
        return (f"ModalityDecision(modality={self.modality}, "
                f"prediction={self.prediction}, confidence={self.confidence:.2f})")


class LateFusion:
    """
    Late Fusion: Process modalities independently, fuse decisions
    
    Survey Reference:
    - Section 2.3: Decision-level fusion
    - Section 4: Ensemble methods for evaluation
    
    Pipeline:
    1. Process each modality independently
    2. Get decision from each modality
    3. Combine decisions using voting, averaging, or weighted sum
    """
    
    def __init__(
        self,
        fusion_method: str = "weighted",  # "voting", "average", "weighted", "max", "product"
        modality_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.5
    ):
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        
        # Default weights (can be learned from validation set)
        self.modality_weights = modality_weights or {
            "visual": 0.4,
            "audio": 0.35,
            "video": 0.25
        }
        
        # Normalize weights
        total = sum(self.modality_weights.values())
        self.modality_weights = {
            k: v/total for k, v in self.modality_weights.items()
        }
        
        # Performance tracking for online learning
        self.modality_performance = defaultdict(list)
        
        print(f"[LateFusion] Initialized with {fusion_method} fusion")
        print(f"  Weights: {self.modality_weights}")
        print(f"  Confidence threshold: {confidence_threshold}")
    
    def fuse_decisions(
        self,
        visual_decision: ModalityDecision,
        audio_decision: ModalityDecision,
        video_decision: ModalityDecision
    ) -> Tuple[str, float, Dict]:
        """
        Fuse decisions from multiple modalities
        
        Args:
            visual_decision: Decision from visual modality
            audio_decision: Decision from audio modality
            video_decision: Decision from video modality
            
        Returns:
            (final_prediction, confidence, fusion_metadata)
        """
        
        decisions = [visual_decision, audio_decision, video_decision]
        
        # Select fusion method
        if self.fusion_method == "voting":
            return self._majority_voting(decisions)
        elif self.fusion_method == "average":
            return self._average_fusion(decisions)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(decisions)
        elif self.fusion_method == "max":
            return self._max_confidence_fusion(decisions)
        elif self.fusion_method == "product":
            return self._product_fusion(decisions)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _majority_voting(
        self, 
        decisions: List[ModalityDecision]
    ) -> Tuple[str, float, Dict]:
        """
        Majority voting (Survey: simple ensemble)
        Each modality gets one vote
        
        Rule: Select prediction with most votes
        Confidence: Proportion of votes
        """
        
        # Count votes
        votes = defaultdict(int)
        vote_details = defaultdict(list)
        
        for decision in decisions:
            pred = decision.prediction
            votes[pred] += 1
            vote_details[pred].append({
                "modality": decision.modality,
                "confidence": decision.confidence
            })
        
        # Get majority
        final_pred = max(votes, key=votes.get)
        vote_count = votes[final_pred]
        
        # Confidence = proportion of votes
        confidence = vote_count / len(decisions)
        
        # Average confidence of agreeing modalities
        agreeing_confidences = [
            d['confidence'] for d in vote_details[final_pred]
        ]
        avg_agreeing_confidence = np.mean(agreeing_confidences)
        
        # Combine vote proportion with average confidence
        final_confidence = (confidence + avg_agreeing_confidence) / 2
        
        metadata = {
            "method": "voting",
            "votes": dict(votes),
            "vote_details": dict(vote_details),
            "modality_predictions": {
                d.modality: d.prediction for d in decisions
            },
            "agreement_rate": confidence
        }
        
        return final_pred, final_confidence, metadata
    
    def _average_fusion(
        self, 
        decisions: List[ModalityDecision]
    ) -> Tuple[str, float, Dict]:
        """
        Average confidence scores
        
        Rule: Select prediction with highest average confidence
        Confidence: Average of supporting confidences
        """
        
        # Group by prediction
        pred_confidences = defaultdict(list)
        pred_modalities = defaultdict(list)
        
        for decision in decisions:
            pred = decision.prediction
            pred_confidences[pred].append(decision.confidence)
            pred_modalities[pred].append(decision.modality)
        
        # Average confidences
        avg_confidences = {
            pred: np.mean(confs) 
            for pred, confs in pred_confidences.items()
        }
        
        # Select best
        final_pred = max(avg_confidences, key=avg_confidences.get)
        confidence = avg_confidences[final_pred]
        
        metadata = {
            "method": "average",
            "avg_confidences": avg_confidences,
            "supporting_modalities": pred_modalities[final_pred],
            "modality_confidences": {
                d.modality: d.confidence for d in decisions
            }
        }
        
        return final_pred, confidence, metadata
    
    def _weighted_fusion(
        self, 
        decisions: List[ModalityDecision]
    ) -> Tuple[str, float, Dict]:
        """
        Weighted fusion (Survey: learned weights)
        
        Rule: Weight decisions by modality importance
        Confidence: Weighted sum of confidences
        
        This is the most common approach in literature.
        """
        
        # Group by prediction
        pred_scores = defaultdict(float)
        pred_contributions = defaultdict(list)
        
        for decision in decisions:
            pred = decision.prediction
            weight = self.modality_weights.get(decision.modality, 1.0)
            weighted_conf = decision.confidence * weight
            
            pred_scores[pred] += weighted_conf
            pred_contributions[pred].append({
                "modality": decision.modality,
                "weight": weight,
                "confidence": decision.confidence,
                "contribution": weighted_conf
            })
        
        # Normalize scores (sum of weights)
        total_possible = sum(self.modality_weights.values())
        pred_scores = {k: v/total_possible for k, v in pred_scores.items()}
        
        # Select best
        final_pred = max(pred_scores, key=pred_scores.get)
        confidence = pred_scores[final_pred]
        
        metadata = {
            "method": "weighted",
            "weights": self.modality_weights,
            "weighted_scores": dict(pred_scores),
            "contributions": dict(pred_contributions),
            "modality_decisions": {
                d.modality: {
                    "prediction": d.prediction,
                    "confidence": d.confidence,
                    "weight": self.modality_weights.get(d.modality, 1.0)
                }
                for d in decisions
            }
        }
        
        return final_pred, confidence, metadata
    
    def _max_confidence_fusion(
        self, 
        decisions: List[ModalityDecision]
    ) -> Tuple[str, float, Dict]:
        """
        Select modality with highest confidence
        
        Rule: Trust the most confident expert
        Confidence: Max confidence value
        
        Useful when one modality is clearly more reliable.
        """
        
        # Find max confidence
        best_decision = max(decisions, key=lambda d: d.confidence)
        
        # Check if confidence is above threshold
        if best_decision.confidence < self.confidence_threshold:
            # If no confident decision, fall back to weighted fusion
            return self._weighted_fusion(decisions)
        
        metadata = {
            "method": "max_confidence",
            "selected_modality": best_decision.modality,
            "all_confidences": {
                d.modality: d.confidence for d in decisions
            },
            "confidence_gap": best_decision.confidence - sorted(
                [d.confidence for d in decisions], reverse=True
            )[1] if len(decisions) > 1 else 0
        }
        
        return best_decision.prediction, best_decision.confidence, metadata
    
    def _product_fusion(
        self, 
        decisions: List[ModalityDecision]
    ) -> Tuple[str, float, Dict]:
        """
        Product of confidences (multiplicative fusion)
        
        Rule: Multiply confidences (penalizes disagreement)
        Confidence: Normalized product
        
        All modalities must agree for high confidence.
        """
        
        # Group by prediction
        pred_products = {}
        
        for pred in set(d.prediction for d in decisions):
            # Get confidences for this prediction
            confidences = []
            for decision in decisions:
                if decision.prediction == pred:
                    confidences.append(decision.confidence)
                else:
                    # Penalize disagreement
                    confidences.append(1 - decision.confidence)
            
            # Product of confidences
            product = np.prod(confidences)
            pred_products[pred] = product
        
        # Normalize
        total = sum(pred_products.values())
        if total > 0:
            pred_products = {k: v/total for k, v in pred_products.items()}
        
        # Select best
        final_pred = max(pred_products, key=pred_products.get)
        confidence = pred_products[final_pred]
        
        metadata = {
            "method": "product",
            "products": pred_products,
            "modality_contributions": {
                d.modality: d.confidence for d in decisions
            },
            "note": "Product fusion penalizes disagreement"
        }
        
        return final_pred, confidence, metadata
    
    def update_weights(
        self, 
        modality: str, 
        performance: float,
        learning_rate: float = 0.1
    ):
        """
        Dynamically update modality weights based on performance
        (Online learning / Adaptive weighting)
        
        Args:
            modality: Modality name
            performance: Performance metric (0-1)
            learning_rate: How fast to adapt (0-1)
        """
        
        if modality not in self.modality_weights:
            print(f"[LateFusion] Warning: Unknown modality {modality}")
            return
        
        # Record performance
        self.modality_performance[modality].append(performance)
        
        # Exponential moving average
        current = self.modality_weights[modality]
        updated = (1 - learning_rate) * current + learning_rate * performance
        self.modality_weights[modality] = updated
        
        # Renormalize all weights
        total = sum(self.modality_weights.values())
        self.modality_weights = {
            k: v/total for k, v in self.modality_weights.items()
        }
        
        print(f"[LateFusion] Updated {modality}: {current:.3f} → {updated:.3f}")
        print(f"  New weights: {self.modality_weights}")
    
    def get_weight_statistics(self) -> Dict:
        """Get statistics about weight evolution"""
        return {
            "current_weights": self.modality_weights,
            "performance_history": {
                mod: {
                    "samples": len(perfs),
                    "mean": np.mean(perfs) if perfs else 0,
                    "std": np.std(perfs) if perfs else 0,
                    "recent": perfs[-5:] if perfs else []
                }
                for mod, perfs in self.modality_performance.items()
            }
        }
    
    def reset_weights(self):
        """Reset to default equal weights"""
        self.modality_weights = {"visual": 1/3, "audio": 1/3, "video": 1/3}
        self.modality_performance.clear()
        print("[LateFusion] Weights reset to default")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Late Fusion Demo")
    print("=" * 60)
    
    # Test all fusion methods
    methods = ["voting", "average", "weighted", "max", "product"]
    
    # Create sample decisions
    visual_dec = ModalityDecision(
        modality="visual",
        confidence=0.85,
        prediction="threat",
        features=np.random.randn(512)
    )
    
    audio_dec = ModalityDecision(
        modality="audio",
        confidence=0.75,
        prediction="threat",
        features=np.random.randn(512)
    )
    
    video_dec = ModalityDecision(
        modality="video",
        confidence=0.60,
        prediction="safe",
        features=np.random.randn(512)
    )
    
    print("\nInput Decisions:")
    print(f"  {visual_dec}")
    print(f"  {audio_dec}")
    print(f"  {video_dec}")
    print()
    
    for method in methods:
        print(f"### {method.upper()} Fusion ###\n")
        
        # Initialize late fusion
        late_fusion = LateFusion(
            fusion_method=method,
            modality_weights={"visual": 0.4, "audio": 0.4, "video": 0.2}
        )
        
        # Fuse decisions
        prediction, confidence, metadata = late_fusion.fuse_decisions(
            visual_dec, audio_dec, video_dec
        )
        
        print(f"Result:")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Method: {metadata['method']}")
        
        # Show method-specific details
        if method == "voting":
            print(f"  Votes: {metadata['votes']}")
            print(f"  Agreement: {metadata['agreement_rate']:.2%}")
        elif method == "weighted":
            print(f"  Weighted scores: {metadata['weighted_scores']}")
        elif method == "max":
            print(f"  Selected: {metadata['selected_modality']}")
        
        print("\n" + "-" * 60 + "\n")
    
    # Test online weight adaptation
    print("### Online Weight Adaptation ###\n")
    
    late_fusion = LateFusion(fusion_method="weighted")
    
    print("Simulating feedback...")
    # Simulate that visual was very accurate
    late_fusion.update_weights("visual", performance=0.95)
    
    # Simulate that audio was less reliable
    late_fusion.update_weights("audio", performance=0.60)
    
    # Simulate that video was okay
    late_fusion.update_weights("video", performance=0.75)
    
    print("\nWeight Statistics:")
    stats = late_fusion.get_weight_statistics()
    for mod, perf in stats['performance_history'].items():
        print(f"  {mod}:")
        print(f"    Mean performance: {perf['mean']:.2%}")
        print(f"    Samples: {perf['samples']}")
    
    print("\n" + "=" * 60)
"""
Security System Evaluation Metrics
Survey Section 4: "accuracy, precision, recall, and F1-Score"

Extended for security applications:
- False Alarm Rate (FAR)
- Detection Rate (DR)
- Response Time
- Cross-Modal Agreement
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Security-specific metrics
    false_alarm_rate: float
    detection_rate: float
    avg_response_time: float
    
    # Cross-modal metrics
    cross_modal_agreement: float
    modality_contributions: Dict[str, float]
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Temporal metrics
    avg_detection_latency: float = 0.0
    
    def __str__(self):
        """Human-readable report"""
        lines = [
            "=" * 60,
            "SECURITY SYSTEM PERFORMANCE REPORT",
            "=" * 60,
            "",
            "📊 BASIC METRICS:",
            f"  Accuracy:  {self.accuracy:.2%}",
            f"  Precision: {self.precision:.2%}",
            f"  Recall:    {self.recall:.2%}",
            f"  F1-Score:  {self.f1_score:.2%}",
            "",
            "🔒 SECURITY METRICS:",
            f"  False Alarm Rate:  {self.false_alarm_rate:.2%}",
            f"  Detection Rate:    {self.detection_rate:.2%}",
            f"  Avg Response Time: {self.avg_response_time:.3f}s",
            "",
            "🔗 CROSS-MODAL METRICS:",
            f"  Agreement Score: {self.cross_modal_agreement:.2%}",
            "  Modality Contributions:",
        ]
        
        for modality, contrib in self.modality_contributions.items():
            lines.append(f"    {modality}: {contrib:.2%}")
        
        lines.extend([
            "",
            "📈 CONFUSION MATRIX:",
            f"  True Positives:  {self.true_positives}",
            f"  False Positives: {self.false_positives}",
            f"  True Negatives:  {self.true_negatives}",
            f"  False Negatives: {self.false_negatives}",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


class SecurityMetrics:
    """
    Comprehensive metrics for multimodal security systems
    
    Tracks:
    - Detection performance (Survey Section 4.1)
    - False alarms (critical for security)
    - Response times (real-time requirement)
    - Cross-modal agreement
    """
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.timestamps = []
        self.modality_decisions = []
        self.response_times = []
        
        # Per-modality tracking
        self.modality_correct = defaultdict(int)
        self.modality_total = defaultdict(int)
        
    def add_prediction(
        self,
        prediction: str,
        ground_truth: str,
        modality_decisions: Dict[str, str],
        response_time: float = None
    ):
        """
        Add a prediction for evaluation
        
        Args:
            prediction: Final system prediction
            ground_truth: True label
            modality_decisions: Individual modality predictions
            response_time: Time taken for prediction (seconds)
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.timestamps.append(time.time())
        self.modality_decisions.append(modality_decisions)
        
        if response_time:
            self.response_times.append(response_time)
        
        # Track per-modality performance
        for modality, decision in modality_decisions.items():
            self.modality_total[modality] += 1
            if decision == ground_truth:
                self.modality_correct[modality] += 1
    
    def compute_metrics(self) -> PerformanceReport:
        """
        Compute comprehensive metrics
        
        Returns:
            PerformanceReport with all metrics
        """
        
        if len(self.predictions) == 0:
            raise ValueError("No predictions to evaluate")
        
        # Convert to numpy arrays
        y_pred = np.array(self.predictions)
        y_true = np.array(self.ground_truths)
        
        # Basic metrics (Survey Section 4)
        accuracy = np.mean(y_pred == y_true)
        
        # For binary classification (threat vs no-threat)
        tp = np.sum((y_pred == "threat") & (y_true == "threat"))
        fp = np.sum((y_pred == "threat") & (y_true != "threat"))
        tn = np.sum((y_pred != "threat") & (y_true != "threat"))
        fn = np.sum((y_pred != "threat") & (y_true == "threat"))
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Security-specific metrics
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Response time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # Cross-modal agreement
        cross_modal_agreement = self._compute_cross_modal_agreement()
        
        # Modality contributions
        modality_contributions = {
            modality: self.modality_correct[modality] / self.modality_total[modality]
            if self.modality_total[modality] > 0 else 0.0
            for modality in self.modality_total.keys()
        }
        
        # Per-class metrics
        per_class = self._compute_per_class_metrics(y_pred, y_true)
        
        return PerformanceReport(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_alarm_rate=false_alarm_rate,
            detection_rate=detection_rate,
            avg_response_time=avg_response_time,
            cross_modal_agreement=cross_modal_agreement,
            modality_contributions=modality_contributions,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            per_class_metrics=per_class
        )
    
    def _compute_cross_modal_agreement(self) -> float:
        """
        Compute agreement between different modalities
        Higher agreement = more confident prediction
        """
        
        if not self.modality_decisions:
            return 0.0
        
        agreements = []
        for decisions in self.modality_decisions:
            if len(decisions) < 2:
                continue
            
            # Count how many modalities agree
            decision_list = list(decisions.values())
            most_common = max(set(decision_list), key=decision_list.count)
            agreement = decision_list.count(most_common) / len(decision_list)
            agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _compute_per_class_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class"""
        
        classes = np.unique(np.concatenate([y_pred, y_true]))
        per_class = {}
        
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(np.sum(y_true == cls))
            }
        
        return per_class
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            # Compute confusion matrix
            cm = confusion_matrix(self.ground_truths, self.predictions)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path)
                print(f"Confusion matrix saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib and seaborn required for plotting")
    
    def export_results(self) -> Dict:
        """Export results for further analysis"""
        return {
            "predictions": self.predictions,
            "ground_truths": self.ground_truths,
            "timestamps": self.timestamps,
            "modality_decisions": self.modality_decisions,
            "response_times": self.response_times
        }
    
    def compare_with_baseline(
        self,
        baseline_predictions: List[str],
        metric: str = "f1_score"
    ) -> Dict[str, float]:
        """
        Compare with baseline system
        
        Args:
            baseline_predictions: Predictions from baseline
            metric: Metric to compare
            
        Returns:
            Comparison dict with improvement percentage
        """
        
        # Current system metrics
        current_report = self.compute_metrics()
        current_score = getattr(current_report, metric)
        
        # Baseline metrics
        baseline_metrics = SecurityMetrics()
        for pred, gt in zip(baseline_predictions, self.ground_truths):
            baseline_metrics.add_prediction(pred, gt, {})
        
        baseline_report = baseline_metrics.compute_metrics()
        baseline_score = getattr(baseline_report, metric)
        
        # Calculate improvement
        improvement = ((current_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        
        return {
            "current": current_score,
            "baseline": baseline_score,
            "improvement_pct": improvement
        }


# Example usage
if __name__ == "__main__":
    # Initialize metrics
    metrics = SecurityMetrics()
    
    # Simulate predictions
    test_cases = [
        ("threat", "threat", {"visual": "threat", "audio": "threat", "video": "threat"}, 0.123),
        ("safe", "safe", {"visual": "safe", "audio": "safe", "video": "safe"}, 0.098),
        ("threat", "safe", {"visual": "threat", "audio": "safe", "video": "threat"}, 0.156),  # False positive
        ("safe", "threat", {"visual": "safe", "audio": "safe", "video": "threat"}, 0.110),  # False negative
        ("threat", "threat", {"visual": "threat", "audio": "threat", "video": "safe"}, 0.134),
    ]
    
    for pred, gt, modalities, rt in test_cases:
        metrics.add_prediction(pred, gt, modalities, rt)
    
    # Compute and print report
    report = metrics.compute_metrics()
    print(report)
    
    # Export
    print("\n📁 Exported data:")
    export = metrics.export_results()
    print(f"  {len(export['predictions'])} predictions")
    print(f"  {len(export['modality_decisions'])} modality decisions")
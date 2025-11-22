
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import List, Dict


class ModelEvaluator:
    """Evaluate model performance"""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    # =====================================================
    # METHOD 1 → Numeric labels (your original method)
    # =====================================================
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluation using numeric encoded labels"""

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm.tolist(),
            "target_achieved": macro_f1 >= 0.90
        }

    # =====================================================
    # METHOD 2 → Text labels (needed for BaselineClassifier)
    # =====================================================
    def evaluate_text_labels(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """Evaluation using TEXT labels directly"""

        # classification_report handles string labels automatically
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = report["macro avg"]["f1-score"]
        macro_precision = report["macro avg"]["precision"]
        macro_recall = report["macro avg"]["recall"]

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "classification_report": report,
            "target_achieved": macro_f1 >= 0.90
        }

    # =====================================================
    # PRINT REPORT
    # =====================================================
    def print_report(self, results: Dict):
        print("\n" + "="*70)
        print("EVALUATION REPORT")
        print("="*70)

        print(f"\n📊 OVERALL METRICS:")
        print(f"  Accuracy:        {results['accuracy']:.4f}")
        print(f"  Macro F1-Score:  {results['macro_f1']:.4f}", end="")

        if results["target_achieved"]:
            print("  ✓ TARGET ACHIEVED (≥0.90)")
        else:
            print("  ✗ Below Target (needs ≥0.90)")

        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall:    {results['macro_recall']:.4f}")
        print("="*70)


if __name__ == "__main__":
    # Test both numeric + text mode
    class_names = ["Food & Drinks", "Shopping", "Fuel"]
    evaluator = ModelEvaluator(class_names)

    # Numeric test
    y_true_num = np.array([0, 1, 2, 0, 1, 2])
    y_pred_num = np.array([0, 1, 2, 0, 1, 1])
    results_num = evaluator.evaluate(y_true_num, y_pred_num)
    evaluator.print_report(results_num)

    # Text label test
    y_true_txt = ["Food & Drinks", "Shopping", "Fuel"]
    y_pred_txt = ["Food & Drinks", "Shopping", "Shopping"]
    results_txt = evaluator.evaluate_text_labels(y_true_txt, y_pred_txt)
    evaluator.print_report(results_txt)

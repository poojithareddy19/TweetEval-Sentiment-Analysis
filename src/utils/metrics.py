import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

LABELS = [0, 1, 2]
TARGET_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred):
    """Accuracy, macro F1, macro recall (the official TweetEval sentiment metric) and per-class detail.

    labels=[0, 1, 2] is passed everywhere so a class missing from y_pred (or
    y_true) still produces a 3x3 confusion matrix and three per-class scores,
    and zero_division=0 keeps that case warning-free.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    cls_report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=TARGET_NAMES, zero_division=0, output_dict=True
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_macro": recall_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }

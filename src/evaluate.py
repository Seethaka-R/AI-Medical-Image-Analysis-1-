"""
evaluate.py — Binary classifier evaluation (NORMAL vs PNEUMONIA)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    recall_score,
)

# ── Theme ───────────────────────────────────────────────
PAL = {
    "primary": "#1D4ED8",
    "green":   "#059669",
    "red":     "#DC2626",
    "light":   "#EFF6FF",
    "grid":    "#E2E8F0",
}


# ── MAIN FUNCTION ───────────────────────────────────────
def evaluate_model(model, test_gen, class_names, out_dir, binary=True):

    os.makedirs(out_dir, exist_ok=True)

    y_true = []
    y_pred_raw = []

    # 🔥 IMPORTANT FIX — use .predict instead of next()
    test_gen.reset()
    preds = model.predict(test_gen, verbose=0)

    y_true = test_gen.classes

    if binary:
        y_pred_raw = preds[:, 0]
        y_pred = (y_pred_raw >= 0.5).astype(int)
    else:
        y_pred_raw = preds
        y_pred = np.argmax(preds, axis=1)

    # ── Classification report ────────────────────────────
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    acc = float(report["accuracy"])

    # ── Plots ────────────────────────────────────────────
    _confusion_matrix_plot(y_true, y_pred, class_names, out_dir)
    _roc_plot(y_true, y_pred_raw, class_names, out_dir, binary)

    # ── Save JSON ────────────────────────────────────────
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── Extra metrics ────────────────────────────────────
    extras = {}

    if binary:
        extras["sensitivity"] = round(
            recall_score(y_true, y_pred, zero_division=0) * 100, 2
        )
        extras["specificity"] = round(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100, 2
        )

        try:
            extras["roc_auc"] = round(
                roc_auc_score(y_true, y_pred_raw) * 100, 2
            )
        except:
            extras["roc_auc"] = 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
        "weighted_recall": round(report["weighted avg"]["recall"] * 100, 2),
        "per_class": {
            c: round(report[c]["f1-score"] * 100, 2)
            for c in class_names if c in report
        },
        **extras,
    }


# ── CONFUSION MATRIX ────────────────────────────────────
def _confusion_matrix_plot(y_true, y_pred, class_names, out_dir):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ── ROC CURVE ───────────────────────────────────────────
def _roc_plot(y_true, y_pred_raw, class_names, out_dir, binary):

    plt.figure(figsize=(6, 5))

    try:
        if binary:
            fpr, tpr, _ = roc_curve(y_true, y_pred_raw)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color=PAL["primary"])
        else:
            pass  # multi-class optional

        plt.plot([0, 1], [0, 1], "k--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        path = os.path.join(out_dir, "roc_curves.png")
        plt.savefig(path, dpi=150)

    except Exception as e:
        print(f"[WARN] ROC plot failed: {e}")

    plt.close()


# ── TRAINING CURVES ─────────────────────────────────────
def save_training_curves(history, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(15, 4))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.get("accuracy", []), label="Train")
    plt.plot(epochs, history.get("val_accuracy", []), label="Val")
    plt.title("Accuracy")
    plt.legend()

    # AUC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.get("auc", []), label="Train")
    plt.plot(epochs, history.get("val_auc", []), label="Val")
    plt.title("AUC")
    plt.legend()

    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.get("loss", []), label="Train")
    plt.plot(epochs, history.get("val_loss", []), label="Val")
    plt.title("Loss")
    plt.legend()

    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path

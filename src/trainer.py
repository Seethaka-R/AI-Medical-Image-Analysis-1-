"""
Training orchestration for the binary NORMAL vs PNEUMONIA classifier.
"""

import json
import os
import threading

import numpy as np
import tensorflow as tf

from src.evaluate import evaluate_model, save_training_curves
from src.model import build_model, get_callbacks, unfreeze_top_layers
from src.preprocessing import CLASS_NAMES, dataset_stats, get_generators


def json_safe(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


class LiveMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, store=None, emit_fn=None, epoch_offset=0):
        super().__init__()
        self.store = store if store is not None else []
        self.emit_fn = emit_fn
        self.epoch_offset = epoch_offset

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        payload = {
            "epoch": self.epoch_offset + epoch + 1,
            "accuracy": round(float(logs.get("accuracy", 0.0)) * 100, 2),
            "val_accuracy": round(float(logs.get("val_accuracy", 0.0)) * 100, 2),
            "auc": round(float(logs.get("auc", 0.0)) * 100, 2),
            "val_auc": round(float(logs.get("val_auc", 0.0)) * 100, 2),
            "precision": round(float(logs.get("precision", 0.0)) * 100, 2),
            "recall": round(float(logs.get("recall", 0.0)) * 100, 2),
            "loss": round(float(logs.get("loss", 0.0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0.0)), 4),
        }
        self.store.append(payload)
        if self.emit_fn:
            self.emit_fn(payload)


def _merge_histories(*histories):
    merged = {}
    for history in histories:
        if not history:
            continue
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def _fit_phase(model, train_gen, val_gen, epochs, callbacks_list):
    if epochs <= 0:
        return {}
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )
    return history.history


def train(
    data_dir: str,
    model_path: str,
    out_dir: str,
    img_size: int = 224,
    batch: int = 8,
    epochs_phase1: int = 10,
    epochs_phase2: int = 0,
    metrics_store: list = None,
    done_event: threading.Event = None,
    progress_callback=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if metrics_store is None:
        metrics_store = []

    train_gen, val_gen, test_gen = get_generators(data_dir, img_size, batch)

    print(f"\n[Trainer] Classes : {train_gen.class_indices}")
    print(f"[Trainer] Train   : {train_gen.samples}")
    print(f"[Trainer] Val     : {val_gen.samples}")
    print(f"[Trainer] Test    : {test_gen.samples}\n")

    meta_dir = os.path.dirname(model_path)
    os.makedirs(meta_dir, exist_ok=True)

    with open(os.path.join(meta_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f)

    model = build_model(img_size=img_size)

    print("[Trainer] Phase 1 training")
    phase1_history = _fit_phase(
        model,
        train_gen,
        val_gen,
        epochs_phase1,
        get_callbacks(model_path) + [LiveMetricsCallback(metrics_store, progress_callback, 0)],
    )

    phase2_history = {}
    if epochs_phase2 > 0:
        print("[Trainer] Phase 2 fine-tuning")
        model = unfreeze_top_layers(model, n=30, lr=1e-5)
        phase2_history = _fit_phase(
            model,
            train_gen,
            val_gen,
            epochs_phase2,
            get_callbacks(model_path) + [
                LiveMetricsCallback(metrics_store, progress_callback, epochs_phase1)
            ],
        )

    clean_history = _merge_histories(phase1_history, phase2_history)

    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(clean_history, f, default=json_safe, indent=2)

    curves_path = save_training_curves(clean_history, out_dir)

    print("[Trainer] Evaluating on test set...")
    eval_metrics = evaluate_model(model, test_gen, CLASS_NAMES, out_dir, binary=True)

    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, default=json_safe, indent=2)

    stats = dataset_stats(data_dir)

    if done_event:
        done_event.set()

    return {
        "class_names": CLASS_NAMES,
        "eval_metrics": eval_metrics,
        "curves_path": curves_path,
        "model_path": model_path,
        "total_epochs": int(epochs_phase1 + epochs_phase2),
        "dataset_stats": stats,
    }

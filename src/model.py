"""
model.py — Binary Chest X-Ray Classifier (NORMAL vs PNEUMONIA)
Architecture: ResNet50 base + custom sigmoid head
Loss:         Binary Crossentropy
Output:       Single sigmoid neuron → threshold 0.5 → NORMAL(0) or PNEUMONIA(1)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2


# ── Constants ────────────────────────────────────────────────────
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
THRESHOLD   = 0.5          # sigmoid decision boundary


# ── Model builder ────────────────────────────────────────────────
def build_model(img_size: int = 224, lr: float = 1e-4) -> tf.keras.Model:
    """
    ResNet50 (ImageNet weights) + binary classification head.

    Phase 1 — base frozen:   train the dense head quickly.
    Phase 2 — top unfrozen:  fine-tune top 30 ResNet layers for higher AUC.
    """
    base = ResNet50(
        weights      = "imagenet",
        include_top  = False,
        input_shape  = (img_size, img_size, 3),
    )
    base.trainable = False          # Phase 1: freeze entire base

    inp = tf.keras.Input(shape=(img_size, img_size, 3), name="xray_input")
    x   = preprocess_input(inp)    # ImageNet normalisation built in
    x   = base(x, training=False)

    # ── Custom head ──────────────────────────────────────────────
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.BatchNormalization(name="bn")(x)
    x   = layers.Dense(
              512, activation="relu", name="dense_512",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
          )(x)
    x   = layers.Dropout(0.4, name="drop_1")(x)
    x   = layers.Dense(
              256, activation="relu", name="dense_256",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
          )(x)
    x   = layers.Dropout(0.3, name="drop_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    # ─────────────────────────────────────────────────────────────

    model = tf.keras.Model(inp, out, name="PneumoniaResNet50")
    _compile(model, lr, binary=True)
    return model


def unfreeze_top_layers(model: tf.keras.Model,
                        n: int = 30,
                        lr: float = 1e-5) -> tf.keras.Model:
    """Phase 2: unfreeze the top N layers of the ResNet50 base."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):        # the ResNet50 sub-model
            for sub in layer.layers[-n:]:
                sub.trainable = True
    _compile(model, lr, binary=True)
    return model


def _compile(model: tf.keras.Model, lr: float, binary: bool = True):
    model.compile(
        optimizer = optimizers.Adam(lr),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ── Training callbacks ───────────────────────────────────────────
def get_callbacks(model_path: str, patience: int = 8) -> list:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            model_path,
            save_best_only = True,
            monitor        = "val_auc",
            mode           = "max",
            verbose        = 1,
        ),
        callbacks.EarlyStopping(
            monitor             = "val_auc",
            patience            = patience,
            restore_best_weights= True,
            mode                = "max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 4,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]


# ── Grad-CAM ─────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Maps for binary classifier.
    Highlights WHICH lung regions drove the PNEUMONIA prediction.
    Uses the last conv layer of ResNet50: conv5_block3_out.
    """

    def __init__(self, model: tf.keras.Model):
        self.model      = model
        self.preprocess_model = None
        self.feature_model = None
        self.classifier_layers = []
        self.grad_model = self._build_grad_model()

    def _build_grad_model(self):
        try:
            resnet = self.model.get_layer("resnet50")
            last_conv = resnet.get_layer("conv5_block3_out")
            self.preprocess_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=resnet.input,
            )
            self.feature_model = tf.keras.Model(
                inputs=resnet.input,
                outputs=[last_conv.output, resnet.output],
            )
            resnet_index = self.model.layers.index(resnet)
            self.classifier_layers = self.model.layers[resnet_index + 1 :]
            return True
        except Exception as e:
            print(f"[GradCAM] Could not build grad model: {e}")
            return None

    def heatmap(self, img_array: np.ndarray) -> np.ndarray:
        """
        img_array: (1, H, W, 3) float32
        Returns (H', W') float32 heatmap normalised 0-1.
        """
        if self.grad_model is None:
            return np.zeros(img_array.shape[1:3], dtype=np.float32)

        try:
            processed = self.preprocess_model(img_array, training=False)
            with tf.GradientTape() as tape:
                conv_out, pred_features = self.feature_model(processed, training=False)
                x = pred_features
                for layer in self.classifier_layers:
                    x = layer(x, training=False)
                # Binary: gradient w.r.t. the single sigmoid output
                loss = x[:, 0]

            grads  = tape.gradient(loss, conv_out)              # (1,H,W,C)
            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))     # (C,)
            cam    = conv_out[0] @ pooled[..., tf.newaxis]     # (H,W,1)
            cam    = tf.squeeze(cam)                            # (H,W)
            cam    = tf.maximum(cam, 0)
            cam    = cam / (tf.reduce_max(cam) + 1e-8)
            return cam.numpy()
        except Exception as e:
            print(f"[GradCAM] Falling back to empty heatmap: {e}")
            return np.zeros(img_array.shape[1:3], dtype=np.float32)

    def overlay(self, rgb_img: np.ndarray,
                heatmap: np.ndarray,
                alpha: float = 0.4) -> np.ndarray:
        """Superimpose JET heatmap on the original X-ray."""
        h, w   = rgb_img.shape[:2]
        cam_r  = cv2.resize(heatmap, (w, h))
        colored= cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        base   = rgb_img if rgb_img.dtype == np.uint8 else np.uint8(rgb_img * 255)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.shape[2] == 3:
            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        result = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# ── Inference helper ─────────────────────────────────────────────
def predict_single(model: tf.keras.Model,
                   img_array: np.ndarray) -> dict:
    """
    img_array: (1, H, W, 3) float32
    Returns dict with label, confidence, raw_score.
    """
    raw        = float(model.predict(img_array, verbose=0)[0][0])
    label      = CLASS_NAMES[1] if raw >= THRESHOLD else CLASS_NAMES[0]
    confidence = raw * 100 if raw >= THRESHOLD else (1 - raw) * 100
    return {
        "label":       label,
        "confidence":  round(confidence, 2),
        "raw_score":   round(raw, 4),
        "probabilities": {
            "NORMAL":    round((1 - raw) * 100, 2),
            "PNEUMONIA": round(raw * 100, 2),
        },
    }


def model_info(model: tf.keras.Model) -> dict:
    return {
        "name":             model.name,
        "total_params":     int(model.count_params()),
        "trainable_params": int(
            sum(tf.size(w).numpy() for w in model.trainable_weights)
        ),
        "num_layers":       len(model.layers),
        "task":             "Binary — NORMAL vs PNEUMONIA",
    }

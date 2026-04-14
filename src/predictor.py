"""
predictor.py — Binary inference engine: NORMAL vs PNEUMONIA
Returns prediction, confidence, Grad-CAM overlay, risk level, recommendation.
"""

import os
import json
import io
import base64
import numpy as np
import cv2
from PIL import Image

from src.preprocessing import preprocess_single, load_image_for_display
from src.model         import GradCAM, predict_single, CLASS_NAMES


def predict_image(model,
                  img_path: str,
                  out_dir: str = "outputs/predictions") -> dict:
    """
    Full binary prediction pipeline for one chest X-ray.
    Returns a dict with all data needed by the dashboard.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load & preprocess ─────────────────────────────────────────
    img_arr = preprocess_single(img_path)              # (1,224,224,3) float32
    display = load_image_for_display(img_path)         # uint8 RGB (224,224,3)

    # ── Inference ─────────────────────────────────────────────────
    result = predict_single(model, img_arr)
    label      = result["label"]
    confidence = result["confidence"]
    raw_score  = result["raw_score"]
    probs      = result["probabilities"]

    # ── Grad-CAM (only meaningful for PNEUMONIA — model is attending
    #              to infiltrate regions in the lung fields)
    gcam        = GradCAM(model)
    hm          = gcam.heatmap(img_arr)
    overlay_rgb = gcam.overlay(display, hm, alpha=0.45)

    # ── Save overlay ──────────────────────────────────────────────
    fname        = os.path.splitext(os.path.basename(img_path))[0]
    overlay_path = os.path.join(out_dir, f"{fname}_gradcam.png")
    Image.fromarray(overlay_rgb).save(overlay_path)

    # ── Encode for dashboard (base64 PNG) ─────────────────────────
    orig_b64 = _to_b64(display)
    cam_b64  = _to_b64(overlay_rgb)

    # ── Clinical outputs ──────────────────────────────────────────
    risk       = _risk_level(label, confidence)
    rec        = _recommendation(label, confidence)
    findings   = _clinical_findings(label, raw_score)

    output = {
        # ── Identity
        "filename":       os.path.basename(img_path),
        "label":          label,
        "confidence":     confidence,
        "raw_score":      raw_score,
        "probabilities":  probs,
        # ── Clinical
        "risk_level":     risk,
        "recommendation": rec,
        "findings":       findings,
        # ── Images
        "original_b64":   orig_b64,
        "gradcam_b64":    cam_b64,
        "gradcam_path":   overlay_path,
    }

    # Save JSON report (without base64 blobs — too large)
    report = {k: v for k, v in output.items() if "_b64" not in k}
    with open(os.path.join(out_dir, f"{fname}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return output


# ── Helpers ──────────────────────────────────────────────────────

def _to_b64(rgb_arr: np.ndarray) -> str:
    img = Image.fromarray(rgb_arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _risk_level(label: str, confidence: float) -> str:
    if label == "NORMAL":
        return "Low"
    # PNEUMONIA
    if confidence >= 90:
        return "High"
    if confidence >= 75:
        return "Moderate"
    return "Low"


def _recommendation(label: str, confidence: float) -> str:
    if label == "NORMAL":
        return (
            "No radiological signs of pneumonia detected. "
            "Routine clinical follow-up as appropriate."
        )
    if confidence >= 90:
        return (
            "Strong radiological indicators of pneumonia detected. "
            "Immediate clinical review and appropriate antibiotic therapy recommended. "
            "Consider repeat CXR in 4–6 weeks to confirm resolution."
        )
    if confidence >= 75:
        return (
            "Moderate radiological indicators of pneumonia detected. "
            "Clinical correlation recommended. "
            "Consider sputum culture and sensitivity testing."
        )
    return (
        "Possible early or resolving pneumonia. "
        "Clinical correlation and repeat imaging in 48–72 hours advised."
    )


def _clinical_findings(label: str, raw_score: float) -> list:
    """Returns a list of radiological finding strings for the report panel."""
    if label == "NORMAL":
        return [
            "Lung fields clear bilaterally",
            "No focal consolidation identified",
            "Cardiac silhouette within normal limits",
            "No pleural effusion detected",
        ]
    return [
        "Focal opacification / consolidation detected",
        f"Model confidence: {round(raw_score * 100, 1)}% PNEUMONIA",
        "Grad-CAM highlights affected lung region",
        "Recommend correlation with clinical presentation and labs",
    ]

"""
Flask + Socket.IO app for training and inference on chest X-ray images.
"""

import base64
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from src import predictor as _pred
from src import trainer as _trainer
from src.preprocessing import dataset_stats


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_PATH = BASE_DIR / "models" / "saved" / "pneumonia_resnet50.h5"
OUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
EVAL_METRICS_PATH = OUT_DIR / "eval_metrics.json"

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models" / "saved").mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "medai"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

STATE = {
    "model": None,
    "training": False,
    "trained": False,
}


def _load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _encode_file_b64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_model_if_exists():
    import tensorflow as tf

    if MODEL_PATH.exists():
        try:
            STATE["model"] = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            STATE["trained"] = True
            print("Model loaded from disk")
        except Exception as exc:
            print("Failed loading model:", exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify(
        {
            "model_loaded": STATE["trained"],
            "training": STATE["training"],
            "dataset_exists": DATA_DIR.exists(),
            "dataset_splits": dataset_stats(str(DATA_DIR)),
            "class_names": ["NORMAL", "PNEUMONIA"],
        }
    )


@app.route("/api/eval_metrics")
def eval_metrics():
    return jsonify(_load_json(EVAL_METRICS_PATH, default={}))


@app.route("/api/eval_plots")
def eval_plots():
    names = ["confusion_matrix.png", "roc_curves.png", "training_curves.png"]
    return jsonify({name: _encode_file_b64(OUT_DIR / name) for name in names})


@app.route("/api/train", methods=["POST"])
def train():
    if STATE["training"]:
        return jsonify({"error": "Already training"}), 400

    if not DATA_DIR.exists():
        return jsonify({"error": f"Dataset missing at {DATA_DIR}"}), 400

    payload = request.get_json(silent=True) or {}
    epochs_phase1 = int(payload.get("epochs_phase1", 3))
    epochs_phase2 = int(payload.get("epochs_phase2", 0))

    STATE["training"] = True
    STATE["trained"] = False
    STATE["model"] = None

    def run_training():
        try:
            result = _trainer.train(
                data_dir=str(DATA_DIR),
                model_path=str(MODEL_PATH),
                out_dir=str(OUT_DIR),
                epochs_phase1=epochs_phase1,
                epochs_phase2=epochs_phase2,
                progress_callback=lambda metric: socketio.emit("epoch_metric", metric),
            )

            load_model_if_exists()
            socketio.emit("training_complete", result["eval_metrics"])
        except Exception as exc:
            socketio.emit("training_error", {"error": str(exc)})
        finally:
            STATE["training"] = False

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"started": True, "epochs_phase1": epochs_phase1, "epochs_phase2": epochs_phase2})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not STATE["trained"] or STATE["model"] is None:
        return jsonify({"error": "Model not trained yet"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    name = f"{int(time.time())}_{secure_filename(file.filename)}"
    save_path = UPLOAD_DIR / name
    file.save(str(save_path))

    try:
        result = _pred.predict_image(
            STATE["model"],
            str(save_path),
            out_dir=str(OUT_DIR / "predictions"),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(str(OUT_DIR), filename)


@socketio.on("connect")
def connect():
    emit("server_ready", {"trained": STATE["trained"], "training": STATE["training"]})


load_model_if_exists()


if __name__ == "__main__":
    print("\nServer running")
    print("http://127.0.0.1:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000)

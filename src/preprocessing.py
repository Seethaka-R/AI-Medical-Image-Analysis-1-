"""
preprocessing.py — Optimized Chest X-Ray Pipeline
Binary classification: NORMAL vs PNEUMONIA
"""

import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ─────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 8
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_MODE  = "binary"


# ── Data Generators ───────────────────────────────────────
TRAIN_DATAGEN = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
)

VAL_DATAGEN  = ImageDataGenerator()
TEST_DATAGEN = ImageDataGenerator()


# ── Generator Factory ─────────────────────────────────────
def get_generators(data_dir: str = None,
                   img_size: int = IMG_SIZE,
                   batch: int = BATCH_SIZE):

    # Always resolve clean root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    root = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "raw"

    train_path = root / "train"
    val_path   = root / "val"
    test_path  = root / "test"

    # ── Validation ────────────────────────────────────────
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    print(f"\n[DATA] Using dataset at: {root}")

    # ── Generators ────────────────────────────────────────
    train_gen = TRAIN_DATAGEN.flow_from_directory(
        str(train_path),
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode=CLASS_MODE,
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )

    val_gen = VAL_DATAGEN.flow_from_directory(
        str(val_path),
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode=CLASS_MODE,
        classes=CLASS_NAMES,
        shuffle=False,
    )

    test_gen = TEST_DATAGEN.flow_from_directory(
        str(test_path),
        target_size=(img_size, img_size),
        batch_size=batch,
        class_mode=CLASS_MODE,
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ── Single Image Preprocessing ────────────────────────────
def preprocess_single(img_path: str, img_size: int = IMG_SIZE):
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img, dtype=np.float32)
    return np.expand_dims(img, axis=0)


# ── Display Helper ────────────────────────────────────────
def load_image_for_display(img_path: str, img_size: int = IMG_SIZE):
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    return np.array(img.resize((img_size, img_size)))


# ── Dataset Stats ─────────────────────────────────────────
def dataset_stats(data_dir: str = None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / "raw"

    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        split_path = root / split
        counts = {}

        if not split_path.exists():
            stats[split] = {}
            continue

        for cls in CLASS_NAMES:
            cls_path = split_path / cls
            counts[cls] = len(list(cls_path.glob("*.*"))) if cls_path.exists() else 0

        stats[split] = counts

    return stats

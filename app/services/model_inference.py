
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.services.inference_preprocessing import (
    prepare_inference_features,
    normalize_inference_features,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best_landslide_unet.keras"
STATS_PATH = BASE_DIR / "models" / "normalization_stats_26ch.npz"

# Load once when this module is imported.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)


def predict_landslide(
    bbox: list[float],
    time_from: str,
    time_to: str,
) -> np.ndarray:
    """Return a 128x128 landslide probability mask."""

    features = prepare_inference_features(
        bbox=bbox,
        time_from=time_from,
        time_to=time_to,
    )

    features = normalize_inference_features(
        features,
        stats_path=STATS_PATH,
    )

    # Model expects a batch: (1, 128, 128, 26).
    batch = np.expand_dims(features, axis=0)

    predictions = model.predict(batch, verbose=0)

    # The model has coarse_mask, medium_mask, and final_mask outputs.
    if isinstance(predictions, dict):
        mask = predictions["final_mask"][0, ..., 0]
    else:
        mask = predictions[model.output_names.index("final_mask")][0, ..., 0]

    if not np.isfinite(mask).all():
        raise ValueError("Model returned non-finite predictions.")

    return np.clip(mask, 0.0, 1.0)

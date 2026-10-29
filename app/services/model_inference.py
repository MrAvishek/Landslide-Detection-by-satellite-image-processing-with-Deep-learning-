# app/services/model_inference.py

from pathlib import Path

import numpy as np
import tensorflow as tf

from app.services.preprocessing import create_features
from app.services.inference_preprocessing import normalize_inference_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_landslide_unet.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)


def predict_image(image: np.ndarray) -> np.ndarray:
    """Predict a 128x128 landslide probability mask from a raw 14-channel image."""
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (128, 128, 14):
        raise ValueError(
            f"Expected raw image shape (128, 128, 14); got {image.shape}."
        )

    features = create_features(image)
    features = normalize_inference_features(features)

    batch = np.expand_dims(features, axis=0)
    predictions = model.predict(batch, verbose=0)

    if isinstance(predictions, dict):
        if "final_mask" not in predictions:
            raise ValueError("Model output does not contain 'final_mask'.")
        mask = predictions["final_mask"][0, ..., 0]
    else:
        output_index = model.output_names.index("final_mask")
        mask = predictions[output_index][0, ..., 0]

    if mask.shape != (128, 128) or not np.isfinite(mask).all():
        raise ValueError("Model returned an invalid segmentation mask.")

    return np.clip(mask, 0.0, 1.0).astype(np.float32)

import numpy as np
import tensorflow as tf

from app.services.inference_preprocessing import (
    prepare_inference_features,
    normalize_inference_features,
)

bbox = [88.35, 27.30, 88.36, 27.31]
time_from = "2025-10-01T00:00:00Z"
time_to = "2025-11-01T23:59:59Z"

features = prepare_inference_features(bbox, time_from, time_to)
x = normalize_inference_features(
    features,
    stats_path="models/normalization_stats_26ch.npz",
)

print("Normalized shape:", x.shape)
print("Normalized min/max:", float(x.min()), float(x.max()))
print("Per-channel means:", np.round(x.mean(axis=(0, 1)), 3))
print("Per-channel stds:", np.round(x.std(axis=(0, 1)), 3))

model = tf.keras.models.load_model(
    "models/best_landslide_unet.keras",
    compile=False,
)

outputs = model.predict(x[np.newaxis, ...], verbose=0)

for name, output in zip(model.output_names, outputs):
    print(
        name,
        "min:", float(output.min()),
        "max:", float(output.max()),
        "mean:", float(output.mean()),
    )

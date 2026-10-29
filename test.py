import numpy as np

from app.services.inference_preprocessing import (
    STATS_PATH,
    normalize_inference_features,
)

print("Statistics file:", STATS_PATH)
print("File exists:", STATS_PATH.exists())

features = np.random.default_rng(42).random(
    (128, 128, 26), dtype=np.float32
)

normalized = normalize_inference_features(features)

print("Output shape:", normalized.shape)
print("Output dtype:", normalized.dtype)
print("All finite:", np.isfinite(normalized).all())
print("Normalization test passed.")
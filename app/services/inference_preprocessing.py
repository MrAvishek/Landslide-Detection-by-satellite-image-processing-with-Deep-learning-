
import numpy as np

from app.services.inference_input import build_model_input
from app.services.preprocessing import create_features


EXPECTED_RAW_SHAPE = (128, 128, 14)
EXPECTED_FEATURE_SHAPE = (128, 128, 26)


def prepare_inference_features(
    bbox: list[float],
    time_from: str,
    time_to: str,
) -> np.ndarray:
    """Retrieve satellite data and create the model's 26 input features."""

    raw_input = build_model_input(
        bbox=bbox,
        time_from=time_from,
        time_to=time_to,
    )

    if raw_input.shape != EXPECTED_RAW_SHAPE:
        raise ValueError(
            f"Expected raw input {EXPECTED_RAW_SHAPE}, "
            f"got {raw_input.shape}"
        )

    features = create_features(raw_input)

    if features.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(
            f"Expected features {EXPECTED_FEATURE_SHAPE}, "
            f"got {features.shape}"
        )

    if features.dtype != np.float32:
        features = features.astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values.")

    return features


# def normalize_inference_features(
#     features: np.ndarray,
#     stats_path: str = "normalization_stats_26ch.npz",
# ) -> np.ndarray:
#     """Apply the saved training statistics to one 26-channel image."""

#     if features.shape != EXPECTED_FEATURE_SHAPE:
#         raise ValueError(
#             f"Expected features {EXPECTED_FEATURE_SHAPE}, "
#             f"got {features.shape}"
#         )

#     with np.load(stats_path) as stats:
#         mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
#         std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)

#     if mean.size != 26 or std.size != 26:
#         raise ValueError(
#             f"Expected 26 means and standard deviations; "
#             f"got {mean.size} means and {std.size} standard deviations."
#         )

#     if not np.isfinite(mean).all() or not np.isfinite(std).all():
#         raise ValueError("Normalization statistics contain invalid values.")

#     if np.any(std <= 0):
#         raise ValueError("All standard deviations must be positive.")

#     normalized = (
#         features - mean.reshape(1, 1, 26)
#     ) / std.reshape(1, 1, 26)

#     normalized = normalized.astype(np.float32)

#     if not np.isfinite(normalized).all():
#         raise ValueError("Normalized features contain invalid values.")

#     return normalized


def normalize_inference_features(
    features: np.ndarray,
    stats_path: str = "normalization_stats_26ch.npz",
) -> np.ndarray:
    """Apply the same 26-channel normalization used during training."""

    if features.shape != (128, 128, 26):
        raise ValueError(
            f"Expected features (128, 128, 26), got {features.shape}"
        )

    with np.load(stats_path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

    # Training statistics may be stored as (1, 1, 1, 26).
    if mean.size != 26 or std.size != 26:
        raise ValueError("Expected 26 means and 26 standard deviations.")

    mean = mean.reshape(1, 1, 26)
    std = std.reshape(1, 1, 26)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain invalid values.")

    if np.any(std <= 0):
        raise ValueError("Standard deviations must be positive.")

    normalized = (features - mean) / std

    if not np.isfinite(normalized).all():
        raise ValueError("Normalized features contain invalid values.")

    return normalized.astype(np.float32)


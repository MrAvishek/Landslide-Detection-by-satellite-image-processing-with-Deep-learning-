
import numpy as np

from app.services.satellite import (
    fetch_sentinel2_bands,
    fetch_dem,
    calculate_slope,
)


def build_model_input(
    bbox: list[float],
    time_from: str,
    time_to: str,
) -> np.ndarray:
    """
    Build a (128, 128, 14) input:
    Sentinel-2 B01-B12, slope, DEM.
    """

    bands = fetch_sentinel2_bands(
        bbox=bbox,
        time_from=time_from,
        time_to=time_to,
        width=128,
        height=128,
    )

    dem = fetch_dem(bbox=bbox, width=128, height=128)
    slope = calculate_slope(dem, bbox)

    model_input = np.concatenate(
        [
            bands,
            slope[..., np.newaxis],
            dem[..., np.newaxis],
        ],
        axis=-1,
    ).astype(np.float32)

    if model_input.shape != (128, 128, 14):
        raise ValueError(
            f"Expected (128, 128, 14), got {model_input.shape}"
        )

    if not np.isfinite(model_input).all():
        raise ValueError("Model input contains NaN or infinite values.")

    return model_input

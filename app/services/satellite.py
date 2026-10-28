from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
from rasterio.io import MemoryFile
import requests

from app.services.copernicus import get_access_token


PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

BANDS = [f"B{i:02d}" for i in range(1, 13)]

EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: [
                "B01", "B02", "B03", "B04",
                "B05", "B06", "B07", "B08",
                "B09", "B10", "B11", "B12"
            ],
            units: "REFLECTANCE"
        }],
        output: {
            bands: 12,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B01, sample.B02, sample.B03, sample.B04,
        sample.B05, sample.B06, sample.B07, sample.B08,
        sample.B09, sample.B10, sample.B11, sample.B12
    ];
}
"""


def fetch_sentinel2_bands(
    bbox: list[float],
    time_from: str,
    time_to: str,
    width: int = 128,
    height: int = 128,
    max_cloud_coverage: int = 20,
) -> np.ndarray:
    """
    Fetch Sentinel-2 L1C spectral data.

    bbox order: [min_longitude, min_latitude,
                 max_longitude, max_latitude]

    Returns:
        Float32 array shaped (height, width, 12),
        ordered B01 through B12.
    """

    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates.")

    min_lon, min_lat, max_lon, max_lat = bbox

    if not (
        -180 <= min_lon < max_lon <= 180
        and -90 <= min_lat < max_lat <= 90
    ):
        raise ValueError("Invalid geographic bounding box.")

    token = get_access_token()

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l1c",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to,
                        },
                        "maxCloudCoverage": max_cloud_coverage,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": EVALSCRIPT,
    }

    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )
    if not response.ok:
        print("HTTP Status:", response.status_code)
        print("API Error:", response.text)
        response.raise_for_status()

    with MemoryFile(response.content) as memory_file:
        with memory_file.open() as dataset:
            bands_first = dataset.read()

    if bands_first.shape != (12, height, width):
        raise ValueError(
            f"Unexpected satellite raster shape: {bands_first.shape}"
        )

    # Convert (bands, height, width) to (height, width, bands).
    image = np.moveaxis(bands_first, 0, -1).astype(np.float32)

    if not np.isfinite(image).all():
        raise ValueError("Satellite data contains NaN or infinite values.")

    return image

DEM_EVALSCRIPT = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["DEM"],
            units: "METERS"
        }],
        output: {
            bands: 1,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.DEM];
}
"""


def fetch_dem(
    bbox: list[float],
    width: int = 128,
    height: int = 128,
) -> np.ndarray:
    """Retrieve elevation in metres as a (height, width) array."""

    token = get_access_token()

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [{"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_90"}}],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": DEM_EVALSCRIPT,
    }

    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )
    if not response.ok:
        print("HTTP Status:", response.status_code)
        print("API Error:", response.text)
        response.raise_for_status()

    with MemoryFile(response.content) as memory_file:
        with memory_file.open() as dataset:
            dem = dataset.read(1).astype(np.float32)

    if dem.shape != (height, width):
        raise ValueError(f"Unexpected DEM shape: {dem.shape}")

    if not np.isfinite(dem).all():
        raise ValueError("DEM contains NaN or infinite values.")

    return dem


def calculate_slope(
    dem: np.ndarray,
    bbox: list[float],
) -> np.ndarray:
    """Estimate slope in degrees from an elevation grid."""

    height, width = dem.shape
    min_lon, min_lat, max_lon, max_lat = bbox

    latitude = (min_lat + max_lat) / 2
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = 111_320.0 * np.cos(np.radians(latitude))

    pixel_width = (
        (max_lon - min_lon) * metres_per_degree_lon / width
    )
    pixel_height = (
        (max_lat - min_lat) * metres_per_degree_lat / height
    )

    dz_dy, dz_dx = np.gradient(dem, pixel_height, pixel_width)

    slope = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    )

    return slope.astype(np.float32)

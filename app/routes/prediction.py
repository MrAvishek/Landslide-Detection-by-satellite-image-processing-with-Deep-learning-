
import io
import logging

import h5py
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.model_inference import predict_image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict-upload")
async def predict_uploaded_file(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    """
    Upload an HDF5 satellite image and generate a landslide segmentation mask.

    Expected HDF5 structure:
        img:  (128, 128, 14) satellite image
        mask: optional ground-truth mask

    The predict_image() function is responsible for feature engineering,
    normalization, and model inference.
    """

    # 1. Validate the threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0.0 and 1.0.",
        )

    # 2. Validate the uploaded filename
    filename = file.filename or "uploaded_file.h5"

    if not filename.lower().endswith((".h5", ".hdf5")):
        raise HTTPException(
            status_code=400,
            detail="Upload an HDF5 satellite image (.h5 or .hdf5).",
        )

    # 3. Read the uploaded file
    try:
        content = await file.read()
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 4. Read the HDF5 image and optional ground-truth mask
    try:
        with h5py.File(io.BytesIO(content), "r") as h5_file:
            if "img" not in h5_file:
                raise HTTPException(
                    status_code=400,
                    detail="The HDF5 file must contain an 'img' dataset.",
                )

            image = np.asarray(h5_file["img"], dtype=np.float32)

            ground_truth_mask = None
            if "mask" in h5_file:
                ground_truth_mask = np.asarray(h5_file["mask"])

    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not read uploaded HDF5 file: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not read the file as a valid HDF5 satellite image.",
        ) from exc

    # 5. Validate the model's expected raw input shape
    expected_shape = (128, 128, 14)

    if image.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image shape: {image.shape}. "
                f"Expected {expected_shape}."
            ),
        )

    if not np.isfinite(image).all():
        raise HTTPException(
            status_code=400,
            detail="The image contains NaN or infinite values.",
        )

    # 6. Run model inference without blocking the async event loop
    try:
        probability_mask = await run_in_threadpool(predict_image, image)

    except Exception as exc:
        logger.exception("Landslide model inference failed")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check the server logs.",
        ) from exc

    probability_mask = np.asarray(
        probability_mask, dtype=np.float32
    ).squeeze()

    if probability_mask.shape != (128, 128):
        logger.error(
            "Unexpected prediction shape: %s",
            probability_mask.shape,
        )
        raise HTTPException(
            status_code=500,
            detail="The model returned an unexpected prediction shape.",
        )

    if not np.isfinite(probability_mask).all():
        raise HTTPException(
            status_code=500,
            detail="The model returned invalid prediction values.",
        )

    # 7. Convert probabilities into a binary segmentation mask
    binary_mask = (probability_mask >= threshold).astype(np.uint8)

    positive_pixels = int(np.count_nonzero(binary_mask))
    total_pixels = int(binary_mask.size)
    positive_percentage = round(
        (positive_pixels / total_pixels) * 100, 2
    )

    # 8. Prepare optional ground-truth data for the frontend
    ground_truth_result = None

    if ground_truth_mask is not None:
        ground_truth_mask = np.squeeze(ground_truth_mask)

        if ground_truth_mask.shape == (128, 128):
            ground_truth_result = (
                ground_truth_mask > 0
            ).astype(np.uint8).tolist()

    # 9. Return results to the frontend
    return {
        "filename": filename,
        "mask": binary_mask.tolist(),
        "probability_mask": probability_mask.tolist(),
        "mask_shape": list(binary_mask.shape),
        "positive_pixels": positive_pixels,
        "positive_percentage": positive_percentage,
        "threshold": threshold,
        "ground_truth_mask": ground_truth_result,
    }

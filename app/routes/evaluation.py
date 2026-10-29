# app/routes/evaluation.py

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.services.dataset import load_sample
from app.services.model_inference import predict_image

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


class EvaluationRequest(BaseModel):
    split: Literal["train", "valid", "test"] = "test"
    sample_ids: list[int] = Field(min_length=1, max_length=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sample_ids(self):
        if any(sample_id < 1 for sample_id in self.sample_ids):
            raise ValueError("Every sample ID must be at least 1.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must not contain duplicates.")
        return self


@router.post("")
def evaluate_dataset(payload: EvaluationRequest):
    tp = fp = fn = tn = 0
    evaluated_count = 0
    missing_mask_ids = []
    inference_errors = []

    for sample_id in payload.sample_ids:
        try:
            image, ground_truth = load_sample(payload.split, sample_id)

            if ground_truth is None:
                missing_mask_ids.append(sample_id)
                continue

            probabilities = predict_image(image)
            predicted = probabilities >= payload.threshold
            actual = ground_truth.astype(bool)

            tp += int(np.logical_and(predicted, actual).sum())
            fp += int(np.logical_and(predicted, ~actual).sum())
            fn += int(np.logical_and(~predicted, actual).sum())
            tn += int(np.logical_and(~predicted, ~actual).sum())

            evaluated_count += 1

        except FileNotFoundError:
            inference_errors.append({
                "sample_id": sample_id,
                "error": "Image file not found.",
            })
        except ValueError as exc:
            inference_errors.append({
                "sample_id": sample_id,
                "error": str(exc),
            })
        except Exception:
            inference_errors.append({
                "sample_id": sample_id,
                "error": "Prediction or evaluation failed.",
            })

    metrics = None

    if evaluated_count > 0:
        union = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        precision_denominator = tp + fp
        recall_denominator = tp + fn

        metrics = {
            "iou": float(tp / union) if union else 1.0,
            "dice": (
                float(2 * tp / dice_denominator)
                if dice_denominator else 1.0
            ),
            "precision": (
                float(tp / precision_denominator)
                if precision_denominator else 0.0
            ),
            "recall": (
                float(tp / recall_denominator)
                if recall_denominator else 0.0
            ),
            "f1": (
                float(2 * tp / (2 * tp + fp + fn))
                if (2 * tp + fp + fn) else 1.0
            ),
            "pixel_accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        }

    return {
        "split": payload.split,
        "threshold": payload.threshold,
        "requested_count": len(payload.sample_ids),
        "evaluated_count": evaluated_count,
        "missing_mask_count": len(missing_mask_ids),
        "missing_mask_sample_ids": missing_mask_ids,
        "failed_count": len(inference_errors),
        "errors": inference_errors,
        "confusion_counts": (
            {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            if evaluated_count else None
        ),
        "metrics": metrics,
        "status": (
            "completed"
            if evaluated_count == len(payload.sample_ids)
            else "partial"
            if evaluated_count > 0
            else "ground_truth_unavailable"
            if not inference_errors
            else "no_samples_evaluated"
        ),
        "note": (
            "Metrics are computed only for samples with ground-truth masks. "
            "IoU and Dice are defined as 1 when both prediction and truth "
            "contain no positive pixels; precision and recall are 0 when "
            "their denominators are zero."
        ),
    }
"""
Colour correction model loader.
Loads the trained XGBoost model at startup and provides a single
predict() function used by the pre_shoot and post_correction routes.
"""

import os
import json
import numpy as np
import joblib
from typing import Optional

_MODEL_DIR = os.path.join(os.path.dirname(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "colour_correction_model.pkl")
_METADATA_PATH = os.path.join(_MODEL_DIR, "model_metadata.json")

_model = None
_metadata = None


def _load():
    global _model, _metadata
    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        _model = joblib.load(_MODEL_PATH)
        with open(_METADATA_PATH) as f:
            _metadata = json.load(f)
        return True
    except Exception:
        return False


# Load once at import time
_model_available = _load()


def model_available() -> bool:
    return _model_available and _model is not None


def get_metadata() -> Optional[dict]:
    return _metadata


def predict_correction(scene_profile: dict, reference_profile: dict) -> dict:
    """
    Predict colour correction values for a scene frame given a reference profile.

    Returns a dict with correction values for R, G, B channels,
    exposure EV, colour temperature, and saturation.

    Falls back to delta calculations if the model is not available.
    """
    if not model_available():
        # Graceful fallback: return raw deltas as corrections
        return {
            "correct_r": round(reference_profile["mean_r"] - scene_profile["mean_r"], 2),
            "correct_g": round(reference_profile["mean_g"] - scene_profile["mean_g"], 2),
            "correct_b": round(reference_profile["mean_b"] - scene_profile["mean_b"], 2),
            "correct_exposure_ev": round(
                reference_profile["exposure_ev"] - scene_profile["exposure_ev"], 3
            ),
            "correct_temp_k": round(
                reference_profile["colour_temperature_k"] - scene_profile["colour_temperature_k"]
            ),
            "correct_saturation": round(
                reference_profile["saturation_pct"] - scene_profile["saturation_pct"], 1
            ),
            "source": "fallback_delta",
        }

    # Build feature vector matching training schema
    delta_r = scene_profile["mean_r"] - reference_profile["mean_r"]
    delta_g = scene_profile["mean_g"] - reference_profile["mean_g"]
    delta_b = scene_profile["mean_b"] - reference_profile["mean_b"]
    delta_temp = scene_profile["colour_temperature_k"] - reference_profile["colour_temperature_k"]
    delta_ev = scene_profile["exposure_ev"] - reference_profile["exposure_ev"]
    delta_sat = scene_profile["saturation_pct"] - reference_profile["saturation_pct"]

    features = np.array([[
        # Scene features
        scene_profile["mean_r"],
        scene_profile["mean_g"],
        scene_profile["mean_b"],
        scene_profile["colour_temperature_k"],
        scene_profile["exposure_ev"],
        scene_profile["saturation_pct"],
        scene_profile["contrast_ratio"],
        # Reference features
        reference_profile["mean_r"],
        reference_profile["mean_g"],
        reference_profile["mean_b"],
        reference_profile["colour_temperature_k"],
        reference_profile["exposure_ev"],
        reference_profile["saturation_pct"],
        reference_profile["contrast_ratio"],
        # Delta features
        delta_r,
        delta_g,
        delta_b,
        delta_temp,
        delta_ev,
        delta_sat,
    ]])

    prediction = _model.predict(features)[0]
    target_names = _metadata["target_names"]
    result = {name: round(float(val), 4) for name, val in zip(target_names, prediction)}
    result["source"] = "xgboost_model"
    return result

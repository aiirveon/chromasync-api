from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.colour_engine import analyse_image, compute_scene_drift
from app.models.predictor import predict_correction, model_available

router = APIRouter()


@router.post("/analyse")
async def analyse_footage(
    reference: UploadFile = File(...),
    scenes: List[UploadFile] = File(...),
):
    if not reference.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Reference must be an image")

    if len(scenes) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 scenes per analysis")

    reference_bytes = await reference.read()
    results = []

    for i, scene in enumerate(scenes):
        if not scene.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Scene {i+1} must be an image")

        scene_bytes = await scene.read()

        try:
            drift = compute_scene_drift(scene_bytes, reference_bytes)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Scene {i+1}: {str(e)}")

        delta_e = drift["delta_e"]

        if delta_e < 2.0:
            status = "Accepted"
        elif delta_e < 5.0:
            status = "Corrected"
        else:
            status = "Needs Review"

        # ML model prediction for correction values
        correction = predict_correction(
            drift["scene_profile"],
            drift["reference_profile"]
        )

        exp = drift["scene_profile"]["exposure_ev"]
        results.append({
            "scene_number": i + 1,
            "scene_name": scene.filename or f"SC{str(i+1).zfill(3)}",
            "delta_e": delta_e,
            "status": status,
            "temperature": f"{drift['scene_profile']['colour_temperature_k']}K",
            "exposure": f"{'+' if exp >= 0 else ''}{exp} EV",
            "temp_delta": drift["temp_delta"],
            "exposure_delta": drift["exposure_delta"],
            "saturation_delta": drift["saturation_delta"],
            "ml_correction": {
                "correct_r": correction.get("correct_r"),
                "correct_g": correction.get("correct_g"),
                "correct_b": correction.get("correct_b"),
                "correct_exposure_ev": correction.get("correct_exposure_ev"),
                "correct_temp_k": correction.get("correct_temp_k"),
                "correct_saturation": correction.get("correct_saturation"),
                "source": correction.get("source"),
            },
        })

    avg_delta_e = sum(r["delta_e"] for r in results) / len(results)
    max_delta_e = max(r["delta_e"] for r in results)
    needs_review = sum(1 for r in results if r["status"] == "Needs Review")
    ref_profile = analyse_image(reference_bytes)
    ref_exp = ref_profile["exposure_ev"]

    return {
        "scenes": results,
        "summary": {
            "total_scenes": len(results),
            "avg_delta_e": round(avg_delta_e, 2),
            "max_delta_e": round(max_delta_e, 2),
            "needs_review_count": needs_review,
            "reference_temp": f"{ref_profile['colour_temperature_k']}K",
            "reference_exposure": f"{'+' if ref_exp >= 0 else ''}{ref_exp} EV",
            "correction_source": "xgboost_model" if model_available() else "fallback_delta",
        },
    }

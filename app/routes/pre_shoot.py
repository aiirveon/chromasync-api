from fastapi import APIRouter, UploadFile, File, HTTPException
from app.colour_engine import analyse_image
from app.recommendations import recommend_camera_settings

router = APIRouter()


@router.post("/analyse")
async def analyse_reference_frame(file: UploadFile = File(...)):
    """
    Upload a reference frame image.
    Returns colour profile analysis and recommended camera settings.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    if len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")

    try:
        profile = analyse_image(image_bytes)
        settings = recommend_camera_settings(profile)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "colour_profile": {
            "colour_temperature_k": profile["colour_temperature_k"],
            "exposure_ev": profile["exposure_ev"],
            "saturation_pct": profile["saturation_pct"],
            "contrast_ratio": profile["contrast_ratio"],
            "mean_r": profile["mean_r"],
            "mean_g": profile["mean_g"],
            "mean_b": profile["mean_b"],
        },
        "camera_settings": settings,
        "histogram": profile["histogram"],
    }

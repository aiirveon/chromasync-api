from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.recommendations import recommend_on_shoot_adjustments

router = APIRouter()


class OnShootRequest(BaseModel):
    location: str = "Outdoor"
    time_of_day: str = "Golden Hour"
    lighting_source: str = "Natural"
    reference_temp_k: float = 5600
    reference_iso: int = 800


@router.post("/recommend")
async def get_on_shoot_recommendations(request: OnShootRequest):
    """
    Given current shooting conditions, return live parameter recommendations.
    """
    valid_locations = ["Indoor", "Outdoor", "Mixed"]
    valid_times = ["Golden Hour", "Midday", "Overcast", "Night"]
    valid_lighting = ["Natural", "Tungsten", "Fluorescent", "Mixed"]

    if request.location not in valid_locations:
        raise HTTPException(status_code=400, detail=f"Invalid location. Choose from: {valid_locations}")
    if request.time_of_day not in valid_times:
        raise HTTPException(status_code=400, detail=f"Invalid time of day. Choose from: {valid_times}")
    if request.lighting_source not in valid_lighting:
        raise HTTPException(status_code=400, detail=f"Invalid lighting source. Choose from: {valid_lighting}")

    recommendations = recommend_on_shoot_adjustments(
        location=request.location,
        time_of_day=request.time_of_day,
        lighting_source=request.lighting_source,
        reference_temp_k=request.reference_temp_k,
        reference_iso=request.reference_iso,
    )

    return {
        "conditions": {
            "location": request.location,
            "time_of_day": request.time_of_day,
            "lighting_source": request.lighting_source,
        },
        "recommendations": recommendations,
    }

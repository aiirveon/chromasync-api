def recommend_camera_settings(profile: dict) -> dict:
    """
    Given a colour profile from colour_engine.analyse_image,
    return recommended camera settings with plain English explanations.
    """
    temp_k = profile["colour_temperature_k"]
    exposure_ev = profile["exposure_ev"]
    saturation_pct = profile["saturation_pct"]
    contrast = profile["contrast_ratio"]

    # White balance recommendation
    wb = recommend_white_balance(temp_k)

    # ISO recommendation based on brightness
    iso = recommend_iso(exposure_ev)

    # Picture profile recommendation based on contrast range
    picture_profile = recommend_picture_profile(contrast, saturation_pct)

    # Exposure compensation
    exp_comp = recommend_exposure_comp(exposure_ev)

    return {
        "white_balance": wb,
        "iso": iso,
        "picture_profile": picture_profile,
        "exposure_compensation": exp_comp,
    }


def recommend_white_balance(temp_k: float) -> dict:
    temp_rounded = round(temp_k / 100) * 100

    if temp_k < 3200:
        label = "Tungsten / Warm Indoor"
        explanation = "Scene has a warm orange cast typical of tungsten lighting. Set white balance to match to avoid blue correction in post."
    elif temp_k < 4500:
        label = "Fluorescent / Early Morning"
        explanation = "Slightly cool-warm light. Common in mixed indoor environments or overcast dawn conditions."
    elif temp_k < 5500:
        label = "Daylight / Cloudy"
        explanation = "Natural daylight range. Standard outdoor setting for most shooting conditions."
    elif temp_k < 6500:
        label = "Sunny Daylight"
        explanation = "Bright midday sun. Your reference frame has a clean neutral colour balance."
    else:
        label = "Shade / Open Sky"
        explanation = "Cool blue cast from open shade or blue sky. Set white balance warmer to compensate."

    return {
        "value": f"{temp_rounded}K",
        "label": label,
        "explanation": explanation,
    }


def recommend_iso(exposure_ev: float) -> dict:
    if exposure_ev < -1.0:
        iso_value = 3200
        explanation = "Scene is underexposed. Higher ISO needed — watch for noise in shadows."
    elif exposure_ev < -0.3:
        iso_value = 1600
        explanation = "Slightly underexposed scene. ISO 1600 recovers detail without heavy noise."
    elif exposure_ev < 0.3:
        iso_value = 800
        explanation = "Well-exposed scene. ISO 800 is a clean, balanced choice for this light level."
    elif exposure_ev < 1.0:
        iso_value = 400
        explanation = "Scene is bright. Lower ISO keeps noise minimal and preserves highlight detail."
    else:
        iso_value = 100
        explanation = "Very bright scene. Base ISO recommended — consider ND filter if shooting wide open."

    return {
        "value": str(iso_value),
        "explanation": explanation,
    }


def recommend_picture_profile(contrast: float, saturation_pct: float) -> dict:
    if contrast > 0.7:
        profile = "S-Log3 / Log"
        explanation = "High contrast scene. Log profile preserves both highlights and shadows for maximum grading flexibility."
    elif contrast > 0.4:
        profile = "Cine Profile"
        explanation = "Moderate contrast. Cine profile gives a gentle roll-off in highlights — good balance of latitude and usable SOOC image."
    else:
        profile = "Neutral / Flat"
        explanation = "Low contrast scene. Neutral profile is sufficient — no need for heavy log encoding."

    return {
        "value": profile,
        "explanation": explanation,
    }


def recommend_exposure_comp(exposure_ev: float) -> dict:
    if exposure_ev < -0.5:
        comp = f"+{abs(round(exposure_ev, 1))} EV"
        explanation = "Push exposure up to match your reference brightness level."
    elif exposure_ev > 0.5:
        comp = f"-{round(exposure_ev, 1)} EV"
        explanation = "Pull exposure down slightly to prevent blown highlights."
    else:
        comp = "0 EV"
        explanation = "Exposure matches reference closely. No compensation needed."

    return {
        "value": comp,
        "explanation": explanation,
    }


def recommend_on_shoot_adjustments(
    location: str,
    time_of_day: str,
    lighting_source: str,
    reference_temp_k: float = 5600,
    reference_iso: int = 800,
) -> list:
    """
    Given current shooting conditions, return parameter adjustment recommendations.
    """
    recommendations = []

    # White balance adjustment
    target_temp = get_condition_temperature(location, time_of_day, lighting_source)
    temp_delta = target_temp - reference_temp_k
    recommendations.append({
        "parameter": "White Balance",
        "current": f"{int(reference_temp_k)}K",
        "recommended": f"{int(target_temp)}K",
        "delta": f"{'+' if temp_delta >= 0 else ''}{int(temp_delta)}K",
        "direction": "up" if temp_delta > 0 else "down" if temp_delta < 0 else "neutral",
        "explanation": f"Adjust for {lighting_source.lower()} light in {location.lower()} conditions.",
    })

    # ISO adjustment
    target_iso = get_condition_iso(time_of_day, lighting_source)
    iso_delta = target_iso - reference_iso
    recommendations.append({
        "parameter": "ISO",
        "current": str(reference_iso),
        "recommended": str(target_iso),
        "delta": f"{'+' if iso_delta >= 0 else ''}{iso_delta}",
        "direction": "up" if iso_delta > 0 else "down" if iso_delta < 0 else "neutral",
        "explanation": f"ISO adjusted for {time_of_day.lower()} lighting conditions.",
    })

    # ND filter
    nd = get_nd_recommendation(time_of_day, location)
    if nd:
        recommendations.append(nd)

    return recommendations


def get_condition_temperature(location: str, time_of_day: str, lighting_source: str) -> float:
    base = 5600.0

    temp_map = {
        "Golden Hour": 3800,
        "Midday": 5500,
        "Overcast": 6500,
        "Night": 3200,
    }
    lighting_map = {
        "Tungsten": 3200,
        "Fluorescent": 4500,
        "Natural": 5600,
        "Mixed": 4800,
    }

    time_temp = temp_map.get(time_of_day, base)
    light_temp = lighting_map.get(lighting_source, base)

    if location == "Indoor":
        return light_temp
    return time_temp


def get_condition_iso(time_of_day: str, lighting_source: str) -> int:
    iso_map = {
        "Golden Hour": 800,
        "Midday": 200,
        "Overcast": 1600,
        "Night": 3200,
    }
    if lighting_source == "Tungsten":
        return 1600
    return iso_map.get(time_of_day, 800)


def get_nd_recommendation(time_of_day: str, location: str) -> dict | None:
    if time_of_day == "Midday" and location == "Outdoor":
        return {
            "parameter": "ND Filter",
            "current": "None",
            "recommended": "ND 0.9",
            "delta": "3 stops",
            "direction": "neutral",
            "explanation": "Bright midday sun requires ND to maintain correct exposure at cinematic shutter speed.",
        }
    return None

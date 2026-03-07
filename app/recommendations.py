def recommend_camera_settings(profile: dict) -> dict:
    """
    Given a colour profile from colour_engine.analyse_image,
    return recommended camera settings with plain English explanations.
    """
    temp_k = profile["colour_temperature_k"]
    exposure_ev = profile["exposure_ev"]
    saturation_pct = profile["saturation_pct"]
    contrast = profile["contrast_ratio"]

    wb = recommend_white_balance(temp_k)
    iso = recommend_iso(exposure_ev)
    picture_profile = recommend_picture_profile(contrast, saturation_pct)
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
        plain = "Scene has a warm orange cast typical of tungsten lighting. Set white balance to match to avoid blue correction in post."
        technical = "Set WB to 2700–3200K or use Tungsten preset. Add slight magenta trim (+2–3) to counteract green cast common in older tungsten bulbs."
    elif temp_k < 4500:
        label = "Fluorescent / Early Morning"
        plain = "Slightly cool-warm light. Common in mixed indoor environments or overcast dawn conditions."
        technical = "Set WB to 4000–4500K. Add +3–5 Magenta in WB fine-tune to counteract the green spike of fluorescent tubes."
    elif temp_k < 5500:
        label = "Daylight / Cloudy"
        plain = "Natural daylight range. Standard outdoor setting for most shooting conditions."
        technical = "Set WB to 5000–5500K or use Cloudy/Daylight preset. Lock in Kelvin manually to prevent drift between shots."
    elif temp_k < 6500:
        label = "Sunny Daylight"
        plain = "Bright midday sun. Your reference frame has a clean neutral colour balance."
        technical = "Set WB to 5500–6000K. Use Daylight preset or lock Kelvin manually. Avoid Auto WB — it shifts as clouds pass."
    else:
        label = "Shade / Open Sky"
        plain = "Cool blue cast from open shade or blue sky. Set white balance warmer to compensate."
        technical = "Set WB to 6500–7500K or use Shade preset. In-camera: WB > Kelvin > dial up until skin tones look neutral."

    return {
        "value": f"{temp_rounded}K",
        "label": label,
        "plainEnglish": plain,
        "technical": technical,
    }


def recommend_iso(exposure_ev: float) -> dict:
    if exposure_ev < -1.0:
        iso_value = 3200
        plain = "Scene is underexposed. Higher ISO needed — watch for noise in shadows."
        technical = "ISO 3200: check your camera's native ISO values — on Sony/Canon dual-native cameras, 3200 is often a clean native ISO. Review shadow noise in playback."
    elif exposure_ev < -0.3:
        iso_value = 1600
        plain = "Slightly underexposed scene. ISO 1600 recovers detail without heavy noise."
        technical = "ISO 1600: 1 stop above base. Acceptable on most modern mirrorless cameras. Check grain on a calibrated monitor before delivery."
    elif exposure_ev < 0.3:
        iso_value = 800
        plain = "Well-exposed scene. ISO 800 is a clean, balanced choice for this light level."
        technical = "ISO 800: common native ISO for Sony/Canon/Panasonic cameras. Clean signal with full dynamic range available."
    elif exposure_ev < 1.0:
        iso_value = 400
        plain = "Scene is bright. Lower ISO keeps noise minimal and preserves highlight detail."
        technical = "ISO 400: 1 stop below base ISO on most cameras. Use ND filter or stop down aperture if highlights are clipping."
    else:
        iso_value = 100
        plain = "Very bright scene. Base ISO recommended — consider ND filter if shooting wide open."
        technical = "ISO 100: maximum signal quality. Pair with ND filter outdoors to maintain cinematic shutter speed (1/50s at 25fps)."

    return {
        "value": str(iso_value),
        "plainEnglish": plain,
        "technical": technical,
    }


def recommend_picture_profile(contrast: float, saturation_pct: float) -> dict:
    if contrast > 0.7:
        profile = "S-Log3 / Log"
        plain = "High contrast scene. Log profile preserves both highlights and shadows for maximum grading flexibility."
        technical = "S-Log3 or V-Log: +800 ISO minimum. Expose 1–2 stops brighter than metered (ETTR). Apply LUT in post for correct gamma."
    elif contrast > 0.4:
        profile = "Cine Profile"
        plain = "Moderate contrast. Cine profile gives a gentle roll-off in highlights — good balance of latitude and usable SOOC image."
        technical = "Cinelike D2 (Panasonic), S-Cinetone (Sony), or Canon Cinema. These profiles clip highlights gently and require minimal grading."
    else:
        profile = "Neutral / Flat"
        plain = "Low contrast scene. Neutral profile is sufficient — no need for heavy log encoding."
        technical = "Set Picture Style/Profile to Neutral or Flat. Reduce contrast -3 and saturation -2 for extra grading headroom without full log."

    return {
        "value": profile,
        "plainEnglish": plain,
        "technical": technical,
    }


def recommend_exposure_comp(exposure_ev: float) -> dict:
    if exposure_ev < -0.5:
        comp = f"+{abs(round(exposure_ev, 1))} EV"
        plain = "Push exposure up to match your reference brightness level."
        technical = f"Apply {abs(round(exposure_ev, 1))} EV positive exposure compensation. Check histogram — peaks should sit in the upper third without clipping."
    elif exposure_ev > 0.5:
        comp = f"-{round(exposure_ev, 1)} EV"
        plain = "Pull exposure down slightly to prevent blown highlights."
        technical = f"Apply {round(exposure_ev, 1)} EV negative exposure compensation. Use zebras at 95–100 IRE to monitor highlight rolloff."
    else:
        comp = "0 EV"
        plain = "Exposure matches reference closely. No compensation needed."
        technical = "Exposure is within ±0.5 EV of reference. Histogram peaks should align with your reference frame. No adjustment required."

    return {
        "value": comp,
        "plainEnglish": plain,
        "technical": technical,
    }


# ─── On-Shoot Live Recommendations ───────────────────────────────────────────

def recommend_on_shoot_adjustments(
    location: str,
    time_of_day: str,
    lighting_source: str,
    reference_temp_k: float = 5600,
    reference_iso: int = 800,
) -> list:
    """
    Given current shooting conditions, return parameter adjustment recommendations
    with plain English, technical detail, and a practical tip.
    """
    recommendations = []

    # White balance
    target_temp = get_condition_temperature(location, time_of_day, lighting_source)
    temp_delta = target_temp - reference_temp_k
    wb_technical, wb_tip = get_wb_detail(target_temp, lighting_source)
    recommendations.append({
        "parameter": "White Balance",
        "current": f"{int(reference_temp_k)}K",
        "recommended": f"{int(target_temp)}K",
        "delta": f"{'+' if temp_delta >= 0 else ''}{int(temp_delta)}K",
        "direction": "up" if temp_delta > 0 else "down" if temp_delta < 0 else "neutral",
        "explanation": f"Adjust white balance for {lighting_source.lower()} light in {location.lower()} conditions.",
        "technical_detail": wb_technical,
        "tip": wb_tip,
    })

    # ISO
    target_iso = get_condition_iso(time_of_day, lighting_source)
    iso_delta = target_iso - reference_iso
    iso_technical, iso_tip = get_iso_detail(target_iso, time_of_day)
    recommendations.append({
        "parameter": "ISO",
        "current": str(reference_iso),
        "recommended": str(target_iso),
        "delta": f"{'+' if iso_delta >= 0 else ''}{iso_delta}",
        "direction": "up" if iso_delta > 0 else "down" if iso_delta < 0 else "neutral",
        "explanation": f"ISO adjusted for {time_of_day.lower()} {lighting_source.lower()} conditions.",
        "technical_detail": iso_technical,
        "tip": iso_tip,
    })

    # Shutter speed
    shutter_tech, shutter_tip, shutter_rec = get_shutter_detail(time_of_day, location)
    recommendations.append({
        "parameter": "Shutter Speed",
        "current": "1/50s (180° rule)",
        "recommended": shutter_rec,
        "delta": "check",
        "direction": "neutral",
        "explanation": "Shutter speed should follow the 180° rule: set to double your frame rate for natural motion blur.",
        "technical_detail": shutter_tech,
        "tip": shutter_tip,
    })

    # ND filter
    nd = get_nd_recommendation(time_of_day, location)
    if nd:
        recommendations.append(nd)

    return recommendations


def get_wb_detail(target_temp: float, lighting_source: str) -> tuple:
    tech_map = {
        "Tungsten":    (
            f"Set WB to {int(target_temp)}K or use the Tungsten preset. On Sony: Camera Settings 1 > White Balance > Kelvin. On Canon: Shooting Menu > White Balance > K.",
            "Don't fully correct tungsten to neutral — the warm glow is often the look. Set WB to match the dominant source, then grade selectively in post."
        ),
        "Fluorescent": (
            f"Set WB to {int(target_temp)}K or use Fluorescent preset. Add +3 to +5 Magenta in WB fine-tune (A-B / G-M grid) to kill the green cast.",
            "Custom white balance using a grey card under your lights gives far more accurate results than any preset in fluorescent environments."
        ),
        "Natural":     (
            f"Set WB manually to {int(target_temp)}K. Avoid Auto WB — it drifts between shots, creating colour inconsistency that's difficult to match in post.",
            "Lock your white balance before rolling. Even small Auto WB shifts create a colour-matching headache across a sequence of shots."
        ),
        "Mixed":       (
            f"Target WB is {int(target_temp)}K — a blend of your mixed sources. Choose the dominant light and set WB to match it. Correct other sources in post.",
            "Shoot a grey card under each individual light source at the start. This gives you a reference for every colour temperature in the scene."
        ),
    }
    return tech_map.get(lighting_source, (
        f"Set white balance to {int(target_temp)}K manually for consistent colour across the shoot.",
        "Always lock white balance manually when possible — Auto WB shifts between shots and creates colour inconsistency."
    ))


def get_iso_detail(target_iso: int, time_of_day: str) -> tuple:
    tech_map = {
        "Golden Hour": (
            f"ISO {target_iso}: light drops rapidly at golden hour. Start here and increase as the sun falls. Monitor histogram — don't clip sky highlights.",
            "Shoot in manual mode during golden hour. Light changes every 2–3 minutes — adjust ISO incrementally rather than letting Auto ISO shift mid-scene."
        ),
        "Midday":      (
            f"ISO {target_iso}: bright midday light allows base ISO for maximum image quality and dynamic range. Pair with ND filter to maintain cinematic shutter speed.",
            "Shoot at base ISO in midday sun. A Log or flat picture profile protects highlights — bright skies blow out fast on standard colour profiles."
        ),
        "Overcast":    (
            f"ISO {target_iso}: soft overcast light provides even, diffused exposure. This ISO gives clean shadows without noise risk.",
            "Overcast is ideal for consistent exposure. No harsh shadows means you can focus entirely on composition and performance."
        ),
        "Night":       (
            f"ISO {target_iso}: night shooting requires high ISO to gather enough light. Test your camera's noise ceiling — most modern cameras handle ISO 3200–6400 well.",
            "Open aperture first (f/1.8–2.8) before pushing ISO. A wide aperture gives 2–3 stops of extra light and keeps noise lower than ISO alone."
        ),
    }
    return tech_map.get(time_of_day, (
        f"ISO {target_iso} recommended for current conditions. Check your camera's native ISO values for cleanest signal.",
        "Test your camera's dual native ISO values — shooting at native ISO (e.g. 800 or 3200 on Sony) gives the cleanest signal-to-noise ratio."
    ))


def get_shutter_detail(time_of_day: str, location: str) -> tuple:
    shutter_map = {
        ("Golden Hour", "Outdoor"): (
            "1/50s at 25fps or 1/48s at 24fps (180° rule). Golden hour light drops fast — adjust ISO rather than changing shutter to maintain consistent motion blur.",
            "Keep shutter locked at double frame rate. Adjust only ISO or aperture as light fades — changing shutter mid-scene creates an unnatural look.",
            "1/50s (25fps) or 1/48s (24fps)"
        ),
        ("Midday", "Outdoor"):      (
            "1/50s at 25fps is correct. Bright conditions will require ND filter to maintain this shutter speed. Without ND, you'll need to stop down or raise shutter — both look wrong.",
            "An ND 0.6 (2 stops) or ND 0.9 (3 stops) lets you hold 1/50s at base ISO in full sun. Variable ND is flexible but check for X-pattern at extreme settings.",
            "1/50s + ND filter"
        ),
        ("Overcast", "Outdoor"):    (
            "1/50s at 25fps. Soft overcast rarely requires ND outdoors. You may need to open aperture slightly — overcast can drop 1–2 stops versus direct sun.",
            "Overcast is the most forgiving condition for shutter speed. Use it to experiment with depth of field — you can often shoot wide open without ND.",
            "1/50s (25fps)"
        ),
        ("Night", "Outdoor"):       (
            "1/50s at 25fps. Maintain double frame rate even at night. Push ISO and open aperture before considering a slower shutter — below 1/25s motion blur becomes unnatural.",
            "Avoid slowing shutter below your frame rate. If you're still underexposed at max ISO and wide aperture, you need more practical light in the scene.",
            "1/50s (25fps) + wide aperture"
        ),
        ("Night", "Indoor"):        (
            "1/50s at 25fps. Control exposure indoors by adjusting your lights, not shutter speed. Shutter speed should always follow the 180° rule.",
            "Use dimmable LEDs or practical lamps to manage exposure indoors at night. Shutter speed is not an exposure tool in cinematic work.",
            "1/50s (25fps)"
        ),
        ("Golden Hour", "Indoor"):  (
            "1/50s at 25fps. Window light at golden hour gives warm directional light — very cinematic. Position subject facing the window as a natural key light.",
            "Golden hour window light fades in minutes. Set exposure manually, frame your shot, and roll before the light shifts.",
            "1/50s (25fps)"
        ),
        ("Overcast", "Indoor"):     (
            "1/50s at 25fps. Overcast window light is soft and even — great for interviews and close-ups. No harsh shadows to manage.",
            "Overcast indoor window light is consistent for extended periods — perfect for long takes or interviews.",
            "1/50s (25fps)"
        ),
    }
    key = (time_of_day, location)
    return shutter_map.get(key, (
        "1/50s at 25fps or 1/48s at 24fps following the 180° rule for natural cinematic motion blur.",
        "The 180° shutter rule (double your frame rate) is the cinematic standard for natural-looking motion. Stick to it unless you deliberately want a different look.",
        "1/50s (25fps)"
    ))


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
            "recommended": "ND 0.9 (3 stops)",
            "delta": "3 stops",
            "direction": "neutral",
            "explanation": "Bright midday sun requires ND to maintain cinematic shutter speed (1/50s) at base ISO.",
            "technical_detail": "ND 0.9 = 3 stops of light reduction. Allows ISO 200 + f/4 + 1/50s in direct sunlight. Variable ND (ND 0.6–1.8) is versatile but check for X-pattern artefacts at extreme settings.",
            "tip": "Fixed ND gives better optical quality than variable ND. Carry ND 0.6 (2 stop) and ND 0.9 (3 stop) as a starting kit — they cover 90% of outdoor daylight situations.",
        }
    return None

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.colour_engine import analyse_image
import json

router = APIRouter()


def drift_status(delta: float, tight: float, loose: float) -> str:
    abs_delta = abs(delta)
    if abs_delta <= tight:
        return "on_target"
    if abs_delta <= loose:
        return "slight"
    return "significant"


def advise_white_balance(delta_k: float, live_k: float, ref_k: float) -> str:
    if abs(delta_k) <= 200:
        return "White balance is well matched to your reference."
    direction = "warmer" if delta_k < 0 else "cooler"
    stops = abs(round(delta_k / 500, 1))
    return (
        f"Your live frame is {abs(int(delta_k))}K {direction} than the reference. "
        f"Shift white balance {'up' if direction == 'warmer' else 'down'} by roughly {stops} stop{'s' if stops != 1 else ''} "
        f"(target: {int(ref_k)}K)."
    )


def advise_exposure(delta_ev: float) -> str:
    if abs(delta_ev) <= 0.3:
        return "Exposure matches the reference closely - no adjustment needed."
    direction = "brighter" if delta_ev < 0 else "darker"
    return (
        f"Your live frame is {abs(round(delta_ev, 1))} EV {direction} than the reference. "
        f"{'Increase' if direction == 'brighter' else 'Decrease'} exposure by {abs(round(delta_ev, 1))} EV using "
        f"{'aperture, ISO, or exposure compensation' if direction == 'brighter' else 'ND filter, stop down, or reduce exposure comp'}."
    )


def advise_saturation(delta_sat: float) -> str:
    if abs(delta_sat) <= 5:
        return "Saturation is close to your reference - no camera adjustment needed."
    direction = "more saturated" if delta_sat > 0 else "less saturated"
    action = (
        "lower saturation in your picture profile by 1-2 steps"
        if delta_sat > 0
        else "increase saturation in your picture profile by 1-2 steps, or consider a warmer light source"
    )
    return f"Your live frame is {abs(round(delta_sat, 1))}% {direction} than the reference. Try to {action}."


def advise_contrast(delta_contrast: float) -> str:
    if abs(delta_contrast) <= 0.05:
        return "Contrast is well matched - no picture profile change needed."
    direction = "more contrast" if delta_contrast > 0 else "less contrast"
    action = (
        "reduce contrast in your picture profile, or switch to a flatter profile"
        if delta_contrast > 0
        else "increase contrast slightly in your picture profile, or use a less flat profile"
    )
    return f"Your live frame has {direction} than the reference. On camera, {action}."


def advise_channel_balance(delta_r: float, delta_g: float, delta_b: float) -> str:
    dominant = max(
        [("red", delta_r), ("green", delta_g), ("blue", delta_b)],
        key=lambda x: abs(x[1])
    )
    channel, value = dominant
    if abs(value) <= 8:
        return "Colour channel balance is well matched across RGB channels."
    direction = "higher" if value > 0 else "lower"
    fix_map = {
        ("red", "higher"):   "shift white balance slightly cooler, or add a subtle cyan grade in post",
        ("red", "lower"):    "shift white balance slightly warmer",
        ("green", "higher"): "add magenta in your white balance fine-tune (+2 to +5 magenta)",
        ("green", "lower"):  "reduce magenta or add slight green in your WB fine-tune",
        ("blue", "higher"):  "shift white balance warmer (higher Kelvin value)",
        ("blue", "lower"):   "shift white balance cooler (lower Kelvin value)",
    }
    fix = fix_map.get((channel, direction), "adjust white balance to correct the channel imbalance")
    return (
        f"Your {channel} channel is {abs(round(value, 1))} points {direction} than the reference - "
        f"a visible colour cast. On camera: {fix}."
    )


def advise_tonal_distribution(live_hist: list, ref_hist: list) -> tuple:
    live_shadows = sum(live_hist[:2])
    live_highs   = sum(live_hist[6:])
    ref_shadows  = sum(ref_hist[:2])
    ref_highs    = sum(ref_hist[6:])
    shadow_delta = live_shadows - ref_shadows
    high_delta   = live_highs   - ref_highs
    issues = []
    if shadow_delta > 0.15:
        issues.append("shadows are heavier - lift shadows or increase exposure")
    elif shadow_delta < -0.15:
        issues.append("shadows are lighter - crush them slightly or reduce exposure")
    if high_delta > 0.12:
        issues.append("highlights are brighter - reduce exposure or use ND")
    elif high_delta < -0.12:
        issues.append("highlights are duller - increase exposure or open aperture")
    if not issues:
        return "Tonal distribution matches your reference well across shadows, midtones, and highlights.", "on_target"
    status = "significant" if len(issues) >= 2 else "slight"
    return "Tonal drift: " + "; ".join(issues) + ".", status


def advise_picture_profile_flatness(live_contrast: float, ref_contrast: float) -> str:
    delta = live_contrast - ref_contrast
    if abs(delta) <= 0.06:
        return "Picture profile flatness appears well matched to your reference."
    if delta > 0:
        return (
            "Your live frame has more contrast than the reference, suggesting a less flat picture profile. "
            "Switch to a flatter profile (Cine/Log) to match the reference latitude."
        )
    return (
        "Your live frame is flatter than the reference. "
        "If shooting Log intentionally, this is correct - apply your LUT in post. "
        "Otherwise, increase contrast in your picture profile."
    )


def compute_overall_status(metrics: list) -> str:
    statuses = [m["status"] for m in metrics]
    if all(s == "on_target" for s in statuses):
        return "on_target"
    if any(s == "significant" for s in statuses):
        return "significant"
    return "slight"


@router.post("/compare")
async def compare_live_frame(
    live_frame: UploadFile = File(...),
    reference_profile: str = Form(...),
):
    image_bytes = await live_frame.read()
    if len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 50MB.")
    try:
        ref = json.loads(reference_profile)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid reference_profile JSON")
    try:
        live = analyse_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    delta_k        = live["colour_temperature_k"] - ref["colour_temperature_k"]
    delta_ev       = live["exposure_ev"]           - ref["exposure_ev"]
    delta_sat      = live["saturation_pct"]        - ref["saturation_pct"]
    delta_contrast = live["contrast_ratio"]        - ref["contrast_ratio"]
    delta_r        = live["mean_r"]                - ref["mean_r"]
    delta_g        = live["mean_g"]                - ref["mean_g"]
    delta_b        = live["mean_b"]                - ref["mean_b"]

    ref_hist  = ref.get("histogram",  [0.125] * 8)
    live_hist = live.get("histogram", [0.125] * 8)

    tonal_advice, tonal_status = advise_tonal_distribution(live_hist, ref_hist)

    metrics_unordered = [
        {
            "id":        "white_balance",
            "label":     "White Balance",
            "ref_value": f"{int(ref['colour_temperature_k'])}K",
            "live_value":f"{int(live['colour_temperature_k'])}K",
            "delta":     f"{'+' if delta_k >= 0 else ''}{int(delta_k)}K",
            "delta_raw": delta_k,
            "status":    drift_status(delta_k, 200, 600),
            "advice":    advise_white_balance(delta_k, live["colour_temperature_k"], ref["colour_temperature_k"]),
        },
        {
            "id":        "exposure",
            "label":     "Exposure",
            "ref_value": f"{'+' if ref['exposure_ev'] >= 0 else ''}{round(ref['exposure_ev'], 2)} EV",
            "live_value":f"{'+' if live['exposure_ev'] >= 0 else ''}{round(live['exposure_ev'], 2)} EV",
            "delta":     f"{'+' if delta_ev >= 0 else ''}{round(delta_ev, 2)} EV",
            "delta_raw": delta_ev,
            "status":    drift_status(delta_ev, 0.3, 0.8),
            "advice":    advise_exposure(delta_ev),
        },
        {
            "id":        "saturation",
            "label":     "Saturation",
            "ref_value": f"{round(ref['saturation_pct'], 1)}%",
            "live_value":f"{round(live['saturation_pct'], 1)}%",
            "delta":     f"{'+' if delta_sat >= 0 else ''}{round(delta_sat, 1)}%",
            "delta_raw": delta_sat,
            "status":    drift_status(delta_sat, 5, 15),
            "advice":    advise_saturation(delta_sat),
        },
        {
            "id":        "contrast",
            "label":     "Contrast",
            "ref_value": f"{round(ref['contrast_ratio'], 3)}x",
            "live_value":f"{round(live['contrast_ratio'], 3)}x",
            "delta":     f"{'+' if delta_contrast >= 0 else ''}{round(delta_contrast, 3)}x",
            "delta_raw": delta_contrast,
            "status":    drift_status(delta_contrast, 0.05, 0.15),
            "advice":    advise_contrast(delta_contrast),
        },
        {
            "id":        "channel_balance",
            "label":     "Colour Cast",
            "ref_value": f"R{int(ref['mean_r'])} G{int(ref['mean_g'])} B{int(ref['mean_b'])}",
            "live_value":f"R{int(live['mean_r'])} G{int(live['mean_g'])} B{int(live['mean_b'])}",
            "delta":     f"R{int(delta_r):+} G{int(delta_g):+} B{int(delta_b):+}",
            "delta_raw": max(abs(delta_r), abs(delta_g), abs(delta_b)),
            "status":    drift_status(max(abs(delta_r), abs(delta_g), abs(delta_b)), 8, 20),
            "advice":    advise_channel_balance(delta_r, delta_g, delta_b),
        },
        {
            "id":        "tonal_distribution",
            "label":     "Tonal Distribution",
            "ref_value": "reference tones",
            "live_value":"live tones",
            "delta":     "--",
            "delta_raw": 0,
            "status":    tonal_status,
            "advice":    tonal_advice,
        },
        {
            "id":        "picture_profile",
            "label":     "Picture Profile Flatness",
            "ref_value": f"{round(ref['contrast_ratio'], 3)}x",
            "live_value":f"{round(live['contrast_ratio'], 3)}x",
            "delta":     f"{'+' if delta_contrast >= 0 else ''}{round(delta_contrast, 3)}x",
            "delta_raw": abs(delta_contrast),
            "status":    drift_status(abs(delta_contrast), 0.06, 0.18),
            "advice":    advise_picture_profile_flatness(live["contrast_ratio"], ref["contrast_ratio"]),
        },
    ]

    # Reorder by importance: WB > Colour Cast > Exposure > Tonal Distribution > Saturation > Contrast > Picture Profile
    order = ["white_balance", "channel_balance", "exposure", "tonal_distribution", "saturation", "contrast", "picture_profile"]
    metrics = sorted(metrics_unordered, key=lambda m: order.index(m["id"]) if m["id"] in order else 99)

    overall = compute_overall_status(metrics)
    overall_label = {
        "on_target":   "On Target",
        "slight":      "Slight Drift",
        "significant": "Significant Drift",
    }[overall]

    return {
        "overall_status": overall,
        "overall_label":  overall_label,
        "live_profile": {
            "colour_temperature_k": live["colour_temperature_k"],
            "exposure_ev":          live["exposure_ev"],
            "saturation_pct":       live["saturation_pct"],
            "contrast_ratio":       live["contrast_ratio"],
            "mean_r": live["mean_r"],
            "mean_g": live["mean_g"],
            "mean_b": live["mean_b"],
            "histogram": live["histogram"],
        },
        "metrics": metrics,
    }

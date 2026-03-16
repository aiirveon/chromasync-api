import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.colour_engine import analyse_image, decode_image

router = APIRouter()


def delta_e_verdict(delta_e: float) -> dict:
    """
    Translates a Delta E score into a plain English verdict and status.
    Delta E is the industry standard colour difference metric.
    Below 1: invisible to the human eye.
    1 to 2: visible only to trained eyes.
    2 to 5: visible to most people on close inspection.
    Above 5: clearly visible drift between scenes.
    """
    if delta_e < 1.0:
        return {
            "status": "excellent",
            "label": "Invisible",
            "plain": "The colour difference between these two frames is invisible to the human eye. Your scenes will cut together seamlessly.",
            "recommendation": "No correction needed. Maintain your current camera settings.",
        }
    elif delta_e < 2.0:
        return {
            "status": "good",
            "label": "Minimal",
            "plain": "A very small colour difference exists. Only a trained eye looking carefully will notice it.",
            "recommendation": "No correction needed for most projects. For high-end work, a minor white balance tweak may help.",
        }
    elif delta_e < 5.0:
        return {
            "status": "warning",
            "label": "Noticeable",
            "plain": "A visible colour difference exists between these frames. Viewers may notice if scenes cut directly against each other.",
            "recommendation": "Adjust white balance or exposure before the next shot. Check the detailed breakdown below for which parameter has drifted most.",
        }
    elif delta_e < 10.0:
        return {
            "status": "significant",
            "label": "Significant Drift",
            "plain": "Clear colour drift between these frames. These scenes will look inconsistent when edited together.",
            "recommendation": "Correct camera settings now. Check white balance first, then exposure. Post correction will also be needed.",
        }
    else:
        return {
            "status": "critical",
            "label": "Critical Drift",
            "plain": "Severe colour drift. These frames look like they were shot under completely different conditions.",
            "recommendation": "Significant post correction will be required. Reset your camera settings to match the reference before continuing.",
        }



@router.post("/drift")
async def compute_drift(
    scene: UploadFile = File(..., description="The scene frame to check for drift"),
    reference: UploadFile = File(..., description="The reference frame to compare against"),
):
    """
    Compute colour drift between two frames using Delta E.

    Upload a scene frame and a reference frame.
    Returns a Delta E score, plain English verdict, and a per-parameter breakdown
    showing exactly which colour property has drifted and by how much.

    Delta E is the industry standard colour difference metric:
    - Below 2: acceptable continuity
    - 2 to 5: noticeable, correct before next shot
    - Above 5: significant drift requiring correction
    """
    scene_bytes = await scene.read()
    reference_bytes = await reference.read()

    if len(scene_bytes) > 50 * 1024 * 1024 or len(reference_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Files too large. Maximum 50MB per image.")

    try:
        scene_bgr = decode_image(scene_bytes)
        ref_bgr = decode_image(reference_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Analyse both frames
    scene_profile = analyse_image(scene_bytes)
    ref_profile = analyse_image(reference_bytes)

    # Compute Delta E in CIE Lab space (perceptually accurate)
    scene_lab = cv2.cvtColor(scene_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(float)
    ref_lab = cv2.cvtColor(ref_bgr.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(float)
    mean_scene_lab = np.mean(scene_lab.reshape(-1, 3), axis=0)
    mean_ref_lab = np.mean(ref_lab.reshape(-1, 3), axis=0)
    delta_e = float(np.sqrt(np.sum((mean_scene_lab - mean_ref_lab) ** 2)))
    delta_e = round(delta_e, 2)

    # Per-parameter deltas
    delta_k = scene_profile["colour_temperature_k"] - ref_profile["colour_temperature_k"]
    delta_ev = scene_profile["exposure_ev"] - ref_profile["exposure_ev"]
    delta_sat = scene_profile["saturation_pct"] - ref_profile["saturation_pct"]
    delta_r = scene_profile["mean_r"] - ref_profile["mean_r"]
    delta_g = scene_profile["mean_g"] - ref_profile["mean_g"]
    delta_b = scene_profile["mean_b"] - ref_profile["mean_b"]

    # Identify the primary driver of drift
    drivers = {
        "white_balance": abs(delta_k) / 500,
        "exposure": abs(delta_ev) / 0.5,
        "saturation": abs(delta_sat) / 10,
        "colour_cast": max(abs(delta_r), abs(delta_g), abs(delta_b)) / 15,
    }
    primary_driver = max(drivers, key=drivers.get)

    driver_labels = {
        "white_balance": "White balance",
        "exposure": "Exposure",
        "saturation": "Saturation",
        "colour_cast": "Colour cast (RGB channel imbalance)",
    }

    verdict = delta_e_verdict(delta_e)

    return {
        "delta_e": delta_e,
        "verdict": verdict,
        "primary_driver": {
            "parameter": primary_driver,
            "label": driver_labels[primary_driver],
            "plain": f"{driver_labels[primary_driver]} is the largest contributor to the colour drift between these frames.",
        },
        "breakdown": {
            "white_balance": {
                "label": "White Balance",
                "scene_value": f"{scene_profile['colour_temperature_k']}K",
                "reference_value": f"{ref_profile['colour_temperature_k']}K",
                "delta": f"{'+' if delta_k >= 0 else ''}{int(delta_k)}K",
                "delta_raw": round(delta_k),
            },
            "exposure": {
                "label": "Exposure",
                "scene_value": f"{scene_profile['exposure_ev']:+.2f} EV",
                "reference_value": f"{ref_profile['exposure_ev']:+.2f} EV",
                "delta": f"{'+' if delta_ev >= 0 else ''}{round(delta_ev, 2)} EV",
                "delta_raw": round(delta_ev, 2),
            },
            "saturation": {
                "label": "Saturation",
                "scene_value": f"{scene_profile['saturation_pct']}%",
                "reference_value": f"{ref_profile['saturation_pct']}%",
                "delta": f"{'+' if delta_sat >= 0 else ''}{round(delta_sat, 1)}%",
                "delta_raw": round(delta_sat, 1),
            },
            "colour_cast": {
                "label": "Colour Cast (RGB)",
                "scene_value": f"R{int(scene_profile['mean_r'])} G{int(scene_profile['mean_g'])} B{int(scene_profile['mean_b'])}",
                "reference_value": f"R{int(ref_profile['mean_r'])} G{int(ref_profile['mean_g'])} B{int(ref_profile['mean_b'])}",
                "delta": f"R{int(delta_r):+} G{int(delta_g):+} B{int(delta_b):+}",
                "delta_raw": round(max(abs(delta_r), abs(delta_g), abs(delta_b)), 1),
            },
        },
        "profiles": {
            "scene": scene_profile,
            "reference": ref_profile,
        },
        "target": {
            "delta_e_target": 5.0,
            "target_label": "Professional continuity threshold",
            "within_target": delta_e < 5.0,
        },
    }

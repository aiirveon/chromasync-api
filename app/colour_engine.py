import cv2
import numpy as np


def analyse_image(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    mean_r = float(np.mean(img_rgb[:, :, 0]))
    mean_g = float(np.mean(img_rgb[:, :, 1]))
    mean_b = float(np.mean(img_rgb[:, :, 2]))

    colour_temp_k = estimate_colour_temperature(mean_r, mean_g, mean_b)
    brightness = float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))
    exposure_ev = round((brightness - 128) / 64, 2)
    saturation = float(np.mean(img_hsv[:, :, 1])) / 255 * 100
    l_channel = img_lab[:, :, 0].astype(float)
    contrast = float(np.std(l_channel)) / 128

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist_normalised = [float(v[0]) for v in hist / hist.sum()]

    return {
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "colour_temperature_k": round(colour_temp_k),
        "exposure_ev": exposure_ev,
        "saturation_pct": round(saturation, 1),
        "contrast_ratio": round(contrast, 3),
        "histogram": hist_normalised,
    }


def estimate_colour_temperature(r: float, g: float, b: float) -> float:
    total = r + g + b
    if total == 0:
        return 5500.0

    rn = r / total
    gn = g / total
    bn = b / total

    x = 0.4124 * rn + 0.3576 * gn + 0.1805 * bn
    y = 0.2126 * rn + 0.7152 * gn + 0.0722 * bn

    if y == 0:
        return 5500.0

    n = (x - 0.3320) / (0.1858 - y)
    cct = 449 * (n ** 3) + 3525 * (n ** 2) + 6823.3 * n + 5520.33

    return max(2500.0, min(10000.0, cct))


def calculate_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def compute_scene_drift(scene_bytes: bytes, reference_bytes: bytes) -> dict:
    scene_data = analyse_image(scene_bytes)
    ref_data = analyse_image(reference_bytes)

    scene_vec = np.array([scene_data["mean_r"], scene_data["mean_g"], scene_data["mean_b"]])
    ref_vec = np.array([ref_data["mean_r"], ref_data["mean_g"], ref_data["mean_b"]])
    delta_e = calculate_delta_e(scene_vec, ref_vec)

    return {
        "delta_e": round(delta_e, 2),
        "temp_delta": round(scene_data["colour_temperature_k"] - ref_data["colour_temperature_k"]),
        "exposure_delta": round(scene_data["exposure_ev"] - ref_data["exposure_ev"], 2),
        "saturation_delta": round(scene_data["saturation_pct"] - ref_data["saturation_pct"], 1),
        "scene_profile": scene_data,
        "reference_profile": ref_data,
    }

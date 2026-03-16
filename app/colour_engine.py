import cv2
import numpy as np
from PIL import Image
import io
try:
    import pillow_avif  # registers AVIF support with Pillow
except ImportError:
    pass


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes to BGR numpy array.
    Falls back to Pillow for formats OpenCV can't handle (AVIF, HEIC, WebP edge cases).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        return img_bgr

    # Fallback: Pillow handles AVIF, HEIC, TIFF, and other formats
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        raise ValueError(
            f"Could not decode image. Supported formats: JPEG, PNG, WebP, AVIF, TIFF. Error: {str(e)}"
        )


def analyse_image(image_bytes: bytes) -> dict:
    img_bgr = decode_image(image_bytes)

    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format")

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

    # Compute Delta E in CIE Lab space — perceptually accurate.
    # RGB Euclidean distance is NOT Delta E and produces misleading scores.
    # Two similar images in RGB can score 100+ because RGB is not perceptually uniform.
    # Lab space is designed so equal numerical distances = equal perceived colour differences.
    scene_bgr = decode_image(scene_bytes)
    ref_bgr = decode_image(reference_bytes)

    scene_lab = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2Lab).astype(float)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2Lab).astype(float)

    mean_scene_lab = np.mean(scene_lab.reshape(-1, 3), axis=0)
    mean_ref_lab = np.mean(ref_lab.reshape(-1, 3), axis=0)

    delta_e = float(np.sqrt(np.sum((mean_scene_lab - mean_ref_lab) ** 2)))

    return {
        "delta_e": round(delta_e, 2),
        "temp_delta": round(scene_data["colour_temperature_k"] - ref_data["colour_temperature_k"]),
        "exposure_delta": round(scene_data["exposure_ev"] - ref_data["exposure_ev"], 2),
        "saturation_delta": round(scene_data["saturation_pct"] - ref_data["saturation_pct"], 1),
        "scene_profile": scene_data,
        "reference_profile": ref_data,
    }

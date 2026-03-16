import io
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.colour_engine import analyse_image

router = APIRouter()

LUT_SIZE = 33  # 33x33x33 is industry standard for .cube files


def decode_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image. Supported formats: JPEG, PNG, WebP.")
    return bgr


def sample_colour_mapping(scene_bgr: np.ndarray, reference_bgr: np.ndarray) -> tuple:
    """
    Sample the colour relationship between scene and reference images.

    Samples pixels from both images in Lab colour space and builds
    a per-channel polynomial mapping from scene Lab to reference Lab.
    This captures the non-linear colour shift across the tonal range.

    Returns three polynomial functions: one per Lab channel (L, a, b).
    """
    # Resize both to same small size for efficient sampling
    size = (64, 64)
    scene_small = cv2.resize(scene_bgr, size)
    ref_small = cv2.resize(reference_bgr, size)

    # Convert to Lab
    scene_lab = cv2.cvtColor(scene_small, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(float)
    ref_lab = cv2.cvtColor(ref_small, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(float)

    # Fit a polynomial mapping from scene Lab values to reference Lab values
    # Degree 2 captures the non-linear relationship without overfitting
    polys = []
    for ch in range(3):
        scene_ch = scene_lab[:, ch]
        ref_ch = ref_lab[:, ch]
        # Sort by scene value for stable polynomial fit
        sort_idx = np.argsort(scene_ch)
        s_sorted = scene_ch[sort_idx]
        r_sorted = ref_ch[sort_idx]
        coeffs = np.polyfit(s_sorted, r_sorted, deg=2)
        polys.append(np.poly1d(coeffs))

    return polys


def build_lut(polys: list) -> np.ndarray:
    """
    Build a 33x33x33 LUT by applying the colour mapping to every grid point.

    Each grid point represents an RGB input value from 0 to 1.
    For each point:
      1. Convert from linear RGB to Lab
      2. Apply the polynomial mapping (scene Lab to reference Lab)
      3. Convert back to RGB
      4. Clamp to [0, 1]

    The result is a 3D array of shape (LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    where the last dimension is the corrected RGB output.
    """
    step = 1.0 / (LUT_SIZE - 1)
    lut = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE, 3), dtype=float)

    for bi in range(LUT_SIZE):
        for gi in range(LUT_SIZE):
            for ri in range(LUT_SIZE):
                # Input RGB in [0, 1]
                r = ri * step
                g = gi * step
                b = bi * step

                # Convert to uint8 BGR for OpenCV Lab conversion
                bgr_pixel = np.array([[[b * 255, g * 255, r * 255]]], dtype=np.uint8)
                lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0, 0].astype(float)

                # Apply polynomial mapping per channel
                mapped_lab = np.array([
                    polys[0](lab_pixel[0]),
                    polys[1](lab_pixel[1]),
                    polys[2](lab_pixel[2]),
                ], dtype=float)

                # Clamp Lab values to valid range
                mapped_lab[0] = np.clip(mapped_lab[0], 0, 255)
                mapped_lab[1] = np.clip(mapped_lab[1], 0, 255)
                mapped_lab[2] = np.clip(mapped_lab[2], 0, 255)

                # Convert back to BGR
                mapped_lab_pixel = np.array(
                    [[[mapped_lab[0], mapped_lab[1], mapped_lab[2]]]], dtype=np.uint8
                )
                bgr_out = cv2.cvtColor(mapped_lab_pixel, cv2.COLOR_Lab2BGR)[0, 0].astype(float)

                # Store as normalised RGB
                lut[bi, gi, ri] = np.clip(bgr_out / 255.0, 0.0, 1.0)

    return lut


def write_cube(lut: np.ndarray, title: str = "ChromaSync Scene to Reference") -> str:
    """
    Serialise the LUT to .cube file format.

    The .cube format iterates R fastest, then G, then B.
    Each line is: R_out G_out B_out
    """
    lines = []
    lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {LUT_SIZE}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    lines.append("")

    for bi in range(LUT_SIZE):
        for gi in range(LUT_SIZE):
            for ri in range(LUT_SIZE):
                r_out, g_out, b_out = lut[bi, gi, ri]
                lines.append(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}")

    return "\n".join(lines)


@router.post("/lut")
async def generate_lut(
    scene: UploadFile = File(..., description="The on-shoot scene frame with colour drift"),
    reference: UploadFile = File(..., description="The reference frame representing the target look"),
):
    """
    Generate a scene-to-reference LUT (.cube) from two frames.

    Upload the drifted scene frame and the reference frame.
    Returns a downloadable .cube LUT file that corrects the scene
    toward the reference look.

    Compatible with DaVinci Resolve and Adobe Premiere Pro.

    Apply in DaVinci Resolve: Colour tab > LUTs > Import LUT > apply to node.
    Apply in Premiere Pro: Lumetri Colour panel > Creative > Look > browse for .cube file.
    """
    scene_bytes = await scene.read()
    reference_bytes = await reference.read()

    if len(scene_bytes) > 50 * 1024 * 1024 or len(reference_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Files too large. Maximum 50MB per image.")

    try:
        scene_bgr = decode_bytes(scene_bytes)
        ref_bgr = decode_bytes(reference_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build the colour mapping from scene to reference
    polys = sample_colour_mapping(scene_bgr, ref_bgr)

    # Build the 3D LUT
    lut = build_lut(polys)

    # Get scene profile for the filename
    try:
        scene_profile = analyse_image(scene_bytes)
        scene_temp = scene_profile["colour_temperature_k"]
        ref_profile = analyse_image(reference_bytes)
        ref_temp = ref_profile["colour_temperature_k"]
        title = f"ChromaSync {scene_temp}K to {ref_temp}K"
        filename = f"chromasync_{scene_temp}K_to_{ref_temp}K.cube"
    except Exception:
        title = "ChromaSync Scene to Reference"
        filename = "chromasync_correction.cube"

    # Serialise to .cube format
    cube_content = write_cube(lut, title)

    # Return as downloadable file
    return StreamingResponse(
        io.BytesIO(cube_content.encode("utf-8")),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-LUT-Size": str(LUT_SIZE),
            "X-LUT-Title": title,
        },
    )

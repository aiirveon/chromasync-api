"""
ChromaSync Colour Correction Model Training Script
====================================================
Trains an XGBoost model to predict correction values for colour drift.

Target variable: correction values for R, G, B channels and exposure.
Features: colour profile statistics from the drifted scene and the reference frame.

Run from the chromasync-api directory:
    python train_model.py

Outputs:
    app/models/colour_correction_model.pkl   -- trained XGBoost model
    app/models/model_metadata.json           -- feature names, performance metrics
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# ── Output paths ─────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "colour_correction_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Synthetic data generation ─────────────────────────────────────────────────

RANDOM_SEED = 42
N_SAMPLES = 6000

np.random.seed(RANDOM_SEED)


def generate_reference_profiles(n: int) -> pd.DataFrame:
    """
    Generate plausible reference colour profiles for a range of shooting conditions.
    These represent what a correctly exposed, colour-calibrated frame looks like.
    """
    colour_temps = np.random.uniform(2800, 8000, n)

    # Derive mean RGB from colour temperature (approximate)
    r_base = np.clip(255 - (colour_temps - 2800) * 0.015, 180, 255)
    g_base = np.clip(200 + np.random.normal(0, 10, n), 170, 230)
    b_base = np.clip(100 + (colour_temps - 2800) * 0.020, 80, 240)

    return pd.DataFrame({
        "ref_mean_r": r_base + np.random.normal(0, 5, n),
        "ref_mean_g": g_base,
        "ref_mean_b": b_base + np.random.normal(0, 5, n),
        "ref_colour_temp_k": colour_temps,
        "ref_exposure_ev": np.random.uniform(-1.5, 1.5, n),
        "ref_saturation_pct": np.random.uniform(20, 80, n),
        "ref_contrast_ratio": np.random.uniform(0.15, 0.65, n),
    })


def apply_drift(ref: pd.DataFrame) -> tuple:
    """
    Apply simulated camera drift to the reference profiles.
    Returns drifted scene profiles and the ground truth corrections needed.

    Drift types modelled:
    - White balance shift (colour temperature change)
    - Exposure drift (ISO creep, aperture change, ND removal)
    - Saturation shift (picture profile change)
    - Channel-specific colour cast (mixed lighting, lens flare)
    """
    n = len(ref)

    # Drift magnitudes
    temp_drift = np.random.normal(0, 600, n)           # Kelvin drift
    exposure_drift = np.random.normal(0, 0.6, n)       # EV drift
    sat_drift = np.random.normal(0, 8, n)              # % saturation drift
    r_cast = np.random.normal(0, 12, n)                # RGB channel cast
    g_cast = np.random.normal(0, 8, n)
    b_cast = np.random.normal(0, 12, n)

    # Apply drift to reference
    scene = pd.DataFrame({
        "scene_mean_r": np.clip(ref["ref_mean_r"] + r_cast + temp_drift * (-0.008), 0, 255),
        "scene_mean_g": np.clip(ref["ref_mean_g"] + g_cast, 0, 255),
        "scene_mean_b": np.clip(ref["ref_mean_b"] + b_cast + temp_drift * 0.012, 0, 255),
        "scene_colour_temp_k": np.clip(ref["ref_colour_temp_k"] + temp_drift, 2000, 10000),
        "scene_exposure_ev": ref["ref_exposure_ev"] + exposure_drift,
        "scene_saturation_pct": np.clip(ref["ref_saturation_pct"] + sat_drift, 0, 100),
        "scene_contrast_ratio": np.clip(
            ref["ref_contrast_ratio"] + np.random.normal(0, 0.06, n), 0.05, 0.9
        ),
    })

    # Ground truth corrections: what needs to be applied to scene to match reference
    corrections = pd.DataFrame({
        "correct_r": ref["ref_mean_r"] - scene["scene_mean_r"],
        "correct_g": ref["ref_mean_g"] - scene["scene_mean_g"],
        "correct_b": ref["ref_mean_b"] - scene["scene_mean_b"],
        "correct_exposure_ev": ref["ref_exposure_ev"] - scene["scene_exposure_ev"],
        "correct_temp_k": ref["ref_colour_temp_k"] - scene["scene_colour_temp_k"],
        "correct_saturation": ref["ref_saturation_pct"] - scene["scene_saturation_pct"],
    })

    return scene, corrections


def add_noise(df: pd.DataFrame, noise_factor: float = 0.03) -> pd.DataFrame:
    """
    Add realistic sensor noise and measurement uncertainty.
    3% noise by default mirrors real camera sensor variation.
    """
    noisy = df.copy()
    for col in df.columns:
        std = df[col].std() * noise_factor
        noisy[col] = df[col] + np.random.normal(0, std, len(df))
    return noisy


# ── Build dataset ─────────────────────────────────────────────────────────────

print("Generating synthetic training data...")
reference_profiles = generate_reference_profiles(N_SAMPLES)
scene_profiles, corrections = apply_drift(reference_profiles)

# Features: scene statistics + reference statistics (delta context)
features = pd.concat([
    scene_profiles,
    reference_profiles,
    pd.DataFrame({
        "delta_r": scene_profiles["scene_mean_r"] - reference_profiles["ref_mean_r"],
        "delta_g": scene_profiles["scene_mean_g"] - reference_profiles["ref_mean_g"],
        "delta_b": scene_profiles["scene_mean_b"] - reference_profiles["ref_mean_b"],
        "delta_temp_k": scene_profiles["scene_colour_temp_k"] - reference_profiles["ref_colour_temp_k"],
        "delta_exposure_ev": scene_profiles["scene_exposure_ev"] - reference_profiles["ref_exposure_ev"],
        "delta_saturation": scene_profiles["scene_saturation_pct"] - reference_profiles["ref_saturation_pct"],
    })
], axis=1)

features = add_noise(features)
targets = corrections

FEATURE_NAMES = list(features.columns)
TARGET_NAMES = list(targets.columns)

print(f"Dataset: {len(features)} samples, {len(FEATURE_NAMES)} features, {len(TARGET_NAMES)} targets")

# ── Train / test split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=RANDOM_SEED
)

# ── Model training ────────────────────────────────────────────────────────────

print("Training XGBoost model...")

base_xgb = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)

model = MultiOutputRegressor(base_xgb)
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=TARGET_NAMES)

print("\nModel Performance:")
print("-" * 50)
metrics = {}
for col in TARGET_NAMES:
    mae = mean_absolute_error(y_test[col], y_pred_df[col])
    r2 = r2_score(y_test[col], y_pred_df[col])
    metrics[col] = {"mae": round(mae, 4), "r2": round(r2, 4)}
    print(f"  {col:30s}  MAE: {mae:.4f}  R²: {r2:.4f}")

overall_mae = mean_absolute_error(y_test, y_pred_df)
overall_r2 = r2_score(y_test, y_pred_df)
print(f"\n  {'Overall':30s}  MAE: {overall_mae:.4f}  R²: {overall_r2:.4f}")

# ── Save model and metadata ───────────────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

metadata = {
    "model_type": "MultiOutputRegressor(XGBRegressor)",
    "n_samples_train": len(X_train),
    "n_samples_test": len(X_test),
    "feature_names": FEATURE_NAMES,
    "target_names": TARGET_NAMES,
    "overall_mae": round(overall_mae, 4),
    "overall_r2": round(overall_r2, 4),
    "per_target_metrics": metrics,
    "noise_factor": 0.03,
    "random_seed": RANDOM_SEED,
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {METADATA_PATH}")
print("\nDone.")

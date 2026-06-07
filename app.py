 

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class StarRanges:
    temp_k: Tuple[float, float]
    lum: Tuple[float, float]
    radius: Tuple[float, float]
    abs_mag: Tuple[float, float]


STAR_CLASSES: Dict[str, StarRanges] = {
    # Intentionally broad, "toy" astrophysics-inspired ranges.
    # Units: Temperature(K), Luminosity(L/Lo), Radius(R/Ro), Absolute Magnitude(Mv)
    # Widened slightly so your UI sliders can reach each class more easily.
    # NOTE: These ranges are designed to be *distinct* in (L, R) space:
    # White dwarfs are very small and dim (tiny radius + low luminosity),
    # while giants have large radius and high luminosity.
    "Red Dwarf": StarRanges((2200, 5200), (0.00001, 0.30), (0.06, 1.00), (7.0, 18.5)),
    "Sun-like": StarRanges((4300, 7200), (0.18, 6.0), (0.70, 2.20), (2.2, 7.2)),
    "Red Giant": StarRanges((2600, 5600), (25, 15000), (5.0, 350), (-3.5, 3.5)),
    "White Dwarf": StarRanges((9000, 60000), (0.000001, 0.02), (0.004, 0.03), (9.0, 17.2)),
    "Blue Giant": StarRanges((9000, 60000), (250, 600000), (2.2, 160), (-10.5, 1.0)),
}


def _log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    if low <= 0 or high <= 0:
        return float(rng.uniform(low, high))
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_mock_star_dataset(
    n_per_class: int = 120, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates a small offline-ready dataset for training.
    Features: [Temperature, Luminosity, Radius, Absolute Magnitude]
    """
    rng = np.random.default_rng(seed)
    X_parts: List[np.ndarray] = []
    y_parts: List[str] = []

    for label, r in STAR_CLASSES.items():
        temps = rng.uniform(r.temp_k[0], r.temp_k[1], size=n_per_class)
        lums = np.array([_log_uniform(r.lum[0], r.lum[1], rng) for _ in range(n_per_class)])
        radii = np.array([_log_uniform(r.radius[0], r.radius[1], rng) for _ in range(n_per_class)])
        mags = rng.uniform(r.abs_mag[0], r.abs_mag[1], size=n_per_class)

        # Add light noise to simulate measurement variability
        # Slightly higher noise so boundaries are less "knife-edge"
        temps = np.clip(temps + rng.normal(0, 180, size=n_per_class), 1500, 60000)
        lums = np.clip(lums * np.exp(rng.normal(0, 0.26, size=n_per_class)), 1e-6, 1e6)
        radii = np.clip(radii * np.exp(rng.normal(0, 0.22, size=n_per_class)), 1e-4, 1e4)
        mags = np.clip(mags + rng.normal(0, 0.55, size=n_per_class), -12, 20)

        X_parts.append(np.column_stack([temps, lums, radii, mags]).astype(np.float64))
        y_parts.extend([label] * n_per_class)

    X = np.vstack(X_parts)
    y = np.array(y_parts, dtype=object)
    labels = sorted(STAR_CLASSES.keys())
    return X, y, labels


def build_model(seed: int = 42) -> Pipeline:
    X, y, _labels = generate_mock_star_dataset(seed=seed)

    def feature_map(arr: np.ndarray) -> np.ndarray:
        """
        Stabilize wide-range astrophysical features for a simple tree:
        - temperature: raw
        - luminosity: log10
        - radius: log10
        - abs magnitude: raw
        """
        a = np.asarray(arr, dtype=np.float64)
        out = a.copy()
        out[:, 1] = np.log10(np.clip(out[:, 1], 1e-12, None))
        out[:, 2] = np.log10(np.clip(out[:, 2], 1e-12, None))
        return out

    model = Pipeline(
        steps=[
            ("map", FunctionTransformer(feature_map, validate=False)),
            ("scaler", StandardScaler()),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=8,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


app = Flask(__name__)
MODEL = build_model()


@app.get("/")
def index():
    return render_template("index.html")


def _coerce_float(v: Any, field: str) -> float:
    try:
        return float(v)
    except Exception as exc:
        raise ValueError(f"Invalid '{field}': expected a number.") from exc


def _fits_any_known_class(temp: float, lum: float, rad: float, mag: float) -> bool:
    for ranges in STAR_CLASSES.values():
        if (
            ranges.temp_k[0] <= temp <= ranges.temp_k[1]
            and ranges.lum[0] <= lum <= ranges.lum[1]
            and ranges.radius[0] <= rad <= ranges.radius[1]
            and ranges.abs_mag[0] <= mag <= ranges.abs_mag[1]
        ):
            return True
    return False


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    print(f"Payload received: {payload}")

    try:
        temp = _coerce_float(payload.get("temperature"), "temperature")
        lum = _coerce_float(payload.get("luminosity"), "luminosity")
        rad = _coerce_float(payload.get("radius"), "radius")
        mag = _coerce_float(payload.get("abs_magnitude"), "abs_magnitude")
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # Ensure positive for log-mapped features inside the pipeline.
    lum = max(lum, 1e-12)
    rad = max(rad, 1e-12)
    X = np.array([[temp, lum, rad, mag]], dtype=np.float64)
    pred = str(MODEL.predict(X)[0])

    proba: Dict[str, float] = {}
    try:
        if hasattr(MODEL[-1], "classes_") and hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X)[0]
            classes = [str(c) for c in MODEL[-1].classes_]
            proba = {cls: float(p) for cls, p in zip(classes, probs)}
            proba = dict(sorted(proba.items(), key=lambda kv: kv[1], reverse=True))
    except Exception:
        proba = {}

    max_conf = max(proba.values()) if proba else None
    if (max_conf is not None and max_conf < 0.45) or not _fits_any_known_class(temp, lum, rad, mag):
        pred = "Unknown Planet"
        if proba:
            proba = {"Unknown Planet": 1.0, **proba}
        else:
            proba = {"Unknown Planet": 1.0}
        max_conf = 1.0

    return jsonify(
        {
            "ok": True,
            "prediction": pred,
            "confidence": max_conf,
            "probabilities": proba,
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
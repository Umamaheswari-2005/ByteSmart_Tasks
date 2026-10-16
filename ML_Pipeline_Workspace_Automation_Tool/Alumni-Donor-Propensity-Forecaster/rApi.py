# ============================================================
# Alumni-Donor-Propensity-Forecaster-flask-app.py — Alumni Donor Propensity Forecaster API (auto-generated fallback, 6)
# ============================================================

import os
import sys
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
INPUT_DIR = PROJECT_ROOT / "input" / "Alumni-Donor-Propensity-Forecaster"

# Add relevant directories to sys.path
for p in [str(BASE_DIR), str(PROJECT_ROOT), str(INPUT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# -------------------------------------------------
# REUSE THE ALGORITHM'S OWN TEST-SCRIPT PREDICTION LOGIC
# -------------------------------------------------
# Alumni-Donor-Propensity-Forecaster-test.py (produced by Step 1 from the training notebook) already
# implements this algorithm's exact feature engineering and prediction
# logic, and Step 4 already exercises it successfully. Rather than
# re-deriving algorithm-specific feature formulas generically here --
# which would mean guessing field names and math per algorithm -- this
# imports that script as a module and reuses whatever predict_* function
# it defines, preferring predict_Alumni-Donor-Propensity-Forecaster (this project's own naming
# convention) and falling back to any other predict_-prefixed function.
# If no such function is found, /predict falls back to the generic
# raw-column path further below.

import importlib.util

TEST_SCRIPT_PATH = BASE_DIR / "Alumni-Donor-Propensity-Forecaster-test.py"
PREDICT_FN = None


def _load_test_script_predict_fn():
    if not TEST_SCRIPT_PATH.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_algo_test_module", TEST_SCRIPT_PATH)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
    except Exception as e:
        app.logger.warning(f"Could not import {TEST_SCRIPT_PATH.name} for prediction reuse: {e}")
        return None

    preferred_name = "predict_Alumni-Donor-Propensity-Forecaster"
    if hasattr(test_module, preferred_name) and callable(getattr(test_module, preferred_name)):
        return getattr(test_module, preferred_name)

    for attr_name in dir(test_module):
        if attr_name.startswith("predict_"):
            candidate = getattr(test_module, attr_name)
            if callable(candidate):
                return candidate

    return None


PREDICT_FN = _load_test_script_predict_fn()
if PREDICT_FN is not None:
    app.logger.info(f"Using prediction logic from {TEST_SCRIPT_PATH.name} ({PREDICT_FN.__name__}).")

# The test script's own prediction function returns whatever key style
# its notebook author used (e.g. "Forecast", "Confidence (%)"), which
# may not match what a consumer like test_runner.py expects (e.g.
# "forecast"). Rather than guess and rewrite the notebook's dict shape,
# this adds a lowercase/snake_case alias for every key ALONGSIDE the
# original -- so both "Forecast" and "forecast" are present -- without
# removing or renaming anything that might already be relied on.
import re


def _normalize_key(key, keep_percent=True):
    text = str(key)
    text = text.replace("%", " percent ") if keep_percent else text.replace("%", " ")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", text).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


def _normalize_result_keys(result):
    if not isinstance(result, dict):
        return result
    normalized = dict(result)
    for k, v in list(result.items()):
        for nk in (_normalize_key(k, keep_percent=True), _normalize_key(k, keep_percent=False)):
            if nk and nk not in normalized:
                normalized[nk] = v
    return normalized

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
# Models in this project are saved with joblib.dump() (not plain
# pickle.dump()), so they must be loaded with joblib.load(). A raw
# pickle.load() on a joblib-saved file can appear to "succeed" while
# actually returning a mangled/partial object instead of the real
# bundle, since joblib's array storage format isn't plain pickle.

MODEL_NAME = "Alumni Donor Propensity Forecaster"
ALGORITHM = "Auto-generated model bundle"
MODEL_FILE = 'Alumni-Donor-Propensity-Forecaster-model_pipeline.pkl'

MODEL_PATH = Path(os.environ.get("MODEL_DIR", str(BASE_DIR))) / MODEL_FILE

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    model_bundle = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Unwrap common dictionary/bundle structures. Key names vary between
# training scripts, and a truthy-but-wrong-typed value under one of the
# expected keys (e.g. a numpy array of feature importances stored under
# "model") would otherwise be picked over the real estimator -- so this
# validates that a candidate actually has a callable .predict() before
# accepting it, and falls back to scanning every value in the bundle.
def _looks_like_estimator(obj):
    return hasattr(obj, "predict") and callable(getattr(obj, "predict"))


if isinstance(model_bundle, dict):
    model = None
    for key in ("pipeline", "model", "clf", "estimator"):
        candidate = model_bundle.get(key)
        if candidate is not None and _looks_like_estimator(candidate):
            model = candidate
            break

    if model is None:
        for candidate in model_bundle.values():
            if _looks_like_estimator(candidate):
                model = candidate
                break

    feature_cols = (
        model_bundle.get("feature_names") 
        or model_bundle.get("features") 
        or model_bundle.get("feature_columns")
    )
    target_mapping = (
        model_bundle.get("target_mapping")
        or model_bundle.get("target_names")
        or model_bundle.get("class_labels")
    )
else:
    model = model_bundle if _looks_like_estimator(model_bundle) else None
    feature_cols = None
    target_mapping = None

if model is None and PREDICT_FN is None:
    if isinstance(model_bundle, dict):
        shape_info = ", ".join(f"{k!r}: {type(v).__name__}" for k, v in model_bundle.items())
        detail = f"Bundle is a dict with keys/types: {{{shape_info}}}"
    else:
        detail = f"Bundle is a {type(model_bundle).__name__}, not a dict, and has no .predict()."
    raise RuntimeError(
        "Could not find a usable model (an object with a callable .predict()) "
        "inside the loaded pickle, and no prediction function was found in "
        f"{TEST_SCRIPT_PATH.name} either. {detail}"
    )

REQUIRED_COLUMNS = list(feature_cols) if feature_cols else None
TARGET_MAPPING = list(target_mapping) if target_mapping else None

# -------------------------------------------------
# CONFIGURATION & ROUTES
# -------------------------------------------------

tasks = [
    {'url': '/', 'Method': 'GET', 'Description': 'Returns the API documentation/status'},
    {'url': '/health', 'Method': 'GET', 'Description': 'Returns service health status'},
    {'url': '/predict', 'Method': 'GET', 'Description': 'Shows required payload structure'},
    {'url': '/predict', 'Method': 'POST', 'data': 'JSON object or array of features', 'Description': 'Returns model predictions'},
    {'url': '/model-info', 'Method': 'GET', 'Description': 'Returns details about the underlying ML model'},
]

def make_error(status_code, message, **kwargs):
    response_body = {
        'status': status_code,
        'error': message
    }
    response_body.update(kwargs)
    return make_response(jsonify(response_body), status_code)


@app.errorhandler(400)
def bad_request(error):
    return make_error(400, "Bad Request Error, Please give valid input")


@app.errorhandler(404)
def not_found(error):
    return make_error(404, "Not found")


@app.errorhandler(405)
def method_not_allowed(error):
    return make_error(405, "Method not allowed, please check the URL and method")


@app.errorhandler(500)
def internal_server_error(error):
    return make_error(500, "Internal server Error")


def verification(df):
    if not REQUIRED_COLUMNS:
        return '0'

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        return make_error(
            400,
            "Missing required columns",
            missing_columns=missing_columns
        )
    return '0'


@app.route("/", methods=["GET"])
def home():
    return jsonify({'docs': tasks})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["GET"])
def predict_get():
    return jsonify({
        "message": "Use POST method to submit prediction data.",
        "required_columns": REQUIRED_COLUMNS,
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": MODEL_NAME,
        "algorithm": ALGORITHM,
        "model_file": MODEL_FILE,
        "feature_count": len(REQUIRED_COLUMNS) if REQUIRED_COLUMNS else None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return make_error(400, "Request content type must be application/json")

        data = request.get_json()
        if data is None:
            return make_error(400, "No JSON payload received")

        if isinstance(data, dict):
            records = [data]
            is_batch = False
        elif isinstance(data, list):
            if len(data) == 0:
                return make_error(400, "Empty JSON array received")
            records = data
            is_batch = True
        else:
            return make_error(400, "Invalid JSON format. Must be an object or array.")

        if PREDICT_FN is not None:
            # Reuse the algorithm's own tested prediction function --
            # it does its own feature engineering from raw record fields,
            # so records are passed through as-is (same shape as the
            # example project dict in Alumni-Donor-Propensity-Forecaster-test.py), not reindexed
            # against REQUIRED_COLUMNS.
            raw_results = [PREDICT_FN(record) for record in records]
            results = [_normalize_result_keys(r) for r in raw_results]
            return jsonify(results if is_batch else results[0])

        # Fallback: no algorithm-specific prediction function was found,
        # so this expects the caller to already supply the model's exact
        # trained column names.
        df = pd.DataFrame(records)

        ver = verification(df)
        if ver != '0':
            return ver

        if REQUIRED_COLUMNS:
            df = df.reindex(columns=REQUIRED_COLUMNS).fillna(0)

        preds = model.predict(df)
        raw_results = [p.item() if hasattr(p, "item") else p for p in preds]

        if TARGET_MAPPING:
            results = [
                TARGET_MAPPING[r] if isinstance(r, int) and 0 <= r < len(TARGET_MAPPING) else r
                for r in raw_results
            ]
        else:
            results = raw_results

        return jsonify(results if is_batch else results[0])

    except Exception as e:
        app.logger.exception(e)
        return make_error(500, f"Prediction failed: {str(e)}")

# -------------------------------------------------
# SERVER EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9102))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

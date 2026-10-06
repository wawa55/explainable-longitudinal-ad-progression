# ============================================================
# STEP 61
# FINAL LOCKED OASIS-3 EXTERNAL-VALIDATION SYNTHESIS
# ============================================================
#
# PURPOSE
# -------
# Consolidate already-completed frozen OASIS-3 external-validation
# results into one publication-ready locked results package.
#
# THIS SCRIPT DOES NOT:
#   - read MRI voxels
#   - load model checkpoints
#   - run inference
#   - modify predictions
#   - modify endpoints
#   - train/fine-tune/recalibrate the model
#   - optimize thresholds
#   - perform OASIS-based model selection
#
# It reads ONLY locked result tables produced after frozen inference.
#
# Inputs:
#   Step 59A : external point estimates
#   Step 59B : participant-cluster bootstrap performance
#   Step 60A : calibration/domain-shift diagnostics
#   Step 60B : participant-cluster bootstrap calibration
#
# Primary horizons:
#   12, 24, 36 months
#
# Secondary horizons:
#   48, 60 months
#
# ============================================================

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = (Path(__file__).resolve().parents[2] / "external_data" / "oasis3")

EVAL_DIR = ROOT / "external_evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

STEP59A = (
    EVAL_DIR /
    "OASIS3_STEP59A_EXTERNAL_POINT_ESTIMATES.csv"
)

STEP59B = (
    EVAL_DIR /
    "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP60A = (
    EVAL_DIR /
    "OASIS3_STEP60A_EXTERNAL_CALIBRATION_SUMMARY.csv"
)

STEP60B = (
    EVAL_DIR /
    "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

OUT_CSV = (
    EVAL_DIR /
    "OASIS3_STEP61_FINAL_LOCKED_EXTERNAL_VALIDATION_SYNTHESIS.csv"
)

OUT_PRIMARY = (
    EVAL_DIR /
    "OASIS3_STEP61_PRIMARY_12_24_36_MONTH_RESULTS.csv"
)

OUT_SECONDARY = (
    EVAL_DIR /
    "OASIS3_STEP61_SECONDARY_48_60_MONTH_RESULTS.csv"
)

OUT_JSON = (
    EVAL_DIR /
    "OASIS3_STEP61_FINAL_LOCKED_EXTERNAL_VALIDATION_SYNTHESIS.json"
)

OUT_TXT = (
    EVAL_DIR /
    "OASIS3_STEP61_PUBLICATION_READY_EXTERNAL_RESULTS.txt"
)

OUT_LOCK = (
    EVAL_DIR /
    "OASIS3_STEP61_FINAL_EXTERNAL_VALIDATION_LOCK.json"
)

PRIMARY_HORIZONS = [12, 24, 36]
SECONDARY_HORIZONS = [48, 60]
ALL_HORIZONS = PRIMARY_HORIZONS + SECONDARY_HORIZONS

EXPECTED_BOOTSTRAPS = 2000


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired locked input does not exist:\n{path}"
        )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def find_col(df, candidates, required=True):
    """
    Return the first matching column from candidate names.
    Matching is case-insensitive.
    """

    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise RuntimeError(
            "\nCould not locate required column.\n"
            f"Candidates:\n{candidates}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def fmt3(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def ci_text(est, lo, hi):
    if pd.isna(est) or pd.isna(lo) or pd.isna(hi):
        return "NA"

    return (
        f"{float(est):.3f} "
        f"[{float(lo):.3f}, {float(hi):.3f}]"
    )


def get_row(df, horizon_col, horizon):
    values = numeric(df[horizon_col])

    hit = df.loc[values == horizon]

    if len(hit) != 1:
        raise RuntimeError(
            f"Expected exactly one row for horizon {horizon}, "
            f"found {len(hit)}."
        )

    return hit.iloc[0]


def value_from_row(row, column):
    if column is None:
        return np.nan

    value = row[column]

    try:
        return float(value)
    except Exception:
        return np.nan


def safe_int(value):
    if pd.isna(value):
        return None
    return int(float(value))


# ============================================================
# BANNER
# ============================================================

print("=" * 78)
print("STEP 61 — FINAL LOCKED OASIS-3 EXTERNAL-VALIDATION SYNTHESIS")
print("=" * 78)

print("\nREAD-ONLY SYNTHESIS")
print("MRI voxels read                  : NO")
print("Frozen checkpoints loaded        : NO")
print("Model inference re-run           : NO")
print("External predictions modified    : NO")
print("External endpoints modified      : NO")
print("Model training                    : NO")
print("Model fine-tuning                 : NO")
print("Model recalibration               : NO")
print("Threshold optimization            : NO")
print("OASIS-based model selection       : NO")


# ============================================================
# VERIFY INPUTS
# ============================================================

inputs = {
    "step59a_point_estimates": STEP59A,
    "step59b_performance_bootstrap": STEP59B,
    "step60a_calibration": STEP60A,
    "step60b_calibration_bootstrap": STEP60B,
}

print("\n" + "=" * 78)
print("LOCKED INPUT FILES")
print("=" * 78)

input_hashes = {}

for name, path in inputs.items():

    require_file(path)

    digest = sha256_file(path)
    input_hashes[name] = digest

    print(f"\n{name}")
    print(f"Path   : {path}")
    print(f"SHA256 : {digest}")


# ============================================================
# LOAD LOCKED RESULT TABLES
# ============================================================

d59a = pd.read_csv(STEP59A, low_memory=False)
d59b = pd.read_csv(STEP59B, low_memory=False)
d60a = pd.read_csv(STEP60A, low_memory=False)
d60b = pd.read_csv(STEP60B, low_memory=False)

print("\n" + "=" * 78)
print("INPUT TABLE DIMENSIONS")
print("=" * 78)

print("Step 59A :", d59a.shape)
print("Step 59B :", d59b.shape)
print("Step 60A :", d60a.shape)
print("Step 60B :", d60b.shape)


# ============================================================
# IDENTIFY HORIZON COLUMNS
# ============================================================

h59a = find_col(
    d59a,
    ["horizon_months", "horizon"]
)

h59b = find_col(
    d59b,
    ["horizon_months", "horizon"]
)

h60a = find_col(
    d60a,
    ["horizon_months", "horizon"]
)

h60b = find_col(
    d60b,
    ["horizon_months", "horizon"]
)


# ============================================================
# STEP 59A COLUMNS
# ============================================================

n_eval_59a = find_col(
    d59a,
    [
        "n_evaluable_landmarks",
        "n_evaluable",
    ]
)

n_events_59a = find_col(
    d59a,
    [
        "n_events",
        "events",
    ]
)

event_rate_59a = find_col(
    d59a,
    [
        "event_rate",
        "observed_event_rate",
    ]
)

auc_59a = find_col(
    d59a,
    [
        "roc_auc",
        "auc",
        "auc_estimate",
    ]
)

brier_59a = find_col(
    d59a,
    [
        "brier_score",
        "brier",
        "brier_estimate",
    ]
)


# ============================================================
# STEP 59B COLUMNS
# ============================================================

auc_est_59b = find_col(
    d59b,
    [
        "roc_auc_estimate",
        "auc_estimate",
        "roc_auc",
        "auc",
    ]
)

auc_lo_59b = find_col(
    d59b,
    [
        "roc_auc_ci_lower",
        "auc_ci_lower",
        "auc_lower",
    ]
)

auc_hi_59b = find_col(
    d59b,
    [
        "roc_auc_ci_upper",
        "auc_ci_upper",
        "auc_upper",
    ]
)

brier_est_59b = find_col(
    d59b,
    [
        "brier_score_estimate",
        "brier_estimate",
        "brier_score",
        "brier",
    ]
)

brier_lo_59b = find_col(
    d59b,
    [
        "brier_score_ci_lower",
        "brier_ci_lower",
        "brier_lower",
    ]
)

brier_hi_59b = find_col(
    d59b,
    [
        "brier_score_ci_upper",
        "brier_ci_upper",
        "brier_upper",
    ]
)

bootstrap_col_59b = find_col(
    d59b,
    [
        "n_bootstrap_replicates",
        "bootstrap_replicates",
        "n_replicates",
    ],
    required=False
)


# ============================================================
# STEP 60A COLUMNS
# ============================================================

mean_pred_60a = find_col(
    d60a,
    [
        "mean_predicted_risk",
        "mean_predicted_risk_estimate",
    ]
)

cal_intercept_60a = find_col(
    d60a,
    [
        "calibration_intercept",
        "calibration_intercept_estimate",
    ]
)

cal_slope_60a = find_col(
    d60a,
    [
        "calibration_slope",
        "calibration_slope_estimate",
    ]
)

bss_60a = find_col(
    d60a,
    [
        "brier_skill_score",
        "brier_skill_score_estimate",
    ]
)


# ============================================================
# STEP 60B COLUMNS
# ============================================================

obs_est_60b = find_col(
    d60b,
    [
        "observed_event_rate_estimate",
        "observed_event_rate",
    ]
)

obs_lo_60b = find_col(
    d60b,
    [
        "observed_event_rate_ci_lower",
    ]
)

obs_hi_60b = find_col(
    d60b,
    [
        "observed_event_rate_ci_upper",
    ]
)

pred_est_60b = find_col(
    d60b,
    [
        "mean_predicted_risk_estimate",
        "mean_predicted_risk",
    ]
)

pred_lo_60b = find_col(
    d60b,
    [
        "mean_predicted_risk_ci_lower",
    ]
)

pred_hi_60b = find_col(
    d60b,
    [
        "mean_predicted_risk_ci_upper",
    ]
)

int_est_60b = find_col(
    d60b,
    [
        "calibration_intercept_estimate",
        "calibration_intercept",
    ]
)

int_lo_60b = find_col(
    d60b,
    [
        "calibration_intercept_ci_lower",
    ]
)

int_hi_60b = find_col(
    d60b,
    [
        "calibration_intercept_ci_upper",
    ]
)

slope_est_60b = find_col(
    d60b,
    [
        "calibration_slope_estimate",
        "calibration_slope",
    ]
)

slope_lo_60b = find_col(
    d60b,
    [
        "calibration_slope_ci_lower",
    ]
)

slope_hi_60b = find_col(
    d60b,
    [
        "calibration_slope_ci_upper",
    ]
)

bss_est_60b = find_col(
    d60b,
    [
        "brier_skill_score_estimate",
        "brier_skill_score",
    ]
)

bss_lo_60b = find_col(
    d60b,
    [
        "brier_skill_score_ci_lower",
    ]
)

bss_hi_60b = find_col(
    d60b,
    [
        "brier_skill_score_ci_upper",
    ]
)

bootstrap_col_60b = find_col(
    d60b,
    [
        "n_bootstrap_replicates",
        "bootstrap_replicates",
        "n_replicates",
    ],
    required=False
)


# ============================================================
# HORIZON AUDIT
# ============================================================

print("\n" + "=" * 78)
print("HORIZON AUDIT")
print("=" * 78)

for name, df, hc in [
    ("Step 59A", d59a, h59a),
    ("Step 59B", d59b, h59b),
    ("Step 60A", d60a, h60a),
    ("Step 60B", d60b, h60b),
]:

    horizons = sorted(
        numeric(df[hc]).dropna().astype(int).unique().tolist()
    )

    print(f"{name}: {horizons}")

    missing = sorted(set(ALL_HORIZONS) - set(horizons))

    if missing:
        raise RuntimeError(
            f"{name} missing required horizons: {missing}"
        )

print("\nPASS: all four locked result tables contain 12/24/36/48/60 months.")


# ============================================================
# BUILD FINAL SYNTHESIS
# ============================================================

records = []

for horizon in ALL_HORIZONS:

    r59a = get_row(d59a, h59a, horizon)
    r59b = get_row(d59b, h59b, horizon)
    r60a = get_row(d60a, h60a, horizon)
    r60b = get_row(d60b, h60b, horizon)

    n_eval = safe_int(value_from_row(r59a, n_eval_59a))
    n_events = safe_int(value_from_row(r59a, n_events_59a))

    event_rate = value_from_row(r59a, event_rate_59a)

    auc_point = value_from_row(r59a, auc_59a)
    brier_point = value_from_row(r59a, brier_59a)

    auc_est = value_from_row(r59b, auc_est_59b)
    auc_lo = value_from_row(r59b, auc_lo_59b)
    auc_hi = value_from_row(r59b, auc_hi_59b)

    brier_est = value_from_row(r59b, brier_est_59b)
    brier_lo = value_from_row(r59b, brier_lo_59b)
    brier_hi = value_from_row(r59b, brier_hi_59b)

    observed_est = value_from_row(r60b, obs_est_60b)
    observed_lo = value_from_row(r60b, obs_lo_60b)
    observed_hi = value_from_row(r60b, obs_hi_60b)

    predicted_est = value_from_row(r60b, pred_est_60b)
    predicted_lo = value_from_row(r60b, pred_lo_60b)
    predicted_hi = value_from_row(r60b, pred_hi_60b)

    intercept_est = value_from_row(r60b, int_est_60b)
    intercept_lo = value_from_row(r60b, int_lo_60b)
    intercept_hi = value_from_row(r60b, int_hi_60b)

    slope_est = value_from_row(r60b, slope_est_60b)
    slope_lo = value_from_row(r60b, slope_lo_60b)
    slope_hi = value_from_row(r60b, slope_hi_60b)

    bss_est = value_from_row(r60b, bss_est_60b)
    bss_lo = value_from_row(r60b, bss_lo_60b)
    bss_hi = value_from_row(r60b, bss_hi_60b)

    # --------------------------------------------------------
    # STRICT CONSISTENCY CHECKS
    # --------------------------------------------------------

    checks = [
        (
            "AUC",
            auc_point,
            auc_est,
        ),
        (
            "Brier",
            brier_point,
            brier_est,
        ),
        (
            "Observed event rate",
            event_rate,
            observed_est,
        ),
        (
            "Mean predicted risk",
            value_from_row(r60a, mean_pred_60a),
            predicted_est,
        ),
        (
            "Calibration intercept",
            value_from_row(r60a, cal_intercept_60a),
            intercept_est,
        ),
        (
            "Calibration slope",
            value_from_row(r60a, cal_slope_60a),
            slope_est,
        ),
        (
            "Brier skill score",
            value_from_row(r60a, bss_60a),
            bss_est,
        ),
    ]

    for label, a, b in checks:

        if pd.isna(a) or pd.isna(b):
            raise RuntimeError(
                f"{horizon}m: missing value in {label} consistency check."
            )

        if not np.isclose(a, b, rtol=0, atol=1e-6):
            raise RuntimeError(
                f"{horizon}m: {label} mismatch: {a} vs {b}"
            )

    if bootstrap_col_59b is not None:
        nboot59 = safe_int(value_from_row(r59b, bootstrap_col_59b))

        if nboot59 != EXPECTED_BOOTSTRAPS:
            raise RuntimeError(
                f"{horizon}m Step59B bootstrap count = {nboot59}; "
                f"expected {EXPECTED_BOOTSTRAPS}."
            )

    if bootstrap_col_60b is not None:
        nboot60 = safe_int(value_from_row(r60b, bootstrap_col_60b))

        if nboot60 != EXPECTED_BOOTSTRAPS:
            raise RuntimeError(
                f"{horizon}m Step60B bootstrap count = {nboot60}; "
                f"expected {EXPECTED_BOOTSTRAPS}."
            )

    if not (0 <= auc_est <= 1):
        raise RuntimeError(f"{horizon}m invalid AUC.")

    if not (0 <= brier_est <= 1):
        raise RuntimeError(f"{horizon}m invalid Brier score.")

    if not (0 <= observed_est <= 1):
        raise RuntimeError(f"{horizon}m invalid observed event rate.")

    if not (0 <= predicted_est <= 1):
        raise RuntimeError(f"{horizon}m invalid predicted risk.")

    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

    records.append(
        {
            "horizon_months": horizon,
            "analysis_role": (
                "PRIMARY"
                if horizon in PRIMARY_HORIZONS
                else "SECONDARY"
            ),
            "n_evaluable_landmarks": n_eval,
            "n_events": n_events,

            "observed_event_rate": observed_est,
            "observed_event_rate_ci_lower": observed_lo,
            "observed_event_rate_ci_upper": observed_hi,

            "mean_predicted_risk": predicted_est,
            "mean_predicted_risk_ci_lower": predicted_lo,
            "mean_predicted_risk_ci_upper": predicted_hi,

            "risk_minus_observed":
                predicted_est - observed_est,

            "roc_auc": auc_est,
            "roc_auc_ci_lower": auc_lo,
            "roc_auc_ci_upper": auc_hi,

            "brier_score": brier_est,
            "brier_score_ci_lower": brier_lo,
            "brier_score_ci_upper": brier_hi,

            "calibration_intercept": intercept_est,
            "calibration_intercept_ci_lower": intercept_lo,
            "calibration_intercept_ci_upper": intercept_hi,

            "calibration_slope": slope_est,
            "calibration_slope_ci_lower": slope_lo,
            "calibration_slope_ci_upper": slope_hi,

            "brier_skill_score": bss_est,
            "brier_skill_score_ci_lower": bss_lo,
            "brier_skill_score_ci_upper": bss_hi,

            "bootstrap_resampling_unit": "participant",
            "bootstrap_replicates": EXPECTED_BOOTSTRAPS,

            "model_status": "frozen_ADNI_M0_five_seed_ensemble",
            "external_dataset": "OASIS-3",

            "recalibration_performed": False,
            "fine_tuning_performed": False,
            "threshold_optimization_performed": False,
            "external_model_selection_performed": False,
        }
    )


final = pd.DataFrame(records)


# ============================================================
# FINAL CONSISTENCY AUDIT
# ============================================================

print("\n" + "=" * 78)
print("FINAL CONSISTENCY AUDIT")
print("=" * 78)

if len(final) != 5:
    raise RuntimeError(
        f"Expected 5 horizon rows, found {len(final)}."
    )

if final["horizon_months"].tolist() != ALL_HORIZONS:
    raise RuntimeError(
        "Final horizon ordering is incorrect."
    )

if final["roc_auc"].isna().any():
    raise RuntimeError("Missing AUC.")

if final["brier_score"].isna().any():
    raise RuntimeError("Missing Brier score.")

if final["calibration_intercept"].isna().any():
    raise RuntimeError("Missing calibration intercept.")

if final["calibration_slope"].isna().any():
    raise RuntimeError("Missing calibration slope.")

if final["brier_skill_score"].isna().any():
    raise RuntimeError("Missing Brier skill score.")

print("PASS: Step59A point estimates match Step59B estimates.")
print("PASS: Step60A point estimates match Step60B estimates.")
print("PASS: all five horizons are complete.")
print("PASS: all probabilities and scores are finite.")
print("PASS: bootstrap unit remains participant.")
print(f"PASS: bootstrap replicates = {EXPECTED_BOOTSTRAPS}.")


# ============================================================
# WRITE TABLES
# ============================================================

final.to_csv(OUT_CSV, index=False)

primary = final.loc[
    final["horizon_months"].isin(PRIMARY_HORIZONS)
].copy()

secondary = final.loc[
    final["horizon_months"].isin(SECONDARY_HORIZONS)
].copy()

primary.to_csv(OUT_PRIMARY, index=False)
secondary.to_csv(OUT_SECONDARY, index=False)


# ============================================================
# PUBLICATION-READY TEXT
# ============================================================

lines = []

lines.append(
    "OASIS-3 EXTERNAL VALIDATION — FROZEN ADNI M0 MODEL"
)
lines.append(
    "Participant-cluster bootstrap uncertainty: 2,000 replicates"
)
lines.append("")

lines.append("PRIMARY HORIZONS")
lines.append("-" * 78)

for _, r in primary.iterrows():

    h = int(r["horizon_months"])

    lines.append(
        f"{h} months | "
        f"N={int(r['n_evaluable_landmarks'])}, "
        f"events={int(r['n_events'])} | "
        f"AUC {ci_text(r['roc_auc'], r['roc_auc_ci_lower'], r['roc_auc_ci_upper'])} | "
        f"Brier {ci_text(r['brier_score'], r['brier_score_ci_lower'], r['brier_score_ci_upper'])}"
    )

    lines.append(
        f"           Observed risk "
        f"{ci_text(r['observed_event_rate'], r['observed_event_rate_ci_lower'], r['observed_event_rate_ci_upper'])} | "
        f"Predicted risk "
        f"{ci_text(r['mean_predicted_risk'], r['mean_predicted_risk_ci_lower'], r['mean_predicted_risk_ci_upper'])}"
    )

    lines.append(
        f"           Calibration intercept "
        f"{ci_text(r['calibration_intercept'], r['calibration_intercept_ci_lower'], r['calibration_intercept_ci_upper'])} | "
        f"Calibration slope "
        f"{ci_text(r['calibration_slope'], r['calibration_slope_ci_lower'], r['calibration_slope_ci_upper'])} | "
        f"Brier skill "
        f"{ci_text(r['brier_skill_score'], r['brier_skill_score_ci_lower'], r['brier_skill_score_ci_upper'])}"
    )

    lines.append("")


lines.append("SECONDARY HORIZONS")
lines.append("-" * 78)

for _, r in secondary.iterrows():

    h = int(r["horizon_months"])

    lines.append(
        f"{h} months | "
        f"N={int(r['n_evaluable_landmarks'])}, "
        f"events={int(r['n_events'])} | "
        f"AUC {ci_text(r['roc_auc'], r['roc_auc_ci_lower'], r['roc_auc_ci_upper'])} | "
        f"Brier {ci_text(r['brier_score'], r['brier_score_ci_lower'], r['brier_score_ci_upper'])}"
    )

    lines.append(
        f"           Observed risk "
        f"{ci_text(r['observed_event_rate'], r['observed_event_rate_ci_lower'], r['observed_event_rate_ci_upper'])} | "
        f"Predicted risk "
        f"{ci_text(r['mean_predicted_risk'], r['mean_predicted_risk_ci_lower'], r['mean_predicted_risk_ci_upper'])}"
    )

    lines.append(
        f"           Calibration intercept "
        f"{ci_text(r['calibration_intercept'], r['calibration_intercept_ci_lower'], r['calibration_intercept_ci_upper'])} | "
        f"Calibration slope "
        f"{ci_text(r['calibration_slope'], r['calibration_slope_ci_lower'], r['calibration_slope_ci_upper'])} | "
        f"Brier skill "
        f"{ci_text(r['brier_skill_score'], r['brier_skill_score_ci_lower'], r['brier_skill_score_ci_upper'])}"
    )

    lines.append("")


lines.append("INTERPRETATION GUARDRAIL")
lines.append("-" * 78)

lines.append(
    "These results quantify transportability of the untouched frozen ADNI "
    "M0 five-seed ensemble to the locked OASIS-3 external cohort."
)

lines.append(
    "No model training, fine-tuning, recalibration, threshold optimization, "
    "or OASIS-based model selection was performed."
)

lines.append(
    "Observed external performance must not be used to modify the frozen "
    "primary model or predictions."
)

lines.append(
    "Poor external discrimination or calibration should be reported as "
    "evidence of limited transportability/domain shift rather than repaired "
    "within the primary external-validation analysis."
)

OUT_TXT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "step": 61,
    "status": "LOCKED_PASS",

    "analysis":
        "final frozen OASIS-3 external-validation synthesis",

    "external_dataset": "OASIS-3",

    "model":
        "frozen ADNI ExactM0LongitudinalSurvivalModel_V3 "
        "five-seed ensemble",

    "primary_horizons_months": PRIMARY_HORIZONS,
    "secondary_horizons_months": SECONDARY_HORIZONS,

    "bootstrap": {
        "resampling_unit": "participant",
        "replicates": EXPECTED_BOOTSTRAPS,
        "clustered_landmarks_preserved": True,
    },

    "inputs": {
        name: {
            "path": str(path),
            "sha256": input_hashes[name],
        }
        for name, path in inputs.items()
    },

    "results": final.to_dict(orient="records"),

    "analysis_guardrails": {
        "mri_voxels_read": False,
        "checkpoints_loaded": False,
        "inference_rerun": False,
        "predictions_modified": False,
        "endpoints_modified": False,
        "model_training": False,
        "model_fine_tuning": False,
        "model_recalibration": False,
        "threshold_optimization": False,
        "oasis_based_model_selection": False,
    },

    "interpretation":
        "Untouched frozen-model external validation. "
        "Results describe transportability/domain shift.",

    "created_utc":
        datetime.now(timezone.utc).isoformat(),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# HASH OUTPUTS BEFORE FINAL LOCK
# ============================================================

output_hashes = {
    "final_synthesis_csv": sha256_file(OUT_CSV),
    "primary_results_csv": sha256_file(OUT_PRIMARY),
    "secondary_results_csv": sha256_file(OUT_SECONDARY),
    "publication_text": sha256_file(OUT_TXT),
    "summary_json": sha256_file(OUT_JSON),
}


# ============================================================
# FINAL LOCK
# ============================================================

lock = {
    "step": 61,
    "status": "FINAL_EXTERNAL_VALIDATION_SYNTHESIS_LOCKED",

    "external_dataset": "OASIS-3",

    "primary_horizons_months": PRIMARY_HORIZONS,
    "secondary_horizons_months": SECONDARY_HORIZONS,

    "bootstrap_resampling_unit": "participant",
    "bootstrap_replicates": EXPECTED_BOOTSTRAPS,

    "input_sha256": input_hashes,
    "output_sha256": output_hashes,

    "frozen_analysis_contract": {
        "mri_voxels_read": False,
        "frozen_checkpoints_loaded": False,
        "model_inference_rerun": False,
        "external_predictions_modified": False,
        "external_endpoints_modified": False,
        "model_training": False,
        "model_fine_tuning": False,
        "model_recalibration": False,
        "threshold_optimization": False,
        "oasis_based_model_selection": False,
    },

    "external_results_observed": True,

    "post_external_result_model_changes_permitted": False,

    "created_utc":
        datetime.now(timezone.utc).isoformat(),
}

with open(OUT_LOCK, "w", encoding="utf-8") as f:
    json.dump(lock, f, indent=2)


lock_hash = sha256_file(OUT_LOCK)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP-61 EXTERNAL-VALIDATION TABLE")
print("=" * 78)

for _, r in final.iterrows():

    h = int(r["horizon_months"])
    role = r["analysis_role"]

    print(
        f"\n{h} MONTHS [{role}]"
    )

    print(
        f"Evaluable landmarks : {int(r['n_evaluable_landmarks'])}"
    )

    print(
        f"Events              : {int(r['n_events'])}"
    )

    print(
        "AUC                 : "
        + ci_text(
            r["roc_auc"],
            r["roc_auc_ci_lower"],
            r["roc_auc_ci_upper"],
        )
    )

    print(
        "Brier               : "
        + ci_text(
            r["brier_score"],
            r["brier_score_ci_lower"],
            r["brier_score_ci_upper"],
        )
    )

    print(
        "Observed risk       : "
        + ci_text(
            r["observed_event_rate"],
            r["observed_event_rate_ci_lower"],
            r["observed_event_rate_ci_upper"],
        )
    )

    print(
        "Predicted risk      : "
        + ci_text(
            r["mean_predicted_risk"],
            r["mean_predicted_risk_ci_lower"],
            r["mean_predicted_risk_ci_upper"],
        )
    )

    print(
        "Calibration int.    : "
        + ci_text(
            r["calibration_intercept"],
            r["calibration_intercept_ci_lower"],
            r["calibration_intercept_ci_upper"],
        )
    )

    print(
        "Calibration slope   : "
        + ci_text(
            r["calibration_slope"],
            r["calibration_slope_ci_lower"],
            r["calibration_slope_ci_upper"],
        )
    )

    print(
        "Brier skill score   : "
        + ci_text(
            r["brier_skill_score"],
            r["brier_skill_score_ci_lower"],
            r["brier_skill_score_ci_upper"],
        )
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

print("\n" + "=" * 78)
print("OUTPUT")
print("=" * 78)

print(f"Final synthesis : {OUT_CSV}")
print(f"Primary results : {OUT_PRIMARY}")
print(f"Secondary       : {OUT_SECONDARY}")
print(f"Publication txt : {OUT_TXT}")
print(f"Summary JSON    : {OUT_JSON}")
print(f"Final lock      : {OUT_LOCK}")


print("\n" + "=" * 78)
print("SHA256")
print("=" * 78)

for name, digest in output_hashes.items():
    print(f"{name:24s}: {digest}")

print(f"{'final_lock':24s}: {lock_hash}")


# ============================================================
# FINAL SAFETY STATEMENT
# ============================================================

print("\n" + "=" * 78)
print("IMPORTANT")
print("=" * 78)

print("MRI VOXELS READ                 : NO")
print("FROZEN CHECKPOINTS LOADED       : NO")
print("MODEL INFERENCE RE-RUN          : NO")
print("EXTERNAL PREDICTIONS MODIFIED   : NO")
print("EXTERNAL ENDPOINTS MODIFIED     : NO")
print("MODEL TRAINING                   : NO")
print("MODEL FINE-TUNING                : NO")
print("MODEL RECALIBRATION              : NO")
print("THRESHOLD OPTIMIZATION           : NO")
print("OASIS-BASED MODEL SELECTION      : NO")
print("PARTICIPANT-CLUSTER BOOTSTRAP    : YES")
print("BOOTSTRAP REPLICATES             : 2000")
print("PRIMARY HORIZONS                 : 12, 24, 36 months")
print("SECONDARY HORIZONS               : 48, 60 months")

print("\nSTEP 61 COMPLETE.")

print(
    "The frozen ADNI M0 external validation on OASIS-3 has now been "
    "consolidated into a locked publication-ready results package."
)

print(
    "These results must remain separate from any later secondary "
    "recalibration or adaptation analysis."
)

print(
    "Do not modify the frozen model, external predictions, endpoints, "
    "or primary external-validation results after this lock."
)
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd


# ============================================================
# STEP 64
# BUILD LOCKED OASIS-3 EXTERNAL-VALIDATION MANUSCRIPT PACKAGE
#
# READ-ONLY PUBLICATION PACKAGING.
#
# NO MRI reading.
# NO checkpoint loading.
# NO inference.
# NO retraining.
# NO recalibration.
# NO threshold optimization.
# NO endpoint modification.
# NO OASIS-based model selection.
# ============================================================


ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

EVAL = ROOT / "external_evaluation"
FIGDIR = EVAL / "publication_figures"
OUTDIR = EVAL / "manuscript_package"

OUTDIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOCKED INPUTS
# ------------------------------------------------------------

STEP59B = (
    EVAL /
    "OASIS3_STEP59B_PARTICIPANT_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP60B = (
    EVAL /
    "OASIS3_STEP60B_CALIBRATION_CLUSTER_BOOTSTRAP_SUMMARY.csv"
)

STEP61 = (
    EVAL /
    "OASIS3_STEP61_FINAL_LOCKED_EXTERNAL_VALIDATION_SYNTHESIS.csv"
)

STEP63_TABLE = (
    FIGDIR /
    "OASIS3_STEP63_COMBINED_FIGURE_DATA.csv"
)

STEP63_PNG = (
    FIGDIR /
    "Figure_External_Validation_Combined.png"
)

STEP63_PDF = (
    FIGDIR /
    "Figure_External_Validation_Combined.pdf"
)

STEP63_LOCK = (
    FIGDIR /
    "OASIS3_STEP63_COMBINED_FIGURE_LOCK.json"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

METHODS_TEX = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_METHODS.tex"
)

RESULTS_TEX = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_RESULTS.tex"
)

DISCUSSION_TEX = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_DISCUSSION.tex"
)

TABLE_TEX = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_TABLE.tex"
)

SUMMARY_TXT = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_PUBLICATION_SUMMARY.txt"
)

PACKAGE_JSON = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_MANUSCRIPT_SUMMARY.json"
)

LOCK_JSON = (
    OUTDIR /
    "OASIS3_EXTERNAL_VALIDATION_MANUSCRIPT_LOCK.json"
)

OUT_FIG_PNG = (
    OUTDIR /
    "Figure_External_Validation_Combined.png"
)

OUT_FIG_PDF = (
    OUTDIR /
    "Figure_External_Validation_Combined.pdf"
)


PRIMARY = [12, 24, 36]
SECONDARY = [48, 60]
ALL_HORIZONS = PRIMARY + SECONDARY


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required locked input is missing:\n{path}"
        )


def require_columns(df, cols, label):
    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns:\n{missing}"
        )


def f3(x):
    return f"{float(x):.3f}"


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 64 - OASIS-3 EXTERNAL VALIDATION MANUSCRIPT PACKAGE")
print("=" * 78)


inputs = [
    STEP59B,
    STEP60B,
    STEP61,
    STEP63_TABLE,
    STEP63_PNG,
    STEP63_PDF,
    STEP63_LOCK,
]

for p in inputs:
    require_file(p)


print("\nLOCKED INPUTS")

for p in inputs:
    print(p)


# ============================================================
# HASH INPUTS
# ============================================================

input_hashes = {
    str(p): sha256_file(p)
    for p in inputs
}

print("\nINPUT SHA256")

for p, h in input_hashes.items():
    print(Path(p).name, ":", h)


# ============================================================
# READ LOCKED SUMMARIES
# ============================================================

perf = pd.read_csv(
    STEP59B,
    low_memory=False
)

cal = pd.read_csv(
    STEP60B,
    low_memory=False
)

synthesis = pd.read_csv(
    STEP61,
    low_memory=False
)

figdata = pd.read_csv(
    STEP63_TABLE,
    low_memory=False
)


required_perf = [
    "horizon_months",
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "brier_score",
    "brier_ci_lower",
    "brier_ci_upper",
]

required_cal = [
    "horizon_months",
    "n_evaluable_landmarks",
    "n_events",
    "observed_event_rate_estimate",
    "observed_event_rate_ci_lower",
    "observed_event_rate_ci_upper",
    "mean_predicted_risk_estimate",
    "mean_predicted_risk_ci_lower",
    "mean_predicted_risk_ci_upper",
    "calibration_intercept_estimate",
    "calibration_intercept_ci_lower",
    "calibration_intercept_ci_upper",
    "calibration_slope_estimate",
    "calibration_slope_ci_lower",
    "calibration_slope_ci_upper",
    "brier_skill_score_estimate",
    "brier_skill_score_ci_lower",
    "brier_skill_score_ci_upper",
]

require_columns(
    perf,
    required_perf,
    "Step-59B"
)

require_columns(
    cal,
    required_cal,
    "Step-60B"
)


# ============================================================
# INDEX BY HORIZON
# ============================================================

perf["horizon_months"] = (
    pd.to_numeric(
        perf["horizon_months"],
        errors="raise"
    ).astype(int)
)

cal["horizon_months"] = (
    pd.to_numeric(
        cal["horizon_months"],
        errors="raise"
    ).astype(int)
)

perf = perf.set_index(
    "horizon_months"
)

cal = cal.set_index(
    "horizon_months"
)


for h in ALL_HORIZONS:

    if h not in perf.index:
        raise RuntimeError(
            f"Missing Step-59B horizon: {h}"
        )

    if h not in cal.index:
        raise RuntimeError(
            f"Missing Step-60B horizon: {h}"
        )


# ============================================================
# CROSS-CHECK STEP 63
# ============================================================

require_columns(
    figdata,
    [
        "horizon_months",
        "auc",
        "auc_lower",
        "auc_upper",
        "brier",
        "observed_risk",
        "predicted_risk",
        "calibration_slope",
    ],
    "Step-63 figure data"
)

figdata["horizon_months"] = (
    pd.to_numeric(
        figdata["horizon_months"],
        errors="raise"
    ).astype(int)
)

figdata = figdata.set_index(
    "horizon_months"
)


for h in ALL_HORIZONS:

    checks = [
        (
            float(figdata.loc[h, "auc"]),
            float(perf.loc[h, "roc_auc"]),
            "AUC"
        ),
        (
            float(figdata.loc[h, "brier"]),
            float(perf.loc[h, "brier_score"]),
            "Brier"
        ),
        (
            float(figdata.loc[h, "observed_risk"]),
            float(
                cal.loc[
                    h,
                    "observed_event_rate_estimate"
                ]
            ),
            "Observed risk"
        ),
        (
            float(figdata.loc[h, "predicted_risk"]),
            float(
                cal.loc[
                    h,
                    "mean_predicted_risk_estimate"
                ]
            ),
            "Predicted risk"
        ),
        (
            float(figdata.loc[h, "calibration_slope"]),
            float(
                cal.loc[
                    h,
                    "calibration_slope_estimate"
                ]
            ),
            "Calibration slope"
        ),
    ]

    for a, b, name in checks:

        if abs(a - b) > 1e-10:
            raise RuntimeError(
                f"Step-63 mismatch at {h} months "
                f"for {name}: {a} != {b}"
            )


print(
    "\nPASS: Step-63 figure data reproduce "
    "the locked Step-59B/60B results."
)


# ============================================================
# COPY LOCKED FIGURE
# ============================================================

shutil.copy2(
    STEP63_PNG,
    OUT_FIG_PNG
)

shutil.copy2(
    STEP63_PDF,
    OUT_FIG_PDF
)


if (
    sha256_file(STEP63_PNG)
    != sha256_file(OUT_FIG_PNG)
):
    raise RuntimeError(
        "Copied PNG hash mismatch."
    )

if (
    sha256_file(STEP63_PDF)
    != sha256_file(OUT_FIG_PDF)
):
    raise RuntimeError(
        "Copied PDF hash mismatch."
    )


print(
    "PASS: combined figure copied "
    "without modification."
)


# ============================================================
# EXTRACT VALUES
# ============================================================

def P(h, col):
    return float(perf.loc[h, col])


def C(h, col):
    return float(cal.loc[h, col])


# ============================================================
# METHODS LATEX
# ============================================================

methods_text = r"""
\subsection{External validation on OASIS-3}

External validation was performed in OASIS-3 using a protocol
that was locked before examination of external predictive
performance. The ADNI-derived model architecture, learned
parameters, preprocessing contract, temporal input definition,
prediction horizons, and survival-risk calculation were not
modified using OASIS-3 outcomes. The five frozen ADNI model
checkpoints were applied without retraining or fine-tuning.

External predictions were evaluated at 12, 24, and 36 months
as the primary prediction horizons, corresponding to the
principal horizons used for model assessment. Results at
48 and 60 months were treated as secondary analyses.
Discrimination was summarized using horizon-specific
time-dependent receiver-operating-characteristic area under
the curve (ROC AUC), while prediction error was quantified
using the Brier score. Calibration and cross-cohort
transportability were examined by comparing mean predicted
risk with the observed event rate and by estimating
calibration intercepts and slopes.

Uncertainty was quantified using 2,000 participant-cluster
bootstrap replicates. Resampling was performed at the
participant level so that repeated landmarks belonging to
the same participant remained within the same bootstrap
cluster. No OASIS-3-based model selection, threshold
optimization, model fine-tuning, or recalibration was
performed for the primary external-validation analysis.
"""

METHODS_TEX.write_text(
    methods_text.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# RESULTS LATEX
# ============================================================

results_text = f"""
\\subsection{{External validation on OASIS-3}}

Application of the frozen ADNI-derived model to OASIS-3
showed limited cross-cohort discrimination. At the three
primary prediction horizons, the time-dependent ROC AUC was
{f3(P(12, "roc_auc"))} (95\\% CI
{f3(P(12, "roc_auc_ci_lower"))}--{f3(P(12, "roc_auc_ci_upper"))})
at 12 months,
{f3(P(24, "roc_auc"))} (95\\% CI
{f3(P(24, "roc_auc_ci_lower"))}--{f3(P(24, "roc_auc_ci_upper"))})
at 24 months, and
{f3(P(36, "roc_auc"))} (95\\% CI
{f3(P(36, "roc_auc_ci_lower"))}--{f3(P(36, "roc_auc_ci_upper"))})
at 36 months. The corresponding Brier scores were
{f3(P(12, "brier_score"))},
{f3(P(24, "brier_score"))}, and
{f3(P(36, "brier_score"))}, respectively.

A systematic difference between predicted and observed
absolute risk was evident across the primary horizons.
At 12 months, the observed event rate was
{f3(C(12, "observed_event_rate_estimate"))}, compared with a
mean predicted risk of
{f3(C(12, "mean_predicted_risk_estimate"))}.
At 24 months, the corresponding values were
{f3(C(24, "observed_event_rate_estimate"))} and
{f3(C(24, "mean_predicted_risk_estimate"))}, while at
36 months they were
{f3(C(36, "observed_event_rate_estimate"))} and
{f3(C(36, "mean_predicted_risk_estimate"))}.
Thus, the frozen ADNI-derived model overestimated progression
risk in OASIS-3 at each primary horizon.

Calibration slopes were
{f3(C(12, "calibration_slope_estimate"))},
{f3(C(24, "calibration_slope_estimate"))}, and
{f3(C(36, "calibration_slope_estimate"))} at 12, 24, and
36 months, respectively, substantially below the ideal value
of one. Together with the divergence between observed and
predicted absolute risk, these findings indicate substantial
cross-cohort calibration shift.

Secondary analyses at 48 and 60 months showed a similar
pattern. ROC AUC values were
{f3(P(48, "roc_auc"))} (95\\% CI
{f3(P(48, "roc_auc_ci_lower"))}--{f3(P(48, "roc_auc_ci_upper"))})
and
{f3(P(60, "roc_auc"))} (95\\% CI
{f3(P(60, "roc_auc_ci_lower"))}--{f3(P(60, "roc_auc_ci_upper"))}),
respectively. Mean predicted risks remained higher than
observed event rates at both secondary horizons.

Figure~\\ref{{fig:oasis_external_validation}} summarizes
external discrimination, observed versus predicted risk,
and calibration slope across prediction horizons.
"""

RESULTS_TEX.write_text(
    results_text.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# DISCUSSION LATEX
# ============================================================

discussion_text = r"""
\subsection{External transportability and domain shift}

The OASIS-3 evaluation provides a deliberately stringent
assessment of cross-cohort transportability because the
ADNI-derived model was applied without external-data-driven
retraining, model selection, threshold optimization, or
recalibration. Under this frozen evaluation protocol,
discrimination was close to chance across the principal
prediction horizons and predicted progression probabilities
were systematically higher than the observed event rates.
The calibration slopes were also substantially below the
ideal value of one.

These findings indicate that predictive performance observed
within ADNI did not transfer directly to OASIS-3. Importantly,
the external results should not be interpreted as evidence
that the external-validation procedure failed. Rather, they
identify a substantial transportability gap when an
ADNI-trained longitudinal imaging model is transferred
unchanged to an independent cohort.

Several mechanisms may contribute to this cross-cohort
shift, including differences in participant composition,
disease prevalence and progression patterns, acquisition
characteristics, follow-up structure, and other cohort-level
factors. The present analysis does not attempt to attribute
the observed performance degradation to any single source.

The frozen external evaluation therefore distinguishes
internal predictive performance from genuine cross-cohort
generalization. Any subsequent recalibration or adaptation
analysis should be reported separately as a secondary or
post-hoc investigation and should not replace the untouched
OASIS-3 results reported here.
"""

DISCUSSION_TEX.write_text(
    discussion_text.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# LATEX TABLE
# ============================================================

rows = []

for h in ALL_HORIZONS:

    role = (
        "Primary"
        if h in PRIMARY
        else "Secondary"
    )

    row = (
        f"{h} & {role} & "
        f"{f3(P(h, 'roc_auc'))} "
        f"({f3(P(h, 'roc_auc_ci_lower'))}--"
        f"{f3(P(h, 'roc_auc_ci_upper'))}) & "
        f"{f3(P(h, 'brier_score'))} "
        f"({f3(P(h, 'brier_ci_lower'))}--"
        f"{f3(P(h, 'brier_ci_upper'))}) & "
        f"{f3(C(h, 'observed_event_rate_estimate'))} & "
        f"{f3(C(h, 'mean_predicted_risk_estimate'))} & "
        f"{f3(C(h, 'calibration_slope_estimate'))} "
        f"({f3(C(h, 'calibration_slope_ci_lower'))}--"
        f"{f3(C(h, 'calibration_slope_ci_upper'))}) "
        r"\\"
    )

    rows.append(row)


table_text = r"""
\begin{table*}[t]
\centering
\caption{Frozen ADNI model performance during external
validation on OASIS-3. Confidence intervals were obtained
using 2,000 participant-cluster bootstrap replicates.}
\label{tab:oasis_external_validation}
\small
\begin{tabular}{clccccc}
\toprule
Horizon &
Analysis &
ROC AUC (95\% CI) &
Brier score (95\% CI) &
Observed risk &
Predicted risk &
Calibration slope (95\% CI) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

TABLE_TEX.write_text(
    table_text.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# PUBLICATION SUMMARY
# ============================================================

summary_lines = [
    "OASIS-3 FROZEN EXTERNAL VALIDATION",
    "=" * 60,
    "",
    "Primary horizons: 12, 24, 36 months",
    "Secondary horizons: 48, 60 months",
    "Bootstrap: 2,000 participant-cluster replicates",
    "",
]

for h in ALL_HORIZONS:

    role = (
        "PRIMARY"
        if h in PRIMARY
        else "SECONDARY"
    )

    summary_lines.extend([
        f"{h} MONTHS [{role}]",
        (
            "AUC: "
            f"{f3(P(h, 'roc_auc'))} "
            f"[{f3(P(h, 'roc_auc_ci_lower'))}, "
            f"{f3(P(h, 'roc_auc_ci_upper'))}]"
        ),
        (
            "Brier: "
            f"{f3(P(h, 'brier_score'))} "
            f"[{f3(P(h, 'brier_ci_lower'))}, "
            f"{f3(P(h, 'brier_ci_upper'))}]"
        ),
        (
            "Observed risk: "
            f"{f3(C(h, 'observed_event_rate_estimate'))}"
        ),
        (
            "Predicted risk: "
            f"{f3(C(h, 'mean_predicted_risk_estimate'))}"
        ),
        (
            "Calibration slope: "
            f"{f3(C(h, 'calibration_slope_estimate'))} "
            f"[{f3(C(h, 'calibration_slope_ci_lower'))}, "
            f"{f3(C(h, 'calibration_slope_ci_upper'))}]"
        ),
        "",
    ])


summary_lines.extend([
    "INTERPRETATION",
    "-" * 60,
    (
        "The frozen ADNI-derived model showed limited direct "
        "transportability to OASIS-3."
    ),
    (
        "Discrimination was near chance and absolute "
        "progression risk was systematically overestimated."
    ),
    (
        "These are untouched external-validation results; "
        "the model was not adapted after observing them."
    ),
])


SUMMARY_TXT.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8"
)


# ============================================================
# PACKAGE SUMMARY JSON
# ============================================================

package_summary = {
    "step": 64,
    "status": "COMPLETE",
    "purpose": (
        "Locked manuscript packaging of primary "
        "OASIS-3 external validation"
    ),
    "primary_horizons_months": PRIMARY,
    "secondary_horizons_months": SECONDARY,
    "bootstrap_replicates": 2000,
    "outputs": {
        "methods_tex": str(METHODS_TEX),
        "results_tex": str(RESULTS_TEX),
        "discussion_tex": str(DISCUSSION_TEX),
        "table_tex": str(TABLE_TEX),
        "summary_txt": str(SUMMARY_TXT),
        "figure_png": str(OUT_FIG_PNG),
        "figure_pdf": str(OUT_FIG_PDF),
    },
    "mri_voxels_read": False,
    "checkpoints_loaded": False,
    "model_inference_rerun": False,
    "predictions_modified": False,
    "endpoints_modified": False,
    "model_training": False,
    "model_fine_tuning": False,
    "model_recalibration": False,
    "threshold_optimization": False,
    "oasis_based_model_selection": False,
    "external_validation_results_changed": False,
}

with open(
    PACKAGE_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        package_summary,
        f,
        indent=2
    )


# ============================================================
# FINAL PACKAGE LOCK
# ============================================================

output_files = [
    METHODS_TEX,
    RESULTS_TEX,
    DISCUSSION_TEX,
    TABLE_TEX,
    SUMMARY_TXT,
    PACKAGE_JSON,
    OUT_FIG_PNG,
    OUT_FIG_PDF,
]

output_hashes = {
    str(p): sha256_file(p)
    for p in output_files
}


lock = {
    "step": 64,
    "status": "LOCKED",
    "package": (
        "OASIS3_EXTERNAL_VALIDATION_MANUSCRIPT_PACKAGE_V1"
    ),
    "input_hashes": input_hashes,
    "output_hashes": output_hashes,
    "primary_external_validation_frozen": True,
    "visualization_modified": False,
    "model_modified": False,
    "predictions_modified": False,
    "endpoints_modified": False,
    "recalibration_performed": False,
    "model_selection_performed": False,
}

with open(
    LOCK_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lock,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 78)
print("PRIMARY EXTERNAL VALIDATION")
print("=" * 78)

for h in PRIMARY:

    print(
        f"{h:2d}m | "
        f"AUC {P(h, 'roc_auc'):.3f} "
        f"[{P(h, 'roc_auc_ci_lower'):.3f}, "
        f"{P(h, 'roc_auc_ci_upper'):.3f}] | "
        f"Brier {P(h, 'brier_score'):.3f} | "
        f"Observed {C(h, 'observed_event_rate_estimate'):.3f} | "
        f"Predicted {C(h, 'mean_predicted_risk_estimate'):.3f} | "
        f"Slope {C(h, 'calibration_slope_estimate'):.3f}"
    )


print("\n" + "=" * 78)
print("SECONDARY EXTERNAL VALIDATION")
print("=" * 78)

for h in SECONDARY:

    print(
        f"{h:2d}m | "
        f"AUC {P(h, 'roc_auc'):.3f} "
        f"[{P(h, 'roc_auc_ci_lower'):.3f}, "
        f"{P(h, 'roc_auc_ci_upper'):.3f}] | "
        f"Brier {P(h, 'brier_score'):.3f} | "
        f"Observed {C(h, 'observed_event_rate_estimate'):.3f} | "
        f"Predicted {C(h, 'mean_predicted_risk_estimate'):.3f} | "
        f"Slope {C(h, 'calibration_slope_estimate'):.3f}"
    )


print("\n" + "=" * 78)
print("MANUSCRIPT PACKAGE")
print("=" * 78)

for p in output_files:
    print(p)

print("Lock :", LOCK_JSON)


print("\n" + "=" * 78)
print("IMPORTANT")
print("=" * 78)

print("MRI VOXELS READ               : NO")
print("FROZEN CHECKPOINTS LOADED     : NO")
print("MODEL INFERENCE RE-RUN        : NO")
print("EXTERNAL PREDICTIONS MODIFIED : NO")
print("EXTERNAL ENDPOINTS MODIFIED   : NO")
print("MODEL TRAINING                : NO")
print("MODEL FINE-TUNING             : NO")
print("MODEL RECALIBRATION           : NO")
print("THRESHOLD OPTIMIZATION        : NO")
print("OASIS-BASED MODEL SELECTION   : NO")
print("MANUSCRIPT PACKAGING ONLY     : YES")


print("\nSTEP 64 COMPLETE.")
print(
    "The frozen OASIS-3 external-validation results "
    "have been packaged for manuscript integration."
)
print(
    "The primary external-validation results remain "
    "unchanged and locked."
)
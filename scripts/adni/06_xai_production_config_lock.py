from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

try:
    import captum
except Exception:
    captum = None


# ==============================================================================================================
# ADNI XAI PRODUCTION CONFIGURATION LOCK
#
# PURPOSE
# -------
# Freeze the XAI method BEFORE any TEST attribution is generated.
#
# This script:
#   1. Verifies the validation-only evidence files.
#   2. Verifies the exact proposed XAI settings.
#   3. Verifies that 16-step blurred-latest IG passed multi-subject validation.
#   4. Records model checkpoints and hashes.
#   5. Writes JSON/TXT lock files + artifact hash table.
#
# IMPORTANT
# ---------
# This script does NOT load TEST data.
# This script does NOT generate any attribution maps.
# This script does NOT retrain or recalibrate the model.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

VALIDATION_BASELINE_COMPARISON = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BASELINE_COMPARISON_V1.csv"
)

VALIDATION_MULTISUBJECT_RESULTS = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BLURRED_BASELINE_MULTISUBJECT_V1.csv"
)

VALIDATION_MULTISUBJECT_JSON = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_BLURRED_BASELINE_MULTISUBJECT_V1.json"
)

VALIDATION_MULTISUBJECT_SELECTION = (
    RESULT_DIR / "ADNI_XAI_VALIDATION_MULTISUBJECT_SELECTION_V1.csv"
)

ZERO_BASELINE_DIAGNOSTIC = (
    RESULT_DIR / "ADNI_XAI_ZERO_BASELINE_PATH_DIAGNOSTIC_V1.json"
)

CHECKPOINTS = {
    2026: MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",
    3407: MODEL_DIR / "ADNI_M0_SEED_3407_BEST_V1.pt",
    5813: MODEL_DIR / "ADNI_M0_SEED_5813_BEST_V1.pt",
    7291: MODEL_DIR / "ADNI_M0_SEED_7291_BEST_V1.pt",
    9109: MODEL_DIR / "ADNI_M0_SEED_9109_BEST_V1.pt",
}

OUTPUT_JSON = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
)

OUTPUT_TXT = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.txt"
)

OUTPUT_HASHES = (
    RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_HASHES_V1.csv"
)


# ==============================================================================================================
# 2. EXPECTED FROZEN MODEL / VALIDATION VALUES
# ==============================================================================================================

EXPECTED_MODEL_PARAMETERS = 4_269_109

EXPECTED_SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

EXPECTED_BEST_EPOCHS = {
    2026: 12,
    3407: 7,
    5813: 11,
    7291: 5,
    9109: 9,
}

EXPECTED_VALIDATION_NLL = {
    2026: 0.9486606425046921,
    3407: 0.9454395517706871,
    5813: 0.9594686004519463,
    7291: 0.9487041115760804,
    9109: 0.9548525424301624,
}

EXPECTED_VALIDATION_PARTICIPANTS = 12
EXPECTED_EVENT_PARTICIPANTS = 6
EXPECTED_CENSORED_PARTICIPANTS = 6

# Validation evidence from the completed multi-subject confirmation.
MAX_ACCEPTABLE_ABS_COMPLETENESS_ERROR = 1e-5
MAX_ACCEPTABLE_16_TO_32_SIGNED_SUM_CHANGE = 1e-5


# ==============================================================================================================
# 3. FROZEN XAI SPECIFICATION
# ==============================================================================================================

XAI_SPEC = {
    "lock_version":
        "ADNI_XAI_PRODUCTION_LOCK_V1_2026-09-11",

    "analysis_status":
        "FROZEN BEFORE TEST XAI",

    "primary_model":
        "M0 — Longitudinal MRI-only CNN–Transformer baseline, V1",

    "xai_method":
        "Integrated Gradients",

    "explained_input":
        "latest MRI only",

    "conditioning_context":
        "all prior MRI embeddings fixed; actual longitudinal timing fixed",

    "spatial_grid":
        [160, 192, 160],

    "baseline":
        "blurred latest MRI",

    "baseline_filter":
        {
            "type": "3D local mean",
            "kernel_size": [9, 9, 9],
            "padding": "replicate",
        },

    "rejected_baseline":
        {
            "name": "all-zero normalized MRI",
            "reason":
                "validation-only zero-to-actual path showed severe near-zero gradient/path pathology and failed IG completeness convergence",
        },

    "targets":
        {
            "24_months":
                "cumulative progression risk through first 2 discrete hazard intervals",

            "36_months":
                "cumulative progression risk through first 3 discrete hazard intervals",
        },

    "horizons_months":
        [24, 36],

    "integration_method":
        "gausslegendre",

    "n_steps":
        16,

    "internal_batch_size":
        1,

    "attribution_sign":
        "signed attribution retained",

    "stability_map_basis":
        "absolute attribution magnitude",

    "top_k_percentages":
        [1, 5, 10],

    "cross_seed_models":
        EXPECTED_SEEDS,

    "cross_seed_primary_stability":
        [
            "voxelwise Spearman correlation",
            "top-k Dice overlap",
        ],

    "faithfulness":
        {
            "method":
                "attribution-guided perturbation",

            "selected_voxels":
                "high positive-risk attribution voxels for deletion analysis",

            "comparison":
                "equal-size random perturbation within valid brain/non-background voxels",

            "replacement_value":
                "normalized background reference consistent with frozen preprocessing",
        },

    "visit_level_analysis":
        "per-visit attribution magnitude / contribution comparison",

    "interpretation_rule":
        "attribution indicates model sensitivity/association, not biological causality",

    "attention_usage":
        "attention weights are not treated as primary anatomical XAI",

    "test_adaptation":
        "PROHIBITED",

    "post_test_baseline_selection":
        "PROHIBITED",

    "post_test_step_selection":
        "PROHIBITED",

    "retraining":
        "PROHIBITED",

    "recalibration":
        "NOT APPLIED",
}


# ==============================================================================================================
# 4. HELPERS
# ==============================================================================================================

def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def require_file(
    path,
    label,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    print(
        f"{label:45s}: PASS | {path}"
    )


def safe_float(
    value,
):

    if value is None:
        return None

    return float(
        value
    )


# ==============================================================================================================
# 5. MAIN
# ==============================================================================================================

def main():

    print("=" * 120)
    print("ADNI XAI PRODUCTION CONFIGURATION LOCK")
    print("=" * 120)

    print(
        "Purpose: freeze XAI settings before any TEST attribution generation"
    )

    print(
        "TEST data accessed:",
        "NO"
    )

    print()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5A. REQUIRED VALIDATION ARTIFACTS
    # ----------------------------------------------------------------------------------------------------------

    print("VALIDATION EVIDENCE FILE AUDIT")
    print("-" * 120)

    required_files = {
        "Baseline comparison CSV":
            VALIDATION_BASELINE_COMPARISON,

        "Multi-subject results CSV":
            VALIDATION_MULTISUBJECT_RESULTS,

        "Multi-subject JSON":
            VALIDATION_MULTISUBJECT_JSON,

        "Multi-subject selection CSV":
            VALIDATION_MULTISUBJECT_SELECTION,

        "Zero-baseline path diagnostic":
            ZERO_BASELINE_DIAGNOSTIC,
    }

    for label, path in required_files.items():

        require_file(
            path,
            label,
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5B. ZERO-BASELINE REJECTION AUDIT
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("ZERO-BASELINE REJECTION AUDIT")
    print("-" * 120)

    with open(
        ZERO_BASELINE_DIAGNOSTIC,
        "r",
        encoding="utf-8",
    ) as f:

        zero_diag = json.load(
            f
        )

    pathology_flag = bool(
        zero_diag.get(
            "zero_baseline_pathology_flag",
            False,
        )
    )

    print(
        "Zero-baseline pathology flag:",
        pathology_flag
    )

    if pathology_flag is not True:

        raise RuntimeError(
            "Zero-baseline diagnostic did not confirm pathology. "
            "Do not freeze revised baseline."
        )

    print(
        "Zero baseline rejection:",
        "PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5C. MULTI-SUBJECT VALIDATION AUDIT
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("MULTI-SUBJECT VALIDATION AUDIT")
    print("-" * 120)

    with open(
        VALIDATION_MULTISUBJECT_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        multi_json = json.load(
            f
        )

    selected_participants = int(
        multi_json[
            "selected_participants"
        ]
    )

    event_participants = int(
        multi_json[
            "event_participants"
        ]
    )

    censored_participants = int(
        multi_json[
            "censored_participants"
        ]
    )

    step16_all_finite = bool(
        multi_json[
            "step16_all_finite"
        ]
    )

    step16_median_abs_error = float(
        multi_json[
            "step16_median_absolute_completeness_error"
        ]
    )

    step16_max_abs_error = float(
        multi_json[
            "step16_max_absolute_completeness_error"
        ]
    )

    median_change_16_to_32 = float(
        multi_json[
            "median_abs_signed_sum_change_16_to_32"
        ]
    )

    max_change_16_to_32 = float(
        multi_json[
            "max_abs_signed_sum_change_16_to_32"
        ]
    )

    print(
        "Selected validation participants:",
        selected_participants
    )

    print(
        "Event participants:",
        event_participants
    )

    print(
        "Censored participants:",
        censored_participants
    )

    print(
        "16-step all finite:",
        step16_all_finite
    )

    print(
        "16-step median absolute completeness error:",
        f"{step16_median_abs_error:.12e}"
    )

    print(
        "16-step maximum absolute completeness error:",
        f"{step16_max_abs_error:.12e}"
    )

    print(
        "Median |signed sum change| 16->32:",
        f"{median_change_16_to_32:.12e}"
    )

    print(
        "Maximum |signed sum change| 16->32:",
        f"{max_change_16_to_32:.12e}"
    )

    if (
        selected_participants
        !=
        EXPECTED_VALIDATION_PARTICIPANTS
    ):
        raise RuntimeError(
            "Unexpected validation participant count."
        )

    if (
        event_participants
        !=
        EXPECTED_EVENT_PARTICIPANTS
    ):
        raise RuntimeError(
            "Unexpected event participant count."
        )

    if (
        censored_participants
        !=
        EXPECTED_CENSORED_PARTICIPANTS
    ):
        raise RuntimeError(
            "Unexpected censored participant count."
        )

    if not step16_all_finite:

        raise RuntimeError(
            "Not all 16-step IG attributions were finite."
        )

    if (
        step16_max_abs_error
        >
        MAX_ACCEPTABLE_ABS_COMPLETENESS_ERROR
    ):
        raise RuntimeError(
            "16-step maximum completeness error exceeds threshold."
        )

    if (
        max_change_16_to_32
        >
        MAX_ACCEPTABLE_16_TO_32_SIGNED_SUM_CHANGE
    ):
        raise RuntimeError(
            "16->32 step stability exceeds threshold."
        )

    print(
        "Multi-subject validation evidence:",
        "PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5D. DIRECT CSV CROSS-CHECK
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("DIRECT CSV CROSS-CHECK")
    print("-" * 120)

    multi_df = pd.read_csv(
        VALIDATION_MULTISUBJECT_RESULTS
    )

    required_columns = [
        "subject_id",
        "event",
        "horizon_months",
        "n_steps",
        "relative_completeness_error",
        "absolute_completeness_error",
        "captum_convergence_delta",
        "finite_fraction",
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in multi_df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"Missing columns in validation results: "
            f"{missing_columns}"
        )

    horizons_observed = sorted(
        multi_df[
            "horizon_months"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    steps_observed = sorted(
        multi_df[
            "n_steps"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    subjects_observed = int(
        multi_df[
            "subject_id"
        ].nunique()
    )

    print(
        "Horizons observed:",
        horizons_observed
    )

    print(
        "Steps observed:",
        steps_observed
    )

    print(
        "Unique validation subjects:",
        subjects_observed
    )

    if horizons_observed != [24, 36]:

        raise RuntimeError(
            "Validation horizons do not match lock."
        )

    if steps_observed != [16, 32]:

        raise RuntimeError(
            "Validation step counts do not match expected confirmation."
        )

    if subjects_observed != 12:

        raise RuntimeError(
            "Validation subject count mismatch."
        )

    step16_df = multi_df.loc[
        multi_df[
            "n_steps"
        ].astype(
            int
        )
        ==
        16
    ].copy()

    csv_step16_max_abs_error = float(
        step16_df[
            "absolute_completeness_error"
        ].max()
    )

    csv_all_finite = bool(
        np.all(
            np.isclose(
                step16_df[
                    "finite_fraction"
                ].to_numpy(
                    dtype=float
                ),
                1.0,
            )
        )
    )

    print(
        "CSV 16-step max absolute completeness error:",
        f"{csv_step16_max_abs_error:.12e}"
    )

    print(
        "CSV 16-step all finite:",
        csv_all_finite
    )

    if not csv_all_finite:

        raise RuntimeError(
            "CSV cross-check detected non-finite IG results."
        )

    if (
        csv_step16_max_abs_error
        >
        MAX_ACCEPTABLE_ABS_COMPLETENESS_ERROR
    ):
        raise RuntimeError(
            "CSV cross-check completeness error exceeds threshold."
        )

    print(
        "CSV cross-check:",
        "PASS"
    )

    # ----------------------------------------------------------------------------------------------------------
    # 5E. FIVE FROZEN CHECKPOINTS
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("FROZEN CHECKPOINT AUDIT")
    print("-" * 120)

    checkpoint_records = []

    for seed in EXPECTED_SEEDS:

        checkpoint_path = CHECKPOINTS[
            seed
        ]

        require_file(
            checkpoint_path,
            f"Seed {seed} checkpoint",
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        epoch = checkpoint.get(
            "epoch"
        )

        val_nll = float(
            checkpoint.get(
                "validation_loss"
            )
        )

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:

            raise RuntimeError(
                f"Seed {seed}: model_state_dict missing."
            )

        if (
            epoch
            !=
            EXPECTED_BEST_EPOCHS[
                seed
            ]
        ):
            raise RuntimeError(
                f"Seed {seed}: epoch mismatch."
            )

        if (
            abs(
                val_nll
                -
                EXPECTED_VALIDATION_NLL[
                    seed
                ]
            )
            >
            1e-10
        ):
            raise RuntimeError(
                f"Seed {seed}: validation NLL mismatch."
            )

        checkpoint_sha256 = sha256_file(
            checkpoint_path
        )

        checkpoint_records.append(
            {
                "seed":
                    seed,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "epoch":
                    int(
                        epoch
                    ),

                "validation_nll":
                    val_nll,

                "sha256":
                    checkpoint_sha256,
            }
        )

        print(
            f"Seed {seed} | "
            f"epoch={epoch} | "
            f"val NLL={val_nll:.12f} | "
            f"SHA256={checkpoint_sha256[:16]}... | PASS"
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5F. BUILD LOCK
    # ----------------------------------------------------------------------------------------------------------

    print()
    print("WRITING XAI PRODUCTION LOCK")
    print("-" * 120)

    lock = {
        **XAI_SPEC,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "environment":
            {
                "python":
                    sys.version.splitlines()[0],

                "platform":
                    platform.platform(),

                "torch":
                    torch.__version__,

                "torch_cuda_build":
                    torch.version.cuda,

                "cuda_available":
                    bool(
                        torch.cuda.is_available()
                    ),

                "numpy":
                    np.__version__,

                "pandas":
                    pd.__version__,

                "captum":
                    (
                        getattr(
                            captum,
                            "__version__",
                            None,
                        )
                        if captum is not None
                        else None
                    ),
            },

        "validation_evidence":
            {
                "selected_participants":
                    selected_participants,

                "event_participants":
                    event_participants,

                "censored_participants":
                    censored_participants,

                "step16_all_finite":
                    step16_all_finite,

                "step16_median_absolute_completeness_error":
                    step16_median_abs_error,

                "step16_max_absolute_completeness_error":
                    step16_max_abs_error,

                "median_abs_signed_sum_change_16_to_32":
                    median_change_16_to_32,

                "max_abs_signed_sum_change_16_to_32":
                    max_change_16_to_32,

                "zero_baseline_pathology_confirmed":
                    pathology_flag,
            },

        "frozen_checkpoints":
            checkpoint_records,

        "test_xai_status":
            "NOT YET RUN",

        "test_data_accessed_by_lock_script":
            False,

        "production_rule":
            (
                "No XAI baseline, blur kernel, IG integration method, "
                "step count, target horizon, seed set, or primary stability metric "
                "may be changed after TEST attribution generation begins."
            ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            lock,
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5G. HUMAN-READABLE TXT
    # ----------------------------------------------------------------------------------------------------------

    text_lines = [
        "=" * 100,
        "ADNI XAI PRODUCTION CONFIGURATION LOCK — V1",
        "=" * 100,
        "",
        "STATUS",
        "------",
        "XAI configuration: FROZEN BEFORE TEST XAI",
        "TEST attribution generation: NOT YET RUN",
        "TEST data accessed by this lock script: NO",
        "",
        "PRIMARY XAI SPECIFICATION",
        "-------------------------",
        f"Method: {XAI_SPEC['xai_method']}",
        f"Explained input: {XAI_SPEC['explained_input']}",
        f"Conditioning context: {XAI_SPEC['conditioning_context']}",
        f"Baseline: {XAI_SPEC['baseline']}",
        "Baseline filter: 3D local mean, 9x9x9, replicate padding",
        f"Horizons: {XAI_SPEC['horizons_months']}",
        f"Integration method: {XAI_SPEC['integration_method']}",
        f"IG steps: {XAI_SPEC['n_steps']}",
        f"Internal batch size: {XAI_SPEC['internal_batch_size']}",
        f"Seeds: {XAI_SPEC['cross_seed_models']}",
        "",
        "VALIDATION EVIDENCE",
        "-------------------",
        f"Participants: {selected_participants}",
        f"Events: {event_participants}",
        f"Censored: {censored_participants}",
        f"16-step all finite: {step16_all_finite}",
        (
            "16-step median absolute completeness error: "
            f"{step16_median_abs_error:.12e}"
        ),
        (
            "16-step maximum absolute completeness error: "
            f"{step16_max_abs_error:.12e}"
        ),
        (
            "Median |signed IG sum change| 16->32: "
            f"{median_change_16_to_32:.12e}"
        ),
        (
            "Maximum |signed IG sum change| 16->32: "
            f"{max_change_16_to_32:.12e}"
        ),
        f"Zero baseline pathology confirmed: {pathology_flag}",
        "",
        "INTERPRETATION",
        "--------------",
        (
            "Integrated Gradients explanations represent model sensitivity of "
            "predicted progression risk to latest-MRI information conditional "
            "on the participant's prior MRI history and longitudinal timing."
        ),
        (
            "Attribution maps are not interpreted as biological causality."
        ),
        "",
        "POST-LOCK PROHIBITIONS",
        "----------------------",
        "No post-TEST XAI baseline selection.",
        "No post-TEST IG step selection.",
        "No post-TEST horizon selection.",
        "No post-TEST seed selection.",
        "No retraining for the primary analysis.",
        "No recalibration for the primary analysis.",
        "",
        "NEXT PHASE",
        "----------",
        "Frozen five-seed production XAI at 24 and 36 months.",
        "",
        "=" * 100,
    ]

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                text_lines
            )
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5H. HASH ALL LOCK INPUTS + OUTPUTS
    # ----------------------------------------------------------------------------------------------------------

    hash_paths = [
        ZERO_BASELINE_DIAGNOSTIC,
        VALIDATION_BASELINE_COMPARISON,
        VALIDATION_MULTISUBJECT_SELECTION,
        VALIDATION_MULTISUBJECT_RESULTS,
        VALIDATION_MULTISUBJECT_JSON,
        OUTPUT_JSON,
        OUTPUT_TXT,
    ]

    hash_rows = []

    for artifact_path in hash_paths:

        hash_rows.append(
            {
                "artifact":
                    artifact_path.name,

                "path":
                    str(
                        artifact_path
                    ),

                "sha256":
                    sha256_file(
                        artifact_path
                    ),
            }
        )

    for record in checkpoint_records:

        hash_rows.append(
            {
                "artifact":
                    Path(
                        record[
                            "checkpoint"
                        ]
                    ).name,

                "path":
                    record[
                        "checkpoint"
                    ],

                "sha256":
                    record[
                        "sha256"
                    ],
            }
        )

    with open(
        OUTPUT_HASHES,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact",
                "path",
                "sha256",
            ],
        )

        writer.writeheader()

        writer.writerows(
            hash_rows
        )

    # ----------------------------------------------------------------------------------------------------------
    # 5I. FINAL STATUS
    # ----------------------------------------------------------------------------------------------------------

    print(
        "Saved JSON :",
        OUTPUT_JSON
    )

    print(
        "Saved TXT  :",
        OUTPUT_TXT
    )

    print(
        "Saved hashes:",
        OUTPUT_HASHES
    )

    print()
    print("=" * 120)
    print("XAI PRODUCTION CONFIGURATION LOCK: COMPLETE")
    print("=" * 120)

    print(
        "Baseline:",
        "blurred latest MRI (9x9x9 local mean, replicate padding)"
    )

    print(
        "IG method:",
        "Gauss-Legendre"
    )

    print(
        "IG steps:",
        16
    )

    print(
        "Horizons:",
        "24m and 36m"
    )

    print(
        "Five frozen seeds:",
        EXPECTED_SEEDS
    )

    print(
        "Validation evidence:",
        "PASS"
    )

    print(
        "TEST XAI:",
        "NOT YET RUN"
    )

    print(
        "Next step:",
        "FIVE-SEED PRODUCTION XAI"
    )

    print("=" * 120)


if __name__ == "__main__":

    main()

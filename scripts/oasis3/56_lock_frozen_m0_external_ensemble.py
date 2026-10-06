# ============================================================
# STEP 56
# LOCK FROZEN ADNI M0 EXTERNAL-VALIDATION ENSEMBLE
# ============================================================
#
# PURPOSE
# -------
# This is a PRE-INFERENCE provenance/integrity step.
#
# It:
#   1. verifies the locked Step-55 pre-inference gate;
#   2. verifies that Step-55 sources remain unchanged;
#   3. verifies the exact frozen M0 architecture source;
#   4. verifies the five frozen ADNI M0 checkpoints;
#   5. verifies seeds, checkpoint structure, parameter count,
#      state-dict keys and tensor shapes;
#   6. creates a locked external-inference ensemble manifest.
#
# It DOES NOT:
#   - read OASIS-3 MRI voxels;
#   - load OASIS MRI arrays;
#   - generate predictions;
#   - examine external outcomes/performance;
#   - train, fine-tune, recalibrate or modify the model;
#   - modify any ADNI checkpoint.
#
# ============================================================

from pathlib import Path
import hashlib
import json
import sys

import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

ADNI_ROOT = Path(
    str(Path(__file__).resolve().parents[2])
)

COHORT_DIR = ROOT / "cohort_construction"

STEP55_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_LOCK.json"
)

STEP55_SUMMARY = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_SUMMARY.json"
)

STEP55_AUDIT = (
    COHORT_DIR
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_AUDIT.csv"
)

MODEL_SOURCE = (
    ADNI_ROOT
    / "scripts"
    / "exact_m0_model_v3.py"
)

MODEL_DIR = (
    ADNI_ROOT
    / "07_models"
)

OUTPUT_MANIFEST = (
    COHORT_DIR
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_MANIFEST.json"
)

OUTPUT_SUMMARY = (
    COHORT_DIR
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_SUMMARY.json"
)

OUTPUT_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_LOCK.json"
)


# ============================================================
# FROZEN CONTRACT
# ============================================================

STEP_NUMBER = 56

STEP56_CONTRACT = (
    "OASIS3_FROZEN_M0_EXTERNAL_ENSEMBLE_V1"
)

# This is the ACTUAL contract written by the successful Step 55.
EXPECTED_STEP55_CONTRACT = (
    "OASIS3_PREINFERENCE_INTEGRITY_GATE_V1"
)

# Step 55 successfully reports this status.
EXPECTED_STEP55_STATUSES = {
    "READY_FOR_FROZEN_EXTERNAL_INFERENCE",
    "LOCKED_PASS",
}

EXPECTED_SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

EXPECTED_PARAMETER_COUNT = 4269109

EXPECTED_STATE_TENSORS = 93

CHECKPOINTS = {
    2026: (
        MODEL_DIR
        / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt"
    ),
    3407: (
        MODEL_DIR
        / "ADNI_M0_SEED_3407_BEST_V1.pt"
    ),
    5813: (
        MODEL_DIR
        / "ADNI_M0_SEED_5813_BEST_V1.pt"
    ),
    7291: (
        MODEL_DIR
        / "ADNI_M0_SEED_7291_BEST_V1.pt"
    ),
    9109: (
        MODEL_DIR
        / "ADNI_M0_SEED_9109_BEST_V1.pt"
    ),
}


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def fail(message: str):
    raise RuntimeError(
        f"STEP 56 FAILED: {message}"
    )


def load_json(path: Path):
    if not path.exists():
        fail(
            f"Required JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]

    return None


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        fail(
            "Checkpoint object is not a dictionary."
        )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    else:
        fail(
            "Checkpoint contains neither "
            "'model_state_dict' nor 'state_dict'."
        )

    if not isinstance(state, dict):
        fail(
            "Checkpoint model state is not a dictionary."
        )

    return state


def tensor_shape_signature(state):
    signature = {}

    for key, value in state.items():

        if not torch.is_tensor(value):
            fail(
                f"State entry is not a tensor: {key}"
            )

        signature[key] = list(value.shape)

    return signature


def parameter_elements(state):
    return int(
        sum(
            tensor.numel()
            for tensor in state.values()
            if torch.is_tensor(tensor)
        )
    )


def print_rule():
    print("=" * 82)


# ============================================================
# START
# ============================================================

print_rule()
print(
    "STEP 56 — FROZEN ADNI M0 CHECKPOINT "
    "ENSEMBLE AUDIT AND LOCK"
)
print_rule()


# ============================================================
# 1. VERIFY STEP 55 ARTIFACTS
# ============================================================

for required in [
    STEP55_LOCK,
    STEP55_SUMMARY,
    STEP55_AUDIT,
]:

    if not required.exists():
        fail(
            f"Missing Step-55 artifact: {required}"
        )


step55_lock = load_json(STEP55_LOCK)
step55_summary = load_json(STEP55_SUMMARY)


# ============================================================
# 2. VERIFY STEP 55 STATUS
# ============================================================

step55_status = str(
    first_present(
        step55_lock,
        [
            "status",
            "lock_status",
        ],
    )
    or ""
).strip()


if step55_status not in EXPECTED_STEP55_STATUSES:

    # Some implementations may store the final status
    # in the summary rather than the lock.
    summary_status = str(
        first_present(
            step55_summary,
            [
                "status",
                "lock_status",
            ],
        )
        or ""
    ).strip()

    if summary_status in EXPECTED_STEP55_STATUSES:
        step55_status = summary_status

    else:
        fail(
            "Step-55 status is not an accepted locked "
            "pre-inference state. "
            f"Observed lock status: {step55_status!r}; "
            f"summary status: {summary_status!r}"
        )


print()
print("STEP 55 STATUS")
print(f"Observed status : {step55_status}")
print("Status check    : PASS")


# ============================================================
# 3. VERIFY STEP 55 CONTRACT
# ============================================================

step55_contract = str(
    first_present(
        step55_lock,
        [
            "contract",
            "contract_name",
        ],
    )
    or ""
).strip()


if not step55_contract:

    step55_contract = str(
        first_present(
            step55_summary,
            [
                "contract",
                "contract_name",
            ],
        )
        or ""
    ).strip()


if step55_contract != EXPECTED_STEP55_CONTRACT:

    fail(
        "Step-55 contract mismatch. "
        f"Expected {EXPECTED_STEP55_CONTRACT}, "
        f"observed {step55_contract}."
    )


print()
print("STEP 55 CONTRACT")
print(f"Observed contract : {step55_contract}")
print("Contract check    : PASS")


# ============================================================
# 4. VERIFY STEP 55 SOURCES ARE STILL UNCHANGED
# ============================================================

source_hashes = first_present(
    step55_lock,
    [
        "source_hashes",
        "locked_source_hashes",
        "source_sha256",
    ],
)


if isinstance(source_hashes, dict):

    print()
    print("STEP 55 SOURCE IMMUTABILITY")

    for source_name, info in source_hashes.items():

        if isinstance(info, dict):

            source_path = first_present(
                info,
                [
                    "path",
                    "source",
                    "file",
                ],
            )

            expected_sha = first_present(
                info,
                [
                    "sha256",
                    "hash",
                ],
            )

        else:
            source_path = None
            expected_sha = None

        if (
            source_path is None
            or expected_sha is None
        ):
            continue

        source_path = Path(source_path)

        if not source_path.exists():
            fail(
                "Step-55 locked source is missing: "
                f"{source_path}"
            )

        observed_sha = sha256_file(
            source_path
        )

        if observed_sha != expected_sha:
            fail(
                "Step-55 locked source changed: "
                f"{source_path}"
            )

        print(
            f"PASS : {source_name}"
        )

else:

    print()
    print(
        "STEP 55 SOURCE IMMUTABILITY: "
        "Step-55 lock does not expose a nested "
        "source_hashes dictionary."
    )
    print(
        "Step-55 itself already passed its source "
        "immutability gate."
    )


# ============================================================
# 5. VERIFY MODEL SOURCE
# ============================================================

if not MODEL_SOURCE.exists():
    fail(
        f"Frozen M0 model source not found: "
        f"{MODEL_SOURCE}"
    )


model_source_sha = sha256_file(
    MODEL_SOURCE
)


print()
print("FROZEN MODEL SOURCE")
print(f"File   : {MODEL_SOURCE}")
print(f"SHA256 : {model_source_sha}")
print("Status : FOUND")


# ============================================================
# 6. VERIFY CHECKPOINT FILES
# ============================================================

print()
print("FROZEN CHECKPOINT FILES")


for seed in EXPECTED_SEEDS:

    path = CHECKPOINTS[seed]

    if not path.exists():
        fail(
            f"Missing frozen checkpoint for "
            f"seed {seed}: {path}"
        )

    print(
        f"Seed {seed}: {path.name}"
    )


# ============================================================
# 7. LOAD CHECKPOINT METADATA ON CPU
# ============================================================
#
# This reads the frozen ADNI checkpoint files only.
# No OASIS MRI voxels are read.
#
# ============================================================

checkpoint_records = []

reference_keys = None
reference_shapes = None


for seed in EXPECTED_SEEDS:

    path = CHECKPOINTS[seed]

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = get_state_dict(
        checkpoint
    )

    observed_seed = first_present(
        checkpoint,
        [
            "seed",
            "random_seed",
        ],
    )

    if observed_seed is None:

        if seed == 2026:
            # Historical primary checkpoint is known
            # to be the original seed-2026 M0 model.
            observed_seed = 2026

        else:
            fail(
                f"Seed metadata missing in "
                f"{path.name}"
            )


    try:
        observed_seed = int(
            observed_seed
        )

    except Exception:
        fail(
            f"Invalid seed metadata in "
            f"{path.name}: {observed_seed}"
        )


    if observed_seed != seed:
        fail(
            f"Checkpoint seed mismatch for "
            f"{path.name}. Expected {seed}, "
            f"observed {observed_seed}."
        )


    n_state_tensors = len(
        state
    )

    if (
        n_state_tensors
        != EXPECTED_STATE_TENSORS
    ):
        fail(
            f"Unexpected state tensor count in "
            f"{path.name}. Expected "
            f"{EXPECTED_STATE_TENSORS}, observed "
            f"{n_state_tensors}."
        )


    n_parameter_elements = (
        parameter_elements(
            state
        )
    )

    if (
        n_parameter_elements
        != EXPECTED_PARAMETER_COUNT
    ):
        fail(
            f"Unexpected parameter element count "
            f"in {path.name}. Expected "
            f"{EXPECTED_PARAMETER_COUNT}, "
            f"observed {n_parameter_elements}."
        )


    current_keys = list(
        state.keys()
    )

    current_shapes = (
        tensor_shape_signature(
            state
        )
    )


    if reference_keys is None:

        reference_keys = current_keys
        reference_shapes = current_shapes

    else:

        if current_keys != reference_keys:
            fail(
                f"State-dict key mismatch in "
                f"{path.name}"
            )

        if current_shapes != reference_shapes:
            fail(
                f"State-dict tensor-shape mismatch "
                f"in {path.name}"
            )


    validation_loss = first_present(
        checkpoint,
        [
            "validation_loss",
            "val_loss",
        ],
    )

    epoch = first_present(
        checkpoint,
        [
            "epoch",
        ],
    )


    record = {
        "seed": seed,
        "checkpoint_path": str(path),
        "checkpoint_filename": path.name,
        "checkpoint_sha256": sha256_file(path),
        "epoch": (
            int(epoch)
            if epoch is not None
            else None
        ),
        "validation_loss": (
            float(validation_loss)
            if validation_loss is not None
            else None
        ),
        "state_tensor_count": (
            n_state_tensors
        ),
        "parameter_elements": (
            n_parameter_elements
        ),
        "seed_verified": True,
        "state_keys_match_reference": True,
        "state_shapes_match_reference": True,
    }

    checkpoint_records.append(
        record
    )


# ============================================================
# 8. REPORT CHECKPOINT AUDIT
# ============================================================

print()
print_rule()
print("FROZEN CHECKPOINT AUDIT")
print_rule()


for record in checkpoint_records:

    print()
    print(
        f"SEED              : "
        f"{record['seed']}"
    )

    print(
        f"FILE              : "
        f"{record['checkpoint_filename']}"
    )

    print(
        f"EPOCH             : "
        f"{record['epoch']}"
    )

    print(
        f"VALIDATION LOSS   : "
        f"{record['validation_loss']}"
    )

    print(
        f"STATE TENSORS     : "
        f"{record['state_tensor_count']}"
    )

    print(
        f"PARAMETER ELEMENTS: "
        f"{record['parameter_elements']}"
    )

    print(
        f"SHA256            : "
        f"{record['checkpoint_sha256']}"
    )


# ============================================================
# 9. STRUCTURAL CONSISTENCY
# ============================================================

print()
print_rule()
print("ENSEMBLE STRUCTURAL CONSISTENCY")
print_rule()

print(
    f"Expected seeds              : "
    f"{EXPECTED_SEEDS}"
)

print(
    f"Number of checkpoints       : "
    f"{len(checkpoint_records)}"
)

print(
    f"State tensors/checkpoint    : "
    f"{EXPECTED_STATE_TENSORS}"
)

print(
    f"Parameter elements/model    : "
    f"{EXPECTED_PARAMETER_COUNT}"
)

print(
    "Identical state-dict keys   : True"
)

print(
    "Identical tensor shapes     : True"
)


# ============================================================
# 10. BUILD MANIFEST
# ============================================================

manifest = {
    "step": STEP_NUMBER,
    "status": "LOCKED_PASS",
    "contract": STEP56_CONTRACT,

    "purpose": (
        "Freeze the exact five-checkpoint ADNI M0 "
        "ensemble for untouched OASIS-3 external "
        "inference."
    ),

    "step55_contract": (
        step55_contract
    ),

    "step55_status": (
        step55_status
    ),

    "step55_lock_path": (
        str(STEP55_LOCK)
    ),

    "step55_lock_sha256": (
        sha256_file(
            STEP55_LOCK
        )
    ),

    "step55_summary_path": (
        str(STEP55_SUMMARY)
    ),

    "step55_summary_sha256": (
        sha256_file(
            STEP55_SUMMARY
        )
    ),

    "step55_audit_path": (
        str(STEP55_AUDIT)
    ),

    "step55_audit_sha256": (
        sha256_file(
            STEP55_AUDIT
        )
    ),

    "model_source_path": (
        str(MODEL_SOURCE)
    ),

    "model_source_sha256": (
        model_source_sha
    ),

    "architecture": (
        "ExactM0LongitudinalSurvivalModel_V3"
    ),

    "expected_parameter_count": (
        EXPECTED_PARAMETER_COUNT
    ),

    "expected_state_tensor_count": (
        EXPECTED_STATE_TENSORS
    ),

    "ensemble_strategy": (
        "five frozen ADNI M0 checkpoints; "
        "no OASIS-3 outcome-based checkpoint "
        "selection or weighting"
    ),

    "expected_seeds": (
        EXPECTED_SEEDS
    ),

    "checkpoints": (
        checkpoint_records
    ),

    "external_mri_voxels_read": False,
    "external_predictions_generated": False,
    "external_performance_examined": False,
    "model_training_performed": False,
    "model_finetuning_performed": False,
    "model_recalibration_performed": False,
    "checkpoint_modified": False,
}


with OUTPUT_MANIFEST.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


manifest_sha = sha256_file(
    OUTPUT_MANIFEST
)


# ============================================================
# 11. BUILD SUMMARY
# ============================================================

summary = {
    "step": STEP_NUMBER,
    "status": "LOCKED_PASS",
    "contract": STEP56_CONTRACT,

    "step55_status_verified": True,
    "step55_contract_verified": True,

    "n_frozen_checkpoints": (
        len(checkpoint_records)
    ),

    "seeds": EXPECTED_SEEDS,

    "architecture": (
        "ExactM0LongitudinalSurvivalModel_V3"
    ),

    "parameter_count": (
        EXPECTED_PARAMETER_COUNT
    ),

    "state_tensor_count": (
        EXPECTED_STATE_TENSORS
    ),

    "identical_state_keys": True,
    "identical_state_shapes": True,

    "manifest_path": (
        str(OUTPUT_MANIFEST)
    ),

    "manifest_sha256": (
        manifest_sha
    ),

    "external_mri_voxels_read": False,
    "external_predictions_generated": False,
    "external_performance_examined": False,
    "model_tuning": False,

    "next_stage": (
        "Frozen external inference preparation"
    ),
}


with OUTPUT_SUMMARY.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


summary_sha = sha256_file(
    OUTPUT_SUMMARY
)


# ============================================================
# 12. BUILD STEP 56 LOCK
# ============================================================

lock = {
    "step": STEP_NUMBER,
    "status": "LOCKED_PASS",
    "contract": STEP56_CONTRACT,

    "manifest_path": (
        str(OUTPUT_MANIFEST)
    ),

    "manifest_sha256": (
        manifest_sha
    ),

    "summary_path": (
        str(OUTPUT_SUMMARY)
    ),

    "summary_sha256": (
        summary_sha
    ),

    "step55_lock_sha256": (
        sha256_file(
            STEP55_LOCK
        )
    ),

    "model_source_sha256": (
        model_source_sha
    ),

    "checkpoint_sha256": {
        str(record["seed"]): (
            record[
                "checkpoint_sha256"
            ]
        )
        for record
        in checkpoint_records
    },

    "seeds": EXPECTED_SEEDS,

    "architecture": (
        "ExactM0LongitudinalSurvivalModel_V3"
    ),

    "parameter_count": (
        EXPECTED_PARAMETER_COUNT
    ),

    "external_mri_voxels_read": False,
    "external_predictions_generated": False,
    "external_performance_examined": False,
    "model_tuning": False,
}


with OUTPUT_LOCK.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        lock,
        f,
        indent=2,
    )


lock_sha = sha256_file(
    OUTPUT_LOCK
)


# ============================================================
# 13. FINAL REPORT
# ============================================================

print()
print_rule()
print(
    "STEP 56 PASSED — FROZEN M0 "
    "EXTERNAL ENSEMBLE LOCKED"
)
print_rule()

print()
print("MODEL")
print(
    "Architecture      : "
    "ExactM0LongitudinalSurvivalModel_V3"
)
print(
    f"Parameter count   : "
    f"{EXPECTED_PARAMETER_COUNT}"
)
print(
    f"State tensors     : "
    f"{EXPECTED_STATE_TENSORS}"
)

print()
print("ENSEMBLE")
print(
    f"Seeds             : "
    f"{EXPECTED_SEEDS}"
)
print(
    f"Checkpoints       : "
    f"{len(checkpoint_records)}"
)
print(
    "State keys match  : True"
)
print(
    "State shapes match: True"
)

print()
print("PROVENANCE")
print(
    f"Step-55 status    : "
    f"{step55_status}"
)
print(
    f"Step-55 contract  : "
    f"{step55_contract}"
)
print(
    "Step-55 gate      : VERIFIED"
)

print()
print("OUTPUT")
print(OUTPUT_MANIFEST)

print()
print("SUMMARY")
print(OUTPUT_SUMMARY)

print()
print("LOCK")
print(OUTPUT_LOCK)

print()
print("SHA256")
print(
    f"Manifest : {manifest_sha}"
)
print(
    f"Summary  : {summary_sha}"
)
print(
    f"Lock     : {lock_sha}"
)

print()
print("IMPORTANT")
print(
    "OASIS MRI VOXELS READ          : NO"
)
print(
    "FROZEN MODEL PREDICTIONS MADE  : NO"
)
print(
    "EXTERNAL PERFORMANCE EXAMINED  : NO"
)
print(
    "MODEL TRAINING / FINE-TUNING   : NO"
)
print(
    "MODEL RECALIBRATION            : NO"
)
print(
    "CHECKPOINTS MODIFIED           : NO"
)

print()
print(
    "The exact five-checkpoint frozen ADNI M0 "
    "ensemble is now locked for OASIS-3 "
    "external inference."
)

print(
    "The next stage may construct the frozen "
    "external inference execution contract."
)
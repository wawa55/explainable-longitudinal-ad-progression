from pathlib import Path
import sys
import json
import hashlib
import ast
import re
import gc

import numpy as np
import pandas as pd
import nibabel as nib
import torch


# ============================================================
# STEP 58
# FIRST FROZEN OASIS-3 EXTERNAL INFERENCE
#
# SCIENTIFIC FIREWALL
# ------------------------------------------------------------
# NO training
# NO fine-tuning
# NO recalibration
# NO OASIS-based model selection
# NO checkpoint modification
# NO preprocessing modification
# NO outcome-based adaptation
#
# This script applies the already frozen five-seed ADNI M0
# ensemble to the already locked OASIS-3 external cohort.
#
# Correct frozen model input:
#     [B, T, C, D, H, W]
#
# External MRI source grid:
#     [197, 233, 189]
#
# Frozen deterministic center crop:
#     [160, 192, 160]
# ============================================================


print("=" * 78)
print("STEP 58 - FIRST FROZEN OASIS-3 EXTERNAL INFERENCE")
print("=" * 78)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT_DIR = ROOT / "cohort_construction"

ADNI_ROOT = Path(
    str(Path(__file__).resolve().parents[2])
)

SCRIPT_DIR = ADNI_ROOT / "scripts"

MODEL_DIR = ADNI_ROOT / "07_models"

PREPROC_ROOT = Path(
    r"E:\OASIS3_EXTERNAL_VALIDATION\03_preprocessed\external_available"
)

MANIFEST_PATH = (
    COHORT_DIR
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_LOCKED.csv"
)

STEP56_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_LOCK.json"
)

STEP57_LOCK = (
    COHORT_DIR
    / "OASIS3_STEP57_EXTERNAL_INFERENCE_EXECUTION_CONTRACT_LOCK.json"
)

OUTPUT_DIR = ROOT / "external_inference"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_PREDICTIONS.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_INFERENCE_SUMMARY.json"
)

LOCK_PATH = (
    OUTPUT_DIR
    / "OASIS3_STEP58_FROZEN_M0_EXTERNAL_INFERENCE_LOCK.json"
)


# ============================================================
# FROZEN MODEL CONTRACT
# ============================================================

EXPECTED_ARCHITECTURE = (
    "ExactM0LongitudinalSurvivalModel_V3"
)

SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

N_INTERVALS = 5

SOURCE_SPATIAL_SHAPE = (
    197,
    233,
    189,
)

MODEL_SPATIAL_SHAPE = (
    160,
    192,
    160,
)

EXPECTED_PARAMETER_COUNT = 4269109

EXPECTED_STATE_TENSORS = 93

DAYS_PER_YEAR = 365.25

HORIZONS_MONTHS = [
    12,
    24,
    36,
    48,
    60,
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"DEVICE: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# BASIC HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(
        f"STEP 58 FAILED: {message}"
    )


def require_file(
    path,
    description
):
    path = Path(path)

    if not path.exists():
        fail(
            f"{description} missing:\n"
            f"{path}"
        )

    return path


def sha256_file(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


# ============================================================
# PARSING HELPERS
# ============================================================

def parse_sequence(
    value,
    cast=float
):

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        )
    ):
        return [
            cast(x)
            for x in value
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
                np.ndarray,
            )
        ):
            return [
                cast(x)
                for x in parsed
            ]

    except Exception:
        pass

    parts = [
        x.strip()
        for x in re.split(
            r"[|,;]",
            text
        )
        if x.strip()
    ]

    return [
        cast(x)
        for x in parts
    ]


def parse_mri_ids(value):

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        )
    ):
        return [
            str(x).strip()
            for x in value
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
                np.ndarray,
            )
        ):
            return [
                str(x).strip()
                for x in parsed
            ]

    except Exception:
        pass

    if "|" in text:
        return [
            x.strip()
            for x in text.split("|")
            if x.strip()
        ]

    if ";" in text:
        return [
            x.strip()
            for x in text.split(";")
            if x.strip()
        ]

    if "," in text:
        return [
            x.strip()
            for x in text.split(",")
            if x.strip()
        ]

    return [text]


# ============================================================
# EXACT FROZEN CENTER CROP
# ============================================================

def center_crop_3d(
    volume,
    target_shape=MODEL_SPATIAL_SHAPE
):

    source_shape = np.asarray(
        volume.shape,
        dtype=int
    )

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    start = (
        source_shape
        - target_shape
    ) // 2

    end = (
        start
        + target_shape
    )

    cropped = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if cropped.shape != tuple(
        target_shape
    ):
        fail(
            "Center crop mismatch. "
            f"Observed {cropped.shape}; "
            f"expected "
            f"{tuple(target_shape)}"
        )

    return cropped


# ============================================================
# MRI PATH RESOLUTION
# ============================================================

def find_normalized_mri(
    mri_id
):

    mri_id = str(
        mri_id
    ).strip()

    subject_dir = (
        PREPROC_ROOT
        / mri_id
    )

    preferred = (
        subject_dir
        / "05_normalized"
        / f"{mri_id}_MNI_NORMALIZED.nii.gz"
    )

    if preferred.exists():
        return preferred

    candidates = list(
        subject_dir.rglob(
            f"{mri_id}_MNI_NORMALIZED.nii.gz"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        fail(
            "No normalized MRI found "
            f"for {mri_id}"
        )

    fail(
        "Multiple normalized MRI files "
        f"found for {mri_id}:\n"
        + "\n".join(
            str(x)
            for x in candidates
        )
    )


# ============================================================
# EXACT MRI LOADER
#
# Returns ONE MRI as:
# [C, D, H, W]
# =
# [1, 160, 192, 160]
#
# The temporal and batch dimensions are added later.
# ============================================================

def load_mri_tensor(path):

    path = Path(path)

    if not path.exists():
        fail(
            "MRI file missing:\n"
            f"{path}"
        )

    nii = nib.load(
        str(path)
    )

    volume = np.asarray(
        nii.dataobj,
        dtype=np.float32
    )

    if (
        volume.shape
        != SOURCE_SPATIAL_SHAPE
    ):
        fail(
            "Unexpected MRI shape "
            f"{volume.shape}; "
            f"expected "
            f"{SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(
        volume
    ).all():
        fail(
            "Non-finite MRI values:\n"
            f"{path}"
        )

    cropped = center_crop_3d(
        volume
    )

    if (
        cropped.shape
        != MODEL_SPATIAL_SHAPE
    ):
        fail(
            "Unexpected cropped MRI "
            f"shape {cropped.shape}"
        )

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            cropped
        )
    )

    # Add channel dimension only:
    #
    # [D,H,W]
    # ->
    # [C,D,H,W]
    #
    # Batch and temporal dimensions
    # are assembled separately.

    tensor = tensor.unsqueeze(0)

    expected_shape = (
        1,
        MODEL_SPATIAL_SHAPE[0],
        MODEL_SPATIAL_SHAPE[1],
        MODEL_SPATIAL_SHAPE[2],
    )

    if tuple(
        tensor.shape
    ) != expected_shape:
        fail(
            "Single MRI tensor shape "
            "mismatch. "
            f"Observed "
            f"{tuple(tensor.shape)}; "
            f"expected "
            f"{expected_shape}"
        )

    return tensor


# ============================================================
# DISCRETE SURVIVAL RISK
# ============================================================

def cumulative_risk_from_logits(
    logits
):

    hazards = torch.sigmoid(
        logits
    )

    survival = torch.cumprod(
        1.0 - hazards,
        dim=-1
    )

    risk = (
        1.0
        - survival
    )

    return risk


# ============================================================
# IMPORT EXACT FROZEN MODEL
# ============================================================

sys.path.insert(
    0,
    str(SCRIPT_DIR)
)

from exact_m0_model_v3 import (
    ExactM0LongitudinalSurvivalModel_V3
)


# ============================================================
# VERIFY LOCKED INPUT FILES
# ============================================================

require_file(
    MANIFEST_PATH,
    "Step-53 locked external inference manifest"
)

require_file(
    STEP56_LOCK,
    "Step-56 frozen ensemble lock"
)

require_file(
    STEP57_LOCK,
    "Step-57 execution contract lock"
)


with open(
    STEP56_LOCK,
    "r",
    encoding="utf-8"
) as f:
    step56 = json.load(f)


with open(
    STEP57_LOCK,
    "r",
    encoding="utf-8"
) as f:
    step57 = json.load(f)


print("\nLOCKED INPUTS")

print(
    "Step-53 manifest :",
    MANIFEST_PATH
)

print(
    "Step-56 lock     :",
    STEP56_LOCK
)

print(
    "Step-57 lock     :",
    STEP57_LOCK
)


# ============================================================
# LOAD LOCKED EXTERNAL MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)


EXPECTED_LANDMARKS = 1559


if len(df) != EXPECTED_LANDMARKS:
    fail(
        "Expected "
        f"{EXPECTED_LANDMARKS} "
        "model-ready landmarks, "
        f"observed {len(df)}"
    )


required_columns = [
    "locked_subject_id",
    "landmark_mri_id",
    "landmark_day",
    "mri_ids",
    "years_from_first",
    "years_before_landmark",
    "sequence_model_input_ready",
]


missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]


if missing_columns:
    fail(
        "Required manifest columns "
        "missing: "
        + ", ".join(
            missing_columns
        )
    )


ready = (
    df[
        "sequence_model_input_ready"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(
        [
            "true",
            "1",
        ]
    )
)


if not ready.all():
    fail(
        "One or more Step-53 rows "
        "are not sequence-model-input "
        "ready."
    )


print("\nEXTERNAL MANIFEST")

print(
    f"Landmarks : {len(df)}"
)

print(
    "Subjects  : "
    f"{df['locked_subject_id'].nunique()}"
)


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

checkpoint_paths = {}


for seed in SEEDS:

    if seed == 2026:

        candidates = [
            MODEL_DIR
            / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",

            MODEL_DIR
            / "ADNI_M0_SEED_2026_BEST_V1.pt",
        ]

    else:

        candidates = [
            MODEL_DIR
            / (
                f"ADNI_M0_SEED_"
                f"{seed}_BEST_V1.pt"
            )
        ]


    existing = [
        p
        for p in candidates
        if p.exists()
    ]


    if len(existing) == 0:
        fail(
            "Frozen checkpoint "
            f"missing for seed "
            f"{seed}"
        )


    checkpoint_paths[
        seed
    ] = existing[0]


print("\nFROZEN CHECKPOINTS")

for seed in SEEDS:
    print(
        f"{seed}: "
        f"{checkpoint_paths[seed]}"
    )


# ============================================================
# VERIFY CHECKPOINT STRUCTURE
# ============================================================

reference_keys = None

reference_shapes = None


for seed in SEEDS:

    checkpoint = torch.load(
        checkpoint_paths[seed],
        map_location="cpu",
        weights_only=False
    )


    if not isinstance(
        checkpoint,
        dict
    ):
        fail(
            f"Checkpoint seed {seed} "
            "is not a dictionary."
        )


    if (
        "model_state_dict"
        not in checkpoint
    ):
        fail(
            f"Checkpoint seed {seed} "
            "has no model_state_dict."
        )


    state = checkpoint[
        "model_state_dict"
    ]


    if (
        len(state)
        != EXPECTED_STATE_TENSORS
    ):
        fail(
            f"Seed {seed}: expected "
            f"{EXPECTED_STATE_TENSORS} "
            "state tensors, observed "
            f"{len(state)}"
        )


    keys = list(
        state.keys()
    )

    shapes = {
        k: tuple(v.shape)
        for k, v in state.items()
    }


    if reference_keys is None:

        reference_keys = keys

        reference_shapes = shapes

    else:

        if (
            keys
            != reference_keys
        ):
            fail(
                f"Seed {seed}: "
                "state-dict keys "
                "mismatch."
            )

        if (
            shapes
            != reference_shapes
        ):
            fail(
                f"Seed {seed}: "
                "state-dict tensor "
                "shapes mismatch."
            )


    del checkpoint

    gc.collect()


print(
    "\nPASS: all five frozen "
    "checkpoints have identical "
    "state structure."
)


# ============================================================
# VERIFY EXACT MODEL PARAMETER COUNT
# ============================================================

audit_model = (
    ExactM0LongitudinalSurvivalModel_V3()
)

parameter_count = sum(
    p.numel()
    for p in audit_model.parameters()
)


if (
    parameter_count
    != EXPECTED_PARAMETER_COUNT
):
    fail(
        "Frozen architecture "
        "parameter mismatch. "
        f"Expected "
        f"{EXPECTED_PARAMETER_COUNT}, "
        f"observed "
        f"{parameter_count}"
    )


del audit_model

gc.collect()


print(
    "PASS: frozen architecture "
    f"parameter count = "
    f"{parameter_count}"
)


# ============================================================
# BUILD EXACT MRI PATH CACHE
# ============================================================

all_mri_ids = set()


for value in df[
    "mri_ids"
]:

    ids = parse_mri_ids(
        value
    )

    for mri_id in ids:
        all_mri_ids.add(
            mri_id
        )


print(
    "\nUnique MRI IDs required "
    f"by {len(df)} landmarks: "
    f"{len(all_mri_ids)}"
)


mri_path_cache = {}


for i, mri_id in enumerate(
    sorted(
        all_mri_ids
    ),
    start=1
):

    mri_path_cache[
        mri_id
    ] = find_normalized_mri(
        mri_id
    )


    if (
        i == 1
        or i % 250 == 0
        or i == len(
            all_mri_ids
        )
    ):
        print(
            "MRI path verification: "
            f"{i}/"
            f"{len(all_mri_ids)}"
        )


print(
    "PASS: every required "
    "normalized MRI has an "
    "exact path."
)


# ============================================================
# RUN ONE FROZEN SEED AT A TIME
# ============================================================

seed_prediction_tables = {}


for seed_index, seed in enumerate(
    SEEDS,
    start=1
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"FROZEN SEED {seed} "
        f"({seed_index}/"
        f"{len(SEEDS)})"
    )

    print(
        "=" * 78
    )


    checkpoint_path = (
        checkpoint_paths[
            seed
        ]
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )


    model = (
        ExactM0LongitudinalSurvivalModel_V3()
        .to(DEVICE)
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )


    model.eval()


    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad_(
            False
        )


    seed_records = []


    with torch.inference_mode():

        for row_number, (_, row) in enumerate(
            df.iterrows(),
            start=1
        ):

            subject_id = str(
                row[
                    "locked_subject_id"
                ]
            ).strip()


            landmark_mri_id = str(
                row[
                    "landmark_mri_id"
                ]
            ).strip()


            landmark_day = float(
                row[
                    "landmark_day"
                ]
            )


            # --------------------------------------------
            # Parse exact locked temporal sequence
            # --------------------------------------------

            mri_ids = parse_mri_ids(
                row[
                    "mri_ids"
                ]
            )


            years_from_first = (
                parse_sequence(
                    row[
                        "years_from_first"
                    ],
                    float
                )
            )


            years_before_landmark = (
                parse_sequence(
                    row[
                        "years_before_landmark"
                    ],
                    float
                )
            )


            n_visits = len(
                mri_ids
            )


            if n_visits < 1:
                fail(
                    f"{subject_id} / "
                    f"{landmark_mri_id}: "
                    "empty MRI sequence."
                )


            if (
                len(
                    years_from_first
                )
                != n_visits
            ):
                fail(
                    f"{subject_id} / "
                    f"{landmark_mri_id}: "
                    "years_from_first "
                    "length mismatch."
                )


            if (
                len(
                    years_before_landmark
                )
                != n_visits
            ):
                fail(
                    f"{subject_id} / "
                    f"{landmark_mri_id}: "
                    "years_before_landmark "
                    "length mismatch."
                )


            # --------------------------------------------
            # Load each MRI
            #
            # Each visit:
            # [C,D,H,W]
            #
            # Stack:
            # [T,C,D,H,W]
            #
            # Add batch:
            # [B,T,C,D,H,W]
            # --------------------------------------------

            visit_tensors = []


            for mri_id in mri_ids:

                if (
                    mri_id
                    not in mri_path_cache
                ):
                    fail(
                        "MRI path not cached "
                        f"for {mri_id}"
                    )


                visit_tensor = (
                    load_mri_tensor(
                        mri_path_cache[
                            mri_id
                        ]
                    )
                )


                visit_tensors.append(
                    visit_tensor
                )


            # [T,C,D,H,W]
            images = torch.stack(
                visit_tensors,
                dim=0
            )


            # [B,T,C,D,H,W]
            images = (
                images
                .unsqueeze(0)
                .to(
                    device=DEVICE,
                    dtype=torch.float32
                )
            )


            expected_image_shape = (
                1,
                n_visits,
                1,
                MODEL_SPATIAL_SHAPE[0],
                MODEL_SPATIAL_SHAPE[1],
                MODEL_SPATIAL_SHAPE[2],
            )


            if (
                tuple(images.shape)
                != expected_image_shape
            ):
                fail(
                    f"{subject_id} / "
                    f"{landmark_mri_id}: "
                    "assembled MRI tensor "
                    "shape mismatch. "
                    f"Expected "
                    f"{expected_image_shape}, "
                    f"observed "
                    f"{tuple(images.shape)}"
                )


            # --------------------------------------------
            # Visit mask
            # [B,T]
            # --------------------------------------------

            visit_mask = torch.ones(
                (
                    1,
                    n_visits,
                ),
                dtype=torch.bool,
                device=DEVICE
            )


            # --------------------------------------------
            # Frozen temporal variables
            # [B,T]
            # --------------------------------------------

            years_from_first_tensor = (
                torch.tensor(
                    years_from_first,
                    dtype=torch.float32,
                    device=DEVICE
                )
                .unsqueeze(0)
            )


            years_before_landmark_tensor = (
                torch.tensor(
                    years_before_landmark,
                    dtype=torch.float32,
                    device=DEVICE
                )
                .unsqueeze(0)
            )


            if tuple(
                years_from_first_tensor.shape
            ) != (
                1,
                n_visits,
            ):
                fail(
                    f"{subject_id}: "
                    "years_from_first "
                    "tensor shape mismatch."
                )


            if tuple(
                years_before_landmark_tensor.shape
            ) != (
                1,
                n_visits,
            ):
                fail(
                    f"{subject_id}: "
                    "years_before_landmark "
                    "tensor shape mismatch."
                )


            # --------------------------------------------
            # FROZEN MODEL INFERENCE
            # --------------------------------------------

            logits = model(
                images,
                visit_mask,
                years_from_first_tensor,
                years_before_landmark_tensor
            )


            if (
                tuple(logits.shape)
                != (
                    1,
                    N_INTERVALS,
                )
            ):
                fail(
                    f"{subject_id} / "
                    f"{landmark_mri_id}: "
                    "unexpected model "
                    "output shape "
                    f"{tuple(logits.shape)}"
                )


            # --------------------------------------------
            # Convert interval logits to cumulative risk
            # --------------------------------------------

            risks = (
                cumulative_risk_from_logits(
                    logits
                )
                .detach()
                .cpu()
                .numpy()[0]
            )


            logits_np = (
                logits
                .detach()
                .cpu()
                .numpy()[0]
            )


            # --------------------------------------------
            # Store prediction
            # --------------------------------------------

            record = {

                "locked_subject_id":
                    subject_id,

                "landmark_mri_id":
                    landmark_mri_id,

                "landmark_day":
                    landmark_day,

                "n_visits":
                    n_visits,

                "seed":
                    seed,

                "checkpoint":
                    checkpoint_path.name,

                "logit_interval_0":
                    float(
                        logits_np[0]
                    ),

                "logit_interval_1":
                    float(
                        logits_np[1]
                    ),

                "logit_interval_2":
                    float(
                        logits_np[2]
                    ),

                "logit_interval_3":
                    float(
                        logits_np[3]
                    ),

                "logit_interval_4":
                    float(
                        logits_np[4]
                    ),

                "risk_12m":
                    float(
                        risks[0]
                    ),

                "risk_24m":
                    float(
                        risks[1]
                    ),

                "risk_36m":
                    float(
                        risks[2]
                    ),

                "risk_48m":
                    float(
                        risks[3]
                    ),

                "risk_60m":
                    float(
                        risks[4]
                    ),
            }


            seed_records.append(
                record
            )


            # --------------------------------------------
            # MEMORY CLEANUP
            # --------------------------------------------

            del images

            del visit_mask

            del years_from_first_tensor

            del years_before_landmark_tensor

            del logits

            del visit_tensors


            if (
                DEVICE.type
                == "cuda"
            ):
                torch.cuda.empty_cache()


            # --------------------------------------------
            # PROGRESS
            # --------------------------------------------

            if (
                row_number == 1
                or row_number % 25 == 0
                or row_number == len(df)
            ):

                print(
                    f"Seed {seed}: "
                    f"{row_number}/"
                    f"{len(df)} "
                    "landmarks"
                )


    # ========================================================
    # VERIFY COMPLETE SEED
    # ========================================================

    seed_df = pd.DataFrame(
        seed_records
    )


    if (
        len(seed_df)
        != len(df)
    ):
        fail(
            f"Seed {seed}: "
            f"expected {len(df)} "
            "predictions, observed "
            f"{len(seed_df)}"
        )


    seed_prediction_tables[
        seed
    ] = seed_df


    del model

    del checkpoint

    gc.collect()


    if (
        DEVICE.type
        == "cuda"
    ):
        torch.cuda.empty_cache()


    print(
        f"PASS: seed {seed} "
        f"completed "
        f"{len(seed_df)} "
        "frozen predictions."
    )


# ============================================================
# BUILD FIVE-SEED FROZEN ENSEMBLE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "BUILDING FIVE-SEED "
    "FROZEN ENSEMBLE"
)

print(
    "=" * 78
)


base = (
    df.copy()
    .reset_index(
        drop=True
    )
)


for seed in SEEDS:

    seed_df = (
        seed_prediction_tables[
            seed
        ]
        .reset_index(
            drop=True
        )
    )


    subject_match = (
        seed_df[
            "locked_subject_id"
        ]
        .astype(str)
        .values
        ==
        base[
            "locked_subject_id"
        ]
        .astype(str)
        .values
    )


    if not subject_match.all():
        fail(
            f"Seed {seed}: "
            "subject ordering "
            "mismatch."
        )


    landmark_match = (
        seed_df[
            "landmark_mri_id"
        ]
        .astype(str)
        .values
        ==
        base[
            "landmark_mri_id"
        ]
        .astype(str)
        .values
    )


    if not landmark_match.all():
        fail(
            f"Seed {seed}: "
            "landmark ordering "
            "mismatch."
        )


    for horizon in (
        HORIZONS_MONTHS
    ):

        base[
            f"risk_{horizon}m_"
            f"seed_{seed}"
        ] = (
            seed_df[
                f"risk_{horizon}m"
            ].values
        )


# ============================================================
# ENSEMBLE DEFINITION
#
# Arithmetic mean of frozen seed-specific cumulative risks.
# ============================================================

for horizon in (
    HORIZONS_MONTHS
):

    seed_columns = [
        (
            f"risk_{horizon}m_"
            f"seed_{seed}"
        )
        for seed in SEEDS
    ]


    base[
        f"risk_{horizon}m_ensemble"
    ] = (
        base[
            seed_columns
        ]
        .mean(
            axis=1
        )
    )


# ============================================================
# PREDICTION SANITY CHECKS
# ============================================================

prediction_columns = [
    c
    for c in base.columns
    if c.startswith(
        "risk_"
    )
]


for column in (
    prediction_columns
):

    values = pd.to_numeric(
        base[column],
        errors="coerce"
    )


    if values.isna().any():
        fail(
            "Non-numeric or missing "
            f"predictions in {column}"
        )


    if not np.isfinite(
        values
    ).all():
        fail(
            "Non-finite predictions "
            f"in {column}"
        )


    if (
        (values < 0).any()
        or
        (values > 1).any()
    ):
        fail(
            "Risk outside [0,1] "
            f"in {column}"
        )


# ============================================================
# CUMULATIVE RISK MONOTONICITY
# ============================================================

for seed in SEEDS:

    arr = base[
        [
            f"risk_12m_seed_{seed}",
            f"risk_24m_seed_{seed}",
            f"risk_36m_seed_{seed}",
            f"risk_48m_seed_{seed}",
            f"risk_60m_seed_{seed}",
        ]
    ].to_numpy()


    if (
        np.diff(
            arr,
            axis=1
        )
        < -1e-7
    ).any():

        fail(
            f"Seed {seed}: "
            "cumulative risk is "
            "not monotonic."
        )


ensemble_arr = base[
    [
        "risk_12m_ensemble",
        "risk_24m_ensemble",
        "risk_36m_ensemble",
        "risk_48m_ensemble",
        "risk_60m_ensemble",
    ]
].to_numpy()


if (
    np.diff(
        ensemble_arr,
        axis=1
    )
    < -1e-7
).any():

    fail(
        "Ensemble cumulative risk "
        "is not monotonic."
    )


print(
    "PASS: all frozen predictions "
    "are finite, bounded, and "
    "monotonic."
)


# ============================================================
# SAVE FROZEN PREDICTIONS
# ============================================================

base.to_csv(
    PREDICTION_PATH,
    index=False
)


prediction_sha = sha256_file(
    PREDICTION_PATH
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "step":
        58,

    "status":
        "FROZEN_EXTERNAL_INFERENCE_COMPLETE",

    "architecture":
        EXPECTED_ARCHITECTURE,

    "parameter_count":
        EXPECTED_PARAMETER_COUNT,

    "state_tensor_count":
        EXPECTED_STATE_TENSORS,

    "seeds":
        SEEDS,

    "checkpoint_count":
        len(SEEDS),

    "external_landmarks":
        int(
            len(base)
        ),

    "external_subjects":
        int(
            base[
                "locked_subject_id"
            ].nunique()
        ),

    "unique_mri_ids":
        int(
            len(
                all_mri_ids
            )
        ),

    "source_mri_shape":
        list(
            SOURCE_SPATIAL_SHAPE
        ),

    "model_crop_shape":
        list(
            MODEL_SPATIAL_SHAPE
        ),

    "model_input_contract":
        "[B,T,C,D,H,W]",

    "horizons_months":
        HORIZONS_MONTHS,

    "risk_definition":
        (
            "1 - cumulative product "
            "of (1 - sigmoid(interval_logit))"
        ),

    "ensemble_definition":
        (
            "arithmetic mean of "
            "frozen five-seed "
            "cumulative risks"
        ),

    "predictions_file":
        str(
            PREDICTION_PATH
        ),

    "predictions_sha256":
        prediction_sha,

    "training_performed":
        False,

    "fine_tuning_performed":
        False,

    "recalibration_performed":
        False,

    "oasis_based_model_selection":
        False,

    "checkpoint_modified":
        False,

    "preprocessed_mri_modified":
        False,

    "external_performance_examined_during_inference":
        False,
}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


summary_sha = sha256_file(
    SUMMARY_PATH
)


# ============================================================
# LOCK
# ============================================================

lock = {

    "step":
        58,

    "contract":
        (
            "OASIS3_FIRST_FROZEN_"
            "M0_EXTERNAL_INFERENCE_V1"
        ),

    "status":
        "LOCKED_COMPLETE",

    "prediction_sha256":
        prediction_sha,

    "summary_sha256":
        summary_sha,

    "step53_manifest_sha256":
        sha256_file(
            MANIFEST_PATH
        ),

    "step56_lock_sha256":
        sha256_file(
            STEP56_LOCK
        ),

    "step57_lock_sha256":
        sha256_file(
            STEP57_LOCK
        ),

    "checkpoint_sha256": {
        str(seed):
            sha256_file(
                checkpoint_paths[
                    seed
                ]
            )
        for seed in SEEDS
    },

    "frozen_model":
        True,

    "no_training":
        True,

    "no_fine_tuning":
        True,

    "no_recalibration":
        True,

    "no_oasis_model_selection":
        True,

    "input_tensor_contract":
        "[B,T,C,D,H,W]",

    "source_grid":
        list(
            SOURCE_SPATIAL_SHAPE
        ),

    "model_crop":
        list(
            MODEL_SPATIAL_SHAPE
        ),
}


with open(
    LOCK_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        lock,
        f,
        indent=2
    )


lock_sha = sha256_file(
    LOCK_PATH
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STEP 58 PASSED - FIRST FROZEN "
    "OASIS-3 EXTERNAL INFERENCE COMPLETE"
)

print(
    "=" * 78
)


print("\nPOPULATION")

print(
    f"Landmarks : "
    f"{len(base)}"
)

print(
    "Subjects  : "
    f"{base['locked_subject_id'].nunique()}"
)

print(
    "MRI IDs   : "
    f"{len(all_mri_ids)}"
)


print("\nMODEL")

print(
    "Architecture : "
    f"{EXPECTED_ARCHITECTURE}"
)

print(
    "Parameters   : "
    f"{EXPECTED_PARAMETER_COUNT}"
)

print(
    "State tensors: "
    f"{EXPECTED_STATE_TENSORS}"
)

print(
    "Seeds        : "
    f"{SEEDS}"
)


print("\nMRI CONTRACT")

print(
    "Source grid : "
    f"{SOURCE_SPATIAL_SHAPE}"
)

print(
    "Model crop  : "
    f"{MODEL_SPATIAL_SHAPE}"
)

print(
    "Model input : "
    "[B,T,C,D,H,W]"
)


print(
    "\nENSEMBLE RISK SUMMARY"
)

for horizon in (
    HORIZONS_MONTHS
):

    column = (
        f"risk_{horizon}m_ensemble"
    )

    print(
        f"{horizon:2d}m : "
        f"mean="
        f"{base[column].mean():.6f}  "
        f"median="
        f"{base[column].median():.6f}  "
        f"min="
        f"{base[column].min():.6f}  "
        f"max="
        f"{base[column].max():.6f}"
    )


print("\nOUTPUT")
print(
    PREDICTION_PATH
)


print("\nSUMMARY")
print(
    SUMMARY_PATH
)


print("\nLOCK")
print(
    LOCK_PATH
)


print("\nSHA256")

print(
    "Predictions : "
    f"{prediction_sha}"
)

print(
    "Summary     : "
    f"{summary_sha}"
)

print(
    "Lock        : "
    f"{lock_sha}"
)


print("\nIMPORTANT")

print(
    "MRI VOXELS READ                : YES"
)

print(
    "FROZEN CHECKPOINTS LOADED      : YES"
)

print(
    "FROZEN MODEL PREDICTIONS MADE  : YES"
)

print(
    "EXTERNAL PERFORMANCE EXAMINED  : NO"
)

print(
    "MODEL TRAINING                 : NO"
)

print(
    "MODEL FINE-TUNING              : NO"
)

print(
    "MODEL RECALIBRATION            : NO"
)

print(
    "OASIS-BASED MODEL SELECTION    : NO"
)

print(
    "CHECKPOINTS MODIFIED           : NO"
)

print(
    "PREPROCESSED MRI MODIFIED      : NO"
)


print(
    "\nThe frozen five-seed ADNI M0 "
    "model has now been applied to "
    "the locked OASIS-3 external "
    "cohort."
)

print(
    "No external discrimination, "
    "calibration, or survival "
    "performance metric has yet "
    "been calculated."
)

print(
    "The next stage may evaluate "
    "these frozen predictions "
    "against the already locked "
    "external endpoints."
)
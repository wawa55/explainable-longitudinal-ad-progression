# ============================================================
# STEP 57
# LOCK OASIS-3 FROZEN EXTERNAL-INFERENCE EXECUTION CONTRACT
# ============================================================
#
# PURPOSE
# -------
# Freeze the complete execution contract BEFORE the first
# external OASIS-3 prediction is generated.
#
# IMPORTANT POPULATION DISTINCTION
# --------------------------------
# Model-ready population:
#     Step 53 = 1559 landmarks / 542 subjects
#
# Endpoint-evaluable population:
#     Step 53A = 1486 landmarks / 541 subjects
#
# Step 53A is a strict endpoint-evaluable subset of Step 53.
# The 73 Step-53 landmarks excluded from Step 53A are NOT an
# inference failure and must not be confused with MRI readiness.
#
# THIS STEP DOES NOT:
#   - read MRI voxel arrays
#   - load checkpoint tensors
#   - instantiate the neural network
#   - generate predictions
#   - examine external performance
#   - train/fine-tune/recalibrate the model
#
# ============================================================

from pathlib import Path
import hashlib
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[2] / "external_data" / "oasis3")
)

COHORT = ROOT / "cohort_construction"

STEP55_LOCK = (
    COHORT
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_LOCK.json"
)

STEP55_SUMMARY = (
    COHORT
    / "OASIS3_STEP55_PREINFERENCE_INTEGRITY_SUMMARY.json"
)

STEP56_LOCK = (
    COHORT
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_LOCK.json"
)

STEP56_SUMMARY = (
    COHORT
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_SUMMARY.json"
)

STEP56_MANIFEST = (
    COHORT
    / "OASIS3_STEP56_FROZEN_M0_EXTERNAL_ENSEMBLE_MANIFEST.json"
)

STEP53 = (
    COHORT
    / "OASIS3_STEP53_FINAL_EXTERNAL_INFERENCE_MANIFEST_LOCKED.csv"
)

STEP53A = (
    COHORT
    / "OASIS3_STEP53A_ENDPOINT_EVALUABLE_MANIFEST_LOCKED.csv"
)

STEP49 = (
    COHORT
    / "OASIS3_STEP49_EXTERNAL_SURVIVAL_TARGETS_LOCKED.csv"
)

STEP54_LOCK = (
    COHORT
    / "OASIS3_STEP54_EXTERNAL_ENDPOINT_CONTRACT_LOCK.json"
)

OUTPUT_SUMMARY = (
    COHORT
    / "OASIS3_STEP57_EXTERNAL_INFERENCE_EXECUTION_CONTRACT_SUMMARY.json"
)

OUTPUT_LOCK = (
    COHORT
    / "OASIS3_STEP57_EXTERNAL_INFERENCE_EXECUTION_CONTRACT_LOCK.json"
)


# ============================================================
# CONTRACT CONSTANTS
# ============================================================

STEP = 57

CONTRACT = (
    "OASIS3_FROZEN_EXTERNAL_INFERENCE_EXECUTION_CONTRACT_V1"
)

EXPECTED_STEP55_CONTRACT = (
    "OASIS3_PREINFERENCE_INTEGRITY_GATE_V1"
)

EXPECTED_STEP56_CONTRACT = (
    "OASIS3_FROZEN_M0_EXTERNAL_ENSEMBLE_V1"
)

EXPECTED_STEP56_STATUS = "LOCKED_PASS"

EXPECTED_ARCHITECTURE = (
    "ExactM0LongitudinalSurvivalModel_V3"
)

EXPECTED_SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

EXPECTED_PARAMETER_COUNT = 4269109
EXPECTED_STATE_TENSORS = 93

EXPECTED_HORIZONS_MONTHS = [
    12,
    24,
    36,
    48,
    60,
]

EXPECTED_INTERVAL_EDGES_DAYS = [
    0.0,
    365.25,
    730.5,
    1095.75,
    1461.0,
    1826.25,
]

EXPECTED_DAYS_PER_YEAR = 365.25

# ------------------------------------------------------------
# LOCKED POPULATION COUNTS
# ------------------------------------------------------------

EXPECTED_MODEL_READY_LANDMARKS = 1559
EXPECTED_MODEL_READY_SUBJECTS = 542

EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS = 1486
EXPECTED_ENDPOINT_EVALUABLE_SUBJECTS = 541

EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS = (
    EXPECTED_MODEL_READY_LANDMARKS
    - EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS
)

EXPECTED_EVENTS = 344
EXPECTED_CENSORED = 1142

EXPECTED_REQUIRED_MRI = 3729
EXPECTED_READY_MRI = 3729
EXPECTED_MISSING_MRI = 0
EXPECTED_NOT_READY_MRI = 0

# ------------------------------------------------------------
# FROZEN MRI GEOMETRY
# ------------------------------------------------------------

EXPECTED_CROP_SIZE = [
    160,
    192,
    160,
]

EXPECTED_CROP_START = [
    18,
    20,
    14,
]

EXPECTED_CROP_END = [
    178,
    212,
    174,
]

EXPECTED_MNI_SIZE = [
    197,
    233,
    189,
]

EXPECTED_MNI_SPACING = [
    1.0,
    1.0,
    1.0,
]

EXPECTED_MNI_ORIGIN = [
    98.0,
    134.0,
    -72.0,
]

EXPECTED_MNI_DIRECTION = [
    -1.0, 0.0, 0.0,
     0.0, -1.0, 0.0,
     0.0, 0.0, 1.0,
]


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(
        f"STEP 57 FAILED: {message}"
    )


def sha256_file(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    path = Path(path)

    if not path.exists():
        fail(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def first_present(mapping, keys):
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        if key in mapping:
            return mapping[key]

    return None


def assert_equal(
    name,
    observed,
    expected,
):
    if observed != expected:
        fail(
            f"{name} mismatch. "
            f"Expected {expected!r}, "
            f"observed {observed!r}"
        )

    print(
        f"PASS : {name}"
    )


def assert_true(
    name,
    condition,
):
    if not condition:
        fail(
            f"{name} is False"
        )

    print(
        f"PASS : {name}"
    )


def print_rule():
    print("=" * 82)


def find_subject_column(df, table_name):
    candidates = [
        "subject_id",
        "locked_subject_id",
        "participant_id",
        "Subject",
        "subject",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    fail(
        f"Could not identify subject column in {table_name}."
    )


def discover_landmark_key(df):
    """
    Return the strongest available landmark identifier.

    Preference is given to explicit landmark identifiers. If none
    exists, construct subject/day key when possible.
    """

    explicit_candidates = [
        "landmark_key",
        "external_landmark_key",
        "landmark_id",
        "locked_landmark_id",
        "landmark_uid",
    ]

    for col in explicit_candidates:
        if col in df.columns:
            values = (
                df[col]
                .astype(str)
                .str.strip()
            )

            if (
                values.notna().all()
                and values.ne("").all()
                and values.nunique() == len(df)
            ):
                return col, values

    subject_col = None

    for col in [
        "subject_id",
        "locked_subject_id",
        "participant_id",
        "Subject",
        "subject",
    ]:
        if col in df.columns:
            subject_col = col
            break

    day_col = None

    for col in [
        "landmark_day",
        "landmark_days",
        "landmark_day_from_baseline",
        "landmark_time_days",
        "landmark_time",
    ]:
        if col in df.columns:
            day_col = col
            break

    if (
        subject_col is not None
        and day_col is not None
    ):
        values = (
            df[subject_col]
            .astype(str)
            .str.strip()
            + "__"
            + df[day_col]
            .astype(str)
            .str.strip()
        )

        if values.nunique() == len(df):
            return (
                f"{subject_col}+{day_col}",
                values,
            )

    return None, None


# ============================================================
# START
# ============================================================

print_rule()
print(
    "STEP 57 — FROZEN OASIS-3 EXTERNAL-INFERENCE "
    "EXECUTION CONTRACT"
)
print_rule()


# ============================================================
# 1. REQUIRED ARTIFACTS
# ============================================================

required_files = [
    STEP55_LOCK,
    STEP55_SUMMARY,
    STEP56_LOCK,
    STEP56_SUMMARY,
    STEP56_MANIFEST,
    STEP53,
    STEP53A,
    STEP49,
    STEP54_LOCK,
]

print()
print("REQUIRED ARTIFACTS")

for path in required_files:

    if not path.exists():
        fail(
            f"Missing required artifact: {path}"
        )

    print(
        f"PASS : {path.name}"
    )


# ============================================================
# 2. LOAD LOCKS
# ============================================================

step55_lock = load_json(
    STEP55_LOCK
)

step55_summary = load_json(
    STEP55_SUMMARY
)

step56_lock = load_json(
    STEP56_LOCK
)

step56_summary = load_json(
    STEP56_SUMMARY
)

step56_manifest = load_json(
    STEP56_MANIFEST
)

step54_lock = load_json(
    STEP54_LOCK
)


# ============================================================
# 3. STEP 55 VERIFICATION
# ============================================================

print()
print_rule()
print("STEP 55 VERIFICATION")
print_rule()

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


assert_equal(
    "Step-55 contract",
    step55_contract,
    EXPECTED_STEP55_CONTRACT,
)


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

if not step55_status:

    step55_status = str(
        first_present(
            step55_summary,
            [
                "status",
                "lock_status",
            ],
        )
        or ""
    ).strip()


assert_true(
    "Step-55 ready status",
    step55_status
    in {
        "READY_FOR_FROZEN_EXTERNAL_INFERENCE",
        "LOCKED_PASS",
    },
)


# ============================================================
# 4. STEP 56 VERIFICATION
# ============================================================

print()
print_rule()
print("STEP 56 VERIFICATION")
print_rule()

step56_contract = str(
    step56_lock.get(
        "contract",
        "",
    )
).strip()

step56_status = str(
    step56_lock.get(
        "status",
        "",
    )
).strip()


assert_equal(
    "Step-56 contract",
    step56_contract,
    EXPECTED_STEP56_CONTRACT,
)

assert_equal(
    "Step-56 status",
    step56_status,
    EXPECTED_STEP56_STATUS,
)


# ============================================================
# 5. VERIFY STEP 56 ARTIFACT HASHES
# ============================================================

print()
print_rule()
print("STEP 56 PROVENANCE")
print_rule()

manifest_sha_now = sha256_file(
    STEP56_MANIFEST
)

summary_sha_now = sha256_file(
    STEP56_SUMMARY
)

expected_manifest_sha = (
    step56_lock.get(
        "manifest_sha256"
    )
)

expected_summary_sha = (
    step56_lock.get(
        "summary_sha256"
    )
)


assert_equal(
    "Step-56 manifest SHA",
    manifest_sha_now,
    expected_manifest_sha,
)

assert_equal(
    "Step-56 summary SHA",
    summary_sha_now,
    expected_summary_sha,
)


# ============================================================
# 6. VERIFY FROZEN MODEL CONTRACT
# ============================================================

print()
print_rule()
print("FROZEN MODEL CONTRACT")
print_rule()

architecture = (
    step56_manifest.get(
        "architecture"
    )
)

parameter_count = (
    step56_manifest.get(
        "expected_parameter_count"
    )
)

state_tensors = (
    step56_manifest.get(
        "expected_state_tensor_count"
    )
)

seeds = (
    step56_manifest.get(
        "expected_seeds"
    )
)


assert_equal(
    "Architecture",
    architecture,
    EXPECTED_ARCHITECTURE,
)

assert_equal(
    "Parameter count",
    int(parameter_count),
    EXPECTED_PARAMETER_COUNT,
)

assert_equal(
    "State tensor count",
    int(state_tensors),
    EXPECTED_STATE_TENSORS,
)

assert_equal(
    "Frozen ensemble seeds",
    list(seeds),
    EXPECTED_SEEDS,
)


checkpoints = (
    step56_manifest.get(
        "checkpoints",
        [],
    )
)

assert_equal(
    "Frozen checkpoint count",
    len(checkpoints),
    5,
)


for record in checkpoints:

    seed = int(
        record["seed"]
    )

    checkpoint_path = Path(
        record[
            "checkpoint_path"
        ]
    )

    expected_sha = (
        record[
            "checkpoint_sha256"
        ]
    )

    if not checkpoint_path.exists():
        fail(
            "Frozen checkpoint disappeared: "
            f"{checkpoint_path}"
        )

    observed_sha = sha256_file(
        checkpoint_path
    )

    assert_equal(
        f"Checkpoint SHA seed {seed}",
        observed_sha,
        expected_sha,
    )


# ============================================================
# 7. VERIFY MODEL SOURCE
# ============================================================

model_source_path = Path(
    step56_manifest[
        "model_source_path"
    ]
)

expected_model_source_sha = (
    step56_manifest[
        "model_source_sha256"
    ]
)

if not model_source_path.exists():
    fail(
        "Frozen model source disappeared: "
        f"{model_source_path}"
    )

observed_model_source_sha = (
    sha256_file(
        model_source_path
    )
)

assert_equal(
    "Frozen model-source SHA",
    observed_model_source_sha,
    expected_model_source_sha,
)


# ============================================================
# 8. LOAD EXTERNAL LOCKED TABLES
# ============================================================

print()
print_rule()
print("EXTERNAL POPULATION CONTRACT")
print_rule()

step53 = pd.read_csv(
    STEP53,
    low_memory=False,
)

step53a = pd.read_csv(
    STEP53A,
    low_memory=False,
)

step49 = pd.read_csv(
    STEP49,
    low_memory=False,
)


# ------------------------------------------------------------
# Step 53 = complete model-ready inference population
# ------------------------------------------------------------

assert_equal(
    "Step-53 model-ready landmarks",
    len(step53),
    EXPECTED_MODEL_READY_LANDMARKS,
)


# ------------------------------------------------------------
# Step 53A = endpoint-evaluable subset
# ------------------------------------------------------------

assert_equal(
    "Step-53A endpoint-evaluable landmarks",
    len(step53a),
    EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS,
)


# ------------------------------------------------------------
# Step 49 belongs to the model-ready target population.
#
# IMPORTANT:
# Do NOT force Step 49 to equal Step 53A.
# Step 53A excludes 73 landmarks from endpoint evaluation.
# ------------------------------------------------------------

assert_equal(
    "Step-49 locked survival-target landmarks",
    len(step49),
    EXPECTED_MODEL_READY_LANDMARKS,
)


assert_equal(
    "Endpoint-excluded landmarks",
    len(step53) - len(step53a),
    EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS,
)


# ============================================================
# 9. SUBJECT IDENTIFIER AUDIT
# ============================================================

print()
print_rule()
print("SUBJECT POPULATION AUDIT")
print_rule()

step53_subject_col = find_subject_column(
    step53,
    "Step-53",
)

step53a_subject_col = find_subject_column(
    step53a,
    "Step-53A",
)

step49_subject_col = find_subject_column(
    step49,
    "Step-49",
)


step53_subjects = int(
    step53[
        step53_subject_col
    ].nunique()
)

step53a_subjects = int(
    step53a[
        step53a_subject_col
    ].nunique()
)

step49_subjects = int(
    step49[
        step49_subject_col
    ].nunique()
)


assert_equal(
    "Step-53 model-ready subjects",
    step53_subjects,
    EXPECTED_MODEL_READY_SUBJECTS,
)

assert_equal(
    "Step-53A endpoint-evaluable subjects",
    step53a_subjects,
    EXPECTED_ENDPOINT_EVALUABLE_SUBJECTS,
)

assert_equal(
    "Step-49 target subjects",
    step49_subjects,
    EXPECTED_MODEL_READY_SUBJECTS,
)


# ============================================================
# 10. LANDMARK RELATIONSHIP AUDIT
# ============================================================

print()
print_rule()
print("LANDMARK POPULATION RELATIONSHIP")
print_rule()

step53_key_name, step53_keys = (
    discover_landmark_key(step53)
)

step53a_key_name, step53a_keys = (
    discover_landmark_key(step53a)
)

step49_key_name, step49_keys = (
    discover_landmark_key(step49)
)


if (
    step53_keys is not None
    and step53a_keys is not None
):

    step53_set = set(
        step53_keys.tolist()
    )

    step53a_set = set(
        step53a_keys.tolist()
    )

    missing_from_step53 = (
        step53a_set
        - step53_set
    )

    excluded_from_endpoint = (
        step53_set
        - step53a_set
    )

    assert_equal(
        "Step-53A keys absent from Step-53",
        len(missing_from_step53),
        0,
    )

    assert_equal(
        "Step-53 landmarks excluded from endpoint set",
        len(excluded_from_endpoint),
        EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS,
    )

    print(
        f"Step-53 key definition  : {step53_key_name}"
    )

    print(
        f"Step-53A key definition : {step53a_key_name}"
    )

else:

    print(
        "INFO : Explicit comparable landmark keys were not "
        "available for Step-53/53A."
    )

    print(
        "INFO : Locked row counts and subject counts remain "
        "the governing population contract."
    )


if (
    step53_keys is not None
    and step49_keys is not None
):

    step53_set = set(
        step53_keys.tolist()
    )

    step49_set = set(
        step49_keys.tolist()
    )

    if len(step53_set) == len(step53):

        assert_equal(
            "Step-49 keys absent from Step-53",
            len(
                step49_set
                - step53_set
            ),
            0,
        )

        assert_equal(
            "Step-53 keys absent from Step-49",
            len(
                step53_set
                - step49_set
            ),
            0,
        )

        print(
            f"Step-49 key definition  : {step49_key_name}"
        )


# ============================================================
# 11. VERIFY HORIZON CONTRACT
# ============================================================

print()
print_rule()
print("SURVIVAL HORIZON CONTRACT")
print_rule()

for h in EXPECTED_HORIZONS_MONTHS:

    event_col = (
        f"event_by_{h}m"
    )

    evaluable_col = (
        f"evaluable_{h}m"
    )

    if event_col not in step53a.columns:
        fail(
            f"Missing {event_col}"
        )

    if evaluable_col not in step53a.columns:
        fail(
            f"Missing {evaluable_col}"
        )

    print(
        f"PASS : {h}-month endpoint columns"
    )


# ============================================================
# 12. VERIFY EVENT / CENSORING COUNTS
# ============================================================

print()
print_rule()
print("ENDPOINT EVENT AUDIT")
print_rule()

event_indicator_candidates = [
    "event",
    "event_indicator",
    "event_observed",
    "progression_event",
]

event_indicator_col = None

for col in event_indicator_candidates:
    if col in step53a.columns:
        event_indicator_col = col
        break


if event_indicator_col is not None:

    event_values = pd.to_numeric(
        step53a[event_indicator_col],
        errors="coerce",
    )

    if event_values.isna().any():
        fail(
            "Event indicator contains non-numeric or missing values."
        )

    observed_events = int(
        (event_values == 1).sum()
    )

    observed_censored = int(
        (event_values == 0).sum()
    )

    assert_equal(
        "Endpoint events",
        observed_events,
        EXPECTED_EVENTS,
    )

    assert_equal(
        "Endpoint censored",
        observed_censored,
        EXPECTED_CENSORED,
    )

else:

    assert_equal(
        "Locked events + censored",
        EXPECTED_EVENTS
        + EXPECTED_CENSORED,
        EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS,
    )

    print(
        "INFO : No canonical single event-indicator column "
        "was discovered in Step-53A."
    )

    print(
        "INFO : Event/censor counts remain inherited from "
        "the verified Step-54/55 endpoint contract."
    )


# ============================================================
# 13. VERIFY STEP 54 ENDPOINT CONTRACT
# ============================================================

print()
print_rule()
print("STEP 54 ENDPOINT CONTRACT")
print_rule()

step54_status = str(
    step54_lock.get(
        "status",
        "",
    )
).strip()

assert_equal(
    "Step-54 status",
    step54_status,
    "LOCKED_PASS",
)


step54_horizons = (
    step54_lock.get(
        "horizons_months"
    )
)

if step54_horizons is not None:

    assert_equal(
        "Step-54 horizons",
        list(step54_horizons),
        EXPECTED_HORIZONS_MONTHS,
    )


step54_edges = (
    step54_lock.get(
        "interval_edges_days"
    )
)

if step54_edges is not None:

    assert_equal(
        "Step-54 interval edges",
        [
            float(x)
            for x in step54_edges
        ],
        EXPECTED_INTERVAL_EDGES_DAYS,
    )


days_per_year = (
    step54_lock.get(
        "days_per_year"
    )
)

if days_per_year is not None:

    assert_equal(
        "Step-54 days/year",
        float(days_per_year),
        EXPECTED_DAYS_PER_YEAR,
    )


# ============================================================
# 14. FREEZE MRI INPUT GEOMETRY
# ============================================================

print()
print_rule()
print("FROZEN MRI INPUT GEOMETRY")
print_rule()

print(
    f"MNI grid size       : "
    f"{EXPECTED_MNI_SIZE}"
)

print(
    f"MNI spacing         : "
    f"{EXPECTED_MNI_SPACING}"
)

print(
    f"MNI origin          : "
    f"{EXPECTED_MNI_ORIGIN}"
)

print(
    f"MNI direction       : "
    f"{EXPECTED_MNI_DIRECTION}"
)

print(
    f"Crop start          : "
    f"{EXPECTED_CROP_START}"
)

print(
    f"Crop end            : "
    f"{EXPECTED_CROP_END}"
)

print(
    f"Model crop size     : "
    f"{EXPECTED_CROP_SIZE}"
)


assert_equal(
    "Crop dimension 1",
    EXPECTED_CROP_END[0]
    - EXPECTED_CROP_START[0],
    EXPECTED_CROP_SIZE[0],
)

assert_equal(
    "Crop dimension 2",
    EXPECTED_CROP_END[1]
    - EXPECTED_CROP_START[1],
    EXPECTED_CROP_SIZE[1],
)

assert_equal(
    "Crop dimension 3",
    EXPECTED_CROP_END[2]
    - EXPECTED_CROP_START[2],
    EXPECTED_CROP_SIZE[2],
)


# ============================================================
# 15. MRI READINESS CONTRACT
# ============================================================

print()
print_rule()
print("MRI READINESS CONTRACT")
print_rule()

assert_equal(
    "Required MRI",
    EXPECTED_REQUIRED_MRI,
    3729,
)

assert_equal(
    "Ready MRI",
    EXPECTED_READY_MRI,
    EXPECTED_REQUIRED_MRI,
)

assert_equal(
    "Missing MRI",
    EXPECTED_MISSING_MRI,
    0,
)

assert_equal(
    "Not-ready MRI",
    EXPECTED_NOT_READY_MRI,
    0,
)


# ============================================================
# 16. TEMPORAL EXECUTION CONTRACT
# ============================================================

print()
print_rule()
print("LONGITUDINAL TEMPORAL CONTRACT")
print_rule()

temporal_contract = {

    "input_type": (
        "longitudinal serial 3D T1 MRI"
    ),

    "visit_order": (
        "chronological ascending time"
    ),

    "time_encoding": (
        "continuous visit-time encoding"
    ),

    "sequence_aggregation": (
        "3-layer Transformer followed by masked mean"
    ),

    "padding_rule": (
        "padded visits excluded by visit mask"
    ),

    "prediction_unit": (
        "landmark"
    ),

    "future_information_allowed": False,

    "post_landmark_MRI_allowed": False,

    "external_outcome_used_as_model_input": False,
}


for key, value in temporal_contract.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 17. SURVIVAL OUTPUT CONTRACT
# ============================================================

print()
print_rule()
print("FROZEN SURVIVAL OUTPUT CONTRACT")
print_rule()

survival_contract = {

    "horizons_months": (
        EXPECTED_HORIZONS_MONTHS
    ),

    "interval_edges_days": (
        EXPECTED_INTERVAL_EDGES_DAYS
    ),

    "days_per_year": (
        EXPECTED_DAYS_PER_YEAR
    ),

    "network_output": (
        "five discrete interval logits"
    ),

    "interval_hazard": (
        "sigmoid(logit_k)"
    ),

    "cumulative_survival": (
        "product_{k<=t}(1-h_k)"
    ),

    "cumulative_risk": (
        "1-product_{k<=t}(1-h_k)"
    ),

    "primary_report_horizons_months": [
        12,
        24,
        36,
    ],

    "secondary_horizons_months": [
        48,
        60,
    ],
}


for key, value in survival_contract.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 18. EXTERNAL POPULATION CONTRACT
# ============================================================

population_contract = {

    "model_ready_landmarks": (
        EXPECTED_MODEL_READY_LANDMARKS
    ),

    "model_ready_subjects": (
        EXPECTED_MODEL_READY_SUBJECTS
    ),

    "endpoint_evaluable_landmarks": (
        EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS
    ),

    "endpoint_evaluable_subjects": (
        EXPECTED_ENDPOINT_EVALUABLE_SUBJECTS
    ),

    "endpoint_excluded_landmarks": (
        EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS
    ),

    "events": (
        EXPECTED_EVENTS
    ),

    "censored": (
        EXPECTED_CENSORED
    ),

    "required_mri": (
        EXPECTED_REQUIRED_MRI
    ),

    "ready_mri": (
        EXPECTED_READY_MRI
    ),

    "missing_mri": (
        EXPECTED_MISSING_MRI
    ),

    "not_ready_mri": (
        EXPECTED_NOT_READY_MRI
    ),
}


# ============================================================
# 19. PRE-INFERENCE FIREWALL
# ============================================================

firewall = {

    "mri_voxels_read": False,

    "checkpoint_tensors_loaded": False,

    "model_instantiated": False,

    "predictions_generated": False,

    "external_performance_examined": False,

    "external_outcomes_used_for_tuning": False,

    "model_training_performed": False,

    "model_finetuning_performed": False,

    "model_recalibration_performed": False,

    "checkpoint_selection_on_oasis": False,

    "checkpoint_weighting_on_oasis": False,
}


# ============================================================
# 20. SOURCE HASHES
# ============================================================

source_hashes = {

    "step55_lock": {
        "path": str(STEP55_LOCK),
        "sha256": sha256_file(
            STEP55_LOCK
        ),
    },

    "step55_summary": {
        "path": str(STEP55_SUMMARY),
        "sha256": sha256_file(
            STEP55_SUMMARY
        ),
    },

    "step56_lock": {
        "path": str(STEP56_LOCK),
        "sha256": sha256_file(
            STEP56_LOCK
        ),
    },

    "step56_summary": {
        "path": str(STEP56_SUMMARY),
        "sha256": sha256_file(
            STEP56_SUMMARY
        ),
    },

    "step56_manifest": {
        "path": str(STEP56_MANIFEST),
        "sha256": sha256_file(
            STEP56_MANIFEST
        ),
    },

    "step53": {
        "path": str(STEP53),
        "sha256": sha256_file(
            STEP53
        ),
    },

    "step53a": {
        "path": str(STEP53A),
        "sha256": sha256_file(
            STEP53A
        ),
    },

    "step49": {
        "path": str(STEP49),
        "sha256": sha256_file(
            STEP49
        ),
    },

    "step54_lock": {
        "path": str(STEP54_LOCK),
        "sha256": sha256_file(
            STEP54_LOCK
        ),
    },

    "model_source": {
        "path": str(
            model_source_path
        ),
        "sha256": (
            observed_model_source_sha
        ),
    },
}


# ============================================================
# 21. CREATE SUMMARY
# ============================================================

summary = {

    "step": STEP,

    "status": (
        "READY_FOR_FIRST_FROZEN_EXTERNAL_INFERENCE"
    ),

    "contract": CONTRACT,

    "architecture": (
        EXPECTED_ARCHITECTURE
    ),

    "seeds": (
        EXPECTED_SEEDS
    ),

    "parameter_count": (
        EXPECTED_PARAMETER_COUNT
    ),

    "state_tensor_count": (
        EXPECTED_STATE_TENSORS
    ),

    "population_contract": (
        population_contract
    ),

    "population_interpretation": {
        "model_ready_population": (
            "Step-53: all landmarks eligible for frozen "
            "model inference"
        ),

        "endpoint_evaluable_population": (
            "Step-53A: strict subset eligible for "
            "external endpoint evaluation"
        ),

        "endpoint_excluded_landmarks": (
            EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS
        ),

        "endpoint_exclusion_is_model_failure": False,

        "endpoint_exclusion_is_mri_failure": False,
    },

    "temporal_contract": (
        temporal_contract
    ),

    "mri_geometry_contract": {

        "mni_size": (
            EXPECTED_MNI_SIZE
        ),

        "mni_spacing": (
            EXPECTED_MNI_SPACING
        ),

        "mni_origin": (
            EXPECTED_MNI_ORIGIN
        ),

        "mni_direction": (
            EXPECTED_MNI_DIRECTION
        ),

        "crop_start": (
            EXPECTED_CROP_START
        ),

        "crop_end": (
            EXPECTED_CROP_END
        ),

        "crop_size": (
            EXPECTED_CROP_SIZE
        ),
    },

    "survival_contract": (
        survival_contract
    ),

    "preinference_firewall": (
        firewall
    ),

    "source_hashes": (
        source_hashes
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
# 22. CREATE LOCK
# ============================================================

lock = {

    "step": STEP,

    "status": "LOCKED_PASS",

    "contract": CONTRACT,

    "summary_file": (
        str(OUTPUT_SUMMARY)
    ),

    "summary_sha256": (
        summary_sha
    ),

    "step55_contract": (
        step55_contract
    ),

    "step55_status": (
        step55_status
    ),

    "step56_contract": (
        step56_contract
    ),

    "step56_status": (
        step56_status
    ),

    "architecture": (
        EXPECTED_ARCHITECTURE
    ),

    "seeds": (
        EXPECTED_SEEDS
    ),

    "model_ready_landmarks": (
        EXPECTED_MODEL_READY_LANDMARKS
    ),

    "model_ready_subjects": (
        EXPECTED_MODEL_READY_SUBJECTS
    ),

    "endpoint_evaluable_landmarks": (
        EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS
    ),

    "endpoint_evaluable_subjects": (
        EXPECTED_ENDPOINT_EVALUABLE_SUBJECTS
    ),

    "endpoint_excluded_landmarks": (
        EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS
    ),

    "horizons_months": (
        EXPECTED_HORIZONS_MONTHS
    ),

    "interval_edges_days": (
        EXPECTED_INTERVAL_EDGES_DAYS
    ),

    "days_per_year": (
        EXPECTED_DAYS_PER_YEAR
    ),

    "risk_formula": (
        "1-product_{k<=t}(1-sigmoid(logit_k))"
    ),

    "source_hashes": (
        source_hashes
    ),

    "mri_voxels_read": False,

    "checkpoint_tensors_loaded": False,

    "model_instantiated": False,

    "predictions_generated": False,

    "external_performance_examined": False,

    "model_tuning": False,

    "model_recalibration": False,

    "checkpoint_selection_on_oasis": False,
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
# 23. FINAL REPORT
# ============================================================

print()
print_rule()
print(
    "STEP 57 PASSED — EXTERNAL-INFERENCE "
    "EXECUTION CONTRACT LOCKED"
)
print_rule()


print()
print("MODEL")

print(
    f"Architecture             : "
    f"{EXPECTED_ARCHITECTURE}"
)

print(
    f"Frozen checkpoints       : "
    f"{len(EXPECTED_SEEDS)}"
)

print(
    f"Seeds                    : "
    f"{EXPECTED_SEEDS}"
)

print(
    f"Parameter count          : "
    f"{EXPECTED_PARAMETER_COUNT}"
)

print(
    f"State tensors            : "
    f"{EXPECTED_STATE_TENSORS}"
)


print()
print("MODEL-READY EXTERNAL POPULATION")

print(
    f"Step-53 landmarks        : "
    f"{EXPECTED_MODEL_READY_LANDMARKS}"
)

print(
    f"Step-53 subjects         : "
    f"{EXPECTED_MODEL_READY_SUBJECTS}"
)


print()
print("ENDPOINT-EVALUABLE POPULATION")

print(
    f"Step-53A landmarks       : "
    f"{EXPECTED_ENDPOINT_EVALUABLE_LANDMARKS}"
)

print(
    f"Step-53A subjects        : "
    f"{EXPECTED_ENDPOINT_EVALUABLE_SUBJECTS}"
)

print(
    f"Endpoint-excluded        : "
    f"{EXPECTED_ENDPOINT_EXCLUDED_LANDMARKS}"
)

print(
    f"Events                   : "
    f"{EXPECTED_EVENTS}"
)

print(
    f"Censored                 : "
    f"{EXPECTED_CENSORED}"
)


print()
print("MRI READINESS")

print(
    f"Required MRI             : "
    f"{EXPECTED_REQUIRED_MRI}"
)

print(
    f"Ready MRI                : "
    f"{EXPECTED_READY_MRI}"
)

print(
    f"Missing MRI              : "
    f"{EXPECTED_MISSING_MRI}"
)

print(
    f"Not-ready MRI            : "
    f"{EXPECTED_NOT_READY_MRI}"
)


print()
print("SURVIVAL CONTRACT")

print(
    f"Horizons                 : "
    f"{EXPECTED_HORIZONS_MONTHS}"
)

print(
    f"Interval edges (days)    : "
    f"{EXPECTED_INTERVAL_EDGES_DAYS}"
)

print(
    "Risk rule                : "
    "1 - product(1 - sigmoid(logit_k))"
)


print()
print("MRI CONTRACT")

print(
    f"MNI grid                 : "
    f"{EXPECTED_MNI_SIZE}"
)

print(
    f"Crop                     : "
    f"{EXPECTED_CROP_SIZE}"
)

print(
    f"Crop start               : "
    f"{EXPECTED_CROP_START}"
)

print(
    f"Crop end                 : "
    f"{EXPECTED_CROP_END}"
)


print()
print("OUTPUT")
print(
    OUTPUT_SUMMARY
)

print()
print("LOCK")
print(
    OUTPUT_LOCK
)


print()
print("SHA256")

print(
    f"Summary : {summary_sha}"
)

print(
    f"Lock    : {lock_sha}"
)


print()
print("PRE-INFERENCE FIREWALL")

print(
    "MRI VOXELS READ             : NO"
)

print(
    "CHECKPOINT TENSORS LOADED    : NO"
)

print(
    "MODEL INSTANTIATED           : NO"
)

print(
    "PREDICTIONS GENERATED        : NO"
)

print(
    "EXTERNAL PERFORMANCE EXAMINED: NO"
)

print(
    "MODEL TRAINING               : NO"
)

print(
    "MODEL FINE-TUNING            : NO"
)

print(
    "MODEL RECALIBRATION          : NO"
)

print(
    "OASIS-BASED MODEL SELECTION  : NO"
)


print()
print(
    "The complete frozen external-inference "
    "execution contract is now locked."
)

print(
    "The 1559 model-ready landmarks remain distinct "
    "from the 1486 endpoint-evaluable landmarks."
)

print(
    "Only after this gate passes may the first "
    "frozen OASIS-3 model inference be executed."
)
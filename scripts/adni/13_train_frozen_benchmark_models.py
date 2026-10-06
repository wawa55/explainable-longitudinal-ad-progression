from pathlib import Path
import ast
import copy
import gc
import json
import math
import os
import random
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


# ==============================================================================================================
# ADNI — TRAIN FROZEN BENCHMARK MODELS B0 / B1 / B2
#
# Frozen benchmark suite:
#
#   B0 — Landmark-only 3D CNN survival
#        latest MRI only
#
#   B1 — Longitudinal 3D CNN + masked mean pooling survival
#        all MRI visits + continuous-time encoding, no sequence model
#
#   B2 — Longitudinal 3D CNN + GRU survival
#        all MRI visits + continuous-time encoding + single-layer GRU
#
# M0 is already frozen and is NOT retrained here.
#
# COMMON RULES
# ------------
# Same frozen subject split.
# Same longitudinal landmark dataset.
# Same 60-month discrete-time survival target.
# Same informative subject-balanced TRAIN sampler.
# Same validation-only model selection.
# Same five seeds.
# No TEST inference in this script.
#
# Resume-safe:
#   - Completed benchmark seed runs are skipped.
#   - BEST and LAST checkpoints are written separately.
#   - Training history is saved after every epoch.
# ==============================================================================================================


# ==============================================================================================================
# 1. PATHS
# ==============================================================================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "06_splits"
MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"

LONGITUDINAL_DATASET_FILE = (
    SPLIT_DIR / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
)

BENCHMARK_LOCK_FILE = (
    RESULT_DIR / "ADNI_BENCHMARK_SUITE_LOCK_V1.json"
)

BENCHMARK_MODEL_DIR = (
    MODEL_DIR / "benchmark_v1"
)

BENCHMARK_RESULT_DIR = (
    RESULT_DIR / "benchmark_v1"
)

BENCHMARK_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BENCHMARK_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MASTER_SUMMARY_CSV = (
    BENCHMARK_RESULT_DIR / "ADNI_BENCHMARK_TRAINING_SUMMARY_V1.csv"
)

RUN_LOG = (
    BENCHMARK_RESULT_DIR / "ADNI_BENCHMARK_TRAINING_V1.log"
)


# ==============================================================================================================
# 2. FROZEN TRAINING SPEC
# ==============================================================================================================

SEEDS = [
    2026,
    3407,
    5813,
    7291,
    9109,
]

MODELS = [
    "B0",
    "B1",
    "B2",
]

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2
GRADIENT_CLIP_NORM = 1.0

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

DAYS_PER_YEAR = 365.25
ADMIN_CUTOFF_DAYS = 5.0 * DAYS_PER_YEAR

INTERVAL_EDGES_DAYS = np.asarray(
    [
        0.0,
        1.0 * DAYS_PER_YEAR,
        2.0 * DAYS_PER_YEAR,
        3.0 * DAYS_PER_YEAR,
        4.0 * DAYS_PER_YEAR,
        5.0 * DAYS_PER_YEAR,
    ],
    dtype=np.float64,
)

EXPECTED = {
    "subjects": 972,
    "landmarks": 3634,
    "train_subjects": 680,
    "validation_subjects": 146,
    "test_subjects": 146,
    "train_landmarks": 2530,
    "validation_landmarks": 521,
    "test_landmarks": 583,
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

torch.backends.cudnn.benchmark = False

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
)


# ==============================================================================================================
# 3. LOGGING
# ==============================================================================================================

def log(message=""):

    text = str(message)

    print(
        text,
        flush=True,
    )

    with open(
        RUN_LOG,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )

        f.flush()


if not RUN_LOG.exists():

    RUN_LOG.write_text(
        "",
        encoding="utf-8",
    )


# ==============================================================================================================
# 4. HELPERS
# ==============================================================================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def parse_list_field(value):

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if pd.isna(
        value
    ):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list,
        ):
            return parsed

    except Exception:
        pass

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            (
                list,
                tuple,
            ),
        ):
            return list(
                parsed
            )

    except Exception:
        pass

    raise ValueError(
        f"Unable to parse list field: {text[:300]}"
    )


def center_crop_3d(
    volume,
    target_shape=MODEL_SPATIAL_SHAPE,
):

    source_shape = np.asarray(
        volume.shape,
        dtype=int,
    )

    target_shape = np.asarray(
        target_shape,
        dtype=int,
    )

    start = (
        source_shape
        -
        target_shape
    ) // 2

    end = (
        start
        +
        target_shape
    )

    cropped = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if (
        cropped.shape
        !=
        tuple(
            target_shape
        )
    ):

        raise RuntimeError(
            f"Crop mismatch: {cropped.shape} != {tuple(target_shape)}"
        )

    return cropped


def load_mri_array(path):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"MRI missing:\n{path}"
        )

    nii = nib.load(
        str(
            path
        )
    )

    volume = np.asarray(
        nii.dataobj,
        dtype=np.float32,
    )

    if (
        volume.shape
        !=
        SOURCE_SPATIAL_SHAPE
    ):

        raise RuntimeError(
            f"Unexpected MRI shape {volume.shape}; "
            f"expected {SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(
        volume
    ).all():

        raise RuntimeError(
            f"Non-finite MRI values:\n{path}"
        )

    return center_crop_3d(
        volume
    )


def make_discrete_survival_target(
    event,
    time_days,
):

    event = int(
        event
    )

    time_days = float(
        time_days
    )

    y = np.zeros(
        5,
        dtype=np.float32,
    )

    mask = np.zeros(
        5,
        dtype=np.float32,
    )

    # Administrative truncation at 60 months.
    observed_time = min(
        time_days,
        ADMIN_CUTOFF_DAYS,
    )

    event_within_60m = (
        event == 1
        and
        time_days <= ADMIN_CUTOFF_DAYS
    )

    if event_within_60m:

        # Right-closed event intervals:
        # (0,12], (12,24], ..., (48,60]
        event_interval = int(
            np.searchsorted(
                INTERVAL_EDGES_DAYS[
                    1:
                ],
                time_days,
                side="left",
            )
        )

        event_interval = min(
            max(
                event_interval,
                0,
            ),
            4,
        )

        # Fully completed event-free intervals before event.
        if event_interval > 0:

            mask[
                :event_interval
            ] = 1.0

            y[
                :event_interval
            ] = 0.0

        # Event interval is observed and positive.
        mask[
            event_interval
        ] = 1.0

        y[
            event_interval
        ] = 1.0

    else:

        # Censored or event after 60m:
        # only fully completed intervals are observed.
        for j in range(
            5
        ):

            upper = INTERVAL_EDGES_DAYS[
                j + 1
            ]

            if observed_time >= upper:

                mask[
                    j
                ] = 1.0

                y[
                    j
                ] = 0.0

    return (
        y,
        mask,
    )


# ==============================================================================================================
# 5. DATASET
# ==============================================================================================================

class LongitudinalBenchmarkDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        model_code,
    ):

        self.df = dataframe.reset_index(
            drop=True
        ).copy()

        self.model_code = str(
            model_code
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        subject_id = str(
            row[
                "subject_id"
            ]
        )

        landmark_id = str(
            row[
                "landmark_id"
            ]
        )

        event = int(
            row[
                "event"
            ]
        )

        time_days = float(
            row[
                "time_to_event_or_censor_days"
            ]
        )

        image_paths = parse_list_field(
            row[
                "image_paths"
            ]
        )

        days_from_first = np.asarray(
            parse_list_field(
                row[
                    "days_from_first_mri"
                ]
            ),
            dtype=np.float32,
        )

        days_before_landmark = np.asarray(
            parse_list_field(
                row[
                    "days_before_landmark"
                ]
            ),
            dtype=np.float32,
        )

        if self.model_code == "B0":

            image_paths = [
                image_paths[
                    -1
                ]
            ]

            days_from_first = np.asarray(
                [
                    days_from_first[
                        -1
                    ]
                ],
                dtype=np.float32,
            )

            days_before_landmark = np.asarray(
                [
                    days_before_landmark[
                        -1
                    ]
                ],
                dtype=np.float32,
            )

        images = []

        for p in image_paths:

            arr = load_mri_array(
                p
            )

            images.append(
                torch.from_numpy(
                    np.ascontiguousarray(
                        arr
                    )
                ).unsqueeze(
                    0
                )
            )

        images = torch.stack(
            images,
            dim=0,
        )

        years_from_first = torch.tensor(
            days_from_first
            /
            DAYS_PER_YEAR,
            dtype=torch.float32,
        )

        years_before_landmark = torch.tensor(
            days_before_landmark
            /
            DAYS_PER_YEAR,
            dtype=torch.float32,
        )

        y, mask = make_discrete_survival_target(
            event,
            time_days,
        )

        return {
            "subject_id":
                subject_id,

            "landmark_id":
                landmark_id,

            "images":
                images,

            "years_from_first":
                years_from_first,

            "years_before_landmark":
                years_before_landmark,

            "event":
                torch.tensor(
                    event,
                    dtype=torch.long,
                ),

            "time_days":
                torch.tensor(
                    time_days,
                    dtype=torch.float32,
                ),

            "target":
                torch.tensor(
                    y,
                    dtype=torch.float32,
                ),

            "target_mask":
                torch.tensor(
                    mask,
                    dtype=torch.float32,
                ),
        }


def collate_longitudinal(
    batch,
):

    batch_size = len(
        batch
    )

    max_t = max(
        item[
            "images"
        ].shape[
            0
        ]
        for item in batch
    )

    spatial = batch[
        0
    ][
        "images"
    ].shape[
        1:
    ]

    images = torch.zeros(
        (
            batch_size,
            max_t,
            *spatial,
        ),
        dtype=torch.float32,
    )

    years_from_first = torch.zeros(
        (
            batch_size,
            max_t,
        ),
        dtype=torch.float32,
    )

    years_before_landmark = torch.zeros(
        (
            batch_size,
            max_t,
        ),
        dtype=torch.float32,
    )

    visit_mask = torch.zeros(
        (
            batch_size,
            max_t,
        ),
        dtype=torch.bool,
    )

    targets = torch.stack(
        [
            item[
                "target"
            ]
            for item in batch
        ],
        dim=0,
    )

    target_masks = torch.stack(
        [
            item[
                "target_mask"
            ]
            for item in batch
        ],
        dim=0,
    )

    event = torch.stack(
        [
            item[
                "event"
            ]
            for item in batch
        ],
        dim=0,
    )

    time_days = torch.stack(
        [
            item[
                "time_days"
            ]
            for item in batch
        ],
        dim=0,
    )

    subject_ids = []
    landmark_ids = []

    for i, item in enumerate(
        batch
    ):

        t = item[
            "images"
        ].shape[
            0
        ]

        images[
            i,
            :t,
        ] = item[
            "images"
        ]

        years_from_first[
            i,
            :t,
        ] = item[
            "years_from_first"
        ]

        years_before_landmark[
            i,
            :t,
        ] = item[
            "years_before_landmark"
        ]

        visit_mask[
            i,
            :t,
        ] = True

        subject_ids.append(
            item[
                "subject_id"
            ]
        )

        landmark_ids.append(
            item[
                "landmark_id"
            ]
        )

    return {
        "subject_id":
            subject_ids,

        "landmark_id":
            landmark_ids,

        "images":
            images,

        "years_from_first":
            years_from_first,

        "years_before_landmark":
            years_before_landmark,

        "visit_mask":
            visit_mask,

        "event":
            event,

        "time_days":
            time_days,

        "target":
            targets,

        "target_mask":
            target_masks,
    }


# ==============================================================================================================
# 6. INFORMATIVE SUBJECT-BALANCED TRAIN SAMPLER
# ==============================================================================================================

class InformativeSubjectBalancedSampler(
    Sampler
):

    def __init__(
        self,
        dataframe,
        target_masks,
        seed,
    ):

        self.df = dataframe.reset_index(
            drop=True
        ).copy()

        self.seed = int(
            seed
        )

        informative = np.asarray(
            [
                float(
                    np.asarray(
                        m
                    ).sum()
                )
                >
                0.0
                for m in target_masks
            ],
            dtype=bool,
        )

        self.subject_to_indices = {}

        for idx, row in self.df.iterrows():

            if not informative[
                idx
            ]:

                continue

            subject_id = str(
                row[
                    "subject_id"
                ]
            )

            self.subject_to_indices.setdefault(
                subject_id,
                [],
            ).append(
                idx
            )

        self.subjects = sorted(
            self.subject_to_indices.keys()
        )

        self.epoch = 0


    def set_epoch(
        self,
        epoch,
    ):

        self.epoch = int(
            epoch
        )


    def __iter__(
        self
    ):

        rng = random.Random(
            self.seed
            +
            self.epoch
        )

        subjects = self.subjects.copy()

        rng.shuffle(
            subjects
        )

        indices = []

        for subject_id in subjects:

            candidates = self.subject_to_indices[
                subject_id
            ]

            # Deterministic cyclic use over epochs.
            position = (
                self.epoch
                %
                len(
                    candidates
                )
            )

            indices.append(
                candidates[
                    position
                ]
            )

        return iter(
            indices
        )


    def __len__(
        self
    ):

        return len(
            self.subjects
        )


# ==============================================================================================================
# 7. LOSS
# ==============================================================================================================

class DiscreteTimeSurvivalNLL(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )


    def forward(
        self,
        logits,
        target,
        target_mask,
    ):

        element = self.bce(
            logits,
            target,
        )

        observed = (
            element
            *
            target_mask
        ).sum(
            dim=1
        )

        informative = (
            target_mask.sum(
                dim=1
            )
            >
            0
        )

        if not bool(
            informative.any()
        ):

            return logits.sum() * 0.0

        return observed[
            informative
        ].mean()


# ==============================================================================================================
# 8. MODELS
# ==============================================================================================================

def make_template_modules():

    template = ExactM0LongitudinalSurvivalModel_V3()

    return {
        "mri_encoder":
            copy.deepcopy(
                template.mri_encoder
            ),

        "time_encoder":
            copy.deepcopy(
                template.time_encoder
            ),

        "final_norm":
            copy.deepcopy(
                template.final_norm
            ),

        "survival_head":
            copy.deepcopy(
                template.survival_head
            ),
    }


class B0LandmarkOnly3DCNN(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        modules = make_template_modules()

        self.mri_encoder = modules[
            "mri_encoder"
        ]

        self.final_norm = modules[
            "final_norm"
        ]

        self.survival_head = modules[
            "survival_head"
        ]


    def forward(
        self,
        images,
        years_from_first=None,
        years_before_landmark=None,
        visit_mask=None,
    ):

        # B0 dataset contains exactly one visit.
        x = self.mri_encoder(
            images[
                :,
                0,
            ]
        )

        x = self.final_norm(
            x
        )

        return self.survival_head(
            x
        )


class B1LongitudinalMeanPool(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        modules = make_template_modules()

        self.mri_encoder = modules[
            "mri_encoder"
        ]

        self.time_encoder = modules[
            "time_encoder"
        ]

        self.final_norm = modules[
            "final_norm"
        ]

        self.survival_head = modules[
            "survival_head"
        ]


    def forward(
        self,
        images,
        years_from_first,
        years_before_landmark,
        visit_mask,
    ):

        batch_size, max_t = images.shape[
            :2
        ]

        flat = images.reshape(
            batch_size
            *
            max_t,
            *images.shape[
                2:
            ],
        )

        flat_mask = visit_mask.reshape(
            -1
        )

        embeddings = torch.zeros(
            (
                batch_size
                *
                max_t,
                256,
            ),
            device=images.device,
            dtype=images.dtype,
        )

        if bool(
            flat_mask.any()
        ):

            embeddings[
                flat_mask
            ] = self.mri_encoder(
                flat[
                    flat_mask
                ]
            )

        embeddings = embeddings.reshape(
            batch_size,
            max_t,
            256,
        )

        time_embeddings = self.time_encoder(
            years_from_first,
            years_before_landmark,
        )

        x = (
            embeddings
            +
            time_embeddings
        )

        mask_float = visit_mask.unsqueeze(
            -1
        ).to(
            x.dtype
        )

        pooled = (
            (
                x
                *
                mask_float
            ).sum(
                dim=1
            )
            /
            mask_float.sum(
                dim=1
            ).clamp_min(
                1.0
            )
        )

        pooled = self.final_norm(
            pooled
        )

        return self.survival_head(
            pooled
        )


class B2LongitudinalGRU(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        modules = make_template_modules()

        self.mri_encoder = modules[
            "mri_encoder"
        ]

        self.time_encoder = modules[
            "time_encoder"
        ]

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )

        self.final_norm = modules[
            "final_norm"
        ]

        self.survival_head = modules[
            "survival_head"
        ]


    def forward(
        self,
        images,
        years_from_first,
        years_before_landmark,
        visit_mask,
    ):

        batch_size, max_t = images.shape[
            :2
        ]

        flat = images.reshape(
            batch_size
            *
            max_t,
            *images.shape[
                2:
            ],
        )

        flat_mask = visit_mask.reshape(
            -1
        )

        embeddings = torch.zeros(
            (
                batch_size
                *
                max_t,
                256,
            ),
            device=images.device,
            dtype=images.dtype,
        )

        if bool(
            flat_mask.any()
        ):

            embeddings[
                flat_mask
            ] = self.mri_encoder(
                flat[
                    flat_mask
                ]
            )

        embeddings = embeddings.reshape(
            batch_size,
            max_t,
            256,
        )

        time_embeddings = self.time_encoder(
            years_from_first,
            years_before_landmark,
        )

        x = embeddings + time_embeddings

        lengths = visit_mask.sum(
            dim=1
        ).to(
            dtype=torch.long
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(
            packed
        )

        pooled = hidden[
            -1
        ]

        pooled = self.final_norm(
            pooled
        )

        return self.survival_head(
            pooled
        )


def build_model(
    model_code,
):

    if model_code == "B0":

        return B0LandmarkOnly3DCNN()

    if model_code == "B1":

        return B1LongitudinalMeanPool()

    if model_code == "B2":

        return B2LongitudinalGRU()

    raise ValueError(
        model_code
    )


# ==============================================================================================================
# 9. EPOCH FUNCTIONS
# ==============================================================================================================

def move_batch_to_device(
    batch,
):

    return {
        "images":
            batch[
                "images"
            ].to(
                DEVICE,
                non_blocking=False,
            ),

        "years_from_first":
            batch[
                "years_from_first"
            ].to(
                DEVICE
            ),

        "years_before_landmark":
            batch[
                "years_before_landmark"
            ].to(
                DEVICE
            ),

        "visit_mask":
            batch[
                "visit_mask"
            ].to(
                DEVICE
            ),

        "target":
            batch[
                "target"
            ].to(
                DEVICE
            ),

        "target_mask":
            batch[
                "target_mask"
            ].to(
                DEVICE
            ),
    }


def train_one_epoch(
    model,
    loader,
    sampler,
    optimizer,
    criterion,
    epoch,
):

    model.train()

    sampler.set_epoch(
        epoch
    )

    total_loss = 0.0
    n_batches = 0

    for batch in loader:

        b = move_batch_to_device(
            batch
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images=b[
                "images"
            ],
            years_from_first=b[
                "years_from_first"
            ],
            years_before_landmark=b[
                "years_before_landmark"
            ],
            visit_mask=b[
                "visit_mask"
            ],
        )

        loss = criterion(
            logits,
            b[
                "target"
            ],
            b[
                "target_mask"
            ],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )

        optimizer.step()

        total_loss += float(
            loss.item()
        )

        n_batches += 1

        del (
            b,
            logits,
            loss,
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return total_loss / max(
        n_batches,
        1,
    )


@torch.inference_mode()
def evaluate_nll(
    model,
    loader,
    criterion,
):

    model.eval()

    total = 0.0
    n = 0

    for batch in loader:

        b = move_batch_to_device(
            batch
        )

        logits = model(
            images=b[
                "images"
            ],
            years_from_first=b[
                "years_from_first"
            ],
            years_before_landmark=b[
                "years_before_landmark"
            ],
            visit_mask=b[
                "visit_mask"
            ],
        )

        target_mask = b[
            "target_mask"
        ]

        informative = (
            target_mask.sum(
                dim=1
            )
            >
            0
        )

        if bool(
            informative.any()
        ):

            loss = criterion(
                logits[
                    informative
                ],
                b[
                    "target"
                ][
                    informative
                ],
                target_mask[
                    informative
                ],
            )

            batch_n = int(
                informative.sum().item()
            )

            total += float(
                loss.item()
            ) * batch_n

            n += batch_n

        del (
            b,
            logits,
            target_mask,
            informative,
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return total / max(
        n,
        1,
    )


# ==============================================================================================================
# 10. MAIN
# ==============================================================================================================

def main():

    log("=" * 120)
    log("ADNI — FROZEN BENCHMARK TRAINING B0 / B1 / B2")
    log("=" * 120)

    log(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        log(
            f"CUDA build: {torch.version.cuda}"
        )

    if not BENCHMARK_LOCK_FILE.exists():

        raise FileNotFoundError(
            BENCHMARK_LOCK_FILE
        )

    if not LONGITUDINAL_DATASET_FILE.exists():

        raise FileNotFoundError(
            LONGITUDINAL_DATASET_FILE
        )

    with open(
        BENCHMARK_LOCK_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        lock = json.load(
            f
        )

    if lock.get(
        "status"
    ) != "FROZEN BEFORE BENCHMARK TRAINING":

        raise RuntimeError(
            "Benchmark suite lock status mismatch."
        )

    df = pd.read_csv(
        LONGITUDINAL_DATASET_FILE
    )

    if (
        len(
            df
        )
        !=
        EXPECTED[
            "landmarks"
        ]
    ):

        raise RuntimeError(
            "Longitudinal landmark count mismatch."
        )

    train_df = df.loc[
        df[
            "split"
        ].astype(
            str
        ).str.upper()
        ==
        "TRAIN"
    ].copy().reset_index(
        drop=True
    )

    val_df = df.loc[
        df[
            "split"
        ].astype(
            str
        ).str.upper()
        ==
        "VALIDATION"
    ].copy().reset_index(
        drop=True
    )

    test_df = df.loc[
        df[
            "split"
        ].astype(
            str
        ).str.upper()
        ==
        "TEST"
    ].copy().reset_index(
        drop=True
    )

    log(
        f"TRAIN landmarks: {len(train_df)} | subjects: {train_df['subject_id'].nunique()}"
    )

    log(
        f"VALIDATION landmarks: {len(val_df)} | subjects: {val_df['subject_id'].nunique()}"
    )

    log(
        f"TEST landmarks present but untouched: {len(test_df)} | subjects: {test_df['subject_id'].nunique()}"
    )

    if (
        len(
            train_df
        )
        !=
        EXPECTED[
            "train_landmarks"
        ]
        or
        len(
            val_df
        )
        !=
        EXPECTED[
            "validation_landmarks"
        ]
        or
        len(
            test_df
        )
        !=
        EXPECTED[
            "test_landmarks"
        ]
    ):

        raise RuntimeError(
            "Frozen landmark split mismatch."
        )

    # Precompute TRAIN masks only for sampler.
    train_target_masks = []

    for _, row in train_df.iterrows():

        _, mask = make_discrete_survival_target(
            row[
                "event"
            ],
            row[
                "time_to_event_or_censor_days"
            ],
        )

        train_target_masks.append(
            mask
        )

    master_rows = (
        pd.read_csv(
            MASTER_SUMMARY_CSV
        ).to_dict(
            orient="records"
        )
        if MASTER_SUMMARY_CSV.exists()
        else []
    )

    completed = {
        (
            str(
                r[
                    "model"
                ]
            ),
            int(
                r[
                    "seed"
                ]
            ),
        )
        for r in master_rows
        if str(
            r.get(
                "status",
                ""
            )
        )
        ==
        "COMPLETE"
    }

    total_runs = len(
        MODELS
    ) * len(
        SEEDS
    )

    log(
        f"Expected benchmark training runs: {total_runs}"
    )

    log(
        f"Already complete: {len(completed)}"
    )

    for model_code in MODELS:

        log("")
        log("=" * 120)
        log(
            f"MODEL {model_code}"
        )
        log("=" * 120)

        train_dataset = LongitudinalBenchmarkDataset(
            train_df,
            model_code=model_code,
        )

        val_dataset = LongitudinalBenchmarkDataset(
            val_df,
            model_code=model_code,
        )

        for seed in SEEDS:

            run_key = (
                model_code,
                seed,
            )

            if run_key in completed:

                log(
                    f"{model_code} seed {seed}: already complete -> SKIP"
                )

                continue

            set_seed(
                seed
            )

            sampler = InformativeSubjectBalancedSampler(
                dataframe=train_df,
                target_masks=train_target_masks,
                seed=seed,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate_longitudinal,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate_longitudinal,
            )

            model = build_model(
                model_code
            ).to(
                DEVICE
            )

            parameter_count = sum(
                p.numel()
                for p in model.parameters()
            )

            criterion = DiscreteTimeSurvivalNLL()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            best_val = math.inf
            best_epoch = None
            epochs_without_improvement = 0

            best_checkpoint = (
                BENCHMARK_MODEL_DIR
                /
                f"ADNI_{model_code}_SEED_{seed}_BEST_V1.pt"
            )

            last_checkpoint = (
                BENCHMARK_MODEL_DIR
                /
                f"ADNI_{model_code}_SEED_{seed}_LAST_V1.pt"
            )

            history_file = (
                BENCHMARK_RESULT_DIR
                /
                f"ADNI_{model_code}_SEED_{seed}_TRAINING_HISTORY_V1.csv"
            )

            history_rows = []

            log("")
            log(
                f"{model_code} | seed={seed} | parameters={parameter_count:,}"
            )

            run_start = time.time()

            for epoch in range(
                1,
                MAX_EPOCHS + 1,
            ):

                epoch_start = time.time()

                train_nll = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    sampler=sampler,
                    optimizer=optimizer,
                    criterion=criterion,
                    epoch=epoch - 1,
                )

                val_nll = evaluate_nll(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                )

                elapsed_epoch = (
                    time.time()
                    -
                    epoch_start
                )

                improved = (
                    val_nll
                    <
                    best_val
                    -
                    1e-10
                )

                if improved:

                    best_val = val_nll
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    torch.save(
                        {
                            "model_code":
                                model_code,

                            "seed":
                                seed,

                            "epoch":
                                epoch,

                            "validation_loss":
                                val_nll,

                            "parameter_count":
                                parameter_count,

                            "model_state_dict":
                                model.state_dict(),

                            "optimizer_state_dict":
                                optimizer.state_dict(),

                            "benchmark_lock":
                                str(
                                    BENCHMARK_LOCK_FILE
                                ),
                        },
                        best_checkpoint,
                    )

                else:

                    epochs_without_improvement += 1

                torch.save(
                    {
                        "model_code":
                            model_code,

                        "seed":
                            seed,

                        "epoch":
                            epoch,

                        "validation_loss":
                            val_nll,

                        "parameter_count":
                            parameter_count,

                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "benchmark_lock":
                            str(
                                BENCHMARK_LOCK_FILE
                            ),
                    },
                    last_checkpoint,
                )

                history_rows.append(
                    {
                        "model":
                            model_code,

                        "seed":
                            seed,

                        "epoch":
                            epoch,

                        "train_nll":
                            train_nll,

                        "validation_nll":
                            val_nll,

                        "best_validation_nll_so_far":
                            best_val,

                        "best_epoch_so_far":
                            best_epoch,

                        "improved":
                            improved,

                        "epoch_seconds":
                            elapsed_epoch,
                    }
                )

                pd.DataFrame(
                    history_rows
                ).to_csv(
                    history_file,
                    index=False,
                )

                log(
                    f"{model_code} | seed={seed} | "
                    f"epoch={epoch:02d} | "
                    f"train NLL={train_nll:.6f} | "
                    f"val NLL={val_nll:.6f} | "
                    f"best={best_val:.6f}@{best_epoch} | "
                    f"patience={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} | "
                    f"{elapsed_epoch:.1f}s"
                )

                if (
                    epochs_without_improvement
                    >=
                    EARLY_STOPPING_PATIENCE
                ):

                    log(
                        f"{model_code} seed {seed}: EARLY STOP at epoch {epoch}"
                    )

                    break

            run_seconds = (
                time.time()
                -
                run_start
            )

            if not best_checkpoint.exists():

                raise RuntimeError(
                    f"No best checkpoint saved for {model_code} seed {seed}"
                )

            master_rows = [
                r
                for r in master_rows
                if not (
                    str(
                        r.get(
                            "model"
                        )
                    )
                    ==
                    model_code
                    and
                    int(
                        r.get(
                            "seed"
                        )
                    )
                    ==
                    seed
                )
            ]

            master_rows.append(
                {
                    "status":
                        "COMPLETE",

                    "model":
                        model_code,

                    "seed":
                        seed,

                    "parameter_count":
                        parameter_count,

                    "best_epoch":
                        best_epoch,

                    "best_validation_nll":
                        best_val,

                    "best_checkpoint":
                        str(
                            best_checkpoint
                        ),

                    "last_checkpoint":
                        str(
                            last_checkpoint
                        ),

                    "history_file":
                        str(
                            history_file
                        ),

                    "training_seconds":
                        run_seconds,

                    "test_inference_performed":
                        False,
                }
            )

            pd.DataFrame(
                master_rows
            ).sort_values(
                [
                    "model",
                    "seed",
                ]
            ).to_csv(
                MASTER_SUMMARY_CSV,
                index=False,
            )

            completed.add(
                run_key
            )

            log(
                f"{model_code} seed {seed}: COMPLETE | "
                f"best epoch={best_epoch} | "
                f"best val NLL={best_val:.6f} | "
                f"{run_seconds / 60.0:.2f} min"
            )

            del (
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                sampler,
            )

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    summary = pd.read_csv(
        MASTER_SUMMARY_CSV
    )

    complete_summary = summary.loc[
        summary[
            "status"
        ]
        ==
        "COMPLETE"
    ].copy()

    log("")
    log("=" * 120)
    log("BENCHMARK TRAINING SUMMARY")
    log("=" * 120)

    log(
        complete_summary[
            [
                "model",
                "seed",
                "parameter_count",
                "best_epoch",
                "best_validation_nll",
                "training_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    if (
        len(
            complete_summary
        )
        !=
        total_runs
    ):

        log("")
        log(
            f"BENCHMARK TRAINING INCOMPLETE: "
            f"{len(complete_summary)}/{total_runs}"
        )

        log(
            "Re-run this same script to resume."
        )

        return

    log("")
    log("=" * 120)
    log("FROZEN BENCHMARK TRAINING: COMPLETE")
    log("=" * 120)

    log(
        "B0 seeds complete: 5/5"
    )

    log(
        "B1 seeds complete: 5/5"
    )

    log(
        "B2 seeds complete: 5/5"
    )

    log(
        "TEST inference performed: NO"
    )

    log(
        "Next step: benchmark validation audit, then one frozen TEST evaluation"
    )

    log("=" * 120)


if __name__ == "__main__":

    try:

        main()

    except Exception:

        log("")
        log("=" * 120)
        log("FROZEN BENCHMARK TRAINING: FAILED")
        log("=" * 120)

        log(
            traceback.format_exc()
        )

        log(
            "This script is resume-safe. Fix the error and rerun the same command."
        )

        raise

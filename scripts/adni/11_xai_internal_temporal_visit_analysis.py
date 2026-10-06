from pathlib import Path
import ast
import gc
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn

from exact_m0_model_v3 import ExactM0LongitudinalSurvivalModel_V3


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "07_models"
RESULT_DIR = ROOT / "08_results"
PRODUCTION_DIR = RESULT_DIR / "ADNI_XAI_TEST_PRODUCTION_V1"

COHORT_CSV = RESULT_DIR / "ADNI_XAI_FROZEN_TEST_COHORT_V1.csv"
XAI_CONFIG_LOCK = RESULT_DIR / "ADNI_XAI_PRODUCTION_CONFIG_LOCK_V1.json"
FAITHFULNESS_JSON = PRODUCTION_DIR / "ADNI_XAI_FAITHFULNESS_SUMMARY_V1.json"
STABILITY_JSON = PRODUCTION_DIR / "ADNI_XAI_CROSS_SEED_STABILITY_SUMMARY_V1.json"

OUTPUT_VISITS_CSV = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_VISIT_CONTRIBUTIONS_V1.csv"
OUTPUT_RUNS_CSV = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_RUN_SUMMARY_V1.csv"
OUTPUT_SUBJECT_CSV = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_SUBJECT_SUMMARY_V1.csv"
OUTPUT_COHORT_CSV = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_COHORT_SUMMARY_V1.csv"
OUTPUT_EVENT_CSV = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_EVENT_STATUS_SUMMARY_V1.csv"
OUTPUT_JSON = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_SUMMARY_V1.json"
OUTPUT_LOG = PRODUCTION_DIR / "ADNI_XAI_INTERNAL_TEMPORAL_V1.log"

CHECKPOINTS = {
    2026: MODEL_DIR / "ADNI_LONGITUDINAL_SURVIVAL_BEST_V1.pt",
    3407: MODEL_DIR / "ADNI_M0_SEED_3407_BEST_V1.pt",
    5813: MODEL_DIR / "ADNI_M0_SEED_5813_BEST_V1.pt",
    7291: MODEL_DIR / "ADNI_M0_SEED_7291_BEST_V1.pt",
    9109: MODEL_DIR / "ADNI_M0_SEED_9109_BEST_V1.pt",
}

EXPECTED_MODEL_PARAMETERS = 4_269_109
SEEDS = [2026, 3407, 5813, 7291, 9109]

EXPECTED_EPOCHS = {
    2026: 12,
    3407: 7,
    5813: 11,
    7291: 5,
    9109: 9,
}

EXPECTED_VAL_NLL = {
    2026: 0.9486606425046921,
    3407: 0.9454395517706871,
    5813: 0.9594686004519463,
    7291: 0.9487041115760804,
    9109: 0.9548525424301624,
}

HORIZONS = {24: 1, 36: 2}
EXPECTED_SUBJECTS = 24
SOURCE_SPATIAL_SHAPE = (197, 233, 189)
MODEL_SPATIAL_SHAPE = (160, 192, 160)
DAYS_PER_YEAR = 365.25
DEVICE = torch.device("cpu")

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def log(message=""):
    text = str(message)
    print(text, flush=True)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


if not OUTPUT_LOG.exists():
    OUTPUT_LOG.write_text("", encoding="utf-8")


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} missing:\n{path}")
    log(f"{label:45s}: PASS | {path}")


def parse_list_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except Exception:
        pass

    raise ValueError(f"Unable to parse list field: {text[:300]}")


def center_crop_3d(volume, target_shape=MODEL_SPATIAL_SHAPE):
    source_shape = np.asarray(volume.shape, dtype=int)
    target_shape = np.asarray(target_shape, dtype=int)

    start = (source_shape - target_shape) // 2
    end = start + target_shape

    cropped = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if cropped.shape != tuple(target_shape):
        raise RuntimeError(f"Crop mismatch: {cropped.shape} != {tuple(target_shape)}")

    return cropped


def load_mri_tensor(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI file missing:\n{path}")

    nii = nib.load(str(path))
    volume = np.asarray(nii.dataobj, dtype=np.float32)

    if volume.shape != SOURCE_SPATIAL_SHAPE:
        raise RuntimeError(
            f"Unexpected MRI shape {volume.shape}; expected {SOURCE_SPATIAL_SHAPE}"
        )

    if not np.isfinite(volume).all():
        raise RuntimeError(f"Non-finite MRI values:\n{path}")

    cropped = center_crop_3d(volume)

    return (
        torch.from_numpy(np.ascontiguousarray(cropped))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=DEVICE, dtype=torch.float32)
    )


def save_records(records, path, sort_columns):
    df = pd.DataFrame(records)

    if not df.empty:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    df.to_csv(path, index=False)


class TemporalEmbeddingRiskModel(nn.Module):
    def __init__(self, frozen_model, horizon_index):
        super().__init__()
        self.frozen_model = frozen_model
        self.horizon_index = int(horizon_index)

    def forward(
        self,
        mri_embeddings,
        visit_mask,
        years_from_first,
        years_before_landmark,
    ):
        time_embeddings = self.frozen_model.time_encoder(
            years_from_first,
            years_before_landmark,
        )

        x = mri_embeddings + time_embeddings

        mask_float = visit_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask_float

        x = self.frozen_model.transformer(
            x,
            src_key_padding_mask=(~visit_mask),
        )

        x = self.frozen_model.final_norm(x)

        pooled = (
            (x * mask_float).sum(dim=1)
            /
            mask_float.sum(dim=1).clamp_min(1.0)
        )

        logits = self.frozen_model.survival_head(pooled)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        cumulative_risk = 1.0 - survival

        return cumulative_risk[:, self.horizon_index]


def main():
    log("=" * 120)
    log("ADNI XAI — INTERNAL TEMPORAL / VISIT-LEVEL INTERPRETABILITY")
    log("=" * 120)

    log("Method: leave-one-visit-out temporal masking")
    log("Attention weights used as explanation: NO")
    log("Integrated Gradients rerun: NO")
    log("Retraining: NO")
    log("")

    require_file(COHORT_CSV, "Frozen TEST XAI cohort")
    require_file(XAI_CONFIG_LOCK, "Frozen XAI configuration")
    require_file(STABILITY_JSON, "Completed stability summary")
    require_file(FAITHFULNESS_JSON, "Completed faithfulness summary")

    cohort = pd.read_csv(COHORT_CSV)

    if cohort["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise RuntimeError("Frozen TEST XAI cohort size mismatch.")

    visit_records = (
        pd.read_csv(OUTPUT_VISITS_CSV).to_dict(orient="records")
        if OUTPUT_VISITS_CSV.exists()
        else []
    )

    run_records = (
        pd.read_csv(OUTPUT_RUNS_CSV).to_dict(orient="records")
        if OUTPUT_RUNS_CSV.exists()
        else []
    )

    completed = {
        (
            str(r["subject_id"]),
            int(r["seed"]),
            int(r["horizon_months"]),
        )
        for r in run_records
        if str(r.get("status", "")) == "COMPLETE"
    }

    expected_runs = EXPECTED_SUBJECTS * len(SEEDS) * len(HORIZONS)

    log(f"Expected subject/seed/horizon runs: {expected_runs}")
    log(f"Already complete: {len(completed)}")

    for seed_index, seed in enumerate(SEEDS, start=1):
        checkpoint_path = CHECKPOINTS[seed]
        require_file(checkpoint_path, f"Seed {seed} checkpoint")

        log("")
        log("=" * 120)
        log(f"SEED {seed} ({seed_index}/{len(SEEDS)})")
        log("=" * 120)

        model = ExactM0LongitudinalSurvivalModel_V3()

        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != EXPECTED_MODEL_PARAMETERS:
            raise RuntimeError(f"Seed {seed}: parameter mismatch.")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        epoch = checkpoint.get("epoch")
        val_nll = float(checkpoint.get("validation_loss"))

        if epoch != EXPECTED_EPOCHS[seed]:
            raise RuntimeError(f"Seed {seed}: epoch mismatch.")

        if abs(val_nll - EXPECTED_VAL_NLL[seed]) > 1e-10:
            raise RuntimeError(f"Seed {seed}: validation NLL mismatch.")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(DEVICE)
        model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

        for subject_index, (_, row) in enumerate(cohort.iterrows(), start=1):
            subject_id = str(row["subject_id"])
            landmark_id = str(row["landmark_id"])
            event_value = int(row["event"])
            n_mri = int(row["n_mri"])
            history_bin = str(row["history_bin"])

            subject_seed_keys = {
                (subject_id, seed, horizon)
                for horizon in HORIZONS
            }

            if subject_seed_keys.issubset(completed):
                log(
                    f"[{subject_index:02d}/{EXPECTED_SUBJECTS}] "
                    f"{subject_id} | seed={seed} | already complete -> SKIP"
                )
                continue

            image_paths = parse_list_field(row["image_paths"])
            image_ids = parse_list_field(row["image_ids"])
            image_dates = parse_list_field(row["image_dates"])

            days_from_first = np.asarray(
                parse_list_field(row["days_from_first_mri"]),
                dtype=np.float32,
            )

            days_before_landmark = np.asarray(
                parse_list_field(row["days_before_landmark"]),
                dtype=np.float32,
            )

            if not (
                len(image_paths)
                ==
                len(image_ids)
                ==
                len(image_dates)
                ==
                len(days_from_first)
                ==
                len(days_before_landmark)
                ==
                n_mri
            ):
                raise RuntimeError(
                    f"{subject_id}: longitudinal arrays do not match n_mri={n_mri}."
                )

            log("")
            log(
                f"[{subject_index:02d}/{EXPECTED_SUBJECTS}] "
                f"{subject_id} | seed={seed} | event={event_value} | "
                f"n_mri={n_mri} | history={history_bin}"
            )

            embedding_list = []

            for image_path in image_paths:
                image_tensor = load_mri_tensor(image_path)

                with torch.inference_mode():
                    embedding = model.mri_encoder(image_tensor).detach().clone()

                embedding_list.append(embedding)

                del image_tensor, embedding
                gc.collect()

            mri_embeddings = torch.stack(
                embedding_list,
                dim=1,
            )

            years_from_first_tensor = (
                torch.tensor(
                    days_from_first / DAYS_PER_YEAR,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                .unsqueeze(0)
            )

            years_before_landmark_tensor = (
                torch.tensor(
                    days_before_landmark / DAYS_PER_YEAR,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                .unsqueeze(0)
            )

            full_visit_mask = torch.ones(
                (1, n_mri),
                dtype=torch.bool,
                device=DEVICE,
            )

            for horizon_months, horizon_index in HORIZONS.items():
                run_key = (subject_id, seed, horizon_months)

                if run_key in completed:
                    log(f"  {horizon_months}m | complete -> SKIP")
                    continue

                temporal_model = TemporalEmbeddingRiskModel(
                    frozen_model=model,
                    horizon_index=horizon_index,
                ).to(DEVICE)

                temporal_model.eval()

                with torch.inference_mode():
                    full_risk = float(
                        temporal_model(
                            mri_embeddings=mri_embeddings,
                            visit_mask=full_visit_mask,
                            years_from_first=years_from_first_tensor,
                            years_before_landmark=years_before_landmark_tensor,
                        ).item()
                    )

                contribution_values = []

                local_record_indices = []

                for visit_index in range(n_mri):
                    masked_visit_mask = full_visit_mask.clone()
                    masked_visit_mask[0, visit_index] = False

                    if int(masked_visit_mask.sum()) < 1:
                        raise RuntimeError(
                            f"{subject_id}: visit masking removed all visits."
                        )

                    with torch.inference_mode():
                        risk_without_visit = float(
                            temporal_model(
                                mri_embeddings=mri_embeddings,
                                visit_mask=masked_visit_mask,
                                years_from_first=years_from_first_tensor,
                                years_before_landmark=years_before_landmark_tensor,
                            ).item()
                        )

                    contribution = full_risk - risk_without_visit
                    contribution_values.append(contribution)

                    visit_records.append(
                        {
                            "status": "COMPLETE",
                            "subject_id": subject_id,
                            "landmark_id": landmark_id,
                            "event": event_value,
                            "n_mri": n_mri,
                            "history_bin": history_bin,
                            "seed": seed,
                            "horizon_months": horizon_months,
                            "visit_index": visit_index,
                            "visit_number_1based": visit_index + 1,
                            "is_first_visit": visit_index == 0,
                            "is_latest_visit": visit_index == (n_mri - 1),
                            "image_id": str(image_ids[visit_index]),
                            "image_date": str(image_dates[visit_index]),
                            "days_from_first_mri": float(days_from_first[visit_index]),
                            "years_from_first_mri": float(
                                days_from_first[visit_index] / DAYS_PER_YEAR
                            ),
                            "days_before_landmark": float(days_before_landmark[visit_index]),
                            "years_before_landmark": float(
                                days_before_landmark[visit_index] / DAYS_PER_YEAR
                            ),
                            "full_risk": full_risk,
                            "risk_without_visit": risk_without_visit,
                            "visit_contribution": contribution,
                            "absolute_visit_contribution": abs(contribution),
                        }
                    )

                    local_record_indices.append(len(visit_records) - 1)

                contribution_array = np.asarray(
                    contribution_values,
                    dtype=np.float64,
                )

                abs_contribution_array = np.abs(contribution_array)
                abs_total = float(abs_contribution_array.sum())

                if abs_total > 1e-12:
                    normalized_abs = abs_contribution_array / abs_total
                else:
                    normalized_abs = np.zeros_like(abs_contribution_array)

                order = np.argsort(-abs_contribution_array)

                ranks = np.empty(
                    n_mri,
                    dtype=np.int64,
                )

                ranks[order] = np.arange(1, n_mri + 1)

                for local_idx, record_idx in enumerate(local_record_indices):
                    visit_records[record_idx]["absolute_contribution_share"] = float(
                        normalized_abs[local_idx]
                    )

                    visit_records[record_idx]["absolute_contribution_rank"] = int(
                        ranks[local_idx]
                    )

                latest_contribution = float(contribution_array[-1])
                latest_abs_share = float(normalized_abs[-1])
                latest_rank = int(ranks[-1])

                most_influential_index = int(
                    np.argmax(abs_contribution_array)
                )

                most_influential_contribution = float(
                    contribution_array[most_influential_index]
                )

                recency_abs_corr = np.nan

                if (
                    n_mri >= 3
                    and np.std(days_before_landmark) > 0
                    and np.std(abs_contribution_array) > 0
                ):
                    recency_abs_corr = float(
                        np.corrcoef(
                            days_before_landmark,
                            abs_contribution_array,
                        )[0, 1]
                    )

                run_records.append(
                    {
                        "status": "COMPLETE",
                        "subject_id": subject_id,
                        "landmark_id": landmark_id,
                        "event": event_value,
                        "n_mri": n_mri,
                        "history_bin": history_bin,
                        "seed": seed,
                        "horizon_months": horizon_months,
                        "full_risk": full_risk,
                        "sum_signed_visit_contributions": float(
                            contribution_array.sum()
                        ),
                        "sum_absolute_visit_contributions": abs_total,
                        "latest_visit_contribution": latest_contribution,
                        "latest_visit_absolute_share": latest_abs_share,
                        "latest_visit_absolute_rank": latest_rank,
                        "most_influential_visit_index": most_influential_index,
                        "most_influential_visit_number_1based": (
                            most_influential_index + 1
                        ),
                        "most_influential_is_latest": (
                            most_influential_index == (n_mri - 1)
                        ),
                        "most_influential_contribution": (
                            most_influential_contribution
                        ),
                        "corr_days_before_landmark_vs_abs_contribution": (
                            float(recency_abs_corr)
                            if np.isfinite(recency_abs_corr)
                            else None
                        ),
                    }
                )

                completed.add(run_key)

                save_records(
                    visit_records,
                    OUTPUT_VISITS_CSV,
                    [
                        "subject_id",
                        "seed",
                        "horizon_months",
                        "visit_index",
                    ],
                )

                save_records(
                    run_records,
                    OUTPUT_RUNS_CSV,
                    [
                        "subject_id",
                        "seed",
                        "horizon_months",
                    ],
                )

                log(
                    f"  {horizon_months}m | "
                    f"full_risk={full_risk:.6f} | "
                    f"latest_contrib={latest_contribution:+.6f} | "
                    f"latest_share={latest_abs_share:.3f} | "
                    f"latest_rank={latest_rank}/{n_mri} | "
                    f"most_influential_visit={most_influential_index + 1}"
                )

                del temporal_model
                gc.collect()

            del (
                mri_embeddings,
                years_from_first_tensor,
                years_before_landmark_tensor,
                full_visit_mask,
            )

            gc.collect()

        del model, checkpoint
        gc.collect()

    runs_df = pd.read_csv(OUTPUT_RUNS_CSV)

    complete_runs = runs_df.loc[
        runs_df["status"] == "COMPLETE"
    ].copy()

    unique_keys = (
        complete_runs[
            [
                "subject_id",
                "seed",
                "horizon_months",
            ]
        ]
        .drop_duplicates()
    )

    if len(unique_keys) != expected_runs:
        log("")
        log("=" * 120)
        log("INTERNAL TEMPORAL XAI: INCOMPLETE")
        log("=" * 120)
        log(f"Completed: {len(unique_keys)}/{expected_runs}")
        log("Re-run this same script to resume.")
        return

    subject_summary = (
        complete_runs.groupby(
            [
                "subject_id",
                "event",
                "n_mri",
                "history_bin",
                "horizon_months",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_full_risk=("full_risk", "mean"),
            mean_latest_visit_contribution=(
                "latest_visit_contribution",
                "mean",
            ),
            median_latest_visit_contribution=(
                "latest_visit_contribution",
                "median",
            ),
            mean_latest_visit_absolute_share=(
                "latest_visit_absolute_share",
                "mean",
            ),
            median_latest_visit_absolute_share=(
                "latest_visit_absolute_share",
                "median",
            ),
            mean_latest_visit_absolute_rank=(
                "latest_visit_absolute_rank",
                "mean",
            ),
            fraction_seeds_latest_most_influential=(
                "most_influential_is_latest",
                "mean",
            ),
            mean_recency_abs_corr=(
                "corr_days_before_landmark_vs_abs_contribution",
                "mean",
            ),
        )
    )

    if not np.all(subject_summary["seeds"] == len(SEEDS)):
        raise RuntimeError(
            "Some participant summaries do not include all five seeds."
        )

    subject_summary.to_csv(
        OUTPUT_SUBJECT_CSV,
        index=False,
    )

    cohort_summary = (
        subject_summary.groupby(
            "horizon_months",
            as_index=False,
        )
        .agg(
            subjects=("subject_id", "nunique"),
            median_latest_visit_contribution=(
                "mean_latest_visit_contribution",
                "median",
            ),
            q1_latest_visit_contribution=(
                "mean_latest_visit_contribution",
                lambda x: np.quantile(x, 0.25),
            ),
            q3_latest_visit_contribution=(
                "mean_latest_visit_contribution",
                lambda x: np.quantile(x, 0.75),
            ),
            median_latest_visit_absolute_share=(
                "mean_latest_visit_absolute_share",
                "median",
            ),
            q1_latest_visit_absolute_share=(
                "mean_latest_visit_absolute_share",
                lambda x: np.quantile(x, 0.25),
            ),
            q3_latest_visit_absolute_share=(
                "mean_latest_visit_absolute_share",
                lambda x: np.quantile(x, 0.75),
            ),
            fraction_subjects_latest_majority_seed_most_influential=(
                "fraction_seeds_latest_most_influential",
                lambda x: float(np.mean(np.asarray(x) >= 0.6)),
            ),
            median_subject_recency_abs_corr=(
                "mean_recency_abs_corr",
                "median",
            ),
        )
    )

    cohort_summary.to_csv(
        OUTPUT_COHORT_CSV,
        index=False,
    )

    event_summary = (
        subject_summary.groupby(
            [
                "horizon_months",
                "event",
            ],
            as_index=False,
        )
        .agg(
            subjects=("subject_id", "nunique"),
            median_latest_visit_contribution=(
                "mean_latest_visit_contribution",
                "median",
            ),
            median_latest_visit_absolute_share=(
                "mean_latest_visit_absolute_share",
                "median",
            ),
            fraction_latest_majority_seed_most_influential=(
                "fraction_seeds_latest_most_influential",
                lambda x: float(np.mean(np.asarray(x) >= 0.6)),
            ),
            median_recency_abs_corr=(
                "mean_recency_abs_corr",
                "median",
            ),
        )
    )

    event_summary.to_csv(
        OUTPUT_EVENT_CSV,
        index=False,
    )

    output = {
        "analysis": "Internal temporal / visit-level interpretability",
        "status": "COMPLETE",
        "method": "leave-one-visit-out temporal masking in the frozen Transformer sequence",
        "model": "Frozen M0 five-seed models",
        "subjects": EXPECTED_SUBJECTS,
        "seeds": SEEDS,
        "horizons_months": list(HORIZONS.keys()),
        "visit_contribution_definition": (
            "full risk minus risk after masking one MRI visit"
        ),
        "positive_contribution_interpretation": (
            "visit increases predicted progression risk in the full longitudinal context"
        ),
        "negative_contribution_interpretation": (
            "visit suppresses predicted progression risk in the full longitudinal context"
        ),
        "attention_weights_used_as_explanation": False,
        "integrated_gradients_rerun": False,
        "retraining": False,
        "cohort_summary": cohort_summary.to_dict(orient="records"),
        "event_status_summary": event_summary.to_dict(orient="records"),
        "next_phase": "benchmark model comparison + publication tables/figures",
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log("")
    log("=" * 120)
    log("INTERNAL TEMPORAL XAI — COHORT SUMMARY")
    log("=" * 120)
    log(cohort_summary.to_string(index=False))

    log("")
    log("=" * 120)
    log("INTERNAL TEMPORAL XAI — EVENT / CENSORED SUMMARY")
    log("=" * 120)
    log(event_summary.to_string(index=False))

    log("")
    log(f"Saved visit contributions: {OUTPUT_VISITS_CSV}")
    log(f"Saved run summary        : {OUTPUT_RUNS_CSV}")
    log(f"Saved subject summary    : {OUTPUT_SUBJECT_CSV}")
    log(f"Saved cohort summary     : {OUTPUT_COHORT_CSV}")
    log(f"Saved event summary      : {OUTPUT_EVENT_CSV}")
    log(f"Saved JSON               : {OUTPUT_JSON}")

    log("")
    log("=" * 120)
    log("INTERNAL TEMPORAL / VISIT-LEVEL XAI: COMPLETE")
    log("=" * 120)
    log("Attention weights used as explanation: NO")
    log("Integrated Gradients rerun: NO")
    log("Next step: fair benchmark model comparison suite")
    log("=" * 120)


if __name__ == "__main__":

    try:
        main()

    except Exception:
        log("")
        log("=" * 120)
        log("INTERNAL TEMPORAL / VISIT-LEVEL XAI: FAILED")
        log("=" * 120)
        log(traceback.format_exc())
        log(
            "This analysis is resume-safe. Fix the error and rerun the same script."
        )
        raise

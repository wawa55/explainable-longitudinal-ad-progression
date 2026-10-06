# -*- coding: utf-8 -*-
"""
21_anatomical_regional_ig_analysis.py

Anatomical validation of frozen Integrated-Gradients (IG) maps using
Harvard-Oxford cortical + subcortical atlases.

Prerequisites
-------------
- Step 19 passed: frozen MRI grid == local MNI grid.
- Step 20B succeeded: Harvard-Oxford 1-mm atlas files and labels exist.
- Frozen IG maps exist under:
  08_results/ADNI_XAI_TEST_PRODUCTION_V1/attributions

What this script does
---------------------
1. Resamples Harvard-Oxford categorical atlas labels to the exact local MNI
   template grid using nearest-neighbour interpolation.
2. Embeds each frozen 160x192x160 IG map back into the full
   197x233x189 MNI grid using the frozen center-crop bounds.
3. Computes region-level absolute IG summaries for every:
      24 participants x 5 seeds x 2 horizons = 240 maps.
4. Computes both:
   - attribution share: sum(|IG| in region) / sum(|IG| over the full IG map)
   - attribution density: mean(|IG| in region)
   - enrichment: region mean(|IG|) / whole-map mean(|IG| within atlas coverage)
5. Aggregates results at participant level first (across the 5 seeds), then
   summarizes across participants for 24m and 36m.
6. Writes CSV tables, JSON metadata, and publication-ready PNG/PDF figures.
7. Does NOT retrain models, run inference, or recompute IG.

Important interpretation
------------------------
These outputs quantify where the frozen model assigns attribution in atlas-defined
MNI regions. They are anatomical-consistency / regional-attribution results, not
proof that the regions are causal biological biomarkers of Alzheimer's disease.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img


# =====================================================================
# PATHS / FROZEN SPEC
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

LOCAL_MNI = (
    ROOT / "03_preprocessed" / "templates" / "MNI152_T1_1mm.nii.gz"
)

ATLAS_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "atlas"
)

CORT_ATLAS = ATLAS_ROOT / "HarvardOxford_cort_maxprob_thr25_1mm.nii.gz"
SUB_ATLAS = ATLAS_ROOT / "HarvardOxford_sub_maxprob_thr25_1mm.nii.gz"
CORT_LABELS = ATLAS_ROOT / "HarvardOxford_cort_labels.json"
SUB_LABELS = ATLAS_ROOT / "HarvardOxford_sub_labels.json"

ATTR_ROOT = (
    ROOT / "08_results" / "ADNI_XAI_TEST_PRODUCTION_V1" / "attributions"
)

OUT_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "regional_analysis"
)

FIG_ROOT = OUT_ROOT / "figures"

EXPECTED_SEEDS = [2026, 3407, 5813, 7291, 9109]
EXPECTED_HORIZONS = [24, 36]
SOURCE_SHAPE = (197, 233, 189)
MODEL_SHAPE = (160, 192, 160)

CROP_START = (18, 20, 14)
CROP_END = (178, 212, 174)

# Minimum fraction of an atlas-defined region represented inside the model crop.
# We DO NOT delete low-coverage regions; we flag them.
COVERAGE_FLAG_THRESHOLD = 0.95

# For publication figures.
TOP_N = 15


# =====================================================================
# HELPERS
# =====================================================================

def load_labels(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(r["index"]): str(r["label"]) for r in rows}


def atlas_resample_nearest(atlas_path: Path, reference_path: Path, out_path: Path):
    atlas_img = nib.load(str(atlas_path))
    ref_img = nib.load(str(reference_path))

    res = resample_to_img(
        atlas_img,
        ref_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    data = np.asanyarray(res.dataobj)
    data = np.rint(data).astype(np.int16)

    out = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
    out.set_data_dtype(np.int16)
    nib.save(out, str(out_path))

    return out


def embed_to_full_grid(attr: np.ndarray) -> np.ndarray:
    if tuple(attr.shape) != MODEL_SHAPE:
        raise RuntimeError(f"Unexpected IG shape {attr.shape}; expected {MODEL_SHAPE}")

    full = np.zeros(SOURCE_SHAPE, dtype=np.float32)
    x0, y0, z0 = CROP_START
    x1, y1, z1 = CROP_END
    full[x0:x1, y0:y1, z0:z1] = attr.astype(np.float32, copy=False)
    return full


def crop_mask_full_grid() -> np.ndarray:
    mask = np.zeros(SOURCE_SHAPE, dtype=bool)
    x0, y0, z0 = CROP_START
    x1, y1, z1 = CROP_END
    mask[x0:x1, y0:y1, z0:z1] = True
    return mask


def read_npz_meta(path: Path):
    with np.load(path, allow_pickle=True) as z:
        attr = np.asarray(z["attribution"], dtype=np.float32)
        subject = str(np.asarray(z["subject_id"]).item())
        landmark = str(np.asarray(z["landmark_id"]).item())
        seed = int(np.asarray(z["seed"]).item())
        horizon = int(np.asarray(z["horizon_months"]).item())
    return attr, subject, landmark, seed, horizon


def safe_quantile(series, q):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else np.nan


def save_table_tex(df: pd.DataFrame, path: Path, caption: str, label: str):
    # Keep this generic and safe for Springer/Overleaf.
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v).replace("&", "\\&").replace("_", "\\_"))
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 110)
    print("STEP 21 — ANATOMICAL REGIONAL IG ANALYSIS")
    print("=" * 110)

    required = [
        LOCAL_MNI, CORT_ATLAS, SUB_ATLAS,
        CORT_LABELS, SUB_LABELS, ATTR_ROOT
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing:\n{p}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    cort_labels = load_labels(CORT_LABELS)
    sub_labels = load_labels(SUB_LABELS)

    # -----------------------------------------------------------------
    # 1. Resample categorical atlas labels to exact local MNI grid.
    # -----------------------------------------------------------------

    print("\n[1/6] Resampling atlas labels to local MNI grid (nearest-neighbour)")

    cort_res_path = OUT_ROOT / "HarvardOxford_cort_on_local_MNI.nii.gz"
    sub_res_path = OUT_ROOT / "HarvardOxford_sub_on_local_MNI.nii.gz"

    cort_img = atlas_resample_nearest(CORT_ATLAS, LOCAL_MNI, cort_res_path)
    sub_img = atlas_resample_nearest(SUB_ATLAS, LOCAL_MNI, sub_res_path)

    cort_data = np.asanyarray(cort_img.dataobj).astype(np.int16)
    sub_data = np.asanyarray(sub_img.dataobj).astype(np.int16)

    if tuple(cort_data.shape) != SOURCE_SHAPE:
        raise RuntimeError(f"Cortical resampled shape mismatch: {cort_data.shape}")
    if tuple(sub_data.shape) != SOURCE_SHAPE:
        raise RuntimeError(f"Subcortical resampled shape mismatch: {sub_data.shape}")

    print("  Cortical shape   :", cort_data.shape)
    print("  Subcortical shape:", sub_data.shape)

    # -----------------------------------------------------------------
    # 2. Region inventory + crop coverage.
    # -----------------------------------------------------------------

    print("\n[2/6] Building atlas region inventory and model-crop coverage")

    crop_mask = crop_mask_full_grid()

    region_rows = []

    for atlas_name, data, labels in [
        ("cortical", cort_data, cort_labels),
        ("subcortical", sub_data, sub_labels),
    ]:
        unique_ids = sorted(int(v) for v in np.unique(data) if int(v) > 0)

        for region_id in unique_ids:
            full_mask = data == region_id
            full_vox = int(full_mask.sum())
            in_crop_vox = int((full_mask & crop_mask).sum())
            coverage = in_crop_vox / full_vox if full_vox else np.nan

            region_rows.append({
                "atlas": atlas_name,
                "region_id": region_id,
                "region_name": labels.get(region_id, f"Label_{region_id}"),
                "atlas_voxels_full": full_vox,
                "atlas_voxels_in_model_crop": in_crop_vox,
                "crop_coverage_fraction": coverage,
                "coverage_ge_95pct": bool(
                    np.isfinite(coverage) and coverage >= COVERAGE_FLAG_THRESHOLD
                ),
            })

    region_inventory = pd.DataFrame(region_rows)
    region_inventory.to_csv(
        OUT_ROOT / "Table_Anatomical_Region_Coverage.csv", index=False
    )

    print("  Cortical regions   :", (region_inventory["atlas"] == "cortical").sum())
    print("  Subcortical regions:", (region_inventory["atlas"] == "subcortical").sum())
    print(
        "  Regions >=95% represented in model crop:",
        int(region_inventory["coverage_ge_95pct"].sum()),
        "/",
        len(region_inventory),
    )

    # Prebuild masks restricted to model crop.
    region_masks = {}
    for _, r in region_inventory.iterrows():
        data = cort_data if r["atlas"] == "cortical" else sub_data
        mask = (data == int(r["region_id"])) & crop_mask
        region_masks[(r["atlas"], int(r["region_id"]))] = mask

    atlas_union = ((cort_data > 0) | (sub_data > 0)) & crop_mask

    # -----------------------------------------------------------------
    # 3. Process all 240 frozen IG maps.
    # -----------------------------------------------------------------

    print("\n[3/6] Computing region-level attribution for all frozen IG maps")

    files = sorted(ATTR_ROOT.glob("seed_*/*m/*.npz"))
    if len(files) != 240:
        raise RuntimeError(f"Expected 240 frozen IG files, found {len(files)}")

    raw_rows = []

    for idx, p in enumerate(files, start=1):
        attr, subject, landmark, seed, horizon = read_npz_meta(p)

        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"Unexpected seed {seed}: {p}")
        if horizon not in EXPECTED_HORIZONS:
            raise RuntimeError(f"Unexpected horizon {horizon}: {p}")
        if not np.isfinite(attr).all():
            raise RuntimeError(f"Non-finite IG values: {p}")

        full_abs = np.abs(embed_to_full_grid(attr))

        total_abs = float(full_abs.sum())
        atlas_abs = float(full_abs[atlas_union].sum())
        atlas_mean = float(full_abs[atlas_union].mean()) if atlas_union.any() else np.nan

        for _, rr in region_inventory.iterrows():
            key = (rr["atlas"], int(rr["region_id"]))
            mask = region_masks[key]
            nvox = int(mask.sum())

            region_sum = float(full_abs[mask].sum()) if nvox else 0.0
            region_mean = float(full_abs[mask].mean()) if nvox else np.nan

            share_total = region_sum / total_abs if total_abs > 0 else np.nan
            share_atlas = region_sum / atlas_abs if atlas_abs > 0 else np.nan
            enrichment = region_mean / atlas_mean if (
                np.isfinite(region_mean)
                and np.isfinite(atlas_mean)
                and atlas_mean > 0
            ) else np.nan

            raw_rows.append({
                "subject_id": subject,
                "landmark_id": landmark,
                "seed": seed,
                "horizon_months": horizon,
                "atlas": rr["atlas"],
                "region_id": int(rr["region_id"]),
                "region_name": rr["region_name"],
                "region_voxels_in_model_crop": nvox,
                "crop_coverage_fraction": float(rr["crop_coverage_fraction"]),
                "region_abs_ig_sum": region_sum,
                "region_mean_abs_ig": region_mean,
                "share_of_total_abs_ig": share_total,
                "share_of_atlas_covered_abs_ig": share_atlas,
                "regional_enrichment_vs_atlas_mean": enrichment,
                "total_abs_ig": total_abs,
                "atlas_covered_abs_ig": atlas_abs,
            })

        if idx == 1 or idx % 40 == 0 or idx == len(files):
            print(f"  Processed {idx:>3}/{len(files)} IG maps")

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        OUT_ROOT / "Anatomical_IG_All_Maps_Region_Level.csv",
        index=False
    )

    # -----------------------------------------------------------------
    # 4. Participant-level aggregation across 5 seeds.
    # -----------------------------------------------------------------

    print("\n[4/6] Aggregating across seeds at participant level")

    metric_cols = [
        "region_abs_ig_sum",
        "region_mean_abs_ig",
        "share_of_total_abs_ig",
        "share_of_atlas_covered_abs_ig",
        "regional_enrichment_vs_atlas_mean",
    ]

    participant = (
        raw
        .groupby(
            [
                "subject_id",
                "landmark_id",
                "horizon_months",
                "atlas",
                "region_id",
                "region_name",
            ],
            as_index=False
        )[metric_cols]
        .mean()
    )

    participant.to_csv(
        OUT_ROOT / "Anatomical_IG_Participant_Level_5SeedMean.csv",
        index=False
    )

    # -----------------------------------------------------------------
    # 5. Cohort summary across participants.
    # -----------------------------------------------------------------

    print("\n[5/6] Summarizing across 24 participants")

    groups = []
    for keys, g in participant.groupby(
        ["horizon_months", "atlas", "region_id", "region_name"],
        sort=True
    ):
        horizon, atlas_name, region_id, region_name = keys

        vals_share = g["share_of_total_abs_ig"].to_numpy(float)
        vals_density = g["region_mean_abs_ig"].to_numpy(float)
        vals_enrich = g["regional_enrichment_vs_atlas_mean"].to_numpy(float)

        groups.append({
            "horizon_months": int(horizon),
            "atlas": atlas_name,
            "region_id": int(region_id),
            "region_name": region_name,
            "participants": int(g["subject_id"].nunique()),
            "median_share_total": float(np.nanmedian(vals_share)),
            "q1_share_total": safe_quantile(vals_share, 0.25),
            "q3_share_total": safe_quantile(vals_share, 0.75),
            "mean_share_total": float(np.nanmean(vals_share)),
            "median_mean_abs_ig": float(np.nanmedian(vals_density)),
            "median_enrichment": float(np.nanmedian(vals_enrich)),
            "q1_enrichment": safe_quantile(vals_enrich, 0.25),
            "q3_enrichment": safe_quantile(vals_enrich, 0.75),
            "fraction_enrichment_gt1": float(np.nanmean(vals_enrich > 1.0)),
        })

    summary = pd.DataFrame(groups)

    # Add crop coverage info.
    summary = summary.merge(
        region_inventory[
            [
                "atlas", "region_id", "atlas_voxels_full",
                "atlas_voxels_in_model_crop",
                "crop_coverage_fraction", "coverage_ge_95pct"
            ]
        ],
        on=["atlas", "region_id"],
        how="left",
    )

    summary.to_csv(
        OUT_ROOT / "Table_Anatomical_Regional_Attribution_Summary.csv",
        index=False
    )

    # Rank independently by horizon + atlas.
    summary["rank_by_median_share"] = (
        summary.groupby(["horizon_months", "atlas"])["median_share_total"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ranked = summary.sort_values(
        ["horizon_months", "atlas", "rank_by_median_share"]
    ).reset_index(drop=True)

    ranked.to_csv(
        OUT_ROOT / "Table_Anatomical_Regional_Attribution_Ranked.csv",
        index=False
    )

    # Compact top-region table for manuscript.
    top_rows = []
    for horizon in EXPECTED_HORIZONS:
        for atlas_name in ["cortical", "subcortical"]:
            g = ranked[
                (ranked["horizon_months"] == horizon)
                & (ranked["atlas"] == atlas_name)
                & (ranked["coverage_ge_95pct"])
            ].head(10)

            top_rows.append(g)

    manuscript_table = pd.concat(top_rows, ignore_index=True)
    manuscript_table.to_csv(
        OUT_ROOT / "Table_Anatomical_Top10_Per_Horizon_Atlas.csv",
        index=False
    )

    # -----------------------------------------------------------------
    # 6. Figures + report.
    # -----------------------------------------------------------------

    print("\n[6/6] Building publication figures and report")

    for horizon in EXPECTED_HORIZONS:
        g = ranked[
            (ranked["horizon_months"] == horizon)
            & (ranked["coverage_ge_95pct"])
        ].copy()

        # Use enrichment for fairer size-adjusted comparison.
        g = g.sort_values("median_enrichment", ascending=False).head(TOP_N)
        g = g.sort_values("median_enrichment", ascending=True)

        labels = [
            f"{r['region_name']} ({r['atlas']})"
            for _, r in g.iterrows()
        ]
        values = g["median_enrichment"].to_numpy(float)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(labels, values)
        ax.axvline(1.0, linewidth=1)
        ax.set_xlabel("Median regional attribution enrichment")
        ax.set_ylabel("Harvard–Oxford region")
        ax.set_title(
            f"Top atlas-defined IG attribution enrichment at {horizon} months"
        )
        fig.tight_layout()

        fig.savefig(
            FIG_ROOT / f"Figure_Anatomical_IG_Enrichment_{horizon}m.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            FIG_ROOT / f"Figure_Anatomical_IG_Enrichment_{horizon}m.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # Horizon consistency: Spearman correlation across region cohort medians.
    consistency = []
    for atlas_name in ["cortical", "subcortical"]:
        a = summary[
            (summary["atlas"] == atlas_name)
            & (summary["horizon_months"] == 24)
        ][["region_id", "median_share_total", "median_enrichment"]].copy()

        b = summary[
            (summary["atlas"] == atlas_name)
            & (summary["horizon_months"] == 36)
        ][["region_id", "median_share_total", "median_enrichment"]].copy()

        m = a.merge(b, on="region_id", suffixes=("_24", "_36"))

        share_corr = float(
            pd.Series(m["median_share_total_24"]).corr(
                pd.Series(m["median_share_total_36"]),
                method="spearman"
            )
        )
        enrich_corr = float(
            pd.Series(m["median_enrichment_24"]).corr(
                pd.Series(m["median_enrichment_36"]),
                method="spearman"
            )
        )

        consistency.append({
            "atlas": atlas_name,
            "n_regions": len(m),
            "spearman_24_vs_36_median_share": share_corr,
            "spearman_24_vs_36_median_enrichment": enrich_corr,
        })

    consistency_df = pd.DataFrame(consistency)
    consistency_df.to_csv(
        OUT_ROOT / "Table_Anatomical_24_vs_36_Horizon_Consistency.csv",
        index=False
    )

    # Save simple manuscript LaTeX top table.
    latex_df = manuscript_table[
        [
            "horizon_months",
            "atlas",
            "region_name",
            "median_share_total",
            "median_enrichment",
            "fraction_enrichment_gt1",
        ]
    ].copy()

    save_table_tex(
        latex_df,
        OUT_ROOT / "Table_Anatomical_Top10_Per_Horizon_Atlas.tex",
        caption=(
            "Atlas-defined regional Integrated-Gradients attribution in the "
            "frozen XAI cohort. Values are based on participant-level five-seed "
            "means and summarized across participants."
        ),
        label="tab:anatomical-ig",
    )

    # JSON metadata.
    meta = {
        "experiment": "ANATOMICAL_VALIDATION_V1",
        "analysis": "Harvard-Oxford regional Integrated-Gradients attribution",
        "participants": int(participant["subject_id"].nunique()),
        "seeds": EXPECTED_SEEDS,
        "horizons_months": EXPECTED_HORIZONS,
        "ig_maps_processed": int(len(files)),
        "source_shape": list(SOURCE_SHAPE),
        "model_shape": list(MODEL_SHAPE),
        "crop_start": list(CROP_START),
        "crop_end": list(CROP_END),
        "atlas_cortical": str(CORT_ATLAS),
        "atlas_subcortical": str(SUB_ATLAS),
        "atlas_resampling": "nearest-neighbour to exact local MNI grid",
        "primary_region_metric": "participant-level 5-seed mean share_of_total_abs_ig",
        "secondary_region_metric": "regional_enrichment_vs_atlas_mean",
        "interpretation_note": (
            "Regional attribution quantifies model reliance in atlas-defined regions; "
            "it does not establish causal or independently validated AD biomarkers."
        ),
    }

    with (OUT_ROOT / "anatomical_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)

    # Human-readable report.
    report = []
    report.append("=" * 110)
    report.append("ANATOMICAL REGIONAL IG ANALYSIS — SUMMARY")
    report.append("=" * 110)
    report.append(f"Frozen IG maps processed : {len(files)} / 240")
    report.append(f"Participants             : {participant['subject_id'].nunique()} / 24")
    report.append(f"Cortical regions         : {(region_inventory['atlas']=='cortical').sum()}")
    report.append(f"Subcortical regions      : {(region_inventory['atlas']=='subcortical').sum()}")
    report.append(
        f"Regions >=95% in crop    : {int(region_inventory['coverage_ge_95pct'].sum())}"
        f" / {len(region_inventory)}"
    )
    report.append("")
    report.append("24-vs-36 month regional consistency:")
    for _, r in consistency_df.iterrows():
        report.append(
            f"  {r['atlas']}: Spearman share={r['spearman_24_vs_36_median_share']:.4f}, "
            f"enrichment={r['spearman_24_vs_36_median_enrichment']:.4f}"
        )

    for horizon in EXPECTED_HORIZONS:
        report.append("")
        report.append(f"TOP REGIONS — {horizon} MONTHS")
        report.append("-" * 110)

        g = ranked[
            (ranked["horizon_months"] == horizon)
            & (ranked["coverage_ge_95pct"])
        ].sort_values("median_enrichment", ascending=False).head(10)

        for rank, (_, r) in enumerate(g.iterrows(), start=1):
            report.append(
                f"{rank:>2}. {r['region_name']} [{r['atlas']}] | "
                f"median share={r['median_share_total']:.6f} | "
                f"median enrichment={r['median_enrichment']:.3f} | "
                f"participants enrichment>1={r['fraction_enrichment_gt1']:.3f}"
            )

    report.append("")
    report.append("STATUS: COMPLETE — atlas-based regional attribution analysis finished.")
    report.append("")
    report.append(
        "Interpretation: these results provide anatomical consistency of the model's "
        "frozen IG attributions. They do not independently validate causal AD biomarkers."
    )
    report.append(
        "No model training, model selection, inference, or IG recomputation was performed."
    )

    report_text = "\n".join(report)
    with (OUT_ROOT / "anatomical_analysis_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    print()
    print(report_text)
    print()
    print("Saved outputs:")
    print(" ", OUT_ROOT / "Table_Anatomical_Region_Coverage.csv")
    print(" ", OUT_ROOT / "Anatomical_IG_All_Maps_Region_Level.csv")
    print(" ", OUT_ROOT / "Anatomical_IG_Participant_Level_5SeedMean.csv")
    print(" ", OUT_ROOT / "Table_Anatomical_Regional_Attribution_Summary.csv")
    print(" ", OUT_ROOT / "Table_Anatomical_Regional_Attribution_Ranked.csv")
    print(" ", OUT_ROOT / "Table_Anatomical_Top10_Per_Horizon_Atlas.csv")
    print(" ", OUT_ROOT / "Table_Anatomical_24_vs_36_Horizon_Consistency.csv")
    print(" ", OUT_ROOT / "anatomical_analysis_report.txt")
    print(" ", FIG_ROOT)


if __name__ == "__main__":
    main()

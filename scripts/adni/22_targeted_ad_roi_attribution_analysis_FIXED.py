# -*- coding: utf-8 -*-
"""
22_targeted_ad_roi_attribution_analysis.py

Targeted, prespecified AD-relevant anatomical ROI analysis of the existing
frozen Integrated-Gradients maps.

Purpose
-------
Step 21 identified the globally highest-enrichment Harvard-Oxford regions.
Step 22 asks a different, prespecified question:

    Do canonical AD-relevant anatomical regions carry meaningful and
    reproducible attribution, even if they are not among the global top-10?

The script evaluates:
- hippocampus
- amygdala
- thalamus
- caudate
- lateral ventricles
- parahippocampal cortex
- temporal pole
- temporal fusiform cortex
- posterior cingulate cortex
- precuneus
- medial frontal cortex / frontal medial cortex

Analysis principles
-------------------
1. Uses ONLY the already frozen 240 IG maps.
2. No model training, inference, tuning, or IG recomputation.
3. Uses Harvard-Oxford cortical/subcortical labels already resampled to the
   exact local MNI grid in Step 21.
4. Computes ROI-family union masks to avoid double-counting voxels.
5. Aggregates across the five seeds WITHIN participant first.
6. Reports:
   - share of total |IG|
   - mean |IG| density
   - enrichment relative to the atlas-covered brain
   - fraction of participants with enrichment > 1
   - cross-horizon Spearman consistency
   - paired participant-level 24-vs-36 comparisons (Wilcoxon, descriptive)
7. Does NOT call the output a causal biomarker validation.

Repository root
---------------
The repository root is determined dynamically from the location of this
script. No user-specific absolute path is required.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


# =====================================================================
# PATHS / FROZEN SPEC
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

ATTR_ROOT = (
    ROOT / "08_results" / "ADNI_XAI_TEST_PRODUCTION_V1" / "attributions"
)

REGION_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "regional_analysis"
)

CORT_ON_MNI = REGION_ROOT / "HarvardOxford_cort_on_local_MNI.nii.gz"
SUB_ON_MNI = REGION_ROOT / "HarvardOxford_sub_on_local_MNI.nii.gz"

ATLAS_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "atlas"
)

CORT_LABELS = ATLAS_ROOT / "HarvardOxford_cort_labels.json"
SUB_LABELS = ATLAS_ROOT / "HarvardOxford_sub_labels.json"

OUT_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "targeted_ad_roi_analysis"
)

FIG_ROOT = OUT_ROOT / "figures"

EXPECTED_SEEDS = [2026, 3407, 5813, 7291, 9109]
EXPECTED_HORIZONS = [24, 36]

SOURCE_SHAPE = (197, 233, 189)
MODEL_SHAPE = (160, 192, 160)

CROP_START = (18, 20, 14)
CROP_END = (178, 212, 174)


# =====================================================================
# PRESPECIFIED ROI FAMILIES
# =====================================================================
#
# Matching is case-insensitive and applied to Harvard-Oxford region labels.
# Left/right homologues are intentionally pooled into bilateral ROI families.
#
# We keep the pattern list explicit and save all actual matched atlas labels
# so the analysis is fully auditable.
# =====================================================================

ROI_FAMILIES = {
    "Hippocampus": {
        "atlases": ["subcortical"],
        "patterns": [r"\bhippocampus\b"],
    },
    "Amygdala": {
        "atlases": ["subcortical"],
        "patterns": [r"\bamygdala\b"],
    },
    "Thalamus": {
        "atlases": ["subcortical"],
        "patterns": [r"\bthalamus\b"],
    },
    "Caudate": {
        "atlases": ["subcortical"],
        "patterns": [r"\bcaudate\b"],
    },
    "Lateral ventricles": {
        "atlases": ["subcortical"],
        "patterns": [
            r"lateral ventricle",
            r"inf lat vent",
            r"inferior lateral ventricle",
        ],
    },
    "Parahippocampal cortex": {
        "atlases": ["cortical"],
        "patterns": [r"parahippocampal"],
    },
    "Temporal pole": {
        "atlases": ["cortical"],
        "patterns": [r"temporal pole"],
    },
    "Temporal fusiform cortex": {
        "atlases": ["cortical"],
        "patterns": [r"temporal fusiform"],
    },
    "Posterior cingulate cortex": {
        "atlases": ["cortical"],
        "patterns": [r"posterior cingulate"],
    },
    "Precuneus": {
        "atlases": ["cortical"],
        "patterns": [r"\bprecune"],
    },
    "Frontal medial cortex": {
        "atlases": ["cortical"],
        "patterns": [
            r"frontal medial cortex",
            r"medial frontal cortex",
        ],
    },
}


# =====================================================================
# HELPERS
# =====================================================================

def load_label_map(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(r["index"]): str(r["label"]) for r in rows}


def crop_mask_full() -> np.ndarray:
    m = np.zeros(SOURCE_SHAPE, dtype=bool)
    x0, y0, z0 = CROP_START
    x1, y1, z1 = CROP_END
    m[x0:x1, y0:y1, z0:z1] = True
    return m


def embed_attr(attr: np.ndarray) -> np.ndarray:
    if tuple(attr.shape) != MODEL_SHAPE:
        raise RuntimeError(f"IG shape {attr.shape} != expected {MODEL_SHAPE}")

    full = np.zeros(SOURCE_SHAPE, dtype=np.float32)
    x0, y0, z0 = CROP_START
    x1, y1, z1 = CROP_END
    full[x0:x1, y0:y1, z0:z1] = attr.astype(np.float32, copy=False)
    return full


def scalar(z, key):
    return np.asarray(z[key]).item()


def read_npz(path: Path):
    with np.load(path, allow_pickle=True) as z:
        attr = np.asarray(z["attribution"], dtype=np.float32)
        subject = str(scalar(z, "subject_id"))
        landmark = str(scalar(z, "landmark_id"))
        seed = int(scalar(z, "seed"))
        horizon = int(scalar(z, "horizon_months"))
    return attr, subject, landmark, seed, horizon


def matched_ids(labels: dict[int, str], patterns: list[str]) -> list[int]:
    ids = []
    for idx, label in labels.items():
        if idx <= 0:
            continue
        for pat in patterns:
            if re.search(pat, label, flags=re.IGNORECASE):
                ids.append(idx)
                break
    return sorted(set(ids))


def quantile(x, q):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else np.nan


def paired_wilcoxon(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) == 0:
        return np.nan, np.nan

    diff = y - x
    if np.allclose(diff, 0):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(
            y,
            x,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 112)
    print("STEP 22 FIXED — TARGETED AD-RELEVANT ROI ATTRIBUTION ANALYSIS")
    print("=" * 112)

    required = [
        ATTR_ROOT,
        CORT_ON_MNI,
        SUB_ON_MNI,
        CORT_LABELS,
        SUB_LABELS,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing:\n{p}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    cort = np.asanyarray(
        nib.load(str(CORT_ON_MNI)).dataobj
    ).astype(np.int16)

    sub = np.asanyarray(
        nib.load(str(SUB_ON_MNI)).dataobj
    ).astype(np.int16)

    if tuple(cort.shape) != SOURCE_SHAPE:
        raise RuntimeError(
            f"Cortical atlas shape mismatch: {cort.shape}"
        )

    if tuple(sub.shape) != SOURCE_SHAPE:
        raise RuntimeError(
            f"Subcortical atlas shape mismatch: {sub.shape}"
        )

    cort_labels = load_label_map(CORT_LABELS)
    sub_labels = load_label_map(SUB_LABELS)

    crop_mask = crop_mask_full()
    atlas_union = ((cort > 0) | (sub > 0)) & crop_mask

    # -----------------------------------------------------------------
    # 1. Resolve actual atlas labels for the prespecified ROI families.
    # -----------------------------------------------------------------

    print(
        "\n[1/6] Resolving prespecified ROI families "
        "against Harvard-Oxford labels"
    )

    roi_masks = {}
    audit_rows = []

    for family, spec in ROI_FAMILIES.items():
        union = np.zeros(SOURCE_SHAPE, dtype=bool)
        matched = []

        if "cortical" in spec["atlases"]:
            ids = matched_ids(
                cort_labels,
                spec["patterns"],
            )

            for idx in ids:
                union |= cort == idx
                matched.append(
                    (
                        "cortical",
                        idx,
                        cort_labels[idx],
                    )
                )

        if "subcortical" in spec["atlases"]:
            ids = matched_ids(
                sub_labels,
                spec["patterns"],
            )

            for idx in ids:
                union |= sub == idx
                matched.append(
                    (
                        "subcortical",
                        idx,
                        sub_labels[idx],
                    )
                )

        union &= crop_mask
        roi_masks[family] = union

        print(
            f"  {family}: "
            f"{int(union.sum())} voxels"
        )

        for atlas_name, idx, label in matched:
            print(
                f"      {atlas_name:11s} "
                f"label {idx:>2}: {label}"
            )

        if not matched:
            print("      WARNING: NO MATCHED LABELS")

        audit_rows.append(
            {
                "roi_family": family,
                "matched_labels_count": len(matched),
                "matched_labels": " | ".join(
                    f"{a}:{i}:{lab}"
                    for a, i, lab in matched
                ),
                "roi_voxels_in_model_crop": int(
                    union.sum()
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    audit_df.to_csv(
        OUT_ROOT
        / "Table_Targeted_AD_ROI_Label_Audit.csv",
        index=False,
    )

    missing = audit_df.loc[
        audit_df["matched_labels_count"] == 0,
        "roi_family",
    ].tolist()

    if missing:
        print(
            "\n  NOTE: The following prespecified ROI families "
            "are not represented"
        )
        print(
            "  by a matching label in the available "
            "Harvard-Oxford atlas and will"
        )
        print(
            "  therefore be recorded as NOT AVAILABLE "
            "rather than substituted:"
        )

        for name in missing:
            print(f"    - {name}")

    available_roi_masks = {
        family: mask
        for family, mask in roi_masks.items()
        if int(mask.sum()) > 0
    }

    if not available_roi_masks:
        raise RuntimeError(
            "No prespecified ROI family could be "
            "resolved in the atlas."
        )

    # Combined canonical AD ROI panel.
    ad_panel_mask = np.zeros(
        SOURCE_SHAPE,
        dtype=bool,
    )

    for m in available_roi_masks.values():
        ad_panel_mask |= m

    # -----------------------------------------------------------------
    # 2. Process all 240 maps.
    # -----------------------------------------------------------------

    print(
        "\n[2/6] Quantifying targeted ROI attribution "
        "for all 240 frozen IG maps"
    )

    files = sorted(
        ATTR_ROOT.glob("seed_*/*m/*.npz")
    )

    if len(files) != 240:
        raise RuntimeError(
            f"Expected 240 IG files, found {len(files)}"
        )

    raw_rows = []

    for n, p in enumerate(files, start=1):
        (
            attr,
            subject,
            landmark,
            seed,
            horizon,
        ) = read_npz(p)

        if (
            seed not in EXPECTED_SEEDS
            or horizon not in EXPECTED_HORIZONS
        ):
            raise RuntimeError(
                f"Unexpected frozen metadata: {p}"
            )

        if not np.isfinite(attr).all():
            raise RuntimeError(
                f"Non-finite attribution: {p}"
            )

        full_abs = np.abs(
            embed_attr(attr)
        )

        total_abs = float(
            full_abs.sum()
        )

        atlas_abs = float(
            full_abs[atlas_union].sum()
        )

        atlas_mean = float(
            full_abs[atlas_union].mean()
        )

        # Families + combined panel.
        masks_to_run = dict(
            available_roi_masks
        )

        masks_to_run[
            "Combined prespecified AD ROI panel"
        ] = ad_panel_mask

        for family, mask in masks_to_run.items():
            nvox = int(mask.sum())

            region_sum = float(
                full_abs[mask].sum()
            )

            region_mean = (
                float(full_abs[mask].mean())
                if nvox
                else np.nan
            )

            share_total = (
                region_sum / total_abs
                if total_abs > 0
                else np.nan
            )

            enrichment = (
                region_mean / atlas_mean
                if (
                    np.isfinite(region_mean)
                    and atlas_mean > 0
                )
                else np.nan
            )

            raw_rows.append(
                {
                    "subject_id": subject,
                    "landmark_id": landmark,
                    "seed": seed,
                    "horizon_months": horizon,
                    "roi_family": family,
                    "roi_voxels": nvox,
                    "roi_abs_ig_sum": region_sum,
                    "roi_mean_abs_ig": region_mean,
                    "share_of_total_abs_ig": share_total,
                    "enrichment_vs_atlas_mean": enrichment,
                    "total_abs_ig": total_abs,
                    "atlas_abs_ig": atlas_abs,
                    "atlas_mean_abs_ig": atlas_mean,
                }
            )

        if (
            n == 1
            or n % 40 == 0
            or n == len(files)
        ):
            print(
                f"  Processed "
                f"{n:>3}/{len(files)} maps"
            )

    raw = pd.DataFrame(raw_rows)

    raw.to_csv(
        OUT_ROOT
        / "Targeted_AD_ROI_All_Maps.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 3. Participant-level five-seed aggregation.
    # -----------------------------------------------------------------

    print(
        "\n[3/6] Aggregating five seeds "
        "within participant"
    )

    participant = (
        raw.groupby(
            [
                "subject_id",
                "landmark_id",
                "horizon_months",
                "roi_family",
            ],
            as_index=False,
        )[
            [
                "roi_abs_ig_sum",
                "roi_mean_abs_ig",
                "share_of_total_abs_ig",
                "enrichment_vs_atlas_mean",
            ]
        ]
        .mean()
    )

    participant.to_csv(
        OUT_ROOT
        / "Targeted_AD_ROI_Participant_5SeedMean.csv",
        index=False,
    )

    if participant["subject_id"].nunique() != 24:
        raise RuntimeError(
            "Expected 24 participants, found "
            f"{participant['subject_id'].nunique()}"
        )

    # -----------------------------------------------------------------
    # 4. Cohort summaries.
    # -----------------------------------------------------------------

    print(
        "\n[4/6] Building cohort-level summaries"
    )

    summary_rows = []

    for (horizon, family), g in participant.groupby(
        [
            "horizon_months",
            "roi_family",
        ],
        sort=True,
    ):
        share = g[
            "share_of_total_abs_ig"
        ].to_numpy(float)

        enrich = g[
            "enrichment_vs_atlas_mean"
        ].to_numpy(float)

        summary_rows.append(
            {
                "horizon_months": int(horizon),
                "roi_family": family,
                "participants": int(
                    g["subject_id"].nunique()
                ),
                "median_share_total": float(
                    np.nanmedian(share)
                ),
                "q1_share_total": quantile(
                    share,
                    0.25,
                ),
                "q3_share_total": quantile(
                    share,
                    0.75,
                ),
                "median_enrichment": float(
                    np.nanmedian(enrich)
                ),
                "q1_enrichment": quantile(
                    enrich,
                    0.25,
                ),
                "q3_enrichment": quantile(
                    enrich,
                    0.75,
                ),
                "fraction_enrichment_gt1": float(
                    np.nanmean(
                        enrich > 1.0
                    )
                ),
                "fraction_enrichment_gt1_25": float(
                    np.nanmean(
                        enrich > 1.25
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT_ROOT
        / "Table_Targeted_AD_ROI_Summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 5. Cross-horizon consistency and paired horizon comparison.
    # -----------------------------------------------------------------

    print(
        "\n[5/6] Evaluating "
        "24-vs-36 month consistency"
    )

    consistency_rows = []

    families = sorted(
        participant[
            "roi_family"
        ].unique()
    )

    for family in families:
        g = participant[
            participant["roi_family"] == family
        ].copy()

        a = g[
            g["horizon_months"] == 24
        ][
            [
                "subject_id",
                "share_of_total_abs_ig",
                "enrichment_vs_atlas_mean",
            ]
        ].rename(
            columns={
                "share_of_total_abs_ig":
                    "share_24",
                "enrichment_vs_atlas_mean":
                    "enrichment_24",
            }
        )

        b = g[
            g["horizon_months"] == 36
        ][
            [
                "subject_id",
                "share_of_total_abs_ig",
                "enrichment_vs_atlas_mean",
            ]
        ].rename(
            columns={
                "share_of_total_abs_ig":
                    "share_36",
                "enrichment_vs_atlas_mean":
                    "enrichment_36",
            }
        )

        m = a.merge(
            b,
            on="subject_id",
            how="inner",
        )

        rho_share, p_share = spearmanr(
            m["share_24"],
            m["share_36"],
        )

        rho_enrich, p_enrich = spearmanr(
            m["enrichment_24"],
            m["enrichment_36"],
        )

        w_stat, w_p = paired_wilcoxon(
            m["enrichment_24"],
            m["enrichment_36"],
        )

        consistency_rows.append(
            {
                "roi_family": family,
                "participants": len(m),
                "spearman_share_24_vs_36":
                    float(rho_share),
                "spearman_share_p":
                    float(p_share),
                "spearman_enrichment_24_vs_36":
                    float(rho_enrich),
                "spearman_enrichment_p":
                    float(p_enrich),
                "median_enrichment_change_36_minus_24":
                    float(
                        np.median(
                            m["enrichment_36"]
                            - m["enrichment_24"]
                        )
                    ),
                "wilcoxon_stat_enrichment_24_vs_36":
                    w_stat,
                "wilcoxon_p_enrichment_24_vs_36":
                    w_p,
            }
        )

    consistency = pd.DataFrame(
        consistency_rows
    )

    consistency.to_csv(
        OUT_ROOT
        / "Table_Targeted_AD_ROI_24_vs_36_Consistency.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 6. Figures + report.
    # -----------------------------------------------------------------

    print(
        "\n[6/6] Creating publication "
        "figures and report"
    )

    plot_families = [
        f
        for f in ROI_FAMILIES.keys()
        if (
            f
            in summary[
                "roi_family"
            ].unique()
            and f not in missing
        )
    ]

    for horizon in EXPECTED_HORIZONS:
        g = summary[
            (
                summary[
                    "horizon_months"
                ]
                == horizon
            )
            & (
                summary[
                    "roi_family"
                ].isin(
                    plot_families
                )
            )
        ].copy()

        g = g.sort_values(
            "median_enrichment",
            ascending=True,
        )

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        ax.barh(
            g["roi_family"],
            g["median_enrichment"],
        )

        ax.axvline(
            1.0,
            linewidth=1,
        )

        ax.set_xlabel(
            "Median attribution enrichment "
            "vs atlas-covered brain"
        )

        ax.set_ylabel(
            "Prespecified AD-relevant "
            "ROI family"
        )

        ax.set_title(
            "Targeted AD-relevant regional "
            f"attribution at {horizon} months"
        )

        fig.tight_layout()

        fig.savefig(
            FIG_ROOT
            / (
                "Figure_Targeted_AD_ROI_"
                f"Enrichment_{horizon}m.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            FIG_ROOT
            / (
                "Figure_Targeted_AD_ROI_"
                f"Enrichment_{horizon}m.pdf"
            ),
            bbox_inches="tight",
        )

        plt.close(fig)

    # Cross-horizon scatter for ROI median enrichments.
    a = summary[
        (
            summary[
                "horizon_months"
            ]
            == 24
        )
        & (
            summary[
                "roi_family"
            ].isin(
                plot_families
            )
        )
    ][
        [
            "roi_family",
            "median_enrichment",
        ]
    ].rename(
        columns={
            "median_enrichment": "e24"
        }
    )

    b = summary[
        (
            summary[
                "horizon_months"
            ]
            == 36
        )
        & (
            summary[
                "roi_family"
            ].isin(
                plot_families
            )
        )
    ][
        [
            "roi_family",
            "median_enrichment",
        ]
    ].rename(
        columns={
            "median_enrichment": "e36"
        }
    )

    m = a.merge(
        b,
        on="roi_family",
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        m["e24"],
        m["e36"],
    )

    lo = float(
        min(
            m["e24"].min(),
            m["e36"].min(),
        )
    )

    hi = float(
        max(
            m["e24"].max(),
            m["e36"].max(),
        )
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linewidth=1,
    )

    for _, r in m.iterrows():
        ax.annotate(
            r["roi_family"],
            (
                r["e24"],
                r["e36"],
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Median enrichment at 24 months"
    )

    ax.set_ylabel(
        "Median enrichment at 36 months"
    )

    ax.set_title(
        "Cross-horizon consistency of "
        "targeted AD-relevant ROIs"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_ROOT
        / "Figure_Targeted_AD_ROI_24_vs_36.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_ROOT
        / "Figure_Targeted_AD_ROI_24_vs_36.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------

    lines = []

    lines.append("=" * 112)

    lines.append(
        "TARGETED AD-RELEVANT ROI "
        "ATTRIBUTION ANALYSIS — SUMMARY"
    )

    lines.append("=" * 112)

    lines.append(
        f"Frozen IG maps processed : "
        f"{len(files)} / 240"
    )

    lines.append(
        f"Participants             : "
        f"{participant['subject_id'].nunique()} / 24"
    )

    lines.append(
        f"Prespecified ROI families: "
        f"{len(ROI_FAMILIES)}"
    )

    lines.append(
        f"Available atlas ROI families: "
        f"{len(available_roi_masks)}"
    )

    lines.append(
        f"Unavailable atlas ROI families: "
        f"{len(missing)}"
    )

    if missing:
        lines.append(
            "Unavailable (not substituted): "
            + ", ".join(missing)
        )

    lines.append("")

    for horizon in EXPECTED_HORIZONS:
        lines.append(
            f"TARGETED ROI RESULTS — "
            f"{horizon} MONTHS"
        )

        lines.append("-" * 112)

        g = summary[
            (
                summary[
                    "horizon_months"
                ]
                == horizon
            )
            & (
                summary[
                    "roi_family"
                ].isin(
                    plot_families
                )
            )
        ].sort_values(
            "median_enrichment",
            ascending=False,
        )

        for _, r in g.iterrows():
            lines.append(
                f"{r['roi_family']:<30s} | "
                f"median share="
                f"{r['median_share_total']:.6f} | "
                f"median enrichment="
                f"{r['median_enrichment']:.3f} "
                f"[Q1={r['q1_enrichment']:.3f}, "
                f"Q3={r['q3_enrichment']:.3f}] | "
                f"frac>1="
                f"{r['fraction_enrichment_gt1']:.3f}"
            )

        panel = summary[
            (
                summary[
                    "horizon_months"
                ]
                == horizon
            )
            & (
                summary[
                    "roi_family"
                ]
                == "Combined prespecified AD ROI panel"
            )
        ].iloc[0]

        lines.append("")

        lines.append(
            "Combined AD ROI panel | "
            f"median share="
            f"{panel['median_share_total']:.6f} | "
            f"median enrichment="
            f"{panel['median_enrichment']:.3f} | "
            f"frac>1="
            f"{panel['fraction_enrichment_gt1']:.3f}"
        )

        lines.append("")

    lines.append(
        "CROSS-HORIZON PARTICIPANT-LEVEL "
        "CONSISTENCY"
    )

    lines.append("-" * 112)

    for _, r in consistency.iterrows():
        lines.append(
            f"{r['roi_family']:<35s} | "
            f"rho_share="
            f"{r['spearman_share_24_vs_36']:.3f} | "
            f"rho_enrichment="
            f"{r['spearman_enrichment_24_vs_36']:.3f} | "
            f"median Δenrichment="
            f"{r['median_enrichment_change_36_minus_24']:.3f} | "
            f"Wilcoxon p="
            f"{r['wilcoxon_p_enrichment_24_vs_36']:.4g}"
        )

    lines.append("")

    lines.append(
        "STATUS: COMPLETE — targeted AD-relevant "
        "ROI analysis finished."
    )

    lines.append("")

    lines.append(
        "Interpretation boundary: enrichment >1 indicates "
        "above-average model attribution density within the "
        "atlas-covered brain. This supports anatomical "
        "consistency of model reliance, not causal disease "
        "localization or independent biomarker validation."
    )

    lines.append(
        "No model training, inference, model selection, "
        "threshold optimization, or IG recomputation was performed."
    )

    report = "\n".join(lines)

    with (
        OUT_ROOT / "targeted_ad_roi_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    metadata = {
        "experiment": "ANATOMICAL_VALIDATION_V1",
        "step": (
            "22_targeted_ad_roi_"
            "attribution_analysis_FIXED"
        ),
        "participants": 24,
        "seeds": EXPECTED_SEEDS,
        "horizons_months": EXPECTED_HORIZONS,
        "roi_families": ROI_FAMILIES,
        "available_roi_families": list(
            available_roi_masks.keys()
        ),
        "unavailable_roi_families": missing,
        "interpretation_boundary": (
            "Atlas-defined model attribution analysis; "
            "not causal biomarker validation."
        ),
    }

    with (
        OUT_ROOT / "targeted_ad_roi_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(report)
    print()

    print("Saved:")

    print(
        " ",
        OUT_ROOT
        / "Table_Targeted_AD_ROI_Label_Audit.csv",
    )

    print(
        " ",
        OUT_ROOT
        / "Targeted_AD_ROI_All_Maps.csv",
    )

    print(
        " ",
        OUT_ROOT
        / "Targeted_AD_ROI_Participant_5SeedMean.csv",
    )

    print(
        " ",
        OUT_ROOT
        / "Table_Targeted_AD_ROI_Summary.csv",
    )

    print(
        " ",
        OUT_ROOT
        / "Table_Targeted_AD_ROI_24_vs_36_Consistency.csv",
    )

    print(
        " ",
        OUT_ROOT
        / "targeted_ad_roi_report.txt",
    )

    print(
        " ",
        FIG_ROOT,
    )


if __name__ == "__main__":
    main()
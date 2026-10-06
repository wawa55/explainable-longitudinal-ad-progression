from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SUBJECT = "002_S_4270"
LANDMARK = "002_S_4270__20151201__00057"
HORIZON = "24m"

SEEDS = [2026, 3407, 5813, 7291, 9109]

MRI_PATH = (
    ROOT
    / "03_preprocessed"
    / "production_final"
    / SUBJECT
    / "I398531_FINAL.nii.gz"
)

ATTR_ROOT = (
    ROOT
    / "08_results"
    / "ADNI_XAI_TEST_PRODUCTION_V1"
    / "attributions"
)

OUT_DIR = (
    ROOT
    / "08_results"
    / "publication_package_v1"
    / "figures"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SHAPE = np.array(
    [160, 192, 160],
    dtype=int,
)


# ============================================================
# EXACT PRODUCTION CENTER CROP
# ============================================================

def center_crop_3d(volume, target_shape=TARGET_SHAPE):

    source_shape = np.asarray(
        volume.shape,
        dtype=int,
    )

    target_shape = np.asarray(
        target_shape,
        dtype=int,
    )

    if np.any(source_shape < target_shape):
        raise ValueError(
            f"Source shape {tuple(source_shape)} "
            f"is smaller than target "
            f"{tuple(target_shape)}."
        )

    start = (
        source_shape - target_shape
    ) // 2

    end = start + target_shape

    cropped = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]

    if cropped.shape != tuple(target_shape):
        raise RuntimeError(
            f"Crop failed: "
            f"{cropped.shape} != "
            f"{tuple(target_shape)}"
        )

    return cropped


# ============================================================
# LOAD MRI
# ============================================================

if not MRI_PATH.exists():
    raise FileNotFoundError(
        f"MRI not found:\n{MRI_PATH}"
    )

mri_img = nib.load(
    str(MRI_PATH)
)

mri = (
    mri_img
    .get_fdata()
    .astype(np.float32)
)

print(
    "Original MRI shape:",
    mri.shape,
)

mri = center_crop_3d(
    mri
)

print(
    "MRI model-space shape:",
    mri.shape,
)


# ============================================================
# LOAD FIVE FROZEN ATTRIBUTION MAPS
# ============================================================

attrs = []

actual_risks = []
baseline_risks = []
risk_differences = []
convergence_deltas = []

for seed in SEEDS:

    filename = (
        f"{SUBJECT}__"
        f"{LANDMARK}__"
        f"seed{seed}__"
        f"{HORIZON}.npz"
    )

    path = (
        ATTR_ROOT
        / f"seed_{seed}"
        / HORIZON
        / filename
    )

    print(
        f"Loading seed {seed}:"
    )

    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing attribution file:\n"
            f"{path}"
        )

    z = np.load(
        path,
        allow_pickle=True,
    )

    attr = (
        z["attribution"]
        .astype(np.float32)
    )

    if attr.shape != tuple(TARGET_SHAPE):
        raise RuntimeError(
            f"Shape mismatch for "
            f"seed {seed}: "
            f"{attr.shape}"
        )

    attrs.append(
        attr
    )

    actual_risks.append(
        float(z["actual_risk"])
    )

    baseline_risks.append(
        float(z["baseline_risk"])
    )

    risk_differences.append(
        float(z["risk_difference"])
    )

    convergence_deltas.append(
        float(z["convergence_delta"])
    )


attrs = np.stack(
    attrs,
    axis=0,
)

print()
print(
    "Attribution stack:",
    attrs.shape,
)


# ============================================================
# FIVE-SEED ATTRIBUTION AGGREGATION
# ============================================================

mean_abs_ig = np.mean(
    np.abs(attrs),
    axis=0,
)


# ============================================================
# BRAIN MASK
# ============================================================

brain_mask = (
    np.isfinite(mri)
    & (np.abs(mri) > 1e-6)
)

if not np.any(brain_mask):
    raise RuntimeError(
        "No brain voxels detected."
    )


brain_attr = (
    mean_abs_ig[brain_mask]
)

if brain_attr.size == 0:
    raise RuntimeError(
        "No IG values inside brain mask."
    )


# ============================================================
# TOP 5% VISUALIZATION THRESHOLD
# ============================================================

threshold = np.percentile(
    brain_attr,
    95.0,
)

vmax_raw = np.percentile(
    brain_attr,
    99.5,
)

if vmax_raw <= threshold:
    vmax_raw = float(
        np.max(brain_attr)
    )

print()
print(
    "Top-5% threshold:",
    threshold,
)

print(
    "Raw display vmax:",
    vmax_raw,
)


# ============================================================
# NORMALIZE DISPLAYED ATTRIBUTION TO 0--1
# ============================================================

scaled_attr = (
    mean_abs_ig - threshold
) / (
    vmax_raw - threshold + 1e-12
)

scaled_attr = np.clip(
    scaled_attr,
    0.0,
    1.0,
)


overlay = np.full(
    mean_abs_ig.shape,
    np.nan,
    dtype=np.float32,
)

keep = (
    brain_mask
    & (mean_abs_ig >= threshold)
)

overlay[keep] = (
    scaled_attr[keep]
)


# ============================================================
# OBJECTIVE SLICE SELECTION
# ============================================================

weights = np.where(
    brain_mask,
    mean_abs_ig,
    0.0,
)

total_weight = np.sum(
    weights
)

if total_weight <= 0:
    raise RuntimeError(
        "Invalid attribution weights."
    )

coords = np.indices(
    weights.shape
)

center = []

for axis in range(3):

    value = (
        np.sum(
            coords[axis]
            * weights
        )
        / total_weight
    )

    center.append(
        int(round(value))
    )

sx, sy, sz = center

print(
    "Attribution-weighted centre:",
    (sx, sy, sz),
)


# ============================================================
# MRI DISPLAY WINDOW
# ============================================================

mri_values = (
    mri[brain_mask]
)

mri_vmin = np.percentile(
    mri_values,
    1.0,
)

mri_vmax = np.percentile(
    mri_values,
    99.0,
)


# ============================================================
# CLEAN MRI BACKGROUND
# ============================================================

mri_display = mri.copy()

mri_display[
    ~brain_mask
] = np.nan


gray_cmap = plt.cm.gray.copy()

gray_cmap.set_bad(
    color="black"
)


# ============================================================
# PREPARE THREE ORTHOGONAL VIEWS
# ============================================================

views = [
    (
        "Sagittal",
        mri_display[sx, :, :],
        overlay[sx, :, :],
    ),
    (
        "Coronal",
        mri_display[:, sy, :],
        overlay[:, sy, :],
    ),
    (
        "Axial",
        mri_display[:, :, sz],
        overlay[:, :, sz],
    ),
]


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(11.5, 7.2),
    constrained_layout=True,
)

fig.patch.set_facecolor(
    "white"
)


for column, (
    view_name,
    base,
    heat,
) in enumerate(views):

    base = np.rot90(
        base
    )

    heat = np.rot90(
        heat
    )

    # --------------------------------------------------------
    # MRI ONLY
    # --------------------------------------------------------

    axes[
        0,
        column,
    ].imshow(
        base,
        cmap=gray_cmap,
        vmin=mri_vmin,
        vmax=mri_vmax,
        interpolation="bilinear",
    )

    axes[
        0,
        column,
    ].set_title(
        view_name,
        fontsize=12,
        fontweight="bold",
    )

    axes[
        0,
        column,
    ].axis(
        "off"
    )


    # --------------------------------------------------------
    # MRI + IG
    # --------------------------------------------------------

    axes[
        1,
        column,
    ].imshow(
        base,
        cmap=gray_cmap,
        vmin=mri_vmin,
        vmax=mri_vmax,
        interpolation="bilinear",
    )

    im = axes[
        1,
        column,
    ].imshow(
        heat,
        cmap="hot",
        alpha=0.82,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axes[
        1,
        column,
    ].set_title(
        f"{view_name} + IG",
        fontsize=12,
        fontweight="bold",
    )

    axes[
        1,
        column,
    ].axis(
        "off"
    )


# ============================================================
# ROW LABELS
# ============================================================

fig.text(
    0.01,
    0.70,
    "T1-weighted MRI",
    rotation=90,
    va="center",
    ha="center",
    fontsize=11,
)

fig.text(
    0.01,
    0.28,
    "MRI + Integrated Gradients",
    rotation=90,
    va="center",
    ha="center",
    fontsize=11,
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Representative post-hoc Integrated-Gradients explanation\n"
    "24-month prediction horizon | "
    "five-seed mean absolute attribution",
    fontsize=14,
    fontweight="bold",
)


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    im,
    ax=axes[1, :],
    orientation="horizontal",
    shrink=0.55,
    pad=0.04,
)

cbar.set_label(
    "Relative five-seed mean absolute IG attribution",
    fontsize=10,
)

cbar.set_ticks(
    [0.0, 0.25, 0.50, 0.75, 1.0]
)


# ============================================================
# SAVE OUTPUT
# ============================================================

png_out = (
    OUT_DIR
    / "Figure_PostHoc_XAI_MRI_Overlay_Publication.png"
)

pdf_out = (
    OUT_DIR
    / "Figure_PostHoc_XAI_MRI_Overlay_Publication.pdf"
)

plt.savefig(
    png_out,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    pdf_out,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(
    fig
)


# ============================================================
# AUDIT OUTPUT
# ============================================================

print()
print("=" * 80)

print(
    "PUBLICATION POST-HOC XAI FIGURE COMPLETE"
)

print("=" * 80)

print(
    "Internal subject:",
    SUBJECT,
)

print(
    "Landmark:",
    LANDMARK,
)

print(
    "Horizon:",
    HORIZON,
)

print(
    "Seeds:",
    SEEDS,
)

print(
    "MRI shape:",
    mri.shape,
)

print(
    "Attribution stack:",
    attrs.shape,
)

print(
    "Weighted centre:",
    (sx, sy, sz),
)

print(
    "Mean five-seed actual risk:",
    f"{np.mean(actual_risks):.6f}",
)

print(
    "Maximum absolute convergence delta:",
    f"{np.max(np.abs(convergence_deltas)):.3e}",
)

print(
    "Visualization threshold:",
    "top 5% attribution magnitude",
)

print(
    "Displayed attribution scale:",
    "relative 0--1 visualization scale",
)

print()
print(
    "Saved:"
)

print(
    png_out
)

print(
    pdf_out
)
# -*- coding: utf-8 -*-
"""
19_anatomical_spatial_alignment_audit.py

READ-ONLY spatial-alignment audit for anatomical validation of frozen IG maps.

Goal
----
Determine whether the frozen production MRI volumes and the local MNI template
share a common spatial grid/affine, so that model-space IG maps can be embedded
back into the full MRI grid and anatomically parcellated without re-registering
the frozen images.

This script:
- reads the frozen longitudinal landmark CSV,
- resolves the latest MRI used by each of the 24 frozen XAI landmarks,
- checks MRI existence, shape, voxel size, orientation, and affine,
- checks consistency against the local MNI template,
- verifies the deterministic center-crop geometry (197x233x189 -> 160x192x160),
- checks that every IG file has the expected spatial shape,
- writes CSV/JSON/TXT audit reports.

This script DOES NOT:
- retrain a model,
- run model inference,
- recompute IG,
- resample or register MRI,
- alter frozen MRI or attribution files.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LANDMARK_FILE = ROOT / "06_splits" / "ADNI_LONGITUDINAL_LANDMARK_DATASET_V1.csv"
ATTR_ROOT = (
    ROOT / "08_results" / "ADNI_XAI_TEST_PRODUCTION_V1" / "attributions"
)
MNI_FILE = (
    ROOT / "03_preprocessed" / "templates" / "MNI152_T1_1mm.nii.gz"
)
OUT_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "spatial_alignment_audit"
)

EXPECTED_SEEDS = [2026, 3407, 5813, 7291, 9109]
EXPECTED_HORIZONS = ["24m", "36m"]
SOURCE_SHAPE = (197, 233, 189)
MODEL_SHAPE = (160, 192, 160)

AFFINE_ATOL = 1e-4
ZOOM_ATOL = 1e-4


def parse_list_field(value):
    if isinstance(value, (list, tuple)):
        return list(value)

    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    # First try Python-literal representation.
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return list(obj)
    except Exception:
        pass

    # Conservative fallback.
    # Handles semicolon- or pipe-separated values if encountered.
    for sep in [";", "|"]:
        if sep in s:
            return [x.strip().strip("'\"") for x in s.split(sep) if x.strip()]

    return [s.strip().strip("'\"")]


def center_crop_bounds(source_shape, target_shape):
    source = np.asarray(source_shape, dtype=int)
    target = np.asarray(target_shape, dtype=int)

    if np.any(target > source):
        raise ValueError(
            f"Target shape {tuple(target)} exceeds source shape {tuple(source)}"
        )

    start = (source - target) // 2
    end = start + target
    return tuple(int(x) for x in start), tuple(int(x) for x in end)


def axis_codes(img):
    return "".join(nib.aff2axcodes(img.affine))


def zooms3(img):
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def affine_max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def load_scalar(z, key):
    x = z[key]
    if np.asarray(x).ndim == 0:
        return np.asarray(x).item()
    return x


def get_xai_records():
    """
    Use the frozen NPZ metadata itself as the authoritative list of XAI subjects/
    landmarks, taking seed 2026 / 24m only to avoid duplicates.
    """
    files = sorted((ATTR_ROOT / "seed_2026" / "24m").glob("*.npz"))
    rows = []

    for p in files:
        with np.load(p, allow_pickle=True) as z:
            subject_id = str(load_scalar(z, "subject_id"))
            landmark_id = str(load_scalar(z, "landmark_id"))
            spatial_shape = tuple(int(v) for v in np.asarray(z["spatial_shape"]).tolist())
            attr_shape = tuple(np.asarray(z["attribution"]).shape)

        rows.append(
            {
                "subject_id": subject_id,
                "landmark_id": landmark_id,
                "seed2026_24m_npz": str(p),
                "npz_spatial_shape": spatial_shape,
                "attribution_shape": attr_shape,
            }
        )

    return rows


def resolve_latest_mri(df, subject_id, landmark_id):
    hit = df.loc[
        (df["subject_id"].astype(str) == str(subject_id))
        & (df["landmark_id"].astype(str) == str(landmark_id))
    ].copy()

    if len(hit) != 1:
        raise RuntimeError(
            f"Expected exactly one landmark row for {subject_id} / {landmark_id}, "
            f"found {len(hit)}"
        )

    row = hit.iloc[0]
    paths = parse_list_field(row["image_paths"])

    if not paths:
        raise RuntimeError(
            f"No image_paths for {subject_id} / {landmark_id}"
        )

    return row, Path(str(paths[-1])), paths


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    print("=" * 110)
    print("ANATOMICAL VALIDATION — SPATIAL ALIGNMENT AUDIT")
    print("=" * 110)

    for required in [LANDMARK_FILE, ATTR_ROOT, MNI_FILE]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found:\n{required}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    start, end = center_crop_bounds(SOURCE_SHAPE, MODEL_SHAPE)

    print(f"Landmark file : {LANDMARK_FILE}")
    print(f"Attribution   : {ATTR_ROOT}")
    print(f"MNI template  : {MNI_FILE}")
    print(f"Output        : {OUT_ROOT}")
    print()
    print(f"Source shape  : {SOURCE_SHAPE}")
    print(f"Model shape   : {MODEL_SHAPE}")
    print(f"Crop start    : {start}")
    print(f"Crop end      : {end}")
    print()

    # ------------------------------------------------------------------
    # MNI reference
    # ------------------------------------------------------------------
    mni = nib.load(str(MNI_FILE))
    mni_shape = tuple(int(x) for x in mni.shape[:3])
    mni_zooms = zooms3(mni)
    mni_orient = axis_codes(mni)
    mni_affine = np.asarray(mni.affine, dtype=float)

    print("LOCAL MNI TEMPLATE")
    print("-" * 110)
    print("Shape       :", mni_shape)
    print("Voxel sizes :", mni_zooms)
    print("Orientation :", mni_orient)
    print("Affine:")
    print(mni_affine)
    print()

    # ------------------------------------------------------------------
    # Frozen longitudinal dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(LANDMARK_FILE)

    required_cols = {
        "subject_id",
        "landmark_id",
        "image_paths",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            "Missing required landmark columns: " + ", ".join(sorted(missing))
        )

    xai_records = get_xai_records()

    print("FROZEN XAI COHORT")
    print("-" * 110)
    print("Subjects/landmarks found from seed 2026, 24m:", len(xai_records))
    print()

    if len(xai_records) != 24:
        print(
            "WARNING: expected 24 frozen XAI records, "
            f"found {len(xai_records)}"
        )

    audit_rows = []
    errors = []

    for i, rec in enumerate(xai_records, start=1):
        subject_id = rec["subject_id"]
        landmark_id = rec["landmark_id"]

        out = {
            "subject_id": subject_id,
            "landmark_id": landmark_id,
            "landmark_match_ok": False,
            "latest_mri_path": "",
            "latest_mri_exists": False,
            "n_mri_history": None,
            "mri_shape": "",
            "mri_zooms": "",
            "mri_orientation": "",
            "shape_matches_expected_source": False,
            "shape_matches_mni": False,
            "zooms_match_mni": False,
            "orientation_matches_mni": False,
            "affine_matches_mni": False,
            "affine_max_abs_diff": None,
            "npz_spatial_shape": str(rec["npz_spatial_shape"]),
            "attribution_shape": str(rec["attribution_shape"]),
            "attribution_shape_ok": rec["attribution_shape"] == MODEL_SHAPE,
            "crop_geometry_ok": False,
            "error": "",
        }

        try:
            row, latest_path, all_paths = resolve_latest_mri(
                df,
                subject_id,
                landmark_id,
            )

            out["landmark_match_ok"] = True
            out["latest_mri_path"] = str(latest_path)
            out["n_mri_history"] = len(all_paths)
            out["latest_mri_exists"] = latest_path.exists()

            if not latest_path.exists():
                raise FileNotFoundError(f"Latest MRI not found: {latest_path}")

            img = nib.load(str(latest_path))

            shape = tuple(int(x) for x in img.shape[:3])
            z = zooms3(img)
            orient = axis_codes(img)
            aff = np.asarray(img.affine, dtype=float)

            out["mri_shape"] = str(shape)
            out["mri_zooms"] = str(z)
            out["mri_orientation"] = orient

            out["shape_matches_expected_source"] = shape == SOURCE_SHAPE
            out["shape_matches_mni"] = shape == mni_shape
            out["zooms_match_mni"] = bool(
                np.allclose(z, mni_zooms, atol=ZOOM_ATOL, rtol=0)
            )
            out["orientation_matches_mni"] = orient == mni_orient

            diff = affine_max_abs_diff(aff, mni_affine)
            out["affine_max_abs_diff"] = diff
            out["affine_matches_mni"] = bool(
                np.allclose(aff, mni_affine, atol=AFFINE_ATOL, rtol=0)
            )

            # Reconstruct full-grid array and crop it again, to prove exact
            # index geometry without writing any output volume.
            dummy = np.zeros(SOURCE_SHAPE, dtype=np.uint8)
            sl = tuple(slice(start[d], end[d]) for d in range(3))
            crop = dummy[sl]
            out["crop_geometry_ok"] = tuple(crop.shape) == MODEL_SHAPE

        except Exception as e:
            out["error"] = repr(e)
            errors.append(
                {
                    "subject_id": subject_id,
                    "landmark_id": landmark_id,
                    "error": repr(e),
                }
            )

        audit_rows.append(out)

        print(
            f"[{i:02d}/{len(xai_records):02d}] "
            f"{subject_id} | "
            f"shape={out['mri_shape']} | "
            f"orient={out['mri_orientation']} | "
            f"affine_match={out['affine_matches_mni']} | "
            f"err={out['error'][:60]}"
        )

    # ------------------------------------------------------------------
    # Check all 240 IG arrays
    # ------------------------------------------------------------------
    ig_files = sorted(ATTR_ROOT.glob("seed_*/*m/*.npz"))
    ig_shape_bad = []
    ig_nonfinite = []

    print()
    print("CHECKING ALL FROZEN IG ARRAYS")
    print("-" * 110)

    for j, p in enumerate(ig_files, start=1):
        with np.load(p, allow_pickle=True) as z:
            a = np.asarray(z["attribution"])

        if tuple(a.shape) != MODEL_SHAPE:
            ig_shape_bad.append(
                {"file": str(p), "shape": tuple(a.shape)}
            )

        if not np.isfinite(a).all():
            ig_nonfinite.append(str(p))

        if j == 1 or j % 40 == 0 or j == len(ig_files):
            print(f"Checked {j:>3}/{len(ig_files)} IG files")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = len(audit_rows)

    def ntrue(key):
        return sum(bool(r[key]) for r in audit_rows)

    summary = {
        "experiment": "ANATOMICAL_VALIDATION_V1_SPATIAL_ALIGNMENT_AUDIT",
        "read_only": True,
        "landmark_file": str(LANDMARK_FILE),
        "mni_file": str(MNI_FILE),
        "attribution_root": str(ATTR_ROOT),
        "source_shape_expected": list(SOURCE_SHAPE),
        "model_shape_expected": list(MODEL_SHAPE),
        "center_crop_start": list(start),
        "center_crop_end": list(end),
        "mni_shape": list(mni_shape),
        "mni_zooms": list(mni_zooms),
        "mni_orientation": mni_orient,
        "xai_records": n,
        "latest_mri_exists": ntrue("latest_mri_exists"),
        "source_shape_matches": ntrue("shape_matches_expected_source"),
        "mri_shape_matches_mni": ntrue("shape_matches_mni"),
        "mri_zooms_match_mni": ntrue("zooms_match_mni"),
        "mri_orientation_matches_mni": ntrue("orientation_matches_mni"),
        "mri_affine_matches_mni": ntrue("affine_matches_mni"),
        "crop_geometry_ok": ntrue("crop_geometry_ok"),
        "attribution_shape_ok_xai_records": ntrue("attribution_shape_ok"),
        "all_ig_files_checked": len(ig_files),
        "all_ig_bad_shape_count": len(ig_shape_bad),
        "all_ig_nonfinite_count": len(ig_nonfinite),
        "errors": errors,
        "bad_ig_shapes": ig_shape_bad,
        "nonfinite_ig_files": ig_nonfinite,
        "direct_mni_parcellation_ready": False,
    }

    # Strict rule: direct MNI parcellation is allowed only if every resolved
    # MRI shares shape, voxel size, orientation and affine with the local MNI.
    direct_ready = (
        n == 24
        and len(errors) == 0
        and ntrue("latest_mri_exists") == n
        and ntrue("shape_matches_expected_source") == n
        and ntrue("shape_matches_mni") == n
        and ntrue("zooms_match_mni") == n
        and ntrue("orientation_matches_mni") == n
        and ntrue("affine_matches_mni") == n
        and ntrue("crop_geometry_ok") == n
        and len(ig_shape_bad) == 0
        and len(ig_nonfinite) == 0
        and len(ig_files) == 240
    )

    summary["direct_mni_parcellation_ready"] = direct_ready

    write_csv(OUT_ROOT / "spatial_alignment_audit.csv", audit_rows)

    with (OUT_ROOT / "spatial_alignment_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    lines = [
        "=" * 110,
        "ANATOMICAL VALIDATION — SPATIAL ALIGNMENT AUDIT SUMMARY",
        "=" * 110,
        f"XAI subject/landmark records       : {n} / 24",
        f"Latest MRI files found             : {ntrue('latest_mri_exists')} / {n}",
        f"Expected source shape matches      : {ntrue('shape_matches_expected_source')} / {n}",
        f"MRI shape matches local MNI        : {ntrue('shape_matches_mni')} / {n}",
        f"MRI voxel sizes match local MNI    : {ntrue('zooms_match_mni')} / {n}",
        f"MRI orientation matches local MNI  : {ntrue('orientation_matches_mni')} / {n}",
        f"MRI affine matches local MNI       : {ntrue('affine_matches_mni')} / {n}",
        f"Center-crop geometry valid         : {ntrue('crop_geometry_ok')} / {n}",
        f"Frozen IG files checked            : {len(ig_files)} / 240",
        f"IG files with wrong shape          : {len(ig_shape_bad)}",
        f"IG files with non-finite values    : {len(ig_nonfinite)}",
        f"Resolution errors                  : {len(errors)}",
        "",
        f"Crop bounds                        : start={start}, end={end}",
        "",
    ]

    if direct_ready:
        lines += [
            "STATUS: PASS — DIRECT MNI PARCELLATION READY",
            "",
            "All frozen XAI MRIs share the local MNI grid under the strict audit.",
            "The next step may safely resample an atlas to this local MNI template",
            "and embed each 160x192x160 IG map back into the full 197x233x189 grid",
            "using the frozen center-crop bounds, without re-registering participants.",
        ]
    else:
        lines += [
            "STATUS: REGISTRATION/GRID CHECK REQUIRED",
            "",
            "Do NOT perform anatomical regional attribution yet.",
            "At least one strict grid/affine requirement failed. Review the CSV report",
            "before deciding whether atlas-to-template resampling or subject registration",
            "is required.",
        ]

    lines += [
        "",
        "READ-ONLY: no model training, inference, IG recomputation, image registration,",
        "resampling, or frozen-file modification was performed.",
    ]

    report = "\n".join(lines)

    with (OUT_ROOT / "spatial_alignment_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print()
    print(report)
    print()
    print("Saved:")
    print(" ", OUT_ROOT / "spatial_alignment_audit.csv")
    print(" ", OUT_ROOT / "spatial_alignment_summary.json")
    print(" ", OUT_ROOT / "spatial_alignment_report.txt")


if __name__ == "__main__":
    main()

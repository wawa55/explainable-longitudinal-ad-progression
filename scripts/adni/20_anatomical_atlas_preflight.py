# -*- coding: utf-8 -*-
"""
20_anatomical_atlas_preflight.py

READ-ONLY atlas preflight for frozen IG anatomical validation.

Purpose
-------
Find a usable anatomical atlas already available on the local machine and
verify that it can be mapped to the exact local MNI152_T1_1mm.nii.gz grid.

Preferred atlas family
----------------------
Harvard-Oxford cortical + subcortical atlases (FSL distribution), because they
provide named cortical and subcortical regions including hippocampus and
amygdala and are commonly used in MNI space.

This script DOES NOT:
- download anything,
- change the frozen MRI/XAI data,
- retrain models,
- run inference,
- recompute IG,
- resample any atlas yet.

It only inventories candidate atlas files and reports what is available.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MNI_FILE = ROOT / "03_preprocessed" / "templates" / "MNI152_T1_1mm.nii.gz"

OUT_ROOT = (
    ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "atlas_preflight"
)

# Directories worth searching on a Windows / conda setup.
SEARCH_ROOTS = [
    ROOT / "03_preprocessed" / "templates",
    ROOT / "atlases",
    ROOT / "resources",
    Path.home() / "nilearn_data",
    Path.home() / ".cache",
    Path(os.environ.get("FSLDIR", "")) / "data" / "atlases"
        if os.environ.get("FSLDIR") else None,
]

# Known/likely atlas filename tokens.
ATLAS_TOKENS = [
    "harvardoxford",
    "harvard_oxford",
    "harvard-oxford",
    "aal",
    "neuromorphometrics",
    "desikan",
    "destrieux",
    "mni_structural",
    "mni-maxprob",
]

LABEL_EXTS = {".xml", ".txt", ".csv", ".tsv", ".json"}
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    s = path.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")


def candidate_score(name: str) -> int:
    x = name.lower()
    score = 0
    for i, token in enumerate(ATLAS_TOKENS):
        if token in x:
            score += 100 - i
    if "maxprob" in x:
        score += 25
    if "cort" in x:
        score += 10
    if "sub" in x:
        score += 10
    if "1mm" in x or "1.0mm" in x:
        score += 10
    return score


def axis_codes(img):
    return "".join(nib.aff2axcodes(img.affine))


def zooms3(img):
    return tuple(float(v) for v in img.header.get_zooms()[:3])


def safe_load_nifti(path: Path):
    try:
        img = nib.load(str(path))
        return {
            "ok": True,
            "shape": tuple(int(v) for v in img.shape[:3]),
            "zooms": zooms3(img),
            "orientation": axis_codes(img),
            "affine": np.asarray(img.affine, dtype=float),
            "dtype": str(img.get_data_dtype()),
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def find_label_sidecars(atlas_path: Path):
    parent = atlas_path.parent
    stem = atlas_path.name.lower().replace(".nii.gz", "").replace(".nii", "")

    matches = []

    # Exact-ish neighbours.
    for p in parent.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in LABEL_EXTS:
            continue

        name = p.name.lower()
        if any(tok in name for tok in ["harvard", "aal", "neuromorph", "label", "atlas"]):
            matches.append(str(p))
        elif stem[:20] in name:
            matches.append(str(p))

    return sorted(set(matches))


def main():
    print("=" * 110)
    print("ANATOMICAL VALIDATION — ATLAS PREFLIGHT")
    print("=" * 110)

    if not MNI_FILE.exists():
        raise FileNotFoundError(f"Local MNI template not found:\n{MNI_FILE}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mni = nib.load(str(MNI_FILE))
    mni_shape = tuple(int(v) for v in mni.shape[:3])
    mni_zooms = zooms3(mni)
    mni_orient = axis_codes(mni)
    mni_aff = np.asarray(mni.affine, dtype=float)

    print("Local MNI template:")
    print(" ", MNI_FILE)
    print(" shape      :", mni_shape)
    print(" voxel size :", mni_zooms)
    print(" orient     :", mni_orient)
    print()

    roots = [p for p in SEARCH_ROOTS if p is not None]

    print("Search roots:")
    for r in roots:
        print(" ", r, "| exists =", r.exists())
    print()

    nifti_candidates = []

    for root in roots:
        if not root.exists():
            continue

        # Recursive, but skip enormous scans by only retaining filenames with atlas tokens.
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if not is_nifti(p):
                    continue

                lname = p.name.lower()
                if any(tok in lname for tok in ATLAS_TOKENS):
                    nifti_candidates.append(p)
        except PermissionError:
            pass

    # De-duplicate.
    nifti_candidates = sorted(
        set(nifti_candidates),
        key=lambda p: (-candidate_score(p.name), str(p).lower())
    )

    rows = []

    print("Candidate atlas NIfTI files found:", len(nifti_candidates))
    print("-" * 110)

    for i, p in enumerate(nifti_candidates, start=1):
        info = safe_load_nifti(p)

        row = {
            "rank": i,
            "path": str(p),
            "name": p.name,
            "score": candidate_score(p.name),
            "load_ok": info.get("ok", False),
            "shape": None,
            "zooms": None,
            "orientation": None,
            "same_shape_as_mni": False,
            "same_zooms_as_mni": False,
            "same_orientation_as_mni": False,
            "same_affine_as_mni": False,
            "affine_max_abs_diff": None,
            "label_sidecars": [],
            "error": info.get("error", ""),
        }

        if info.get("ok"):
            row["shape"] = list(info["shape"])
            row["zooms"] = list(info["zooms"])
            row["orientation"] = info["orientation"]
            row["same_shape_as_mni"] = info["shape"] == mni_shape
            row["same_zooms_as_mni"] = bool(
                np.allclose(info["zooms"], mni_zooms, atol=1e-4, rtol=0)
            )
            row["same_orientation_as_mni"] = info["orientation"] == mni_orient
            diff = float(np.max(np.abs(info["affine"] - mni_aff)))
            row["affine_max_abs_diff"] = diff
            row["same_affine_as_mni"] = bool(
                np.allclose(info["affine"], mni_aff, atol=1e-4, rtol=0)
            )
            row["label_sidecars"] = find_label_sidecars(p)

        rows.append(row)

        print(
            f"[{i:02d}] {p.name}\n"
            f"     path={p}\n"
            f"     shape={row['shape']} zooms={row['zooms']} orient={row['orientation']}\n"
            f"     exact_grid={row['same_shape_as_mni'] and row['same_zooms_as_mni'] and row['same_orientation_as_mni'] and row['same_affine_as_mni']}\n"
            f"     label_sidecars={len(row['label_sidecars'])}"
        )

    # Find strongest Harvard-Oxford cortical and subcortical candidates.
    ho = [r for r in rows if "harvard" in r["name"].lower()]
    ho_cort = [r for r in ho if "cort" in r["name"].lower()]
    ho_sub = [r for r in ho if "sub" in r["name"].lower()]

    selected_cort = ho_cort[0] if ho_cort else None
    selected_sub = ho_sub[0] if ho_sub else None

    status = "NO_SUITABLE_LOCAL_ATLAS_FOUND"
    if selected_cort and selected_sub:
        status = "HARVARD_OXFORD_CORTICAL_AND_SUBCORTICAL_FOUND"
    elif rows:
        status = "OTHER_LOCAL_ATLAS_CANDIDATES_FOUND"

    summary = {
        "experiment": "ANATOMICAL_VALIDATION_V1_ATLAS_PREFLIGHT",
        "read_only": True,
        "mni_file": str(MNI_FILE),
        "mni_shape": list(mni_shape),
        "mni_zooms": list(mni_zooms),
        "mni_orientation": mni_orient,
        "candidate_count": len(rows),
        "status": status,
        "selected_harvard_oxford_cortical": selected_cort,
        "selected_harvard_oxford_subcortical": selected_sub,
        "candidates": rows,
    }

    with (OUT_ROOT / "atlas_preflight_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = []
    report.append("=" * 110)
    report.append("ANATOMICAL VALIDATION — ATLAS PREFLIGHT SUMMARY")
    report.append("=" * 110)
    report.append(f"Candidate atlas NIfTI files : {len(rows)}")
    report.append(f"Status                      : {status}")
    report.append("")

    if selected_cort:
        report.append("Harvard-Oxford cortical candidate:")
        report.append(f"  {selected_cort['path']}")
        report.append("")
    if selected_sub:
        report.append("Harvard-Oxford subcortical candidate:")
        report.append(f"  {selected_sub['path']}")
        report.append("")

    if status == "NO_SUITABLE_LOCAL_ATLAS_FOUND":
        report += [
            "NEXT ACTION:",
            "No usable atlas was found locally.",
            "Do not run regional attribution yet.",
            "Install or download an MNI-space labelled atlas first.",
        ]
    elif status == "HARVARD_OXFORD_CORTICAL_AND_SUBCORTICAL_FOUND":
        report += [
            "NEXT ACTION:",
            "Atlas candidates are available.",
            "Proceed to Step 21, which will resample the atlas labels to the exact local MNI grid",
            "with nearest-neighbour interpolation and compute region-level IG attribution.",
        ]
    else:
        report += [
            "NEXT ACTION:",
            "Local atlas candidates exist, but the preferred Harvard-Oxford cortical+subcortical pair",
            "was not found automatically. Review the candidate list before regional attribution.",
        ]

    report += [
        "",
        "READ-ONLY: no atlas was downloaded or resampled and no frozen data were modified.",
    ]

    text = "\n".join(report)

    with (OUT_ROOT / "atlas_preflight_report.txt").open("w", encoding="utf-8") as f:
        f.write(text)

    print()
    print(text)
    print()
    print("Saved:")
    print(" ", OUT_ROOT / "atlas_preflight_summary.json")
    print(" ", OUT_ROOT / "atlas_preflight_report.txt")


if __name__ == "__main__":
    main()

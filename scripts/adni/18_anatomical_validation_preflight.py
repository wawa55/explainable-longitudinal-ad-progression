# -*- coding: utf-8 -*-
"""READ-ONLY preflight for frozen ADNI XAI anatomical validation."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, re
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTR_ROOT = PROJECT_ROOT / "08_results" / "ADNI_XAI_TEST_PRODUCTION_V1" / "attributions"
OUT_ROOT = PROJECT_ROOT / "08_results" / "ANATOMICAL_VALIDATION_V1" / "preflight"
EXPECTED_SEEDS = [2026, 3407, 5813, 7291, 9109]
EXPECTED_HORIZONS = ["24m", "36m"]
EXPECTED_PARTICIPANTS = 24
EXPECTED_TOTAL_FILES = 240
MODEL_SHAPE = (160, 192, 160)
ATTR_HINTS = ("ig", "attr", "attribution", "integrated_grad")
PATH_HINTS = ("path", "mri", "nii", "image", "scan", "file")


def preview(a, n=6):
    try:
        if a.ndim == 0:
            return repr(a.item())
        x = a.reshape(-1)
        vals = [v.item() if isinstance(v, np.generic) else v for v in x[:n]]
        return repr(vals) + (" ..." if a.size > n else "")
    except Exception as e:
        return f"<preview failed: {e!r}>"


def parse_name(path):
    m = re.search(r"^(?P<subject>\d{3}_S_\d{4})__(?P<landmark>.+)__seed(?P<seed>\d+)__(?P<horizon>\d+m)$", path.stem)
    if m:
        d = m.groupdict(); d["seed"] = int(d["seed"]); return d
    subject = re.match(r"(\d{3}_S_\d{4})", path.stem)
    seed = next((int(x.split('_')[1]) for x in path.parts if re.fullmatch(r"seed_\d+", x)), None)
    horizon = next((x for x in path.parts if re.fullmatch(r"\d+m", x)), None)
    return {"subject": subject.group(1) if subject else None, "landmark": path.stem, "seed": seed, "horizon": horizon}


def is_attr(key, a):
    numeric = np.issubdtype(a.dtype, np.number)
    shape_ok = tuple(a.shape) == MODEL_SHAPE or (a.ndim >= 3 and tuple(a.shape[-3:]) == MODEL_SHAPE)
    return numeric and (shape_ok or any(h in key.lower() for h in ATTR_HINTS))


def inspect(path):
    rows, attr_keys, path_keys = [], [], []
    with np.load(path, allow_pickle=True) as z:
        for key in z.files:
            a = np.asarray(z[key])
            r = {"key": key, "shape": list(a.shape), "dtype": str(a.dtype), "ndim": int(a.ndim), "size": int(a.size), "preview": preview(a)}
            if np.issubdtype(a.dtype, np.number) and a.size:
                f = np.isfinite(a); r["finite_fraction"] = float(f.mean())
                if f.any():
                    v = a[f]; r.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), mean_abs=float(np.abs(v).mean()))
            if is_attr(key, a): attr_keys.append(key)
            if any(h in key.lower() for h in PATH_HINTS) and (a.ndim == 0 or a.size <= 20): path_keys.append(key)
            rows.append(r)
    return rows, attr_keys, path_keys


def main():
    print("="*96)
    print("ANATOMICAL VALIDATION PREFLIGHT — FROZEN XAI OUTPUTS")
    print("="*96)
    print("Attribution root:", ATTR_ROOT)
    if not ATTR_ROOT.exists():
        print("ERROR: attribution root not found. Nothing modified.")
        return 2

    files = sorted(ATTR_ROOT.glob("seed_*/*m/*.npz"))
    print(f"NPZ files found: {len(files)} / {EXPECTED_TOTAL_FILES}")
    if not files: return 3
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    inventory=[]; schemas=Counter(); attr_counts=Counter(); path_counts=Counter(); errors=[]
    subjects=set(); sh=defaultdict(set); ss=defaultdict(set); seed_counts=Counter(); horizon_counts=Counter(); first=None

    for i,p in enumerate(files,1):
        m=parse_name(p); sub=m['subject']; seed=m['seed']; hor=m['horizon']
        if sub:
            subjects.add(sub); sh[sub].add(hor); ss[sub].add(seed)
        seed_counts[seed]+=1; horizon_counts[hor]+=1
        row={"file":str(p),"relative_file":str(p.relative_to(PROJECT_ROOT)),"subject":sub,"landmark":m['landmark'],"seed":seed,"horizon":hor,"bytes":p.stat().st_size,"load_ok":False,"likely_attr_keys":"","likely_path_keys":"","schema":"","error":""}
        try:
            kr, ak, pk = inspect(p); schema=tuple(sorted(x['key'] for x in kr))
            row.update(load_ok=True, likely_attr_keys=';'.join(ak), likely_path_keys=';'.join(pk), schema=';'.join(schema))
            schemas[schema]+=1; attr_counts.update(ak); path_counts.update(pk)
            if first is None: first={"file":str(p),"subject":sub,"seed":seed,"horizon":hor,"keys":kr}
        except Exception as e:
            row['error']=repr(e); errors.append({"file":str(p),"error":repr(e)})
        inventory.append(row)
        if i==1 or i%25==0 or i==len(files): print(f"Inspected {i}/{len(files)}")

    mismatches=[]
    for seed in EXPECTED_SEEDS:
        for hor in EXPECTED_HORIZONS:
            n=sum(1 for r in inventory if r['seed']==seed and r['horizon']==hor and r['load_ok'])
            if n!=EXPECTED_PARTICIPANTS: mismatches.append({"seed":seed,"horizon":hor,"files_found":n,"expected":EXPECTED_PARTICIPANTS})
    incomplete=[]
    for sub in sorted(subjects):
        if sh[sub] != set(EXPECTED_HORIZONS) or ss[sub] != set(EXPECTED_SEEDS):
            incomplete.append({"subject":sub,"horizons":sorted(sh[sub]),"seeds":sorted(ss[sub])})

    csv_path=OUT_ROOT/'preflight_file_inventory.csv'
    fields=list(inventory[0].keys())
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(inventory)

    summary={"experiment":"ANATOMICAL_VALIDATION_V1_PREFLIGHT","read_only":True,"expected_model_shape":list(MODEL_SHAPE),"expected_seeds":EXPECTED_SEEDS,"expected_horizons":EXPECTED_HORIZONS,"expected_participants":EXPECTED_PARTICIPANTS,"expected_total_files":EXPECTED_TOTAL_FILES,"files_found":len(files),"files_load_ok":sum(bool(r['load_ok']) for r in inventory),"files_load_failed":len(errors),"unique_subjects":len(subjects),"seed_counts":{str(k):v for k,v in seed_counts.items()},"horizon_counts":{str(k):v for k,v in horizon_counts.items()},"schema_counts":[{"keys":list(k),"count":v} for k,v in schemas.most_common()],"likely_attribution_key_counts":dict(attr_counts),"likely_path_key_counts":dict(path_counts),"seed_horizon_count_mismatches":mismatches,"incomplete_subjects":incomplete,"errors":errors,"first_file_schema_detail":first}
    (OUT_ROOT/'preflight_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    lines=["="*96,"ANATOMICAL VALIDATION PREFLIGHT — SUMMARY","="*96,
           f"Files found              : {len(files)} / {EXPECTED_TOTAL_FILES}",
           f"Files loaded successfully: {summary['files_load_ok']} / {len(files)}",
           f"Unique participants      : {len(subjects)} / {EXPECTED_PARTICIPANTS}","",
           "Seed counts:"]
    lines += [f"  {k}: {v}" for k,v in sorted(seed_counts.items(),key=lambda x:str(x[0]))]
    lines += ["","Horizon counts:"]+[f"  {k}: {v}" for k,v in sorted(horizon_counts.items(),key=lambda x:str(x[0]))]
    lines += ["","Likely attribution keys:"] + ([f"  {k}: {v} files" for k,v in attr_counts.most_common()] if attr_counts else ["  NONE IDENTIFIED"])
    lines += ["","Likely MRI/path metadata keys:"] + ([f"  {k}: {v} files" for k,v in path_counts.most_common()] if path_counts else ["  NONE IDENTIFIED"])
    lines += ["",f"Distinct NPZ schemas     : {len(schemas)}",f"Seed/horizon mismatches  : {len(mismatches)}",f"Incomplete subjects      : {len(incomplete)}",f"Load errors              : {len(errors)}","","FIRST NPZ FILE — KEY DETAIL","-"*96]
    if first:
        lines += [f"File: {first['file']}",f"Subject: {first['subject']}  Seed: {first['seed']}  Horizon: {first['horizon']}"]
        for k in first['keys']:
            lines.append(f"  key={k['key']!r} shape={k['shape']} dtype={k['dtype']} finite={k.get('finite_fraction')} min={k.get('min')} max={k.get('max')} mean_abs={k.get('mean_abs')} preview={k['preview']}")
    ok=(len(files)==EXPECTED_TOTAL_FILES and not errors and len(subjects)==EXPECTED_PARTICIPANTS and not mismatches and bool(attr_counts))
    lines += ["","STATUS","-"*96,"PASS — frozen IG files are ready for anatomical-registration planning." if ok else "CHECK REQUIRED — review mismatches before anatomical registration.","","READ-ONLY: no model training, inference, IG recomputation, registration, or frozen-file modification was performed."]
    report='\n'.join(lines)
    (OUT_ROOT/'preflight_report.txt').write_text(report,encoding='utf-8')
    print('\n'+report)
    print('\nSaved to:', OUT_ROOT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

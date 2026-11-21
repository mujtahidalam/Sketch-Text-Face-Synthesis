"""Compare two groups of seeded runs (e.g. global-only vs full model).

Each run directory must contain seed_*/aggregate.json and seed_*/per_image_metrics.csv
(as written by evaluate.py).

  * FID is a set-level metric -> Welch t-test over the per-seed FID values
  * per-image metrics -> paired differences over all (seed, image) pairs, with a
    bootstrap 95% CI and a one-sample t-test
  * seed-level paired t-test on the per-seed means (n = number of seeds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("lpips", "ssim", "clip", "identity_cosine")


def bootstrap_ci(x: np.ndarray, iters: int = 10000, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    return [float(v) for v in np.quantile(x[idx].mean(axis=1), [0.025, 0.975])]


def compare(baseline_dir, model_dir, output) -> dict:
    a_dir, b_dir = Path(baseline_dir), Path(model_dir)
    fid_a = np.array([json.loads(p.read_text())["fid"] for p in sorted(a_dir.glob("seed_*/aggregate.json"))])
    fid_b = np.array([json.loads(p.read_text())["fid"] for p in sorted(b_dir.glob("seed_*/aggregate.json"))])
    result = {
        "fid": {
            "baseline_mean": float(fid_a.mean()), "baseline_std": float(fid_a.std(ddof=1)),
            "model_mean": float(fid_b.mean()), "model_std": float(fid_b.std(ddof=1)),
            "welch_p": float(stats.ttest_ind(fid_a, fid_b, equal_var=False).pvalue),
        },
        "pooled": {},
        "seed_level": {},
    }

    diffs = {m: [] for m in METRICS}
    seed_means = {m: ([], []) for m in METRICS}
    files_a = sorted(a_dir.glob("seed_*/per_image_metrics.csv"))
    files_b = sorted(b_dir.glob("seed_*/per_image_metrics.csv"))
    for fa, fb in zip(files_a, files_b):
        merged = pd.read_csv(fa).merge(pd.read_csv(fb), on="id", suffixes=("_a", "_b"))
        for m in METRICS:
            diffs[m].extend((merged[f"{m}_b"] - merged[f"{m}_a"]).tolist())
            seed_means[m][0].append(merged[f"{m}_a"].mean())
            seed_means[m][1].append(merged[f"{m}_b"].mean())

    for m in METRICS:
        d = np.asarray(diffs[m])
        result["pooled"][m] = {
            "mean_difference": float(d.mean()),
            "bootstrap_95_ci": bootstrap_ci(d),
            "ttest_p": float(stats.ttest_1samp(d, 0).pvalue),
            "n": int(d.size),
        }
        sa, sb = map(np.asarray, seed_means[m])
        result["seed_level"][m] = {
            "mean_difference": float((sb - sa).mean()),
            "paired_ttest_p": float(stats.ttest_rel(sb, sa).pvalue) if len(sa) > 1 else None,
            "seeds_improved": int(((sb - sa) * (-1 if m == "lpips" else 1) > 0).sum()),
            "n_seeds": int(len(sa)),
        }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="e.g. outputs/seeds/base")
    ap.add_argument("--model", required=True, help="e.g. outputs/seeds/full_model")
    ap.add_argument("--output", default="outputs/seeds/comparison.json")
    a = ap.parse_args()
    print(json.dumps(compare(a.baseline, a.model, a.output), indent=2))


if __name__ == "__main__":
    main()

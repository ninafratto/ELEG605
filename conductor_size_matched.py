#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conductor: size-matched dev control for the Mapper stability comparison (post hoc; changes no hyperparameter).

The test-set Mapper graph has far fewer samples than the full dev graph, and Mapper stability/connectedness
depend on sample size. This control repeats the TEST protocol on dev subsets of the SAME size as the test set:
  n_sets stratified dev subsets of size n_test -> own reference graph -> n_draws 80% subsamples vs that reference.
Run it after conductor_dev_seeds.py and conductor_test_seeds.py; it reads their outputs for the comparison figure.

Outputs:
  results_size_matched/seed_<s>/size_matched_sets.csv   one row per dev subset (ref_*, mean_*, std_*)
  results_size_matched/size_matched_per_seed.csv
  results_size_matched/stability_comparison.png         dev (full) vs dev (size-matched) vs test
  results_size_matched/seed_summary.xlsx                sheets Comparison, Size-matched, Seeds
  results_size_matched/runtime.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from conductor_dev import CONFIG as DEV_CONFIG
from conductor_dev_seeds import CONFIG as DEV_SEEDS_CONFIG
from conductor_test import CONFIG as TEST_CONFIG
from conductor_test_seeds import CONFIG as TEST_SEEDS_CONFIG
from apply_mapper import run_mapper_size_matched_stability
from seed_summary import plot_group_comparison, summarize_metrics_long, write_sheets_excel
from runtime_log import RuntimeLog


CONFIG = {
    "seeds": DEV_SEEDS_CONFIG["seeds"],                       # same seeds as the dev and test sweeps
    "mapper_params": TEST_CONFIG["mapper"]["params"],          # the final (test) hyperparameters
    "resample": TEST_CONFIG["mapper"]["resample"],             # same subsampling as the test set
    "set_size": None,                                          # None -> number of test samples in the master CSV
    "n_sets": 20,                                              # dev subsets per seed
    "output_root": "./results_size_matched",
    "dev_seed_root": DEV_SEEDS_CONFIG["output_root"],          # full-dev results (per-seed mapper_grid_search.csv)
    "test_seed_root": TEST_SEEDS_CONFIG["output_root"],        # test results (mapper_per_seed.csv)
    "metrics": [("stability", "topological stability"), ("connectedness", "connectedness ratio")],
    "group_styles": [{"marker": "o", "color": "#00539f"}, {"marker": "^", "color": "#ffd200"},
                     {"marker": "s", "color": "#bdbdbd"}],
    "excel_decimals": 3,
}


def _dev_full_rows(dev_root, seeds, params):
    """Per-seed full-dev subsample results for the chosen hyperparameters (from conductor_dev_seeds outputs)."""
    rows = []
    for s in seeds:
        df = pd.read_csv(Path(dev_root) / f"seed_{s}" / "mapper_grid_search.csv")
        mask = np.ones(len(df), dtype=bool)
        for k in ("num_cubes", "perc_overlap", "dbscan_eps"):
            mask &= np.isclose(df[k].astype(float), float(params[k]))
        if mask.sum() != 1:
            raise ValueError(f"seed {s}: chosen params {params} not found once in the dev grid search")
        rows.append({**df[mask].iloc[0].to_dict(), "seed": s})
    return pd.DataFrame(rows)


def main():
    root = Path(CONFIG["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    paths, cols = DEV_CONFIG["paths"], DEV_CONFIG["columns"]
    params, resample = CONFIG["mapper_params"], CONFIG["resample"]
    master = pd.read_csv(paths["master_csv"])
    n_test = int((master["split"] == "test").sum())
    n_dev = int((master["split"] == "dev").sum())
    set_size = n_test if CONFIG["set_size"] is None else CONFIG["set_size"]
    print(f"size-matched control: dev subsets of {set_size} (test n = {n_test}, dev n = {n_dev})")
    timer = RuntimeLog()

    per_seed = []
    for seed in CONFIG["seeds"]:
        print(f"\n################ seed {seed} ################")
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with timer.stage(f"seed {seed}"):
            sets_df = run_mapper_size_matched_stability(
                feature_prefix=f"{paths['feature_prefix']}_dev", csv_path=paths["master_csv"], name_col=cols["name"],
                mapper_params=params, random_state=seed, set_size=set_size, n_sets=CONFIG["n_sets"],
                resample=resample)
        sets_df.to_csv(seed_dir / "size_matched_sets.csv", index=False)
        row = {"seed": seed}
        for m, _ in CONFIG["metrics"]:
            row[f"mean_{m}"] = sets_df[f"mean_{m}"].mean()               # average over subsets and subsamples
            row[f"std_{m}"] = sets_df[f"std_{m}"].mean()                 # typical spread across subsamples
            row[f"between_set_std_{m}"] = sets_df[f"mean_{m}"].std()     # spread across different dev subsets
        per_seed.append(row)
    sm = pd.DataFrame(per_seed)
    sm.to_csv(root / "size_matched_per_seed.csv", index=False)

    dev_full = _dev_full_rows(CONFIG["dev_seed_root"], CONFIG["seeds"], params)
    test = pd.read_csv(Path(CONFIG["test_seed_root"]) / "mapper_per_seed.csv")
    test = test.rename(columns={f"sub_{k}_{m}": f"{k}_{m}" for k in ("mean", "std") for m, _ in CONFIG["metrics"]})

    # value = mean over seeds of the per-seed subsample mean; error = mean over seeds of the subsample std
    groups, comp_rows = [], []
    for label, df in ((f"dev\n(n = {n_dev})", dev_full), (f"dev, size-matched\n(n = {set_size})", sm),
                      (f"test\n(n = {n_test})", test)):
        vals = {m: (df[f"mean_{m}"].mean(), df[f"std_{m}"].mean()) for m, _ in CONFIG["metrics"]}
        groups.append((label, vals))
        for m, mlabel in CONFIG["metrics"]:
            comp_rows.append({"group": label.replace("\n", " "), "metric": mlabel, "mean": vals[m][0],
                              "subsample std": vals[m][1], "across-seed std": df[f"mean_{m}"].std(),
                              "mean ± subsample std": f"{vals[m][0]:.{CONFIG['excel_decimals']}f} ± "
                                                      f"{vals[m][1]:.{CONFIG['excel_decimals']}f}"})

    plot_group_comparison(groups, panels=CONFIG["metrics"], styles=CONFIG["group_styles"],
                          plot_path=root / "stability_comparison.png",
                          show_plot=TEST_CONFIG["plots"]["show"], dpi=TEST_CONFIG["plots"]["dpi"])

    labels = {f"{k}_{m}": f"{ml} ({desc})" for m, ml in CONFIG["metrics"]
              for k, desc in (("mean", "subsample mean"), ("std", "subsample std"),
                              ("between_set_std", "std across dev subsets"))}
    sheets = {
        "Comparison": pd.DataFrame(comp_rows),
        "Size-matched": summarize_metrics_long(sm, value_cols=[c for c in sm.columns if c != "seed"],
                                               value_labels=labels, decimals=CONFIG["excel_decimals"]),
        "Seeds": pd.DataFrame({"seed": CONFIG["seeds"], "output_folder": [f"seed_{s}" for s in CONFIG["seeds"]],
                               "runtime_s": [r["seconds"] for r in timer.rows]}),
    }
    write_sheets_excel(path=root / "seed_summary.xlsx", sheets=sheets)
    print("\n=== Comparison ===")
    print(sheets["Comparison"].to_string(index=False))
    timer.save(root / "runtime.csv")


if __name__ == "__main__":
    main()

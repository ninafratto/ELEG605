#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conductor: repeat the conductor_test.py evaluation over the SAME seeds as conductor_dev_seeds.py and summarize.

Every pipeline setting (final hyperparameters) comes from conductor_test.CONFIG; only the values below live here.
The test set is identical for every seed, so the across-seed std reflects only algorithm randomness
(K-means initialization, randomized PCA, which test subsamples are drawn), not test-sample variability.

Outputs (same layout as results_dev_seed):
  results_test_seed/seed_<s>/                 full conductor_test outputs for seed s
  results_test_seed/{fft,kmeans,mapper}_per_seed.csv
  results_test_seed/fft_seed_summary.png      test accuracy + cross-entropy per seed
  results_test_seed/mapper_seed_summary.png   test topological stability + connectedness per seed (+/- subsample std)
  results_test_seed/test_metric_heatmap.png   pipeline x metric (accuracy, ARI, NMI, silhouette), mean across seeds
  results_test_seed/seed_summary.xlsx        sheets FFT, K-means, Mapper (mean ± std across seeds), Seeds
  results_test_seed/runtime.csv
"""

import copy
from pathlib import Path

import pandas as pd

from conductor_test import CONFIG as TEST_CONFIG, run_test
from conductor_dev_seeds import CONFIG as DEV_SEEDS_CONFIG
from conductor_dev import CONFIG as DEV_CONFIG
from seed_summary import summarize_metrics_long, write_sheets_excel, plot_metrics_per_seed, plot_method_metric_heatmap
from runtime_log import RuntimeLog


CONFIG = {
    "seeds": DEV_SEEDS_CONFIG["seeds"],      # same seeds as the dev sweep
    "output_root": "./results_test_seed",
    # sheet / file key -> (method name as written in test_summary.csv, metrics summarized across seeds)
    "methods": {
        "FFT": ("fft", "FFT + multinomial logistic regression", ["test_accuracy", "test_log_loss", "test_ari"]),
        "K-means": ("kmeans", "MoCo + K-means", ["test_accuracy", "test_ari", "test_nmi", "test_silhouette"]),
        "Mapper": ("mapper", "MoCo + Mapper (majority vote)",
                   ["test_accuracy", "test_ari", "test_nmi", "test_silhouette", "n_unassigned", "n_ambiguous",
                    "topo_b0", "topo_connectedness", "topo_singletons",
                    "sub_mean_stability", "sub_std_stability", "sub_mean_connectedness", "sub_std_connectedness"]),
    },
    "value_labels": {"test_accuracy": "test accuracy", "test_log_loss": "test categorical cross-entropy",
                     "test_ari": "test ARI", "test_nmi": "test NMI", "test_silhouette": "test silhouette",
                     "n_unassigned": "unassigned test samples", "n_ambiguous": "ambiguous (tied) test samples",
                     "topo_b0": "test graph b0", "topo_connectedness": "test graph connectedness ratio",
                     "topo_singletons": "test graph singletons",
                     "sub_mean_stability": "test topological stability (subsample mean)",
                     "sub_std_stability": "test topological stability (subsample std)",
                     "sub_mean_connectedness": "test subsample connectedness ratio (mean)",
                     "sub_std_connectedness": "test subsample connectedness ratio (std)"},
    "excel_decimals": 3,
    # summary figures: (value column, error-bar column or None, y label)
    "fft_panels": [("test_accuracy", None, "test accuracy"), ("test_log_loss", None, "categorical cross-entropy")],
    "mapper_panels": [("sub_mean_stability", "sub_std_stability", "topological stability"),
                      ("sub_mean_connectedness", "sub_std_connectedness", "connectedness ratio")],
    "fft_style": {"marker": "s", "color": "#00539f"},
    "mapper_style": {"marker": "o", "color": "#00539f"},
    "baseline_accuracy": DEV_CONFIG["fft"]["baseline_accuracy"],     # majority-class baseline; None -> no line
    "baseline_color": DEV_CONFIG["fft"]["baseline_color"],
    # pipeline x metric heatmap (cell = mean across seeds); metrics a pipeline lacks are shown as n/a
    "heatmap": {
        "enabled": True,
        "metrics": ["test_accuracy", "test_ari", "test_silhouette"],
        "metric_labels": {"test_accuracy": "accuracy", "test_ari": "ARI", "test_nmi": "NMI",
                          "test_silhouette": "silhouette"},
        "cmap": "RdBu",
        "vmin": -1.0, "vmax": 1.0,      # ARI and silhouette range over [-1, 1]
        "annotate": True,               # write the mean (across seeds) in each cell
        "na_color": "black",
        "na_text_color": "white",
    },
}


def main():
    root = Path(CONFIG["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLog()

    runs = []
    for seed in CONFIG["seeds"]:
        print(f"\n################ seed {seed} ################")
        cfg = copy.deepcopy(TEST_CONFIG)
        cfg["random_seed"] = seed
        cfg["paths"]["output_dir"] = str(root / f"seed_{seed}")
        with timer.stage(f"seed {seed}"):
            runs.append(run_test(cfg).assign(seed=seed))
    per_seed = pd.concat(runs, ignore_index=True)

    sheets, per_method = {}, {}
    for sheet, (file_key, method, values) in CONFIG["methods"].items():
        rows = per_seed[per_seed["method"] == method]
        if rows.empty:
            raise ValueError(f"no rows for method '{method}' in the test summaries")
        per_method[sheet] = rows[["seed", *values]].reset_index(drop=True)
        per_method[sheet].to_csv(root / f"{file_key}_per_seed.csv", index=False)
        sheets[sheet] = summarize_metrics_long(rows, value_cols=values, value_labels=CONFIG["value_labels"],
                                               decimals=CONFIG["excel_decimals"])
        print(f"\n=== {sheet} across seeds ===")
        print(sheets[sheet].to_string(index=False))
    sheets["Seeds"] = pd.DataFrame({"seed": CONFIG["seeds"], "output_folder": [f"seed_{s}" for s in CONFIG["seeds"]],
                                    "runtime_s": [r["seconds"] for r in timer.rows]})
    write_sheets_excel(path=root / "seed_summary.xlsx", sheets=sheets)
    make_summary_figures(root, per_method)
    timer.save(root / "runtime.csv")


def make_summary_figures(root, per_method=None):
    """Summary figures from the per-seed tables (read from <root>/*_per_seed.csv when per_method is None)."""
    root = Path(root)
    if per_method is None:
        per_method = {sheet: pd.read_csv(root / f"{key}_per_seed.csv")
                      for sheet, (key, _, _) in CONFIG["methods"].items()}
    plots = TEST_CONFIG["plots"]
    plot_kw = {"show_plot": plots["show"], "dpi": plots["dpi"]}
    baselines = {} if CONFIG["baseline_accuracy"] is None else {
        "test_accuracy": (CONFIG["baseline_accuracy"], CONFIG["baseline_color"],
                          f"majority baseline ({CONFIG['baseline_accuracy']:.1%})")}
    plot_metrics_per_seed(per_method["FFT"], panels=CONFIG["fft_panels"], style=CONFIG["fft_style"],
                          baselines=baselines, plot_path=root / "fft_seed_summary.png", **plot_kw)
    plot_metrics_per_seed(per_method["Mapper"], panels=CONFIG["mapper_panels"], style=CONFIG["mapper_style"],
                          baselines={}, plot_path=root / "mapper_seed_summary.png", **plot_kw)
    hm = CONFIG["heatmap"]
    if hm["enabled"]:
        plot_method_metric_heatmap(per_method, methods=list(CONFIG["methods"]), metrics=hm["metrics"],
                                   metric_labels=hm["metric_labels"], cmap=hm["cmap"], vmin=hm["vmin"],
                                   vmax=hm["vmax"], annotate=hm["annotate"], decimals=CONFIG["excel_decimals"],
                                   na_color=hm["na_color"], na_text_color=hm["na_text_color"], plot_path=root / "test_metric_heatmap.png", **plot_kw)


if __name__ == "__main__":
    main()

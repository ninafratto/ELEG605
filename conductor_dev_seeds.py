#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conductor: repeat the full conductor_dev.py workflow over several random seeds and summarize across seeds.

Every pipeline setting comes from conductor_dev.CONFIG; only the values below (seeds, output locations, summary
styling) live here. The dev/test split stays fixed (conductor_dev CONFIG["split"]["seed"]), so the test images
never change. The seed varies the CV fold assignment, the Mapper subsamples, and the K-means / solver seeds.

Outputs:
  results_dev_seed/seed_<s>/          full conductor_dev outputs for seed s (+ that seed's fold assignment csv)
  results_dev_seed/fft_seed_summary.png, mapper_seed_heatmap.png, mapper_seed_interaction.png
  results_dev_seed/seed_summary.xlsx  sheets FFT, Mapper (mean ± std across seeds), Seeds
"""

import copy
from pathlib import Path

import pandas as pd

from conductor_dev import CONFIG as DEV_CONFIG, run_dev
from apply_mapper import plot_mapper_grid_heatmap, plot_mapper_grid_interaction
from seed_summary import aggregate_across_seeds, plot_fft_seed_summary, write_seed_summary_excel
from runtime_log import RuntimeLog


CONFIG = {
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "output_root": "./results_dev_seed",
    # per-seed values summarized across seeds (each is that seed's mean over folds / subsamples)
    "fft_values": ["cv_accuracy", "cv_log_loss"],
    "mapper_values": ["stability", "connectedness", "b0", "singletons"],
    # readable column names for the 'mean ± std' columns in the Excel workbook
    "value_labels": {"cv_accuracy": "validation accuracy", "cv_log_loss": "categorical cross-entropy",
                     "stability": "topological stability", "connectedness": "connectedness ratio",
                     "b0": "b0", "singletons": "singletons"},
    "excel_decimals": 3,
    "summary_plots": {"fft": True, "mapper_heatmap": True, "mapper_interaction": True},
}


def main():
    root = Path(CONFIG["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    timer = RuntimeLog()
    grid_cols = list(DEV_CONFIG["mapper"]["grid_search"])

    fft_runs, mapper_runs = [], []
    for seed in CONFIG["seeds"]:
        print(f"\n################ seed {seed} ################")
        cfg = copy.deepcopy(DEV_CONFIG)
        seed_dir = root / f"seed_{seed}"
        cfg["random_seed"] = seed
        cfg["paths"]["output_dir"] = str(seed_dir)
        cfg["paths"]["master_csv"] = str(seed_dir / "master_dataset_split.csv")   # leaves ./master_dataset_split.csv alone
        seed_dir.mkdir(parents=True, exist_ok=True)

        with timer.stage(f"seed {seed}"):
            fft_df, mapper_df = run_dev(cfg)
        fft_runs.append(fft_df.assign(seed=seed))
        mapper_runs.append(mapper_df.assign(seed=seed))

    fft_all = pd.concat(fft_runs, ignore_index=True)
    mapper_all = pd.concat(mapper_runs, ignore_index=True).rename(
        columns={f"mean_{v}": v for v in CONFIG["mapper_values"]})
    fft_all.to_csv(root / "fft_per_seed.csv", index=False)
    mapper_all[["seed", *grid_cols, *CONFIG["mapper_values"]]].to_csv(root / "mapper_per_seed.csv", index=False)

    fft_summary = aggregate_across_seeds(fft_all, group_cols=["l2_lambda"], value_cols=CONFIG["fft_values"])
    mapper_summary = aggregate_across_seeds(mapper_all, group_cols=grid_cols, value_cols=CONFIG["mapper_values"])

    plots = DEV_CONFIG["plots"]
    plot_kw = {"show_plot": plots["show"], "dpi": plots["dpi"]}
    fft_cfg, mp_cfg = DEV_CONFIG["fft"], DEV_CONFIG["mapper"]
    if CONFIG["summary_plots"]["fft"]:
        plot_fft_seed_summary(fft_summary, styles=fft_cfg["compare_styles"], baseline_accuracy=fft_cfg["baseline_accuracy"],
                              baseline_color=fft_cfg["baseline_color"], plot_path=root / "fft_seed_summary.png", **plot_kw)
    if CONFIG["summary_plots"]["mapper_heatmap"]:
        heat = mp_cfg["heatmap_plot"]
        plot_mapper_grid_heatmap(mapper_summary, row_param=heat["row_param"], col_param=heat["col_param"],
                                 panel_param=heat["panel_param"], param_labels=mp_cfg["param_labels"], cmap=heat["cmap"],
                                 plot_path=root / "mapper_seed_heatmap.png", **plot_kw)
    if CONFIG["summary_plots"]["mapper_interaction"]:
        inter = mp_cfg["interaction_plot"]
        plot_mapper_grid_interaction(mapper_summary, x_param=inter["x_param"], color_param=inter["color_param"],
                                     fill_param=inter["fill_param"], color_styles=inter["color_styles"],
                                     param_labels=mp_cfg["param_labels"], x_offset=inter["x_offset"],
                                     plot_path=root / "mapper_seed_interaction.png", **plot_kw)

    seeds_df = pd.DataFrame({"seed": CONFIG["seeds"],
                             "output_folder": [f"seed_{s}" for s in CONFIG["seeds"]],
                             "runtime_s": [r["seconds"] for r in timer.rows]})
    write_seed_summary_excel(path=root / "seed_summary.xlsx", seeds_df=seeds_df,
                             fft_summary=fft_summary, fft_values=CONFIG["fft_values"],
                             mapper_summary=mapper_summary, mapper_values=CONFIG["mapper_values"],
                             value_labels=CONFIG["value_labels"], decimals=CONFIG["excel_decimals"])

    print("\n=== FFT across seeds ===")
    print(fft_summary.to_string(index=False))
    print("\n=== Mapper across seeds ===")
    print(mapper_summary.to_string(index=False))
    timer.save(root / "runtime.csv")


if __name__ == "__main__":
    main()

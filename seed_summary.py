#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow module: aggregate repeated conductor_dev runs (one per random seed) into across-seed summaries.

Each seed contributes ONE value per setting (the mean over its CV folds / Mapper subsamples); the summaries are
mean +/- std of those per-seed values across seeds. All values are supplied by a conductor script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def aggregate_across_seeds(per_seed_df, *, group_cols, value_cols):
    """per_seed_df: one row per (seed, setting). Returns one row per setting with n_seeds, mean_<v>, std_<v>."""
    g = per_seed_df.groupby(group_cols, sort=True)
    out = g.size().rename('n_seeds').reset_index()
    for v in value_cols:
        stats = g[v].agg(['mean', 'std']).reset_index(drop=True)
        out[f'mean_{v}'] = stats['mean'].values
        out[f'std_{v}'] = stats['std'].values
    return out


def plot_fft_seed_summary(fft_summary, *, styles, baseline_accuracy, baseline_color, plot_path, show_plot, dpi):
    """1x2: validation accuracy (with majority baseline) and cross-entropy vs L2 lambda, mean +/- std across seeds."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(fft_summary))
    lambdas = fft_summary['l2_lambda'].values

    for xi, (_, row), style in zip(x, fft_summary.iterrows(), styles):
        kw = dict(fmt=style['marker'], color=style['color'], ecolor=style['color'], markersize=9,
                  linestyle='none', elinewidth=1.2, capsize=4, label=f"$\\lambda$ = {row['l2_lambda']:g}")
        ax_acc.errorbar([xi], [row['mean_cv_accuracy']], yerr=[row['std_cv_accuracy']], **kw)
        ax_loss.errorbar([xi], [row['mean_cv_log_loss']], yerr=[row['std_cv_log_loss']], **kw)

    if baseline_accuracy is not None:
        ax_acc.axhline(baseline_accuracy, color=baseline_color, linestyle='--', linewidth=1.2,
                       label=f"majority baseline ({baseline_accuracy:.1%})")

    for ax, ylabel in ((ax_acc, 'validation accuracy'), (ax_loss, 'categorical cross-entropy')):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lam:g}" for lam in lambdas])
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xlabel('L2 regularization $\\lambda$', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5)

    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plot_path, dpi=dpi)
    print(f"saved FFT seed summary to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _with_text_columns(summary, value_cols, labels, decimals):
    """Adds a readable 'mean ± std' column per value; numeric mean/std columns are kept next to it."""
    out = summary.copy()
    for v in value_cols:
        out[labels.get(v, v)] = [f"{m:.{decimals}f} ± {s:.{decimals}f}"
                                 for m, s in zip(out[f'mean_{v}'], out[f'std_{v}'])]
    return out


def write_seed_summary_excel(*, path, seeds_df, fft_summary, fft_values, mapper_summary, mapper_values,
                             value_labels, decimals):
    """Excel workbook with sheets FFT, Mapper (mean ± std across seeds) and Seeds (which seeds were run)."""
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for sheet, df, values in (('FFT', fft_summary, fft_values), ('Mapper', mapper_summary, mapper_values)):
            _with_text_columns(df, values, value_labels, decimals).to_excel(xw, sheet_name=sheet, index=False)
        seeds_df.to_excel(xw, sheet_name='Seeds', index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(10, min(40, 2 + max(len(str(c.value or ''))
                                                                                          for c in col)))
    print(f"saved seed summary workbook to '{path}'")


def plot_metrics_per_seed(per_seed_df, *, panels, style, baselines, plot_path, show_plot, dpi):
    """
    1 x len(panels) scatter of a metric vs seed. panels = [(value_col, error_col or None, ylabel), ...];
    error_col (e.g. the within-seed subsample std) gives error bars. baselines = {value_col: (y, color, label)}
    draws a dashed horizontal line on that panel. The across-seed mean +/- std is written above each panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
    seeds = per_seed_df['seed'].values
    for ax, (col, err_col, ylabel) in zip(axes[0], panels):
        yerr = per_seed_df[err_col].values if err_col else None
        ax.errorbar(seeds, per_seed_df[col].values, yerr=yerr, fmt=style['marker'], color=style['color'],
                    ecolor=style['color'], markersize=8, linestyle='none', elinewidth=1.2, capsize=3)
        if col in baselines:
            y, color, label = baselines[col]
            ax.axhline(y, color=color, linestyle='--', linewidth=1.2, label=label)
            ax.legend(frameon=False, fontsize=10, loc='best')
        ax.set_title(f"mean across seeds = {per_seed_df[col].mean():.3f} $\\pm$ {per_seed_df[col].std():.3f}",
                     fontsize=10)
        ax.set_xticks(seeds)
        ax.set_xlabel('seed', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    print(f"saved per-seed summary to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_group_comparison(groups, *, panels, styles, plot_path, show_plot, dpi):
    """
    1 x len(panels): one point per group at its value +/- error. groups = [(label, {metric: (value, error)}), ...],
    panels = [(metric, ylabel), ...], styles[i] = {"marker", "color"} for the i-th group.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), squeeze=False)
    x = np.arange(len(groups))
    for ax, (metric, ylabel) in zip(axes[0], panels):
        for xi, (label, vals), style in zip(x, groups, styles):
            v, e = vals[metric]
            ax.errorbar([xi], [v], yerr=[e], fmt=style['marker'], color=style['color'], ecolor=style['color'],
                        markersize=9, linestyle='none', elinewidth=1.2, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    print(f"saved group comparison to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_method_metric_heatmap(per_method, *, methods, metrics, metric_labels, cmap, vmin, vmax, annotate,
                               decimals, na_color, na_text_color, plot_path, show_std=False, show_plot, dpi):
    """
    Heatmap of pipelines (rows) x metrics (columns); each cell = mean across seeds of that pipeline's per-seed value.
    per_method = {row label: per-seed DataFrame}; methods = row labels in order; metrics = column names.
    Metrics a pipeline does not compute are drawn in na_color and marked 'n/a'.
    """
    vals = np.full((len(methods), len(metrics)), np.nan)
    stds = np.full_like(vals, np.nan)
    for i, m in enumerate(methods):
        for j, v in enumerate(metrics):
            if v in per_method[m].columns:
                x = per_method[m][v].astype(float)
                vals[i, j], stds[i, j] = x.mean(), x.std()

    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 2.5, 1.0 * len(methods) + 1.6))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(na_color)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(len(methods)):
        for j in range(len(metrics)):
            if np.isnan(vals[i, j]):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=10, color=na_text_color)
            elif annotate:
                r, g, b, _ = cm(im.norm(vals[i, j]))
                light = 0.299 * r + 0.587 * g + 0.114 * b > 0.5
                txt = f"{vals[i, j]:.{decimals}f}"
                if show_std:
                    txt += f"\n$\\pm$ {stds[i, j]:.{decimals}f}"
                ax.text(j, i, txt, ha='center',
                        va='center', fontsize=10, color='black' if light else 'white')
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([metric_labels.get(v, v) for v in metrics], fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=11, fontweight='bold')
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label='mean across seeds')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    print(f"saved method x metric heatmap to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def pooled_confusion(preds, *, true_col, pred_col, labels, extra_pred):
    """Counts pooled over all rows of preds (e.g. concatenated splits). Rows = true labels; columns = labels plus any
    extra predicted codes (extra_pred = {code: name}, e.g. {-1: 'unassigned'}) that actually occur."""
    cols = list(labels) + [c for c in extra_pred if (preds[pred_col] == c).any()]
    cm = pd.crosstab(preds[true_col], preds[pred_col]).reindex(index=list(labels), columns=cols, fill_value=0)
    return cm.rename(columns={c: extra_pred[c] for c in extra_pred})


def plot_confusion_matrices(matrices, *, cmap, xlabel, ylabel, plot_path, show_plot, dpi):
    """
    1 x len(matrices) confusion matrices. matrices = [(title, count DataFrame), ...] (rows = true, cols = predicted).
    Color = fraction of each true class (row-normalized, 0-1); text = pooled count.
    """
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices) + 1.2, 4.6), squeeze=False)
    im = None
    for ax, (title, cm) in zip(axes[0], matrices):
        counts = cm.values.astype(float)
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        cmap_obj = plt.get_cmap(cmap)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                r, g, b, _ = cmap_obj(frac[i, j])
                ax.text(j, i, f"{int(counts[i, j])}", ha='center', va='center', fontsize=11,
                        color='black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else 'white')
        ax.set_xticks(np.arange(counts.shape[1]))
        ax.set_xticklabels([str(c) for c in cm.columns], fontsize=11, fontweight='bold')
        ax.set_yticks(np.arange(counts.shape[0]))
        ax.set_yticklabels([str(r) for r in cm.index], fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label='fraction of true class')
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"saved confusion matrices to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def select_by_rank_sum(df, *, criteria):
    """
    Pick one row of a tuning table by a fixed rule. criteria = [(column, "max" | "min"), ...].
    Each criterion ranks the rows (1 = best); the row with the lowest summed rank wins. Ties are broken by the
    criteria in the order given. Returns (selected row as a Series, table with rank columns added).
    """
    out = df.copy()
    rank_cols = []
    for col, direction in criteria:
        if direction not in ("max", "min"):
            raise ValueError(f"criterion direction must be 'max' or 'min', got {direction!r}")
        out[f"rank_{col}"] = out[col].rank(ascending=(direction == "min"), method="min")
        rank_cols.append(f"rank_{col}")
    out["rank_sum"] = out[rank_cols].sum(axis=1)
    order = out.sort_values(["rank_sum", *rank_cols], kind="mergesort")
    return order.iloc[0], out


def summarize_metrics_long(per_seed_df, *, value_cols, value_labels, decimals):
    """One row per metric: n_seeds, mean, std (across seeds) and a readable 'mean ± std' string."""
    rows = []
    for v in value_cols:
        x = per_seed_df[v].astype(float)
        rows.append({"metric": value_labels.get(v, v), "n_seeds": int(x.notna().sum()), "mean": x.mean(),
                     "std": x.std(), "mean ± std": f"{x.mean():.{decimals}f} ± {x.std():.{decimals}f}"})
    return pd.DataFrame(rows)


def write_sheets_excel(*, path, sheets):
    """sheets: {sheet name: DataFrame}, written in order with auto-sized columns."""
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(10, min(40, 2 + max(len(str(c.value or ''))
                                                                                          for c in col)))
    print(f"saved summary workbook to '{path}'")

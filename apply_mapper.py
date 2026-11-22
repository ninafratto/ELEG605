#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nefratto

Workflow module: KeplerMapper clustering head on MoCo features + topological evaluation metrics.
All values are supplied by a conductor script. Mapper hyperparameters are passed as a dict
(`mapper_params`) whose keys match `build_mapper_graph`'s keyword arguments:
    metric_pca_components, lens_pca_components, num_cubes, perc_overlap,
    dbscan_eps, dbscan_min_samples, dbscan_metric
Topological stability is the co-membership Jaccard between subsample graphs and the full graph.
"""

from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from apply_kmeans import load_split_features

UNASSIGNED = -1   # label given to test samples that fall in no dev-labelled Mapper node


def kmapper_to_networkx(graph):
    """Converts a KeplerMapper output dictionary into a NetworkX Graph object."""
    G = nx.Graph()
    for node_id in graph['nodes']:
        G.add_node(node_id)
    for u, neighbors in graph['links'].items():
        for v in neighbors:
            G.add_edge(u, v)
    return G


def compute_mapper_topological_metrics(graph):
    """
    Single-graph descriptors: b0 (connected components), connectedness, singletons and edge density
    (fraction of possible node pairs joined by an edge).
    Topological stability under perturbation is a separate, two-graph measure: see `comembership_jaccard`.
    """
    G = kmapper_to_networkx(graph)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {"b0": 0, "connectedness": 0.0, "singletons": 0, "edge_density": 0.0}

    b0 = nx.number_connected_components(G)

    largest_cc_size = len(max(nx.connected_components(G), key=len))
    connectedness = largest_cc_size / num_nodes if num_nodes > 0 else 0.0
    singletons = sum(1 for node, degree in G.degree() if degree == 0)
    edge_density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    return {
        "b0": b0,
        "connectedness": connectedness,
        "singletons": singletons,
        "edge_density": edge_density,
    }


def _comembership_pairs(graph, index_map, n_total, keep=None):
    """
    Encoded set of sample pairs (i < j) that share at least one Mapper node. `index_map` maps the graph's local
    row indices to common sample ids in [0, n_total); `keep` (bool mask over ids) restricts to a subset of samples.
    """
    codes = []
    for members in graph['nodes'].values():
        ids = np.unique(index_map[np.asarray(members)])
        if keep is not None:
            ids = ids[keep[ids]]
        if len(ids) < 2:
            continue
        a, b = np.triu_indices(len(ids), k=1)
        codes.append(ids[a] * n_total + ids[b])
    return np.unique(np.concatenate(codes)) if codes else np.empty(0, dtype=np.int64)


def comembership_jaccard(graph_ref, graph_sub, sub_idx, n_ref):
    """
    Stability of a Mapper graph under perturbation. `graph_ref` is built on n_ref samples; `graph_sub` on the
    subset of them given by `sub_idx` (sub row k = ref sample sub_idx[k]). Over the samples present in both,
    returns |A & B| / |A | B| where A, B are the pairs of samples sharing a node in each graph
    (1 = same local grouping, 0 = nothing in common).
    """
    keep = np.zeros(n_ref, dtype=bool)
    keep[sub_idx] = True
    pairs_ref = _comembership_pairs(graph_ref, np.arange(n_ref), n_ref, keep=keep)
    pairs_sub = _comembership_pairs(graph_sub, np.asarray(sub_idx), n_ref)
    union = np.union1d(pairs_ref, pairs_sub).size
    if union == 0:
        return 1.0
    return np.intersect1d(pairs_ref, pairs_sub, assume_unique=True).size / union


def build_mapper_graph(X, *, fit_X, metric_pca_components, lens_pca_components, num_cubes, perc_overlap,
                       dbscan_eps, dbscan_min_samples, dbscan_metric, random_state):
    """
    Standardize -> PCA metric space -> PCA lens -> KeplerMapper graph on X.
    The scaler, both PCAs, and the lens min-max scaling are fitted on `fit_X` only, then applied to X
    (pass fit_X=X to fit on the graph's own data).
    """
    scaler = StandardScaler().fit(fit_X)
    pca_metric = PCA(n_components=min(metric_pca_components, fit_X.shape[1]), random_state=random_state)
    fit_metric = pca_metric.fit_transform(scaler.transform(fit_X))
    X_metric = pca_metric.transform(scaler.transform(X))

    pca_lens = PCA(n_components=lens_pca_components, random_state=random_state)
    lens_scaler = MinMaxScaler().fit(pca_lens.fit_transform(fit_metric))
    lens = lens_scaler.transform(pca_lens.transform(X_metric))

    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=num_cubes, perc_overlap=perc_overlap)
    clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)

    graph = mapper.map(lens=lens, X=X_metric, cover=cover, clusterer=clusterer)
    return mapper, graph, X_metric


_PANELS = (('stability', 'topological stability'), ('connectedness', 'connectedness ratio'))
_SUMMARY_METRICS = (('b0', '$b_0$'), ('singletons', 'singletons'))


def _save_or_show(fig, plot_path, show_plot, dpi, what):
    fig.savefig(plot_path, dpi=dpi)
    print(f"saved {what} to '{plot_path}'")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mapper_subsample_metrics(draws_df, *, style, plot_path, show_plot, dpi):
    """
    One grid combination: 1x2 scatter of co-membership stability and connectedness ratio for each subsample graph.
    Betti numbers and singletons are summarized as mean +/- std (across subsamples) above.
    style = {"marker": ..., "color": ...}.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draws = draws_df['subsample'].values

    for ax, (col, ylabel) in zip(axes, _PANELS):
        ax.scatter(draws, draws_df[col], marker=style['marker'], color=style['color'], s=30)
        ax.set_xlabel('subsample number', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5)

    summary = "   ".join(f"{label} = {draws_df[col].mean():.2f} $\\pm$ {draws_df[col].std():.2f}"
                         for col, label in _SUMMARY_METRICS)
    fig.text(0.5, 0.985, summary, ha='center', va='top', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_or_show(fig, plot_path, show_plot, dpi, "Mapper subsample plot")


def plot_mapper_grid_comparison(summary_by_combo, *, styles, plot_path, show_plot, dpi):
    """
    All grid combinations on one figure: 1x2 plot of co-membership stability and connectedness ratio,
    one point per combination at mean +/- std across subsamples. `summary_by_combo` maps a legend label ->
    the summary dict from run_mapper_subsample_stability; styles[i] is used for the i-th combination.
    """
    if len(summary_by_combo) > len(styles):
        raise ValueError(f"{len(summary_by_combo)} grid combinations but only {len(styles)} styles")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    positions = np.arange(1, len(summary_by_combo) + 1)

    for x, (label, s), style in zip(positions, summary_by_combo.items(), styles):
        for ax, (col, _) in zip(axes, _PANELS):
            ax.errorbar([x], [s[f'mean_{col}']], yerr=[s[f'std_{col}']], fmt=style['marker'], color=style['color'],
                        ecolor=style['color'], markersize=8, linestyle='none', elinewidth=1.2, capsize=3, label=label)

    for ax, (_, ylabel) in zip(axes, _PANELS):
        ax.set_xlabel('grid combination', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xticks(positions)
        ax.grid(True, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = 4
    fig.legend(handles, labels, loc='upper center', ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * -(-len(positions) // ncol)))
    _save_or_show(fig, plot_path, show_plot, dpi, "Mapper grid comparison plot")


def _param_text(param, value, param_labels):
    v = f"{value:g}" if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
    return f"{param_labels.get(param, param)} = {v}"


def plot_mapper_grid_interaction(summary_df, *, x_param, color_param, fill_param, color_styles, param_labels,
                                 x_offset, plot_path, show_plot, dpi):
    """
    Interaction plot of the grid search: 1x2 (stability, connectedness), mean +/- std across subsamples.
    x = values of x_param; marker/color = values of color_param (color_styles[i] for the i-th sorted value);
    filled vs open marker = 1st vs 2nd+ sorted value of fill_param. A thin line joins each (color, fill) series
    across x. Series are offset in x by x_offset so error bars do not overlap.
    summary_df: one row per grid combination with columns x_param, color_param, fill_param, mean_*, std_*.
    """
    color_vals = sorted(summary_df[color_param].unique())
    fill_vals = sorted(summary_df[fill_param].unique())
    x_vals = sorted(summary_df[x_param].unique())
    if len(color_vals) > len(color_styles):
        raise ValueError(f"{len(color_vals)} values of '{color_param}' but only {len(color_styles)} color_styles")

    n_series = len(color_vals) * len(fill_vals)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * x_offset
    x_pos = {v: i for i, v in enumerate(x_vals)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    k = 0
    for cv, style in zip(color_vals, color_styles):
        for j, fv in enumerate(fill_vals):
            sub = summary_df[(summary_df[color_param] == cv) & (summary_df[fill_param] == fv)].sort_values(x_param)
            xs = np.array([x_pos[v] for v in sub[x_param]]) + offsets[k]
            label = f"{_param_text(color_param, cv, param_labels)}, {_param_text(fill_param, fv, param_labels)}"
            for ax, (col, _) in zip(axes, _PANELS):
                ax.errorbar(xs, sub[f'mean_{col}'], yerr=sub[f'std_{col}'], fmt=style['marker'] + '-',
                            color=style['color'], ecolor=style['color'], markersize=8, linewidth=0.8,
                            elinewidth=1.2, capsize=3, markeredgecolor=style['color'],
                            markerfacecolor=style['color'] if j == 0 else 'white', label=label)
            k += 1

    for ax, (_, ylabel) in zip(axes, _PANELS):
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:g}" for v in x_vals])
        ax.set_xlim(-0.5, len(x_vals) - 0.5)
        ax.set_xlabel(param_labels.get(x_param, x_param), fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(n_series, 4)
    fig.legend(handles, labels, loc='upper center', ncol=ncol, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.05 * -(-n_series // ncol)))
    _save_or_show(fig, plot_path, show_plot, dpi, "Mapper grid interaction plot")


def plot_mapper_grid_heatmap(summary_df, *, row_param, col_param, panel_param, param_labels, cmap,
                             plot_path, show_plot, dpi):
    """
    Heatmaps of the grid search: rows of panels = metrics (stability, connectedness), columns of panels = values
    of panel_param. Each panel is a row_param x col_param grid colored by the mean_<metric> column.
    Each metric shares one color scale (and colorbar) across its panels.
    """
    row_vals = sorted(summary_df[row_param].unique())
    col_vals = sorted(summary_df[col_param].unique())
    panel_vals = sorted(summary_df[panel_param].unique())

    fig, axes = plt.subplots(len(_PANELS), len(panel_vals), figsize=(4.2 * len(panel_vals) + 1.2, 3.6 * len(_PANELS)),
                             squeeze=False)
    colormap = plt.get_cmap(cmap)

    for r, (metric, metric_label) in enumerate(_PANELS):
        vmin, vmax = summary_df[f'mean_{metric}'].min(), summary_df[f'mean_{metric}'].max()
        if vmax == vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        for c, pv in enumerate(panel_vals):
            ax = axes[r, c]
            sub = summary_df[summary_df[panel_param] == pv].set_index([row_param, col_param])
            mean = np.array([[sub.loc[(rv, cv), f'mean_{metric}'] for cv in col_vals] for rv in row_vals])
            im = ax.imshow(mean, cmap=colormap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
            ax.set_xticks(range(len(col_vals)))
            ax.set_yticks(range(len(row_vals)))
            ax.set_yticklabels([f"{v:g}" for v in row_vals])
            if r == len(_PANELS) - 1:   # x labels only on the bottom row of panels
                ax.set_xticklabels([f"{v:g}" for v in col_vals])
                ax.set_xlabel(param_labels.get(col_param, col_param), fontsize=11, fontweight='bold')
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(param_labels.get(row_param, row_param), fontsize=11, fontweight='bold')
            if r == 0:
                ax.set_title(_param_text(panel_param, pv, param_labels), fontsize=11, fontweight='bold')
        cbar = fig.colorbar(im, ax=list(axes[r]), fraction=0.04, pad=0.03)
        cbar.set_label(metric_label, fontsize=11, fontweight='bold')

    _save_or_show(fig, plot_path, show_plot, dpi, "Mapper grid heatmap")


def run_mapper_pipeline(*, feature_prefix, csv_path, name_col, mapper_params, random_state, output_html):
    """
    Executes KeplerMapper on the full development set (optionally writing the interactive HTML).
    Nodes are colored by the first metric-space principal component (label-free), so that labels
    are not seen while tuning; labeled coloring is reserved for the test evaluation.
    """
    X_target, _, _ = load_split_features(feature_prefix=feature_prefix, csv_path=csv_path,
                                         name_col=name_col, split_tag="dev")

    mapper, graph, X_metric = build_mapper_graph(X_target, fit_X=X_target, random_state=random_state, **mapper_params)

    if output_html:
        mapper.visualize(
            graph,
            color_values=X_metric[:, 0],
            color_function_name="PC1 (label-free)",
            title="MoCo ResNet-50 Mapper Graph (Development Set)",
            path_html=str(output_html)
        )

    return mapper, graph, X_metric


def _subsample_stability(X, y, *, mapper_params, random_state, resample):
    """Reference graph on all of X; n_draws stratified subsamples without replacement, each vs the reference.
    Returns (draws_df, summary) with summary = ref_<metric>, mean_<metric>, std_<metric>."""
    n = len(X)
    _, graph_ref, _ = build_mapper_graph(X, fit_X=X, random_state=random_state, **mapper_params)
    ref = compute_mapper_topological_metrics(graph_ref)

    rows = []
    for d in range(resample["n_draws"]):
        idx, _ = train_test_split(np.arange(n), train_size=resample["fraction"], shuffle=True,
                                  random_state=random_state + d, stratify=y if resample["stratify"] else None)
        X_s = X[idx]
        _, g_s, _ = build_mapper_graph(X_s, fit_X=X_s, random_state=random_state, **mapper_params)
        m_s = compute_mapper_topological_metrics(g_s)
        m_s['stability'] = comembership_jaccard(graph_ref, g_s, idx, n)
        m_s['subsample'] = d + 1
        rows.append(m_s)

    draws_df = pd.DataFrame(rows)
    summary = {f"ref_{k}": v for k, v in ref.items()}
    for col in draws_df.columns.drop('subsample'):
        summary[f"mean_{col}"] = float(draws_df[col].mean())
        summary[f"std_{col}"] = float(draws_df[col].std())
    return draws_df, summary


def run_mapper_size_matched_stability(*, feature_prefix, csv_path, name_col, mapper_params, random_state,
                                      set_size, n_sets, resample):
    """
    Size-matched control for the dev set: n_sets stratified dev subsets of exactly `set_size` samples (e.g. the
    test-set size) are drawn without replacement. Each subset gets the SAME protocol as the test set: its own
    reference graph, then resample["n_draws"] subsamples of it, each compared with that reference.
    No hyperparameter is changed and no test data is used.

    Returns a DataFrame with one row per subset: 'set' plus ref_*, mean_* and std_* (std = across subsamples).
    """
    X_dev, y_dev, _ = load_split_features(feature_prefix=feature_prefix, csv_path=csv_path,
                                          name_col=name_col, split_tag="dev")
    print(f"\nsize-matched dev stability: {n_sets} stratified dev subsets of {set_size}/{len(X_dev)}, "
          f"{resample['n_draws']} draws of {resample['fraction']:.0%} each")
    rows = []
    for k in range(n_sets):
        idx, _ = train_test_split(np.arange(len(X_dev)), train_size=set_size, shuffle=True,
                                  random_state=1000 * random_state + k, stratify=y_dev)
        _, summary = _subsample_stability(X_dev[idx], np.asarray(y_dev)[idx], mapper_params=mapper_params,
                                          random_state=random_state, resample=resample)
        rows.append({"set": k + 1, **summary})
        print(f"  subset {k + 1}/{n_sets} | stability: {summary['mean_stability']:.3f} +/- "
              f"{summary['std_stability']:.3f} | connectedness: {summary['mean_connectedness']:.3f} +/- "
              f"{summary['std_connectedness']:.3f}")
    return pd.DataFrame(rows)


def run_mapper_subsample_stability(*, feature_prefix, csv_path, name_col, split_tag, mapper_params,
                                   random_state, resample, plot_style, plot_path, show_plot, dpi):
    """
    Stability of the full-split Mapper graph (split_tag "dev" or "test"; the same graph as that split's
    topology graph) under subsampling. Each graph is fitted on its own samples only.

    Reference graph: all samples of the split. `resample` = {"n_draws", "fraction", "stratify"}: the samples are
    subsampled WITHOUT replacement (`fraction` of them, stratified by label if `stratify`) `n_draws` times and a
    graph is rebuilt on each subsample. Per subsample:
      'stability' = co-membership Jaccard vs the reference graph (over the samples in the subsample)
      plus the single-graph descriptors (b0, connectedness, singletons, edge density).
    Labels only select which points enter a subsample; they are never used to build a graph.

    Returns (draws_df, summary): one row per subsample, and a dict with 'ref_<metric>' for the reference graph
    and 'mean_<metric>' / 'std_<metric>' across subsamples.
    """
    X_dev, y_dev, _ = load_split_features(feature_prefix=feature_prefix, csv_path=csv_path,
                                          name_col=name_col, split_tag=split_tag)

    print(f"\nMapper subsample stability ({split_tag} n={len(X_dev)},{resample['n_draws']} draws of "
          f"{resample['fraction']:.0%}{', stratified' if resample['stratify'] else ''})")
    print("Mapper params: " + ", ".join(f"{k}={v}" for k, v in mapper_params.items()))

    draws_df, summary = _subsample_stability(X_dev, y_dev, mapper_params=mapper_params,
                                             random_state=random_state, resample=resample)
    ref = {k[4:]: v for k, v in summary.items() if k.startswith("ref_")}

    print(f"reference graph | b0: {ref['b0']} | connectedness: {ref['connectedness']:.3f} | "
          f"singletons: {ref['singletons']}")
    if len(draws_df):
        print(f"subsamples      | stability (Jaccard): {summary['mean_stability']:.3f} +/- {summary['std_stability']:.3f}"
              f" | connectedness: {summary['mean_connectedness']:.3f} +/- {summary['std_connectedness']:.3f}")
        plot_mapper_subsample_metrics(draws_df, style=plot_style, plot_path=plot_path, show_plot=show_plot, dpi=dpi)

    return draws_df, summary


def _majority(labels, tie_break_order):
    """Most common label; a tie goes to whichever tied label is most common in the dev set (earliest in tie_break_order)."""
    counts = Counter(labels)
    top = max(counts.values())
    tied = [lab for lab, c in counts.items() if c == top]
    return tied[0] if len(tied) == 1 else min(tied, key=tie_break_order.index)


def evaluate_mapper_on_test(*, dev_prefix, test_prefix, csv_path, name_col, mapper_params,
                            random_state, output_html):
    """
    Test evaluation for the Mapper head.

    (1) Topology: Mapper graph on the test features only (same graph construction as on dev) ->
        b0, connectedness, singletons, edge density.
    (2) Majority-vote labels for ARI: Mapper graph on dev + test features together, with the scaler / PCA / lens
        fitted on dev only (same dev-fitted space as the K-means test silhouette). Each node is labelled
        by the majority label of its DEV members only (nodes with no dev members stay unlabelled). Each test
        sample takes the majority vote of the labelled nodes it belongs to; ties -> the tied label most common
        in the dev set; samples in no labelled node -> UNASSIGNED (-1), scored as its own cluster.
        Test labels are used only for scoring.
    """
    X_dev, y_dev, _ = load_split_features(feature_prefix=dev_prefix, csv_path=csv_path, name_col=name_col, split_tag="dev")
    X_test, y_test, test_df = load_split_features(feature_prefix=test_prefix, csv_path=csv_path, name_col=name_col, split_tag="test")
    y_dev = [int(v) for v in y_dev]
    y_test = np.asarray(y_test, dtype=int)
    n_dev, n_test = len(X_dev), len(X_test)
    print("Mapper params: " + ", ".join(f"{k}={v}" for k, v in mapper_params.items()))

    # (1) topology on the test set alone
    _, graph_test, _ = build_mapper_graph(X_test, fit_X=X_test, random_state=random_state, **mapper_params)
    topo = compute_mapper_topological_metrics(graph_test)
    print(f"test-only graph | nodes: {len(graph_test['nodes'])} | b0: {topo['b0']} | "
          f"connectedness: {topo['connectedness']:.3f} | singletons: {topo['singletons']} | "
          f"edge density: {topo['edge_density']:.3f}")

    # (2) majority-vote labelling on the dev + test graph
    X_comb = np.vstack([X_dev, X_test])
    mapper, graph_comb, X_metric_comb = build_mapper_graph(X_comb, fit_X=X_dev, random_state=random_state, **mapper_params)
    dev_freq_order = [lab for lab, _ in Counter(y_dev).most_common()]

    node_labels = {}
    for node_id, members in graph_comb['nodes'].items():
        dev_members = [y_dev[i] for i in members if i < n_dev]
        if dev_members:
            node_labels[node_id] = _majority(dev_members, dev_freq_order)

    votes = defaultdict(list)
    for node_id, label in node_labels.items():
        for i in graph_comb['nodes'][node_id]:
            if i >= n_dev:
                votes[i - n_dev].append(label)

    y_pred = np.array([_majority(votes[j], dev_freq_order) if votes[j] else UNASSIGNED for j in range(n_test)])
    n_unassigned = int((y_pred == UNASSIGNED).sum())
    n_ambiguous = int(sum(1 for j in range(n_test) if len(set(votes[j])) > 1))

    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    acc = float(np.mean(y_pred == y_test))
    X_metric_test_comb = X_metric_comb[n_dev:]
    sil = (silhouette_score(X_metric_test_comb, y_pred)
           if 1 < len(np.unique(y_pred)) < n_test else float("nan"))

    print(f"dev+test graph | nodes: {len(graph_comb['nodes'])} | dev-labelled nodes: {len(node_labels)}")
    print(f"test samples: {n_test} | ambiguous (nodes disagree): {n_ambiguous} | unassigned: {n_unassigned}")
    print(f"test ARI: {ari:.4f} | test NMI: {nmi:.4f} | majority-vote accuracy: {acc:.4f} | partition silhouette: {sil:.4f}")
    print("\ncontingency (rows = true label, cols = majority-vote label; -1 = unassigned):")
    print(pd.crosstab(pd.Series(y_test, name="true"), pd.Series(y_pred, name="voted")))

    if output_html:
        mapper.visualize(
            graph_comb,
            color_values=np.concatenate([np.asarray(y_dev), y_test]),
            color_function_name="Continuous Phase Target",
            title="MoCo ResNet-50 Mapper Graph (Dev + Test)",
            path_html=str(output_html)
        )
        print(f"saved Mapper graph to '{output_html}'")

    return {
        "method": "MoCo + Mapper (majority vote)",
        "n_test": int(n_test),
        "test_accuracy": acc,
        "test_ari": ari,
        "test_nmi": nmi,
        "test_silhouette": sil,
        "n_unassigned": n_unassigned,
        "n_ambiguous": n_ambiguous,
        **{f"topo_{k}": v for k, v in topo.items()},
        "names": test_df[name_col].values,
        "y_true": y_test,
        "y_pred": y_pred,
    }

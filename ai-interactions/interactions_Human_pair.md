could you please make it ipynb?

#!/usr/bin/env python3
"""
Phase I UMAPs: CellTypist Immune_All_Low on Human Immune Health Atlas cells.

For each age group, draws side-by-side panels on the SAME UMAP coordinates:
  1. atlas cell_type            (ground-truth clusters)
  2. CellTypist predicted_labels (what the model says)
  3. match status               (correct / wrong / unmapped; needs label_mapping.csv)
  4. confidence                 (conf_score, or top decision score if missing)
  5. donor                      (sanity check for donor/batch effects)

Embedding modes:
  joint     - one UMAP on all 54k cells, split by age group (coordinates comparable
              across groups; other groups drawn as grey background)
  per_group - a separate UMAP per age group (cleaner local structure, but
              coordinates are NOT comparable between groups)
  both      - do both

Also writes CSVs: UMAP coordinates, atlas x predicted crosstabs per age group
(helpful for building the label mapping), and error rates per age group / donor /
atlas type.

Requires: scanpy, anndata, pandas, numpy, matplotlib
"""

import os
import shutil

import numpy as np
import pandas as pd
import anndata as ad
import cellxgene_census
import scanpy as sc

import warnings
from anndata import ImplicitModificationWarning

census_version = '2025-11-08'

from celltypist import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------- config ---

path = '/Users/artsvetkova/Studies/_JKU/3rd_semester/AI and Visualization/UE'
base_dir = os.path.join(path, "iha_celltypist")
prepared_path = os.path.join(base_dir, "iha_expression_prepared.h5ad")
# PREPARED_H5AD = "iha_expression_prepared.h5ad"   # log1p, 10k-normalized in .X
PREPARED_H5AD = os.path.join(base_dir, "iha_expression_prepared.h5ad")
# PRED_CSV = "baseline_predictions.csv"
PRED_CSV = os.path.join(base_dir, "baseline_predictions.csv")
DECISION_CSV = os.path.join(base_dir, "baseline_decision_matrix.csv")    # optional
MAPPING_CSV = "label_mapping.csv"                # optional, see add_match_status()
# OUT_DIR = "umap_out"
OUT_DIR = os.path.join(base_dir, "umap_out")

CELL_KEY = "soma_joinid"
AGE_COL = "age_group"          # <- adjust to your .obs column names
DONOR_COL = "donor_id"
ATLAS_COL = "cell_type"
PRED_COL = "predicted_labels"

MODE = "both"                  # "joint", "per_group" or "both"
N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 15
MIN_DIST = 0.3
SEED = 0
LABEL_MIN_CELLS = 100          # write a text label only for groups this large
# ----------------------------------------------------------------------------

STATUS_COLORS = {
    "correct": "#c7c7c7",
    "wrong": "#d62728",
    "unmapped": "#6baed6",
    "no mapping yet": "#c7c7c7",
}
STATUS_ORDER = ["no mapping yet", "correct", "unmapped", "wrong"]  # wrong on top


# ------------------------------------------------------------------- data ---
def load():
    adata = sc.read_h5ad(PREPARED_H5AD)
    ids = adata.obs[CELL_KEY] if CELL_KEY in adata.obs else adata.obs_names.to_series()
    adata.obs_names = ids.astype(str).values
    adata.obs[CELL_KEY] = adata.obs_names

    for col in (AGE_COL, DONOR_COL, ATLAS_COL):
        if col not in adata.obs:
            raise KeyError(f"'{col}' not in .obs. Available: {list(adata.obs.columns)}")

    # predictions
    pred = pd.read_csv(PRED_CSV)
    if CELL_KEY not in pred.columns:
        pred = pred.rename(columns={pred.columns[0]: CELL_KEY})
    pred[CELL_KEY] = pred[CELL_KEY].astype(str)
    pred = pred.set_index(CELL_KEY)
    missing = adata.obs_names.difference(pred.index)
    if len(missing):
        raise ValueError(f"{len(missing)} cells have no prediction, e.g. {list(missing[:5])}")
    pred = pred.loc[adata.obs_names]
    for c in pred.columns:
        name = c if c not in adata.obs else f"pred_{c}"
        adata.obs[name] = pred[c].values

    # decision matrix -> top score and margin between best and second-best type
    if os.path.exists(DECISION_CSV):
        dm = pd.read_csv(DECISION_CSV)
        if CELL_KEY in dm.columns:
            dm = dm.set_index(CELL_KEY)
        else:
            dm = dm.set_index(dm.columns[0])
        dm.index = dm.index.astype(str)
        dm = dm.select_dtypes("number")
        if adata.obs_names.isin(dm.index).all():
            dm = dm.loc[adata.obs_names]
        elif len(dm) == adata.n_obs:
            print("WARNING: decision matrix ids don't match; assuming same row order.")
        else:
            dm = None
        if dm is not None:
            v = np.sort(dm.to_numpy(float), axis=1)
            adata.obs["top_decision_score"] = v[:, -1]
            adata.obs["decision_margin"] = v[:, -1] - v[:, -2]

    for col in (AGE_COL, DONOR_COL, ATLAS_COL, PRED_COL):
        adata.obs[col] = adata.obs[col].astype(str)
    return adata


def add_match_status(adata):
    """label_mapping.csv: columns atlas_cell_type, celltypist_label.
    One row per allowed pair; an atlas type may map to several CellTypist labels."""
    if not os.path.exists(MAPPING_CSV):
        adata.obs["match_status"] = "no mapping yet"
        print(f"No {MAPPING_CSV} found: panel 3 will be empty until the mapping exists.")
        return False
    m = pd.read_csv(MAPPING_CSV).astype(str)
    allowed = m.groupby("atlas_cell_type")["celltypist_label"].apply(set).to_dict()
    status = [
        "unmapped" if a not in allowed else ("correct" if p in allowed[a] else "wrong")
        for a, p in zip(adata.obs[ATLAS_COL], adata.obs[PRED_COL])
    ]
    adata.obs["match_status"] = status
    return True


# -------------------------------------------------------------- embedding ---
def embed(ad):
    """HVG -> scale -> PCA -> kNN -> UMAP. Returns an (n_cells, 2) array."""
    ad = ad.copy()
    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat", batch_key=DONOR_COL)
    ad = ad[:, ad.var["highly_variable"]].copy()
    ad.X = ad.X.astype(np.float32)
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=N_PCS, random_state=SEED)
    sc.pp.neighbors(ad, n_neighbors=N_NEIGHBORS, random_state=SEED)
    sc.tl.umap(ad, min_dist=MIN_DIST, random_state=SEED)
    return ad.obsm["X_umap"]


# --------------------------------------------------------------- plotting ---
def make_palette(values):
    """Fixed color per category, so a type keeps its color in every figure."""
    cats = sorted(pd.unique(values))
    base = [plt.cm.tab20(i) for i in range(20)] + [plt.cm.tab20b(i) for i in range(20)] \
        + [plt.cm.tab20c(i) for i in range(20)] + [plt.cm.Set3(i) for i in range(12)]
    if len(cats) > len(base):
        base += [plt.cm.hsv(x) for x in np.linspace(0, 1, len(cats) - len(base), endpoint=False)]
    return {c: base[i] for i, c in enumerate(cats)}


def _point_size(n):
    return float(np.clip(40000 / max(n, 1), 0.5, 8))


def scatter_categorical(ax, xy, labels, colors, title, label_on_data=True, order=None):
    labels = np.asarray(labels).astype(str)
    if order is None:
        idx = np.random.default_rng(SEED).permutation(len(labels))  # avoid overplot bias
    else:
        rank = {c: i for i, c in enumerate(order)}
        idx = np.argsort([rank.get(l, -1) for l in labels], kind="stable")
    ax.scatter(xy[idx, 0], xy[idx, 1], c=[colors.get(l, "#cccccc") for l in labels[idx]],
               s=_point_size(len(labels)), lw=0, rasterized=True)
    if label_on_data:
        for c in pd.unique(labels):
            m = labels == c
            if m.sum() >= LABEL_MIN_CELLS:
                x, y = np.median(xy[m], axis=0)
                ax.text(x, y, c, fontsize=5, ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    ax.set_title(title, fontsize=10)


def scatter_continuous(ax, xy, values, title, fig):
    values = np.asarray(values, float)
    idx = np.argsort(-values)  # low-confidence cells drawn last (on top)
    sca = ax.scatter(xy[idx, 0], xy[idx, 1], c=values[idx], cmap="viridis",
                     s=_point_size(len(values)), lw=0, rasterized=True)
    fig.colorbar(sca, ax=ax, fraction=0.04, pad=0.01)
    ax.set_title(title, fontsize=10)


def plot_panels(obs, xy, title, fname, palettes, conf_col, bg=None):
    panels = ["atlas", "pred", "status"] + (["conf"] if conf_col else []) + ["donor"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.2))
    for ax, kind in zip(axes, panels):
        if bg is not None:
            ax.scatter(bg[:, 0], bg[:, 1], c="#f0f0f0", s=0.3, lw=0, rasterized=True)
        if kind == "atlas":
            scatter_categorical(ax, xy, obs[ATLAS_COL], palettes["atlas"],
                                f"Atlas {ATLAS_COL} ({obs[ATLAS_COL].nunique()} types)")
        elif kind == "pred":
            scatter_categorical(ax, xy, obs[PRED_COL], palettes["pred"],
                                f"CellTypist prediction ({obs[PRED_COL].nunique()} types)")
        elif kind == "status":
            s = obs["match_status"]
            n_eval = s.isin(["correct", "wrong"]).sum()
            rate = (s == "wrong").sum() / n_eval if n_eval else float("nan")
            scatter_categorical(ax, xy, s, STATUS_COLORS,
                                f"Prediction vs atlas (error rate {rate:.1%})",
                                label_on_data=False, order=STATUS_ORDER)
            present = [k for k in STATUS_ORDER if k in set(s)]
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=STATUS_COLORS[k],
                                      label=f"{k} ({(s == k).sum():,})") for k in present],
                      loc="lower left", fontsize=7, frameon=False)
        elif kind == "conf":
            scatter_continuous(ax, xy, obs[conf_col], f"Confidence ({conf_col})", fig)
        elif kind == "donor":
            scatter_categorical(ax, xy, obs[DONOR_COL], palettes["donor"],
                                "Donor (batch check)", label_on_data=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    fig.suptitle(f"{title}  -  n = {len(obs):,} cells, {obs[DONOR_COL].nunique()} donors")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{fname}.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- summary ---
def write_tables(obs, groups):
    for g in groups:
        o = obs[obs[AGE_COL] == g]
        pd.crosstab(o[ATLAS_COL], o[PRED_COL]).to_csv(
            os.path.join(OUT_DIR, f"crosstab_atlas_vs_pred_{g}.csv"))
    ev = obs[obs["match_status"].isin(["correct", "wrong"])].assign(
        wrong=lambda d: d["match_status"] == "wrong")
    if len(ev):
        for keys, name in [([AGE_COL], "age"), ([AGE_COL, DONOR_COL], "age_donor"),
                           ([AGE_COL, ATLAS_COL], "age_celltype")]:
            (ev.groupby(keys)["wrong"].agg(n_cells="size", error_rate="mean")
               .reset_index().to_csv(os.path.join(OUT_DIR, f"error_rate_by_{name}.csv"),
                                     index=False))


# ------------------------------------------------------------------- main ---

os.makedirs(OUT_DIR, exist_ok=True)
adata = load()
add_match_status(adata)
obs = adata.obs
groups = sorted(obs[AGE_COL].unique())
palettes = {"atlas": make_palette(obs[ATLAS_COL]),
            "pred": make_palette(obs[PRED_COL]),
            "donor": make_palette(obs[DONOR_COL])}
conf_col = next((c for c in ("conf_score", "top_decision_score") if c in obs), None)
coords = pd.DataFrame(index=obs.index)

if MODE in ("joint", "both"):
    print("Computing joint UMAP on all cells ...")
    xy = embed(adata)
    coords["joint_umap1"], coords["joint_umap2"] = xy[:, 0], xy[:, 1]
    for g in groups:
        m = (obs[AGE_COL] == g).to_numpy()
        plot_panels(obs[m], xy[m], f"Age group {g} (joint embedding)",
                    f"joint_{g}", palettes, conf_col, bg=xy)

if MODE in ("per_group", "both"):
    coords["group_umap1"] = np.nan
    coords["group_umap2"] = np.nan
    for g in groups:
        print(f"Computing UMAP for age group {g} ...")
        m = (obs[AGE_COL] == g).to_numpy()
        xy_g = embed(adata[m])
        coords.loc[m, "group_umap1"], coords.loc[m, "group_umap2"] = xy_g[:, 0], xy_g[:, 1]
        plot_panels(obs[m], xy_g, f"Age group {g} (own embedding)",
                    f"per_group_{g}", palettes, conf_col)

keep = [c for c in (AGE_COL, DONOR_COL, ATLAS_COL, PRED_COL, "match_status", conf_col,
                    "decision_margin") if c and c in obs]
coords.join(obs[keep]).to_csv(os.path.join(OUT_DIR, "umap_coords.csv"))
write_tables(obs, groups)
print(f"Done. Outputs in {OUT_DIR}/")



Now we want to filter out the dataset by the genes that are present only in the model. Give us please the code to look up the genes that our model considers for classification.



All right, now we want to filter out our H5AD dataset to leave only the genes that are present in the model. How can we do it?


Now we would like to discard, for example, half of the dataset that has low expressed genes and run the classification again. is it possible to do this filtering within each cell (50% of genes under expressed in cell1, 50% of genes under expressed in cell2, etc.)?

alright and then make this model-gene-filtering in the dataset and filtering 50% or whatever fraction in a separate notebook please


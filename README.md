# CellTypist performance in classification of single immune cells

## Task

We examine how the pretrained CellTypist `Immune_All_Low` model classifies selected cells from the Human Immune Health Atlas.

In **Phase I**, we explore the model's predictions, scores, and patterns across cell types, and age groups of donors. To investigate the model's annotating behaviour, we use some controlled data modifications to the input data, such as filtering the under-expressed genes. 

In **Phase II**, we investigate selected Phase I findings by examining model weights or using explanation techniques. 

Atlas cell labels and model classes do not always have the same names or level of detail. We preserve the original labels and store any harmonized labels separately. Per-cell results can be joined using `soma_joinid`.

## Model

**CellTypist `Immune_All_Low`**, used without retraining.

- Model information: [CellTypist models](https://www.celltypist.org/models)
- Related paper: [Cross-tissue immune cell analysis reveals tissue-specific features in humans](https://www.science.org/doi/10.1126/science.abl5197)
- GitHub repository: [Teichlab/celltypist](https://github.com/Teichlab/celltypist)

The model contains 98 cell types. Its predicted labels are model outputs, not replacements for the atlas annotations.

## Dataset

| Role | Dataset | Description | Link |
|---|---|---|---|
| Main | Human Immune Health Atlas | Selected healthy cells from 24 donors: 8 donors in each age group (`<18`, `18–45`, `>45`) | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/77f9d7e9-5675-49c3-abed-ce02f39eef1b) |

The sample was selected from the global dataset using these criteria:

- Healthy donors only (`disease = normal`).
- At least 2,000 selected cells per donor and at least 8 atlas cell types per donor.
- Up to 100 cells per donor × atlas cell type.
- Donor × cell-type groups with fewer than 10 cells were excluded.
- Random seed: `42`; selected cell IDs are recorded in `iha_sample_ids.csv`.

The selected dataset contains **54,225 cells**.

## Project files

| File | Description |
|---|---|
| `data/iha_sample_ids.csv` | Selected cell IDs (`soma_joinid`) and associated atlas and donor metadata; no gene expression |
| `iha_expression_raw.h5ad` | Selected cells with raw counts in `.X` |
| `iha_expression_prepared.h5ad` | Prepared expression data; `.X` is log1p-normalized to 10,000 counts per cell, and `layers["counts"]` retains raw counts |
| `baseline_predictions.csv` | CellTypist baseline predictions |
| `baseline_decision_matrix.csv` | CellTypist decision scores for the baseline run |
| `iha_expression_prepared_relabelled.h5ad` | Prepared data with an additional partially harmonized label column; the original `cell_type` is preserved |

The raw and prepared AnnData files each contain 54,225 cells × 61,497 genes. Large `.h5ad` files are stored outside GitHub.

## Analysis

### Phase I — Model behavior

We explore prediction patterns using:

- UMAPs colored by original atlas labels, CellTypist predictions, confidence, donor, and age group.
- Cross-tabulations and confusion-style heatmaps comparing atlas labels with model predictions.
- Comparisons across donors and age groups.
- Controlled input changes, such as gene filtering and downsampling.

These comparisons are descriptive. Atlas labels and model classes can differ in granularity, so an unmatched label is not automatically a model error.

### Phase II — What the model learned

We investigate selected Phase I observations using at least two techniques, such as:

- CellTypist model weights or feature contributions.
- Marker-gene inspection.
- LIME or SHAP explanations.
- Nearest-neighbor comparisons.

We compare or combine the findings from the selected techniques.

## Label handling

The original atlas annotation remains in `cell_type`. CellTypist outputs remain in `predicted_labels`. Any harmonized labels are stored separately, for example in `new_dataset_labels`.

Mappings are documented and treated as analysis choices. Types without a clear mapping remain identifiable under their original labels.

## Notebooks

- `cell_IDs_exploration.ipynb` — selects cells and creates `iha_sample_ids.csv`.
- `download_sampled_cells.ipynb` — retrieves selected cells from CELLxGENE Census and prepares expression data.
- `baseline_predictions.ipynb` — runs baseline CellTypist predictions and explores label correspondences and prediction patterns.
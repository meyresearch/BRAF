# BRAF Structural Workflow

Exploiting the wealth of experimental structural data on kinases to determine conformational changes associated with activation loop conformations.

This repository implements an end-to-end modelling pipeline: acquire and curate kinase structures related to BRAF, analyse activation-loop geometry with dimensionality reduction, define conserved-residue distance features, and train Random Forest classifiers that link structural features to conformational states.

The notebooks are organized by **pipeline progression** (`01` → `11`). If you are looking for a specific stage (acquisition, curation, PCA, feature selection, classification), use the **Directory Table** below.

---

## Pipeline overview

![Pipeline schematic](images/fullPipelineSchematic.png)

---

## Directory Table

Use this table to find the notebook that matches the stage of the workflow you want to run or inspect.

**Status legend:**

- **Main** — primary path through the pipeline
- **Experiment** — variant, benchmark, or follow-on analysis
- **Legacy** — older notebook kept for reference; prefer the Main equivalent

| Stage | Notebook | Purpose | Status |
| :--- | :--- | :--- | :--- |
| **Data acquisition** | [`01-DataAcquisition.ipynb`](./01-DataAcquisition.ipynb) | BLASTP against PDB from BRAF reference (`6UAN`); download hit structures | Main |
| **Data curation** | [`02-DataCuration.ipynb`](./02-DataCuration.ipynb) | Initial curation of the kinase dataset | Main |
| **Data curation** | [`03-DataCuration.ipynb`](./03-DataCuration.ipynb) | Chain extraction with DFG/APE motif constraints | Main |
| **Data curation** | [`03b-DataCuration.ipynb`](./03b-DataCuration.ipynb) | Extended curation / filtering of extracted chains | Experiment |
| **Data curation** | [`03c-DataCuration.ipynb`](./03c-DataCuration.ipynb) | Activation-loop filter benchmark | Experiment |
| **Dimensionality reduction** | [`04a-DimensionalityReduction.ipynb`](./04a-DimensionalityReduction.ipynb) | Motif-based alignment of activation loops | Experiment |
| **Dimensionality reduction** | [`04b-DimensionalityReduction.ipynb`](./04b-DimensionalityReduction.ipynb) | FoldMason conservation and multi-N anchored alignment | Main |
| **Dimensionality reduction** | [`05a-DimensionalityReduction.ipynb`](./05a-DimensionalityReduction.ipynb) | Coarse-graining / path sampling of activation loops | Main |
| **Dimensionality reduction** | [`05b-DimensionalityReduction.ipynb`](./05b-DimensionalityReduction.ipynb) | Coarse-graining variant (incl. reconstruction options) | Experiment |
| **Dimensionality reduction** | [`05c-DimensionalityReduction.ipynb`](./05c-DimensionalityReduction.ipynb) | Coarse-graining / low-dimensional representation variant | Experiment |
| **Dimensionality reduction** | [`06-DimensionalityReduction.ipynb`](./06-DimensionalityReduction.ipynb) | KinCore / Dunbrack labels, ligand-type analysis | Main |
| **Dimensionality reduction** | [`07-DimensionalityReduction.ipynb`](./07-DimensionalityReduction.ipynb) | PCA / clustering and conformation labelling | Main |
| **Feature selection** | [`08a-FeatureSelection.ipynb`](./08a-FeatureSelection.ipynb) | Structural conservation setup for feature definition | Main |
| **Feature selection** | [`08b-FeatureSelection.ipynb`](./08b-FeatureSelection.ipynb) | Multi-MSA structural alignment experiment (FoldMason vs MUSTANG) | Experiment |
| **Feature selection** | [`08c-FeatureSelection.ipynb`](./08c-FeatureSelection.ipynb) | Pairwise structural alignment experiment | Experiment |
| **Feature selection** | [`09-FeatureSelection.ipynb`](./09-FeatureSelection.ipynb) | Conserved-residue distance feature matrix (side-chain / Cα) | Main |
| **Feature selection** | [`10-FeatureSelection.ipynb`](./10-FeatureSelection.ipynb) | Outlier, correlation, and ANOVA feature filtering | Main |
| **Feature classification** | [`11a-FeatureClassification.ipynb`](./11a-FeatureClassification.ipynb) | Random Forest on selected features; importances, SHAP, W/KL | Main |
| **Feature classification** | [`11b-FeatureClassification.ipynb`](./11b-FeatureClassification.ipynb) | Data-leakage investigation (Cα + hierarchical tree) | Experiment |
| **Feature classification** | [`11c-FeatureClassification.ipynb`](./11c-FeatureClassification.ipynb) | Feature-selection experiments (ANOVA vs mutual information) | Experiment |
| **Feature classification** | [`11d-FeatureClassification.ipynb`](./11d-FeatureClassification.ipynb) | KinCore-label classifier experiments and comparisons | Experiment |
| **Feature classification** | [`FeatureClassification.ipynb`](./FeatureClassification.ipynb) | Older classification notebook | Legacy |

---

## Repository layout

| Path | Role |
| :--- | :--- |
| [`workflow/`](./workflow/) | Python backends imported by the notebooks (alignment, curation, PCA, feature selection/classification, utilities) |
| [`images/`](./images/) | Figures used in the README and notebooks |
| [`6UAN_chainD.pdb`](./6UAN_chainD.pdb) | BRAF reference structure |
| `Results/`, `PDBs/` | Large local intermediates (**gitignored**; generated when notebooks are run) |
| `*.csv`, `*.pkl` | Feature matrices and reference pickles (**gitignored**; produced by notebooks 09–10) |

---

## How to run

1. Clone the `devel` branch of this repository.
2. Create / activate a conda (or similar) environment with the scientific Python stack used by the notebooks (see **Dependencies**).
3. Run the **Main** notebooks in order for a full pipeline pass:

   `01` → `02` → `03` → `04b` → `05a` → `06` → `07` → `08a` → `09` → `10` → `11a`

4. Notebooks import helpers from [`workflow/`](./workflow/). Keep the repository root as the working directory so those imports resolve.
5. Heavy outputs stay on disk under `Results/` and as local feature exports; they are not tracked in git.

Experiment notebooks (`03b`, `03c`, `05b`/`05c`, `08b`/`08c`, `11b`–`11d`) can be run independently once their upstream artefacts exist.

---

## Dependencies

There is no pinned `environment.yml` in this repository yet. The notebooks and [`workflow/`](./workflow/) package typically require a modern scientific Python stack, including:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `mdtraj`
- `scikit-learn`
- `jupyter` / JupyterLab

Additional tools used in alignment and labelling steps (e.g. FoldMason, MUSTANG, KinCore assignment inputs) are invoked from the relevant notebooks; see those notebooks for stage-specific requirements.

# KinLoopMap

Mapping kinase activation-loop conformational landscapes from experimental structures.

This repository implements an end-to-end modelling pipeline: acquire and curate kinase structures related to a BRAF reference, analyse activation-loop geometry with dimensionality reduction, define conserved-residue distance features, and train Random Forest classifiers that link structural features to conformational states.

The notebooks are organized by **pipeline progression** across global sections **1–6**. If you are looking for a specific stage, use the **Directory Table** below.

---

## Pipeline overview

![Pipeline schematic](images/fullPipelineSchematic.png)

---

## Directory Table

Use this table to find the notebook that matches the stage of the workflow you want to run or inspect.

**Status legend:**

* 🟢 **Main pipeline** (primary end-to-end path)
* 🟡 **Experiment / variant** (benchmark or optional path)
* 🔴 **Legacy** (kept for reference; prefer the Main-pipeline equivalent)

| Section | Notebook | Purpose | Status |
| :--- | :--- | :--- | :--- |
| **1. Data acquisition** | [`01-DataAcquisition.ipynb`](./01-DataAcquisition.ipynb) | BLASTP / PDB download from BRAF reference (`6UAN`) | 🟢 |
| **2. Data curation** | [`02-DataCuration.ipynb`](./02-DataCuration.ipynb) | Extract protein chains and small molecules; KLIFS dirs | 🟢 |
| **2. Data curation** | [`03-DataCuration.ipynb`](./03-DataCuration.ipynb) | Activation-loop filters (fixed bounds) | 🟢 |
| **2. Data curation** | [`03b-DataCuration.ipynb`](./03b-DataCuration.ipynb) | Tukey loop-length filters and k-factor scan | 🟡 |
| **2. Data curation** | [`03c-DataCuration.ipynb`](./03c-DataCuration.ipynb) | Activation-loop filter benchmark (motif / MUSCLE / KLIFS+HMMER) | 🟡 |
| **3. Dimensionality reduction** | [`04a-DimensionalityReduction.ipynb`](./04a-DimensionalityReduction.ipynb) | Motif-based activation-loop alignment | 🟡 |
| **3. Dimensionality reduction** | [`04b-DimensionalityReduction.ipynb`](./04b-DimensionalityReduction.ipynb) | Multi-N anchored alignment (FoldMason conservation) | 🟢 |
| **3. Dimensionality reduction** | [`05a-DimensionalityReduction.ipynb`](./05a-DimensionalityReduction.ipynb) | Coarse-graining activation loops | 🟢 |
| **3. Dimensionality reduction** | [`05b-DimensionalityReduction.ipynb`](./05b-DimensionalityReduction.ipynb) | Coarse-graining with MODELLER comparison | 🟡 |
| **3. Dimensionality reduction** | [`05c-DimensionalityReduction.ipynb`](./05c-DimensionalityReduction.ipynb) | Coarse-graining variant | 🟡 |
| **3. Dimensionality reduction** | [`06-DimensionalityReduction.ipynb`](./06-DimensionalityReduction.ipynb) | KinCore labels and ligand-type analysis | 🟢 |
| **3. Dimensionality reduction** | [`07-DimensionalityReduction.ipynb`](./07-DimensionalityReduction.ipynb) | PCA clustering vs KinCore | 🟢 |
| **4. Feature definition** | [`09-FeatureSelection.ipynb`](./09-FeatureSelection.ipynb) | Conserved-residue distance feature matrix (side-chain / Cα) | 🟢 |
| **5. Feature selection** | [`08a-FeatureSelection.ipynb`](./08a-FeatureSelection.ipynb) | Structural conservation for feature selection | 🟢 |
| **5. Feature selection** | [`08b-FeatureSelection.ipynb`](./08b-FeatureSelection.ipynb) | Multi-MSA alignment experiment (FoldMason vs MUSTANG) | 🟡 |
| **5. Feature selection** | [`08c-FeatureSelection.ipynb`](./08c-FeatureSelection.ipynb) | Pairwise alignment experiment | 🟡 |
| **5. Feature selection** | [`10-FeatureSelection.ipynb`](./10-FeatureSelection.ipynb) | Outlier, correlation, and ANOVA filtering | 🟢 |
| **6. Feature classification** | [`11a-FeatureClassification.ipynb`](./11a-FeatureClassification.ipynb) | RF importances, SHAP, and W/KL analysis | 🟢 |
| **6. Feature classification** | [`11b-FeatureClassification.ipynb`](./11b-FeatureClassification.ipynb) | Data-leakage investigation (Cα + hierarchical tree) | 🟡 |
| **6. Feature classification** | [`11c-FeatureClassification.ipynb`](./11c-FeatureClassification.ipynb) | ANOVA vs mutual-information experiments | 🟡 |
| **6. Feature classification** | [`11d-FeatureClassification.ipynb`](./11d-FeatureClassification.ipynb) | KinCore classifier experiments | 🟡 |
| **6. Feature classification** | [`FeatureClassification.ipynb`](./FeatureClassification.ipynb) | Older classification notebook | 🔴 |

---

## Repository layout

| Path | Role |
| :--- | :--- |
| [`workflow/`](./workflow/) | Python backends imported by the notebooks (alignment, curation, PCA, feature selection/classification, utilities) |
| [`images/`](./images/) | Figures used in the README and notebooks |
| [`6UAN_chainD.pdb`](./6UAN_chainD.pdb) | BRAF reference structure |
| `Results/`, `PDBs/` | Large local intermediates (**gitignored**; generated when notebooks are run) |
| `*.csv`, `*.pkl` | Feature matrices and reference pickles (**gitignored**; produced by notebooks 09–10) |

Each notebook starts with a **table of contents** mirroring its section headings and a **mermaid backend map** of the `workflow/` modules it uses.

---

## How to run

1. Clone the `devel` branch of this repository.
2. Create / activate a conda (or similar) environment with the scientific Python stack used by the notebooks (see **Dependencies**).
3. Run the 🟢 **Main pipeline** notebooks in order:

   `01` → `02` → `03` → `04b` → `05a` → `06` → `07` → `08a` → `09` → `10` → `11a`

4. Notebooks import helpers from [`workflow/`](./workflow/). Keep the repository root as the working directory so those imports resolve.
5. Heavy outputs stay on disk under `Results/` and as local feature exports; they are not tracked in git.

🟡 Experiment notebooks (`03b`, `03c`, `04a`, `05b`/`05c`, `08b`/`08c`, `11b`–`11d`) can be run once their upstream artefacts exist.

---

## Dependencies

There is no pinned `environment.yml` in this repository yet. The notebooks and [`workflow/`](./workflow/) package typically require a modern scientific Python stack, including:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `mdtraj`
- `scikit-learn`
- `jupyter` / JupyterLab

Additional tools used in alignment and labelling steps (e.g. FoldMason, MUSTANG, KinCore assignment inputs) are invoked from the relevant notebooks; see those notebooks for stage-specific requirements.

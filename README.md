# KinLoopMap

Mapping kinase activation-loop conformational landscapes from experimental structures.

This repository implements an end-to-end modelling pipeline: acquire and curate kinase structures related to a BRAF reference, analyse activation-loop geometry with dimensionality reduction, define conserved-residue distance features, and train Random Forest classifiers that link structural features to conformational states.

The notebooks are organized by **pipeline progression** across global sections **1–3** and **5–6**. If you are looking for a specific stage, use the **Directory Table** below.

---

## Pipeline overview

Counts below are from this `workflowAugust2026` run (notebook outputs and `Results/` artefacts). Nodes use Directory Table notebook names and are colored by **Pipeline** role (green = Main, amber = Experiment). Subgraphs use Directory Table section titles. Feature filtering on the edge into `11a` is correlation then MI top-N only.

```mermaid
flowchart TB
  subgraph s1 ["1. Data acquisition"]
    N01["01-DataAcquisition"]
  end
  subgraph s2 ["2. Data curation"]
    N02["02-ChainsAndLigands"]
    N03["03-ActivationLoopFilters"]
  end
  subgraph s3 ["3. Dimensionality reduction"]
    N04a["04a-MotifAlignment"]
    N05a["05a-CoarseGraining"]
    N06["06-KinCoreLabelsAndLigands"]
    N07["07-PCAClusteringVsKinCore"]
    N07b["07b-AutoencoderBenchmark"]
    N04c["04c-CNN2dLatentLandscape"]
  end
  subgraph s5 ["5. Feature selection"]
    direction TB
    N08a["08a-StructuralConservation"]
    N09["09-FeatureMatrix"]
    N10["10-FeatureFiltering"]
  end
  subgraph s6 ["6. Feature classification"]
    N11a["11a-RFImportancesAndWKL"]
  end

  N01 -->|"9007 hits → 8727 structures"| N02
  N02 -->|"12713 chains"| N03
  N03 -->|"6150 → 3960 → 3833"| N04a
  N04a -->|"3831"| N05a
  N05a -->|"3831"| N06
  N05a -->|"3831"| N07
  N05a -->|"fitted CG"| N07b
  N05a -->|"fitted CG"| N04c
  N06 --> N07
  N04a -->|"3831"| N08a
  N08a -->|"165 residues"| N09
  N07 -->|"3831 labels"| N09
  N02 --> N09
  N09 -->|"3831 x 10011"| N10
  N10 -->|"8777 → 300 features"| N11a
  N07 --> N11a
  N06 --> N11a
  N11a --> outNode["Structural changes linked to activation-loop conformations"]

  classDef main fill:#d4edda,stroke:#28a745,color:#000
  classDef experiment fill:#fff3cd,stroke:#ffc107,color:#000
  class N01,N02,N03,N04a,N05a,N06,N07,N08a,N09,N10,N11a main
  class N07b,N04c experiment
```

---

## Directory Table

Use this table to find the notebook that matches the stage of the workflow you want to run or inspect.

**Pipeline** (role in the workflow):

* **Main pipeline**
* **Experiment / variant**
* **Legacy**

**Status** (readiness):

* 🟢 **Full Tutorial** (Running without bugs and self-explanatory)
* 🟠 **Working code** (Running without bugs)
* 🔴 **To debug**

| Section | Notebook | Purpose | Pipeline | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Data acquisition** | [`01-DataAcquisition.ipynb`](./01-DataAcquisition.ipynb) | BLASTP / PDB download from BRAF reference (`6UAN`) | Main pipeline | 🟢 |
| **2. Data curation** | [`02-ChainsAndLigands.ipynb`](./02-ChainsAndLigands.ipynb) | Extract protein chains and small molecules; KLIFS dirs | Main pipeline | 🟢 |
| **2. Data curation** | [`03-ActivationLoopFilters.ipynb`](./03-ActivationLoopFilters.ipynb) | Activation-loop filters (fixed bounds) | Main pipeline | 🟢 |
| **2. Data curation** | [`03b-TukeyLoopLengthFilters.ipynb`](./03b-TukeyLoopLengthFilters.ipynb) | Tukey loop-length filters and k-factor scan | Experiment / variant | 🟠 |
| **2. Data curation** | [`03c-LoopFilterBenchmark.ipynb`](./03c-LoopFilterBenchmark.ipynb) | Activation-loop filter benchmark (motif / MUSCLE / KLIFS+HMMER) | Experiment / variant | 🟠 |
| **3. Dimensionality reduction** | [`04a-MotifAlignment.ipynb`](./04a-MotifAlignment.ipynb) | Motif-based activation-loop alignment | Main pipeline | 🟠 |
| **3. Dimensionality reduction** | [`04b-MultiNAnchoredAlignment.ipynb`](./04b-MultiNAnchoredAlignment.ipynb) | Multi-N anchored alignment (FoldMason conservation) | Experiment / variant | 🔴 |
| **3. Dimensionality reduction** | [`04c-CNN2dLatentLandscape.ipynb`](./04c-CNN2dLatentLandscape.ipynb) | CNN2d AE with a 6D latent space; 15 pairwise latent RMSD landscapes | Experiment / variant | 🟠 |
| **3. Dimensionality reduction** | [`05a-CoarseGraining.ipynb`](./05a-CoarseGraining.ipynb) | Coarse-graining activation loops | Main pipeline | 🟢 |
| **3. Dimensionality reduction** | [`05b-CoarseGrainingModeller.ipynb`](./05b-CoarseGrainingModeller.ipynb) | Coarse-graining with MODELLER comparison | Experiment / variant | 🔴 |
| **3. Dimensionality reduction** | [`05c-CoarseGrainingVariant.ipynb`](./05c-CoarseGrainingVariant.ipynb) | Coarse-graining variant | Experiment / variant | 🟠 |
| **3. Dimensionality reduction** | [`06-KinCoreLabelsAndLigands.ipynb`](./06-KinCoreLabelsAndLigands.ipynb) | KinCore labels and ligand-type analysis | Main pipeline | 🟢 |
| **3. Dimensionality reduction** | [`07-PCAClusteringVsKinCore.ipynb`](./07-PCAClusteringVsKinCore.ipynb) | PCA clustering vs KinCore | Main pipeline | 🟢 |
| **3. Dimensionality reduction** | [`07b-AutoencoderBenchmark.ipynb`](./07b-AutoencoderBenchmark.ipynb) | CNN2d / Small / wr2DCNN AE vs PCA on fitted CG loops | Experiment / variant | 🟠 |
| **5. Feature selection** | [`08a-StructuralConservation.ipynb`](./08a-StructuralConservation.ipynb) | Structural conservation for feature selection | Main pipeline | 🟢 |
| **5. Feature selection** | [`09-FeatureMatrix.ipynb`](./09-FeatureMatrix.ipynb) | Conserved-residue distance feature matrix (side-chain / Cα) | Main pipeline | 🟢 |
| **5. Feature selection** | [`10-FeatureFiltering.ipynb`](./10-FeatureFiltering.ipynb) | Outlier, correlation, and ANOVA filtering | Main pipeline | 🟢 |
| **5. Feature selection** | [`08b-MultiMSAAlignmentExperiment.ipynb`](./08b-MultiMSAAlignmentExperiment.ipynb) | Multi-MSA alignment experiment (FoldMason vs MUSTANG) | Experiment / variant | 🟠 |
| **5. Feature selection** | [`08c-PairwiseAlignmentExperiment.ipynb`](./08c-PairwiseAlignmentExperiment.ipynb) | Pairwise alignment experiment | Experiment / variant | 🟠 |
| **6. Feature classification** | [`11a-RFImportancesAndWKL.ipynb`](./11a-RFImportancesAndWKL.ipynb) | RF importances, SHAP, and W/KL analysis | Main pipeline | 🟢 |
| **6. Feature classification** | [`11b-DataLeakageInvestigation.ipynb`](./11b-DataLeakageInvestigation.ipynb) | Data-leakage investigation (Cα + hierarchical tree) | Experiment / variant | 🟠 |
| **6. Feature classification** | [`11c-ANOVAvsMI.ipynb`](./11c-ANOVAvsMI.ipynb) | ANOVA vs mutual-information experiments | Experiment / variant | 🟠 |
| **6. Feature classification** | [`11d-KinCoreClassifierExperiments.ipynb`](./11d-KinCoreClassifierExperiments.ipynb) | KinCore classifier experiments | Experiment / variant | 🟠 |
| **6. Feature classification** | [`FeatureClassification.ipynb`](./FeatureClassification.ipynb) | Older classification notebook | Legacy | 🔴 |

---

## Repository layout

| Path | Role |
| :--- | :--- |
| [`workflow/`](./workflow/) | Python backends imported by the notebooks (alignment, curation, PCA, feature selection/classification, utilities) |
| [`images/`](./images/) | Figures used in the README and notebooks |
| [`6UAN_chainD.pdb`](./6UAN_chainD.pdb) | BRAF reference structure |
| `Results/`, `PDBs/` | Large local intermediates (**gitignored**; generated when notebooks are run) |
| `*.csv`, `*.pkl` | Feature matrices and reference pickles (**gitignored**; produced by notebooks 09–10) |

Each notebook starts with a **table of contents** mirroring its section headings and a **backend map** of the `workflow/` modules it uses, including transitive `workflow` subdependencies, with arrows pointing into the notebook (pre-rendered SVG under `images/backend_maps/` (`*.v2.svg`), because GitHub does not render Mermaid fences inside `.ipynb` files).

---

## How to run

1. Clone the `devel` branch of this repository.
2. Create / activate a conda (or similar) environment with the scientific Python stack used by the notebooks (see **Dependencies**).
3. Run the **Main pipeline** notebooks in the order of the **Pipeline overview** / Directory Table dataflow (`09` after conservation `08a`, then `10`):

   `01` → `02` → `03` → `04a` → `05a` → `06` → `07` → `08a` → `09` → `10` → `11a`

   - Check the Directory Table **Status** column before relying on a step (`05b` is currently **To debug**).

4. Notebooks import helpers from [`workflow/`](./workflow/). Keep the repository root as the working directory so those imports resolve.
5. Heavy outputs stay on disk under `Results/` and as local feature exports; they are not tracked in git.

Other **Experiment / variant** notebooks (`03b`, `03c`, `04b`, `04c`, `05b`/`05c`, `07b`, `08b`/`08c`, `11b`–`11d`) can be run once their upstream artefacts exist. Skip **Legacy** `FeatureClassification` unless you need the older path.

---

## Dependencies

There is no pinned `environment.yml` in this repository yet. The notebooks and [`workflow/`](./workflow/) package typically require a modern scientific Python stack, including:

- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `mdtraj`
- `scikit-learn`
- `jupyter` / JupyterLab

Additional tools used in alignment and labelling steps (e.g. FoldMason, MUSTANG, KinCore assignment inputs) are invoked from the relevant notebooks; see those notebooks for stage-specific requirements.

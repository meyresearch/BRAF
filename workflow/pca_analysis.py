"""
PCA Analysis Module for Protein Structures

This module provides classes for performing PCA analysis, clustering,
and visualization on protein structure data from PDB files.
"""

import os
import shutil
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.linalg import norm
import seaborn as sns
# Import utility functions from utilities module
import utilities


class PCAAnalyzer:
    """Class for performing PCA analysis on protein structures."""
    
    def __init__(self, n_components: int = 4):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of principal components to compute
        """
        self.n_components = n_components
        self.pca_model = None
        self.projections = None
        self.file_list = None
        
    def fit_transform(self, file_list: List[str]) -> Tuple[PCA, np.ndarray]:
        """
        Perform PCA on a list of PDB files.
        
        Args:
            file_list: List of paths to PDB files
            
        Returns:
            Tuple of (pca_model, projections)
            
        Raises:
            ValueError: If no files provided
        """
        if len(file_list) == 0:
            raise ValueError("No files provided for PCA (check filtering).")
        
        print("Number of PDB files to load:", len(file_list))
        for f in file_list:
            print("  ", os.path.basename(f))
        
        # Load all frames from the list of files
        traj = md.join([md.load(f) for f in file_list])
        
        # Reshape to (#frames, 3 * #atoms)
        xyz = traj.xyz.reshape(-1, 3 * traj.n_atoms)
        
        # Perform PCA
        self.pca_model = PCA(n_components=self.n_components)
        self.projections = self.pca_model.fit_transform(xyz)
        self.file_list = file_list
        
        return self.pca_model, self.projections
    
    def get_explained_variance_ratios(self) -> np.ndarray:
        """
        Get explained variance ratios as percentages.
        
        Returns:
            Array of explained variance ratios (as percentages)
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted yet. Call fit_transform first.")
        return self.pca_model.explained_variance_ratio_ * 100
    
    def print_explained_variance(self, label: str = ""):
        """
        Print explained variance ratios for each component.
        
        Args:
            label: Optional label to include in print statements
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted yet. Call fit_transform first.")
        
        explained_var_ratios = self.get_explained_variance_ratios()
        for i, ratio in enumerate(explained_var_ratios, 1):
            print(f"PC{i} {label} explains {ratio:.2f}% of variance.")
    
    def plot_scores(self, folder_for_residues: str, figure_title: str = "PCA scores",
                   n_comps: Optional[int] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot the PCA 'scores' (i.e., loadings per residue) for the components.
        
        Args:
            folder_for_residues: Folder to get residue names from
            figure_title: Title for the figure
            n_comps: Number of components to plot (default: all)
            
        Returns:
            Tuple of (figure, axes)
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted yet. Call fit_transform first.")
        
        if n_comps is None:
            n_comps = self.n_components
        
        # Reshape the PCA components to [n_atoms, 3], repeated for each principal component.
        # Adjust 27 if needed for your system.
        scores = self.pca_model.components_.reshape(-1, 27, 3)[:n_comps]
        
        x_labels = utilities.braf_res(folder_for_residues)
        x_vals = list(range(len(scores[0])))
        col = ["red", "blue", "green", "yellow"]
        exp_var = self.pca_model.explained_variance_
        
        fig, axes = plt.subplots(n_comps, sharex=True, figsize=(8, 8))
        if n_comps == 1:
            axes = [axes]  # Ensure axes is always iterable if only 1 PC.
        
        for i in range(n_comps):
            # Calculate norm of loadings * explained variance
            load_val = norm(scores[i] * exp_var[i], axis=1)
            axes[i].plot(x_vals, load_val, marker=".", c=col[i % len(col)])
            axes[i].set_ylabel(f"PC{i+1} load")
        
        axes[-1].set_xticks(range(len(x_labels)))
        axes[-1].set_xticklabels(x_labels, rotation=90)
        axes[0].set_title(figure_title)
        plt.tight_layout()
        
        return fig, axes
    
    def save_pc_values(self, pc_index: int, structure_names: List[str], 
                      output_file: str) -> None:
        """
        Save principal component values with filenames to a text file.
        
        Args:
            pc_index: Index of PC to save (0-based)
            structure_names: List of structure names
            output_file: Output file path
        """
        if self.projections is None:
            raise ValueError("No projections available. Call fit_transform first.")
        
        print(f"Saving PC{pc_index+1} + filenames to {output_file}")
        pc_values = self.projections[:, pc_index]
        pc_names = np.array(structure_names)
        pc_data = np.column_stack((pc_values, pc_names))
        np.savetxt(output_file, pc_data, fmt="%s", header=f"PC{pc_index+1}_value,FileName")


class ClusterAnalyzer:
    """Class for performing clustering analysis on PCA projections."""
    
    def __init__(self, n_clusters: int = 2):
        """
        Initialize cluster analyzer.
        
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.cluster_labels = None
        
    def hierarchical_cluster(self, projections: np.ndarray, 
                            linkage: str = 'ward') -> np.ndarray:
        """
        Perform hierarchical clustering on the first two PCs.
        
        Args:
            projections: PCA projections array
            linkage: Linkage method for hierarchical clustering
            
        Returns:
            Array of cluster labels
        """
        hier_clust = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=linkage)
        print('Hierarchical clustering on shape:', np.shape(projections[:, :2]))
        self.cluster_labels = hier_clust.fit_predict(projections[:, :2])
        return self.cluster_labels
    
    def spectral_cluster(self, projections: np.ndarray, affinity: str = 'rbf',
                        gamma: float = 1.0, random_state: int = 42) -> np.ndarray:
        """
        Perform spectral clustering on the first two PCs.
        
        Args:
            projections: PCA projections array
            affinity: Affinity kernel for spectral clustering
            gamma: Kernel coefficient for rbf kernel
            random_state: Random state for reproducibility
            
        Returns:
            Array of cluster labels
        """
        spec_clust = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=affinity,
            gamma=gamma,
            random_state=random_state
        )
        print('Spectral clustering shape:', np.shape(projections[:, :2]))
        self.cluster_labels = spec_clust.fit_predict(projections[:, :2])
        return self.cluster_labels
    
    def plot_clusters(self, projections: np.ndarray, cluster_labels: np.ndarray,
                     highlight_indices: Optional[List[int]] = None,
                     highlight_color: str = 'red',
                     main_title: str = "Clustering in PC1–PC2",
                     explained_var_ratios: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot clustering results in PC1-PC2 space.
        
        Args:
            projections: PCA projections array
            cluster_labels: Array of cluster labels
            highlight_indices: Indices of points to highlight (e.g., outliers)
            highlight_color: Color for highlighting points
            main_title: Title for the plot
            explained_var_ratios: Array of explained variance ratios for axis labels
            
        Returns:
            Tuple of (figure, axes)
        """
        # Prepare discrete color boundaries
        bounds = list(range(self.n_clusters + 1))
        norm_bound = BoundaryNorm(bounds, ncolors=self.n_clusters, clip=True)
        
        # Plot PC1 vs. PC2, colored by cluster label
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=cluster_labels,
            cmap='tab10',
            norm=norm_bound,
            alpha=0.8
        )
        
        # Highlight the outliers by circling them
        if highlight_indices is not None and len(highlight_indices) > 0:
            ax.scatter(
                projections[highlight_indices, 0],
                projections[highlight_indices, 1],
                facecolors='none',
                edgecolors=highlight_color,
                s=100,
                linewidths=1.5,
                label="Excluded in noOutliers"
            )
        
        # Create a discrete colorbar
        tick_positions = [x + 0.5 for x in range(self.n_clusters)]
        cbar = plt.colorbar(scatter, spacing="proportional", ticks=tick_positions)
        cbar.ax.set_yticklabels([f"Cluster {i}" for i in range(self.n_clusters)])
        cbar.set_label("Cluster ID")
        
        # Set axis labels with explained variance if provided
        if explained_var_ratios is not None and len(explained_var_ratios) >= 2:
            ax.set_xlabel(f"PC1 ({explained_var_ratios[0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({explained_var_ratios[1]:.1f}% variance)")
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        
        ax.set_title(main_title)
        if highlight_indices:
            ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def save_cluster_labels(self, cluster_labels: np.ndarray, structure_names: List[str],
                           output_file: str, include_pdb_codes: bool = False) -> None:
        """
        Save cluster labels with structure names to a file.
        
        Args:
            cluster_labels: Array of cluster labels
            structure_names: List of structure names
            output_file: Output file path
            include_pdb_codes: Whether to extract and include 6-char PDB codes
        """
        print(f"Saving cluster labels + filenames to {output_file}")
        
        if include_pdb_codes:
            # Extract just the 6-character PDB codes from filenames
            pdb_codes = [name[:6] for name in structure_names]
            cluster_data = np.column_stack((cluster_labels, pdb_codes, structure_names))
            np.savetxt(output_file, cluster_data, fmt="%s", 
                      header="ClusterLabel,PDBCode,FullName", delimiter=",")
        else:
            cluster_data = np.column_stack((cluster_labels, structure_names))
            np.savetxt(output_file, cluster_data, fmt="%s",
                      header="ClusterLabel,FileName", delimiter=",")
    
    def copy_structures_to_clusters(self, file_list: List[str], cluster_labels: np.ndarray,
                                   output_dirs: List[str]) -> None:
        """
        Copy structures to cluster-specific directories.
        
        Args:
            file_list: List of structure file paths
            cluster_labels: Array of cluster labels
            output_dirs: List of output directories for each cluster
        """
        # Create output directories
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy structures based on clustering results
        for i, label in enumerate(cluster_labels):
            src_file = file_list[i]
            if label < len(output_dirs):
                shutil.copy(src_file, output_dirs[label])
            else:
                print(f"Warning: cluster label {label} exceeds number of output dirs")
    
    def load_kincore_labels(self, structure_names: List[str], 
                           kincore_file: str = 'Results/dunbrack_assignments/kinase_conformation_assignments.csv'
                           ) -> Tuple[np.ndarray, dict]:
        """
        Load activation state labels from KinCore classification file.
        
        Args:
            structure_names: List of structure names to match
            kincore_file: Path to the KinCore classification CSV file
            
        Returns:
            Tuple of (labels array, label_map dict)
        """
        # Load KinCore assignments
        kincore_df = pd.read_csv(kincore_file)
        
        # Create mapping from pdb_code to activation state
        # Active: "Active conformation (Type I)" -> 1
        # Inactive: anything else -> 0
        activation_map = {}
        for _, row in kincore_df.iterrows():
            pdb_code = row['pdb_code']
            description = row['conformation_description']
            if 'Active' in str(description):
                activation_map[pdb_code] = 1  # Active
            else:
                activation_map[pdb_code] = 0  # Inactive
        
        # Map structure names to activation labels
        labels = []
        for name in structure_names:
            # Try to match with or without .pdb extension
            if name in activation_map:
                labels.append(activation_map[name])
            elif name + '.pdb' in activation_map:
                labels.append(activation_map[name + '.pdb'])
            elif name.replace('.pdb', '') in activation_map:
                labels.append(activation_map[name.replace('.pdb', '')])
            else:
                # Default to inactive if not found
                labels.append(0)
                print(f"Warning: {name} not found in KinCore file, defaulting to inactive")
        
        label_map = {0: 'Inactive', 1: 'Active'}
        return np.array(labels), label_map
    
    def plot_activation_states(self, projections: np.ndarray, activation_labels: np.ndarray,
                              main_title: str = "Activation States in PC1–PC2",
                              explained_var_ratios: Optional[np.ndarray] = None,
                              colors: dict = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot PCA projection colored by activation state (active/inactive).
        
        Args:
            projections: PCA projections array
            activation_labels: Array of activation labels (0=inactive, 1=active)
            main_title: Title for the plot
            explained_var_ratios: Array of explained variance ratios for axis labels
            colors: Dict mapping labels to colors (default: {0: 'red', 1: 'green'})
            
        Returns:
            Tuple of (figure, axes)
        """
        if colors is None:
            colors = {0: 'red', 1: 'green'}  # Inactive=red, Active=green
        
        # Create custom colormap: red for inactive (0), green for active (1)
        activation_cmap = ListedColormap([colors[0], colors[1]])
        
        # Prepare discrete color boundaries
        bounds = [0, 1, 2]
        norm_bound = BoundaryNorm(bounds, ncolors=2, clip=True)
        
        # Plot PC1 vs. PC2, colored by activation state
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            c=activation_labels,
            cmap=activation_cmap,
            norm=norm_bound,
            alpha=0.8
        )
        
        # Create a discrete colorbar
        tick_positions = [0.5, 1.5]
        cbar = plt.colorbar(scatter, spacing="proportional", ticks=tick_positions)
        cbar.ax.set_yticklabels(['Inactive', 'Active'])
        cbar.set_label("Activation State")
        
        # Set axis labels with explained variance if provided
        if explained_var_ratios is not None and len(explained_var_ratios) >= 2:
            ax.set_xlabel(f"PC1 ({explained_var_ratios[0]:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({explained_var_ratios[1]:.1f}% variance)")
        else:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        
        ax.set_title(main_title)
        plt.tight_layout()
        return fig, ax

    def plot_pca_cluster_and_activation(
        self,
        results: dict,
        *,
        kincore_file: str = "Results/dunbrack_assignments/kinase_conformation_assignments.csv",
        cluster_plot_path: str = "pca_clustering_labels.png",
        activation_plot_path: str = "pca_activation_states.png",
        show: bool = True,
        main_title_clusters: str = "PCA Projection - Clustering Labels",
        main_title_activation: str = "PCA Projection - Activation States (KinCore)",
    ) -> dict:
        """
        Convenience wrapper for notebooks:
        - plots PCA projection colored by clustering labels (hierarchical or spectral)
        - loads KinCore activation labels and plots PCA projection colored by activation state
        - saves both figures to disk and optionally displays them

        Expects a dict like the output from PCAWorkflow/run_full_analysis with keys:
            - 'projections' (np.ndarray)
            - 'structure_names' (list[str])
            - 'explained_variance' (np.ndarray)
            - 'hierarchical_labels' and/or 'spectral_labels' (np.ndarray)
        """
        projections = results["projections"]
        structure_names = results["structure_names"]
        explained_variance = results.get("explained_variance")

        cluster_labels = results.get("hierarchical_labels", results.get("spectral_labels"))
        if cluster_labels is None:
            raise ValueError("No cluster labels found in results (expected 'hierarchical_labels' or 'spectral_labels').")

        # Plot 1: clustering labels
        fig1, _ = self.plot_clusters(
            projections,
            cluster_labels,
            main_title=main_title_clusters,
            explained_var_ratios=explained_variance,
        )
        fig1.savefig(cluster_plot_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig1)

        # Plot 2: activation state labels (KinCore)
        activation_labels, label_map = self.load_kincore_labels(
            structure_names,
            kincore_file=kincore_file,
        )
        fig2, _ = self.plot_activation_states(
            projections,
            activation_labels,
            main_title=main_title_activation,
            explained_var_ratios=explained_variance,
        )
        fig2.savefig(activation_plot_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig2)

        # Summary
        active_n = int(np.sum(activation_labels == 1))
        inactive_n = int(np.sum(activation_labels == 0))
        print("\nActivation state distribution:")
        print(f"  Active: {active_n}")
        print(f"  Inactive: {inactive_n}")

        return {
            "cluster_plot_path": cluster_plot_path,
            "activation_plot_path": activation_plot_path,
            "activation_labels": activation_labels,
            "label_map": label_map,
            "active_n": active_n,
            "inactive_n": inactive_n,
        }

    def integrate_dunbrack_with_pca_clusters(
        self,
        *,
        pca_labels_file: str = "cluster_labels_my_analysis_hierarchical.txt",
        dunbrack_assignments_csv: str = "Results/dunbrack_assignments/kinase_conformation_assignments.csv",
        prefix_len: int = 6,
        merged_output_csv: str = "Results/dunbrack_assignments/pca_dunbrack_merged.csv",
        print_tables: bool = True,
        print_percentages: bool = True,
    ) -> dict:
        """
        Convenience wrapper for notebooks: merge PCA cluster labels with Dunbrack/KinCore
        conformational assignments, print cross-tabulations, and save the merged table.

        This reproduces the notebook block that:
        - loads PCA cluster labels text file
        - loads KinCore assignment CSV
        - matches structures by a prefix (default first 6 chars)
        - prints cross-tabs for overall/DFG/C-helix conformations
        - prints within-cluster percentages for overall conformations
        - saves the merged CSV used by downstream correlation plots
        """
        # Load PCA cluster labels
        pca_labels = pd.read_csv(
            pca_labels_file,
            skiprows=1,
            header=None,
            names=["ClusterLabel", "PDBCode", "FullName"],
        )

        # Load Dunbrack conformations
        dunbrack_results = pd.read_csv(dunbrack_assignments_csv)

        # Extract prefix for matching
        pca_labels["pdb_prefix"] = pca_labels["FullName"].astype(str).str[:prefix_len]
        dunbrack_results["pdb_prefix"] = dunbrack_results["pdb_code"].astype(str).str[:prefix_len]

        # Merge datasets
        merged = pd.merge(pca_labels, dunbrack_results, on="pdb_prefix", how="inner")

        if print_tables or print_percentages:
            print(f"Successfully matched {len(merged)} structures\n")

        # Cross-tabulations
        crosstab_overall = pd.crosstab(
            merged["ClusterLabel"],
            merged["overall_conformation"],
            margins=True,
            margins_name="Total",
        ) if "overall_conformation" in merged.columns else pd.DataFrame()

        crosstab_dfg = pd.crosstab(
            merged["ClusterLabel"],
            merged["dfg_conformation"],
            margins=True,
            margins_name="Total",
        ) if "dfg_conformation" in merged.columns else pd.DataFrame()

        crosstab_chelix = pd.crosstab(
            merged["ClusterLabel"],
            merged["chelix_conformation"],
            margins=True,
            margins_name="Total",
        ) if "chelix_conformation" in merged.columns else pd.DataFrame()

        if print_tables:
            print("=" * 80)
            print("PCA CLUSTER vs DUNBRACK CONFORMATION CROSS-TABULATION")
            print("=" * 80)
            print(crosstab_overall)

            print("\n" + "=" * 80)
            print("PCA CLUSTER vs DFG CONFORMATION")
            print("=" * 80)
            print(crosstab_dfg)

            print("\n" + "=" * 80)
            print("PCA CLUSTER vs C-HELIX CONFORMATION")
            print("=" * 80)
            print(crosstab_chelix)

        if print_percentages and "overall_conformation" in merged.columns and "ClusterLabel" in merged.columns:
            print("\n" + "=" * 80)
            print("PERCENTAGE OF CONFORMATIONS WITHIN EACH PCA CLUSTER")
            print("=" * 80)

            for cluster in sorted(merged["ClusterLabel"].unique()):
                cluster_data = merged[merged["ClusterLabel"] == cluster]
                total = len(cluster_data)
                print(f"\nCluster {cluster} (n={total}):")

                conf_dist = cluster_data["overall_conformation"].value_counts()
                for conf, count in conf_dist.items():
                    percentage = (count / total) * 100 if total else 0.0
                    print(f"  {conf:.<40} {count:>4} ({percentage:>5.1f}%)")

        # Save merged results
        os.makedirs(os.path.dirname(merged_output_csv), exist_ok=True)
        merged.to_csv(merged_output_csv, index=False)
        if print_tables or print_percentages:
            print(f"\n✅ Merged results saved to: {merged_output_csv}")

        return {
            "merged": merged,
            "crosstab_overall": crosstab_overall,
            "crosstab_dfg": crosstab_dfg,
            "crosstab_chelix": crosstab_chelix,
            "merged_output_csv": merged_output_csv,
            "n_matched": int(len(merged)),
        }

    def analyze_cluster_vs_activity_status(
        self,
        *,
        merged_csv: str = "Results/dunbrack_assignments/pca_dunbrack_merged.csv",
        activity_column: str = "conformation_description",
        cluster_column: str = "ClusterLabel",
        output_column: str = "activity_status",
        keep: tuple = ("Active", "Inactive"),
        print_tables: bool = True,
        print_percentages: bool = True,
        print_enrichment: bool = True,
        enrichment_threshold_pct: float = 10.0,
    ) -> dict:
        """
        Convenience wrapper for notebooks: compare PCA clusters with Active/Inactive state
        derived from a Dunbrack/KinCore conformation description.

        This reproduces the notebook analysis block that:
        - loads the merged PCA/Dunbrack CSV
        - maps conformation_description -> Active/Inactive/Unknown/Other
        - filters to Active/Inactive only
        - prints cross-tabs + within-cluster and within-activity percentages
        - prints a simple enrichment summary per cluster
        """
        merged = pd.read_csv(merged_csv)

        def classify_activity(description):
            if pd.isna(description) or description == "unknown":
                return "Unknown"
            desc = str(description)
            if "Active" in desc:
                return "Active"
            if "inactive" in desc.lower() or "Inactive" in desc:
                return "Inactive"
            return "Other"

        merged[output_column] = merged[activity_column].apply(classify_activity)
        merged_clean = merged[merged[output_column].isin(list(keep))].copy()

        if print_tables or print_percentages or print_enrichment:
            print("=" * 80)
            print("PCA CLUSTER vs ACTIVITY STATUS (ACTIVE/INACTIVE)")
            print("=" * 80)
            print(f"Total structures analyzed: {len(merged_clean)}")
            print(f"  Active: {(merged_clean[output_column] == 'Active').sum()}")
            print(f"  Inactive: {(merged_clean[output_column] == 'Inactive').sum()}")
            print()

        activity_crosstab = pd.crosstab(
            merged_clean[cluster_column],
            merged_clean[output_column],
            margins=True,
            margins_name="Total",
        )

        if print_tables:
            print(activity_crosstab)

        if print_percentages:
            print("\n" + "=" * 80)
            print("PERCENTAGE OF ACTIVE/INACTIVE WITHIN EACH PCA CLUSTER")
            print("=" * 80)
            for cluster in sorted(merged_clean[cluster_column].unique()):
                cluster_data = merged_clean[merged_clean[cluster_column] == cluster]
                total = len(cluster_data)
                if total > 0:
                    n_active = int((cluster_data[output_column] == "Active").sum())
                    n_inactive = int((cluster_data[output_column] == "Inactive").sum())
                    pct_active = (n_active / total) * 100
                    pct_inactive = (n_inactive / total) * 100
                    print(f"\nCluster {cluster} (n={total}):")
                    print(f"  Active........ {n_active:>4} ({pct_active:>5.1f}%)")
                    print(f"  Inactive...... {n_inactive:>4} ({pct_inactive:>5.1f}%)")

            print("\n" + "=" * 80)
            print("PERCENTAGE OF CLUSTERS WITHIN ACTIVE/INACTIVE GROUPS")
            print("=" * 80)
            for activity in ["Active", "Inactive"]:
                activity_data = merged_clean[merged_clean[output_column] == activity]
                total = len(activity_data)
                if total > 0:
                    print(f"\n{activity} conformations (n={total}):")
                    for cluster in sorted(activity_data[cluster_column].unique()):
                        n_cluster = int((activity_data[cluster_column] == cluster).sum())
                        pct = (n_cluster / total) * 100
                        print(f"  Cluster {cluster}... {n_cluster:>4} ({pct:>5.1f}%)")

        enrichment_rows = []
        if print_enrichment:
            print("\n" + "=" * 80)
            print("ENRICHMENT ANALYSIS")
            print("=" * 80)
            print("\nDoes each cluster show enrichment for active or inactive conformations?")
            print()

            overall_active = int((merged_clean[output_column] == "Active").sum())
            overall_total = len(merged_clean)
            overall_pct_active = (overall_active / overall_total) * 100 if overall_total else 0.0

            for cluster in sorted(merged_clean[cluster_column].unique()):
                cluster_data = merged_clean[merged_clean[cluster_column] == cluster]
                total = len(cluster_data)
                if total > 0:
                    n_active = int((cluster_data[output_column] == "Active").sum())
                    pct_active = (n_active / total) * 100
                    enrichment = pct_active - overall_pct_active

                    if enrichment > enrichment_threshold_pct:
                        status = "ENRICHED for Active"
                    elif enrichment < -enrichment_threshold_pct:
                        status = "DEPLETED for Active (enriched for Inactive)"
                    else:
                        status = "Balanced"

                    enrichment_rows.append(
                        {
                            "cluster": cluster,
                            "n": total,
                            "pct_active": pct_active,
                            "overall_pct_active": overall_pct_active,
                            "enrichment": enrichment,
                            "status": status,
                        }
                    )

                    print(
                        f"Cluster {cluster}: {pct_active:>5.1f}% active (overall: {overall_pct_active:.1f}%) - {status}"
                    )

            print("\n" + "=" * 80)

        enrichment_df = pd.DataFrame(enrichment_rows) if enrichment_rows else pd.DataFrame()

        return {
            "merged": merged,
            "merged_clean": merged_clean,
            "activity_crosstab": activity_crosstab,
            "enrichment": enrichment_df,
        }

    def plot_cluster_vs_activation_state_heatmap(
        self,
        *,
        merged_csv: str = "Results/dunbrack_assignments/pca_dunbrack_merged.csv",
        cluster_column: str = "ClusterLabel",
        description_column: str = "conformation_description",
        output_png: str = "Results/dunbrack_assignments/pca_activation_correlation_plot.png",
        figsize: tuple = (6, 4),
        cmap: str = "RdYlGn",
        show: bool = True,
        title: str = "PCA Cluster vs Activation State (Counts)",
        print_saved_message: bool = True,
    ) -> dict:
        """
        Convenience wrapper for notebooks: create the heatmap of PCA cluster vs activation state counts.

        - Loads the merged PCA/Dunbrack CSV
        - Maps conformation_description -> activation_state (Active/Inactive/Unknown/Other)
        - Filters to Active/Inactive only
        - Builds a crosstab and plots a seaborn heatmap, saving it to output_png
        """
        merged = pd.read_csv(merged_csv)

        def classify_activation(description):
            if pd.isna(description) or description == "unknown":
                return "Unknown"
            desc = str(description)
            if "Active" in desc:
                return "Active"
            if "inactive" in desc.lower() or "Inactive" in desc:
                return "Inactive"
            return "Other"

        merged["activation_state"] = merged[description_column].apply(classify_activation)
        merged_clean = merged[merged["activation_state"].isin(["Active", "Inactive"])].copy()

        activation_crosstab = pd.crosstab(merged_clean[cluster_column], merged_clean["activation_state"])

        plt.figure(figsize=figsize)
        sns.heatmap(
            activation_crosstab,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar_kws={"label": "Count"},
        )
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel("Activation State", fontsize=10)
        plt.ylabel("PCA Cluster", fontsize=10)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        if print_saved_message:
            print(f"✅ Saved heatmap to {output_png}")

        return {
            "merged": merged,
            "merged_clean": merged_clean,
            "activation_crosstab": activation_crosstab,
            "output_png": output_png,
        }


class PCAWorkflow:
    """Main workflow class for PCA analysis and clustering of protein structures."""
    
    def __init__(self, n_components: int = 4, n_clusters: int = 2):
        """
        Initialize PCA workflow.
        
        Args:
            n_components: Number of principal components
            n_clusters: Number of clusters
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pca_analyzer = None
        self.cluster_analyzer = None
        
    def run_full_analysis(
        self,
        structures_path: str,
        output_prefix: str = "pca",
        perform_hierarchical: bool = True,
        perform_spectral: bool = True,
        cluster0_output_hier: Optional[str] = None,
        cluster1_output_hier: Optional[str] = None,
        cluster0_output_spec: Optional[str] = None,
        cluster1_output_spec: Optional[str] = None
    ) -> dict:
        """
        Run full PCA + clustering analysis on structures.
        
        This performs:
        1. PCA on structures
        2. Hierarchical and/or spectral clustering
        3. Saving results and plots
        
        Args:
            structures_path: Path to structures directory
            output_prefix: Prefix for output files (default: "pca")
            perform_hierarchical: Whether to perform hierarchical clustering (default: True)
            perform_spectral: Whether to perform spectral clustering (default: True)
            cluster0_output_hier: Output directory for hierarchical cluster 0
            cluster1_output_hier: Output directory for hierarchical cluster 1
            cluster0_output_spec: Output directory for spectral cluster 0
            cluster1_output_spec: Output directory for spectral cluster 1
            
        Returns:
            Dictionary with analysis results
        """
        # Get PDB files, excluding 'combined' files
        pdb_files = utilities.get_pdb_files(structures_path, exclude_combined=True)
        structure_names = [utilities.fname(fp) for fp in pdb_files]
        
        if len(pdb_files) == 0:
            print(f"WARNING: No PDB files found in {structures_path}. Skipping.")
            return {}
        
        print(f"\n======= PCA ANALYSIS WITH CLUSTERING =======")
        print(f"Structures: {len(structure_names)}")
        
        # ===== Perform PCA =====
        print("\n--- Running PCA ---")
        pca = PCAAnalyzer(n_components=self.n_components)
        pca_model, projections = pca.fit_transform(pdb_files)
        pca.print_explained_variance()
        
        results = {
            'structure_names': structure_names,
            'pdb_files': pdb_files,
            'pca_model': pca_model,
            'projections': projections,
            'explained_variance': pca.get_explained_variance_ratios()
        }
        
        # Plot PCA scores
        fig_scores, axes_scores = pca.plot_scores(
            structures_path,
            figure_title=f"PCA Scores - {output_prefix}"
        )
        scores_file = f"scores_{output_prefix}.png"
        plt.savefig(scores_file, bbox_inches='tight', dpi=300)
        plt.close(fig_scores)
        print(f"Saved PCA scores plot: {scores_file}")
        
        # Save PC1 and PC2 values
        pca.save_pc_values(0, structure_names, f"pc1_{output_prefix}.txt")
        pca.save_pc_values(1, structure_names, f"pc2_{output_prefix}.txt")
        
        # ===== Hierarchical Clustering =====
        if perform_hierarchical:
            print("\n--- Hierarchical Clustering ---")
            cluster_hier = ClusterAnalyzer(n_clusters=self.n_clusters)
            labels_hier = cluster_hier.hierarchical_cluster(projections)
            
            fig_hier, ax_hier = cluster_hier.plot_clusters(
                projections,
                labels_hier,
                main_title=f"Hierarchical Clustering - {output_prefix}",
                explained_var_ratios=pca.get_explained_variance_ratios()
            )
            
            hier_file = f"cluster_{output_prefix}_hierarchical.png"
            fig_hier.savefig(hier_file, dpi=300)
            plt.close(fig_hier)
            print(f"Saved hierarchical clustering plot: {hier_file}")
            
            # Save cluster labels
            cluster_labels_file_hier = f"cluster_labels_{output_prefix}_hierarchical.txt"
            cluster_hier.save_cluster_labels(
                labels_hier, structure_names, 
                cluster_labels_file_hier, include_pdb_codes=True
            )
            
            results['hierarchical_labels'] = labels_hier
            
            # Copy structures to cluster directories
            if cluster0_output_hier and cluster1_output_hier:
                cluster_hier.copy_structures_to_clusters(
                    pdb_files, labels_hier,
                    [cluster0_output_hier, cluster1_output_hier]
                )
                print(f"  Cluster 0 → {cluster0_output_hier}")
                print(f"  Cluster 1 → {cluster1_output_hier}")
        
        # ===== Spectral Clustering =====
        if perform_spectral:
            print("\n--- Spectral Clustering ---")
            cluster_spec = ClusterAnalyzer(n_clusters=self.n_clusters)
            labels_spec = cluster_spec.spectral_cluster(projections)
            
            fig_spec, ax_spec = cluster_spec.plot_clusters(
                projections,
                labels_spec,
                main_title=f"Spectral Clustering - {output_prefix}",
                explained_var_ratios=pca.get_explained_variance_ratios()
            )
            
            spec_file = f"cluster_{output_prefix}_spectral.png"
            fig_spec.savefig(spec_file, dpi=300)
            plt.close(fig_spec)
            print(f"Saved spectral clustering plot: {spec_file}")
            
            # Save cluster labels
            cluster_labels_file_spec = f"cluster_labels_{output_prefix}_spectral.txt"
            cluster_spec.save_cluster_labels(
                labels_spec, structure_names,
                cluster_labels_file_spec, include_pdb_codes=True
            )
            
            results['spectral_labels'] = labels_spec
            
            # Copy structures to cluster directories
            if cluster0_output_spec and cluster1_output_spec:
                cluster_spec.copy_structures_to_clusters(
                    pdb_files, labels_spec,
                    [cluster0_output_spec, cluster1_output_spec]
                )
                print(f"  Cluster 0 → {cluster0_output_spec}")
                print(f"  Cluster 1 → {cluster1_output_spec}")
        
        print("\n======= ANALYSIS COMPLETE =======")
        return results


def main_example():
    """
    Example main function showing how to run PCA analysis.
    Customize paths and parameters for your specific use case.
    """
    # Define paths
    structures_path = "Results/activation_segments/reconstructed_mustang_filtered"
    cluster0_output_hier = "Results/pca_cluster0_hierarchical"
    cluster1_output_hier = "Results/pca_cluster1_hierarchical"
    cluster0_output_spec = "Results/pca_cluster0_spectral"
    cluster1_output_spec = "Results/pca_cluster1_spectral"
    
    # Create workflow instance
    workflow = PCAWorkflow(n_components=4, n_clusters=2)
    
    # Run full analysis
    results = workflow.run_full_analysis(
        structures_path=structures_path,
        output_prefix="analysis",
        perform_hierarchical=True,
        perform_spectral=True,
        cluster0_output_hier=cluster0_output_hier,
        cluster1_output_hier=cluster1_output_hier,
        cluster0_output_spec=cluster0_output_spec,
        cluster1_output_spec=cluster1_output_spec
    )
    
    return results


if __name__ == "__main__":
    main_example()


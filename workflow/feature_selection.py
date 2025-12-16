"""
Feature Selection for Protein Structure Analysis

This module provides functionality for extracting distance-based features from
aligned protein structures and preparing datasets for machine learning.
"""

import os
import pickle
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations
from typing import List, Dict, Tuple, Set, Optional


class FeatureSelection:
    """
    Extract distance-based features from protein structures for classification.
    
    This class handles:
    - Identification of conserved residues
    - Calculation of pairwise distances between conserved residues
    - Feature matrix construction with median imputation
    - Dataset balancing for machine learning
    - Quality control and visualization
    """
    
    def __init__(self, 
                 dfg_index: int = 145,
                 ape_index: int = 174,
                 conservation_threshold: float = 0.97):
        """
        Initialize FeatureSelection.
        
        Args:
            dfg_index: Position of DFG motif in reference sequence
            ape_index: Position of APE motif in reference sequence
            conservation_threshold: Minimum conservation level for residue selection
        """
        self.dfg_index = dfg_index
        self.ape_index = ape_index
        self.conservation_threshold = conservation_threshold
        
        # Data storage
        self.fully_conserved: List[Tuple[int, str]] = []
        self.structures: Dict = {}
        self.intra_structure_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.structure_names: List[str] = []
        self.unique_pairs: List[Tuple[int, int]] = []
        self.median_distances: Dict = {}
        
    def identify_conserved_residues(self, 
                                   conservation: np.ndarray,
                                   reference_residues: List[str]) -> None:
        """
        Identify conserved residues outside the activation loop.
        
        Args:
            conservation: Array of conservation scores per position
            reference_residues: List of residue names in reference sequence
        """
        self.fully_conserved = []
        
        for i, cons_value in enumerate(conservation):
            # Check if position is OUTSIDE the activation loop
            if i < self.dfg_index or i > self.ape_index:
                if cons_value >= self.conservation_threshold:
                    self.fully_conserved.append((i, reference_residues[i]))
        
        print(f"Found {len(self.fully_conserved)} conserved residues "
              f"(≥{self.conservation_threshold*100}% conservation) outside activation loop")
        for idx, name in self.fully_conserved:
            print(f"  Position {idx}: {name}")
    
    def calculate_intra_structure_distances(self,
                                           aligned_structures: List,
                                           pdb_directory: str,
                                           alignment_function) -> pd.DataFrame:
        """
        Calculate pairwise distances between conserved residues within each structure.
        
        Args:
            aligned_structures: List of alignment objects with structure information
            pdb_directory: Directory containing PDB files
            alignment_function: Function to create alignment segment from alignment object
            
        Returns:
            DataFrame containing distance measurements
        """
        # Get PDB files
        all_pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith('.pdb')]
        
        # Load structures and map conserved residues
        print("Loading structures and mapping conserved residues...")
        self.structures = {}
        
        for alignment_obj in tqdm(aligned_structures, desc="Processing structures"):
            structure_name = alignment_obj.name
            short_name = structure_name[:6]
            
            # Find matching PDB files
            matching_files = [f for f in all_pdb_files if f.startswith(short_name)]
            if not matching_files:
                continue
            
            pdb_file = os.path.join(pdb_directory, matching_files[0])
            
            try:
                struct = md.load(pdb_file)
                residue_coords = {}
                
                # Map each conserved residue
                for cons_idx, cons_residue_name in self.fully_conserved:
                    segment = alignment_function(alignment_obj)
                    
                    if cons_idx >= len(segment):
                        continue
                    
                    ref_res_code, struct_res_code = segment[cons_idx]
                    if struct_res_code == '-':
                        continue
                    
                    # Find residue index in structure
                    aligned_count = 0
                    struct_res_idx = None
                    
                    for i, (_, aligned_code) in enumerate(segment):
                        if aligned_code != '-':
                            if i == cons_idx:
                                struct_res_idx = aligned_count
                                break
                            aligned_count += 1
                    
                    if struct_res_idx is None or struct_res_idx >= struct.top.n_residues:
                        continue
                    
                    # Get atom information
                    res_atom_indices = [atom.index for atom in 
                                       struct.top.residue(struct_res_idx).atoms]
                    
                    residue_coords[cons_idx] = {
                        'atom_indices': res_atom_indices,
                        'topology': struct.top.residue(struct_res_idx),
                        'coordinates': struct.xyz[0, res_atom_indices]
                    }
                
                if residue_coords:
                    self.structures[structure_name] = residue_coords
                    
            except Exception as e:
                print(f"Error processing {structure_name}: {e}")
        
        print(f"Successfully loaded {len(self.structures)} structures")
        
        # Calculate pairwise distances
        distance_data = {
            'structure': [],
            'residue1_name': [],
            'residue1_position': [],
            'residue2_name': [],
            'residue2_position': [],
            'distance': []
        }
        
        backbone_atoms = ['N', 'CA', 'C', 'O']
        
        for struct_name, residue_coords in tqdm(self.structures.items(), 
                                                desc="Calculating distances"):
            residue_positions = list(residue_coords.keys())
            
            if len(residue_positions) < 2:
                continue
            
            for (pos1, pos2) in combinations(residue_positions, 2):
                res1_name = next((name for p, name in self.fully_conserved if p == pos1), 
                                f"Res{pos1}")
                res2_name = next((name for p, name in self.fully_conserved if p == pos2), 
                                f"Res{pos2}")
                
                # Get side chain atoms (excluding hydrogens and backbone)
                res1_topology = residue_coords[pos1]['topology']
                res2_topology = residue_coords[pos2]['topology']
                
                res1_coords = residue_coords[pos1]['coordinates']
                res2_coords = residue_coords[pos2]['coordinates']
                
                res1_sc_indices = [i for i, atom in enumerate(res1_topology.atoms)
                                  if not atom.name.startswith('H') and 
                                  atom.name not in backbone_atoms]
                res2_sc_indices = [i for i, atom in enumerate(res2_topology.atoms)
                                  if not atom.name.startswith('H') and 
                                  atom.name not in backbone_atoms]
                
                # Fall back to backbone for residues with no side chains (e.g., glycine)
                if not res1_sc_indices:
                    res1_sc_indices = [i for i, atom in enumerate(res1_topology.atoms)
                                      if not atom.name.startswith('H')]
                if not res2_sc_indices:
                    res2_sc_indices = [i for i, atom in enumerate(res2_topology.atoms)
                                      if not atom.name.startswith('H')]
                
                res1_sc_coords = res1_coords[res1_sc_indices]
                res2_sc_coords = res2_coords[res2_sc_indices]
                
                # Vectorized minimum distance calculation
                coords1 = res1_sc_coords[:, np.newaxis, :]
                coords2 = res2_sc_coords[np.newaxis, :, :]
                all_distances = np.sqrt(np.sum((coords1 - coords2)**2, axis=2)) * 10  # Angstroms
                min_distance = np.min(all_distances)
                
                distance_data['structure'].append(struct_name)
                distance_data['residue1_name'].append(res1_name)
                distance_data['residue1_position'].append(pos1)
                distance_data['residue2_name'].append(res2_name)
                distance_data['residue2_position'].append(pos2)
                distance_data['distance'].append(min_distance)
        
        self.intra_structure_df = pd.DataFrame(distance_data)
        return self.intra_structure_df
    
    def filter_complete_structures(self, pdb_directory: str, output_directory: str) -> int:
        """
        Filter and copy structures that have all conserved residues.
        
        Args:
            pdb_directory: Source directory with PDB files
            output_directory: Destination directory for filtered structures
            
        Returns:
            Number of structures copied
        """
        os.makedirs(output_directory, exist_ok=True)
        
        required_positions = set(pos for pos, _ in self.fully_conserved)
        all_pdb_files = [f for f in os.listdir(pdb_directory) if f.endswith('.pdb')]
        
        copied_count = 0
        for struct_name, residue_coords in self.structures.items():
            if required_positions.issubset(residue_coords.keys()):
                short_name = struct_name[:6]
                matching_files = [f for f in all_pdb_files if f.startswith(short_name)]
                if matching_files:
                    import shutil
                    pdb_file = os.path.join(pdb_directory, matching_files[0])
                    shutil.copy(pdb_file, output_directory)
                    copied_count += 1
        
        print(f"Copied {copied_count} structures with all conserved residues")
        return copied_count
    
    def assign_labels_from_clusters(self,
                                    cluster_directories: Dict[int, str]) -> None:
        """
        Assign labels to structures based on cluster directory membership.
        
        Args:
            cluster_directories: Dict mapping cluster label to directory path
                                e.g., {0: "path/to/cluster0", 1: "path/to/cluster1"}
        """
        if self.intra_structure_df is None:
            raise ValueError("Calculate distances first")
        
        # Build label mapping from cluster directories
        label_names = []
        labels = []
        
        for label, directory in cluster_directories.items():
            cluster_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
            for filename in cluster_files:
                structure_name = filename.split('.')[0]
                label_names.append(structure_name)
                labels.append(label)
        
        label_names = np.array(label_names)
        labels = np.array(labels)
        
        print(f"Loaded {len(label_names)} labeled structures")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Create mapping from structure prefix to label
        label_mapping = {}
        for i, label_name in enumerate(label_names):
            prefix = label_name[:6]
            label_mapping[prefix] = {'label': labels[i], 'label_name': label_name}
        
        # Add labels to dataframe
        structure_labels = []
        structure_label_names = []
        
        for struct_name in self.intra_structure_df['structure']:
            struct_prefix = struct_name[:6]
            if struct_prefix in label_mapping:
                structure_labels.append(label_mapping[struct_prefix]['label'])
                structure_label_names.append(label_mapping[struct_prefix]['label_name'])
            else:
                structure_labels.append(-1)
                structure_label_names.append('Unknown')
        
        self.intra_structure_df['label'] = structure_labels
        self.intra_structure_df['label_name'] = structure_label_names
        
        print(f"\nLabel summary:")
        print(self.intra_structure_df['label'].value_counts())
        print(f"Structures with unknown labels: "
              f"{(self.intra_structure_df['label'] == -1).sum()}")
    
    def build_feature_matrix(self, use_median_imputation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build feature matrix from distance measurements.
        
        Args:
            use_median_imputation: Whether to impute missing values with median
            
        Returns:
            Tuple of (feature_matrix, imputation_mask)
        """
        if self.intra_structure_df is None:
            raise ValueError("Calculate distances first")
        
        # Get unique residue pairs
        unique_pairs_set = set()
        for _, row in self.intra_structure_df.iterrows():
            pos1, pos2 = row['residue1_position'], row['residue2_position']
            pair = (min(pos1, pos2), max(pos1, pos2))
            unique_pairs_set.add(pair)
        
        self.unique_pairs = sorted(list(unique_pairs_set))
        pair_to_idx = {pair: idx for idx, pair in enumerate(self.unique_pairs)}
        
        # Calculate median distances for imputation
        if use_median_imputation:
            print("Calculating median distances for imputation...")
            self.median_distances = {}
            for pair in self.unique_pairs:
                pos1, pos2 = pair
                pair_data = self.intra_structure_df[
                    ((self.intra_structure_df['residue1_position'] == pos1) & 
                     (self.intra_structure_df['residue2_position'] == pos2)) |
                    ((self.intra_structure_df['residue1_position'] == pos2) & 
                     (self.intra_structure_df['residue2_position'] == pos1))
                ]
                self.median_distances[pair] = pair_data['distance'].median()
        
        # Filter to labeled structures only
        labeled_df = self.intra_structure_df[self.intra_structure_df['label'] != -1]
        unique_structures = labeled_df['structure'].unique()
        self.structure_names = list(unique_structures)
        
        # Extract labels
        label_dict = {}
        for _, row in labeled_df.iterrows():
            if row['structure'] not in label_dict:
                label_dict[row['structure']] = row['label']
        self.labels = np.array([label_dict[s] for s in self.structure_names])
        
        # Build feature matrix
        n_structures = len(self.structure_names)
        n_features = len(self.unique_pairs)
        
        self.feature_matrix = np.zeros((n_structures, n_features))
        imputation_matrix = np.zeros((n_structures, n_features), dtype=bool)
        
        print(f"Building feature matrix ({n_structures} × {n_features})...")
        
        for i, struct_name in enumerate(tqdm(self.structure_names, desc="Building matrix")):
            struct_data = labeled_df[labeled_df['structure'] == struct_name]
            measured_pairs = set()
            
            for _, row in struct_data.iterrows():
                pos1, pos2 = row['residue1_position'], row['residue2_position']
                pair = (min(pos1, pos2), max(pos1, pos2))
                pair_idx = pair_to_idx[pair]
                self.feature_matrix[i, pair_idx] = row['distance']
                measured_pairs.add(pair)
            
            # Impute missing values
            if use_median_imputation:
                for pair in self.unique_pairs:
                    if pair not in measured_pairs:
                        pair_idx = pair_to_idx[pair]
                        self.feature_matrix[i, pair_idx] = self.median_distances[pair]
                        imputation_matrix[i, pair_idx] = True
        
        # Report imputation statistics
        total_values = self.feature_matrix.size
        total_imputed = np.sum(imputation_matrix)
        percent_imputed = (total_imputed / total_values) * 100
        
        print(f"\nFeature Matrix Statistics:")
        print(f"  Shape: {self.feature_matrix.shape}")
        print(f"  Total values: {total_values}")
        print(f"  Measured values: {total_values - total_imputed} ({100 - percent_imputed:.1f}%)")
        print(f"  Imputed values: {total_imputed} ({percent_imputed:.1f}%)")
        
        return self.feature_matrix, imputation_matrix
    
    def balance_dataset(self, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Balance dataset by undersampling majority class.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (balanced_features, balanced_labels, balanced_structure_names)
        """
        if self.feature_matrix is None or self.labels is None:
            raise ValueError("Build feature matrix first")
        
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)
        print(f"Class distribution before balancing: {dict(zip(unique_classes, class_counts))}")
        
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        minority_count = np.min(class_counts)
        
        # Get indices for each class
        majority_indices = np.where(self.labels == majority_class)[0]
        minority_indices = np.where(self.labels == minority_class)[0]
        
        # Randomly subsample majority class
        np.random.seed(random_seed)
        majority_indices_to_keep = np.random.choice(majority_indices, 
                                                    minority_count, 
                                                    replace=False)
        
        indices_to_keep = np.concatenate([majority_indices_to_keep, minority_indices])
        
        balanced_features = self.feature_matrix[indices_to_keep]
        balanced_labels = self.labels[indices_to_keep]
        balanced_names = [self.structure_names[i] for i in indices_to_keep]
        
        print(f"Class distribution after balancing: "
              f"{dict(zip(*np.unique(balanced_labels, return_counts=True)))}")
        print(f"Total samples: {len(balanced_labels)}")
        
        return balanced_features, balanced_labels, balanced_names
    
    def filter_correlated_features(self, correlation_threshold: float = 0.90, 
                                   plot_histogram: bool = True,
                                   plot_network: bool = False,
                                   use_parallel: bool = True,
                                   n_jobs: int = -1) -> Tuple[List[int], Dict]:
        """
        Identify groups of highly correlated features and keep only the feature
        with highest standard deviation from each group.
        
        Args:
            correlation_threshold: Correlation coefficient threshold (default 0.90)
            plot_histogram: Whether to plot correlation distribution
            plot_network: Whether to plot correlation network graph
            use_parallel: Whether to use parallel processing (default True)
            n_jobs: Number of parallel jobs (-1 = all CPUs, default -1)
            
        Returns:
            Tuple of (selected_feature_indices, analysis_info)
        """
        if self.feature_matrix is None:
            raise ValueError("Build feature matrix first")

        # ------------------------------------------------------------------
        # IMPORTANT FIX:
        # The previous implementation attempted to materialize:
        # 1) a full (n_features x n_features) correlation matrix, AND
        # 2) a Python list of *all* pairwise correlations (~94 million pairs here).
        #
        # Even after the correlation chunks show 100% complete, the code still had to:
        # - build the huge "correlations" list (nested Python loops)
        # - iterate it again to build a graph
        # This can take a very long time and can look like it "never ends".
        #
        # This rewritten version:
        # - does NOT build/store the full correlation matrix
        # - does NOT store all pairwise correlations
        # - only keeps edges above the threshold (what we actually need)
        # - uses a union-find to build correlated groups efficiently
        # - optionally plots an approximate histogram via sampling
        # ------------------------------------------------------------------

        print("\n" + "="*60)
        print("CORRELATION-BASED FEATURE SELECTION")
        print("="*60)

        n_samples, n_features = self.feature_matrix.shape
        print(f"\nCurrent feature matrix shape: ({n_samples}, {n_features})")
        print(f"Has NaN values: {np.any(np.isnan(self.feature_matrix))}")
        if use_parallel:
            # Kept for API compatibility; block-wise BLAS already uses multithreaded BLAS.
            print("Note: block-wise correlation uses BLAS; joblib parallel is not used here to avoid huge memory usage.")

        # Standard deviations (used for picking a representative feature per correlated group)
        feature_std = np.nanstd(self.feature_matrix, axis=0)

        # Median-impute (for fast correlation math) without touching self.feature_matrix
        X = self.feature_matrix.astype(float, copy=True)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)  # all-NaN columns -> 0
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = med[np.where(nan_mask)[1]]

        # Center + scale
        X -= X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std = np.where(std == 0.0, 1.0, std)
        Z = X / std

        # Union-Find for correlated components
        parent = np.arange(n_features)
        rank = np.zeros(n_features, dtype=np.int32)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Optional histogram via sampling (no full correlation storage)
        hist_bins = np.linspace(-1.0, 1.0, 51)
        hist_counts = np.zeros(len(hist_bins) - 1, dtype=np.int64)

        # Only store edges if plot_network is requested
        edges_for_graph = [] if plot_network else None

        block = 512
        n_blocks = (n_features + block - 1) // block
        total_block_pairs = (n_blocks * (n_blocks + 1)) // 2
        print(f"\nComputing correlation blocks (block={block}, blocks={n_blocks}x{n_blocks}, pairs={total_block_pairs})")
        print(f"Threshold: r > {correlation_threshold}")

        pbar = tqdm(total=total_block_pairs, desc="Correlation blocks", unit="block")
        for bi in range(n_blocks):
            i0 = bi * block
            i1 = min((bi + 1) * block, n_features)
            Zi = Z[:, i0:i1]

            for bj in range(bi, n_blocks):
                j0 = bj * block
                j1 = min((bj + 1) * block, n_features)
                Zj = Z[:, j0:j1]

                # (i_block x j_block) correlations
                C = (Zi.T @ Zj) / (n_samples - 1)
                if bi == bj:
                    np.fill_diagonal(C, -np.inf)

                if plot_histogram:
                    flat = C.ravel()
                    # sample to keep histogram fast
                    if flat.size > 200_000:
                        idx = np.random.choice(flat.size, size=200_000, replace=False)
                        flat = flat[idx]
                    flat = flat[np.isfinite(flat)]
                    if flat.size:
                        hist_counts += np.histogram(flat, bins=hist_bins)[0]

                # union correlated features
                mask = C > correlation_threshold
                if np.any(mask):
                    ii, jj = np.where(mask)
                    gi = ii + i0
                    gj = jj + j0
                    for a, b in zip(gi.tolist(), gj.tolist()):
                        union(a, b)
                        if edges_for_graph is not None:
                            edges_for_graph.append((a, b))

                pbar.update(1)
        pbar.close()

        if plot_histogram:
            plt.figure(figsize=(12, 7))
            centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
            plt.bar(centers, hist_counts, width=(hist_bins[1] - hist_bins[0]), edgecolor="black", color="skyblue")
            plt.axvline(x=correlation_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {correlation_threshold}")
            plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
            plt.ylabel("Frequency (sampled)", fontsize=12)
            plt.title("Distribution of Correlation Coefficients Between Features (sampled)", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Build network graph only if requested
        if plot_network:
            import networkx as nx
            print(f"\nBuilding correlation network (threshold = {correlation_threshold})...")
            G = nx.Graph()
            G.add_nodes_from(range(n_features))
            G.add_edges_from(edges_for_graph)
            correlated_classes = list(nx.connected_components(G))
        else:
            groups = {}
            for i in range(n_features):
                root = find(i)
                groups.setdefault(root, set()).add(i)
            correlated_classes = list(groups.values())

        print(f"Found {len(correlated_classes)} correlated feature groups")
        
        # Select feature with highest STD from each correlated class
        selected_features = []
        group_info = []
        
        print("\nCorrelated Feature Classes (showing only groups with >1 feature):")
        print("="*60)
        
        for idx, feature_class in enumerate(correlated_classes):
            feature_list = list(feature_class)
            std_values = [feature_std[f] for f in feature_list]
            
            # Get feature with highest standard deviation
            max_std_idx = np.argmax(std_values)
            max_std_feature = feature_list[max_std_idx]
            
            selected_features.append(max_std_feature)
            
            # Store group info
            feature_pairs = [self.unique_pairs[f_idx] for f_idx in feature_list]
            selected_pair = self.unique_pairs[max_std_feature]
            
            group_info.append({
                'group_id': idx + 1,
                'size': len(feature_class),
                'features': feature_pairs,
                'std_values': std_values,
                'selected_feature': max_std_feature,
                'selected_pair': selected_pair,
                'selected_std': feature_std[max_std_feature]
            })
            
            # Print only groups with multiple features (avoid massive logs)
            if len(feature_class) > 1:
                print(f"\nGroup {idx+1}: {len(feature_class)} correlated features")
                print(f"  Features: {feature_pairs}")
                print(f"  STDs: {[round(s, 4) for s in std_values]}")
                print(f"  Selected: {selected_pair} (STD = {feature_std[max_std_feature]:.4f})")
        
        # Plot network if requested
        if plot_network:
            self._plot_correlation_network(G, correlated_classes, feature_std, correlation_threshold)
        
        print(f"\n📊 Feature Selection Summary:")
        print(f"   Original features: {n_features}")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Features removed: {n_features - len(selected_features)}")
        
        analysis_info = {
            'correlation_matrix': None,  # intentionally not materialized
            'correlated_classes': correlated_classes,
            'group_info': group_info,
            'feature_std': feature_std
        }
        
        return selected_features, analysis_info
    
    def _plot_correlation_network(self, G, correlated_classes, feature_std, threshold):
        """Helper method to plot correlation network."""
        import networkx as nx
        
        plt.figure(figsize=(16, 12))
        
        # Create color map
        color_map = plt.cm.get_cmap('tab20', len(correlated_classes))
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            for idx, component in enumerate(correlated_classes):
                if node in component:
                    node_colors.append(color_map(idx))
                    node_sizes.append(300 * feature_std[node] / np.max(feature_std) + 50)
                    break
        
        # Edge widths based on correlation
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [3 * w for w in edge_weights]
        
        # Node labels
        node_labels = {node: f"{self.unique_pairs[node][0]}-{self.unique_pairs[node][1]}" 
                      for node in G.nodes()}
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, 
                               font_family='sans-serif', font_weight='bold')
        
        plt.title(f'Feature Correlation Network (r > {threshold})', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def apply_feature_selection(self, selected_indices: List[int]) -> None:
        """
        Apply feature selection by keeping only selected features.
        
        Args:
            selected_indices: List of feature indices to keep
        """
        if self.feature_matrix is None:
            raise ValueError("Build feature matrix first")
        
        # Sort indices to maintain order
        selected_indices = sorted(selected_indices)
        
        # Filter feature matrix
        self.feature_matrix = self.feature_matrix[:, selected_indices]
        self.unique_pairs = [self.unique_pairs[i] for i in selected_indices]
        
        print(f"\n✅ Applied feature selection")
        print(f"   New matrix shape: {self.feature_matrix.shape}")
        print(f"   Features retained: {len(self.unique_pairs)}")
    
    def filter_same_mean_features(self, mean_tolerance: float = 0.01, 
                                  remove: bool = True) -> Dict:
        """
        Filter features with the same mean, keeping only the one with highest standard deviation.
        
        Features with similar means (within tolerance) are grouped together, and only
        the feature with the highest STD is retained from each group.
        
        Args:
            mean_tolerance: Tolerance for considering means as "same" (in Angstroms)
            remove: If True, apply filtering to the feature matrix
            
        Returns:
            Dictionary with filtering information
        """
        if self.feature_matrix is None or not self.unique_pairs:
            raise ValueError("Build feature matrix first")
        
        print("\n" + "="*60)
        print("FILTERING SAME-MEAN FEATURES")
        print("="*60)
        
        n_features = len(self.unique_pairs)
        print(f"\nFeatures before filtering: {n_features}")
        
        # Calculate mean and std for each feature
        has_nan = np.any(np.isnan(self.feature_matrix))
        if has_nan:
            feature_means = np.nanmean(self.feature_matrix, axis=0)
            feature_stds = np.nanstd(self.feature_matrix, axis=0)
        else:
            feature_means = np.mean(self.feature_matrix, axis=0)
            feature_stds = np.std(self.feature_matrix, axis=0)
        
        # Group features by similar means
        mean_groups = {}
        for i in range(n_features):
            mean_val = feature_means[i]
            
            # Find if this mean belongs to an existing group
            found_group = False
            for group_mean in list(mean_groups.keys()):
                if abs(mean_val - group_mean) <= mean_tolerance:
                    mean_groups[group_mean].append(i)
                    found_group = True
                    break
            
            if not found_group:
                mean_groups[mean_val] = [i]
        
        # Select feature with highest STD from each group
        features_to_keep = []
        features_removed = []
        
        for group_mean, feature_indices in mean_groups.items():
            if len(feature_indices) == 1:
                features_to_keep.append(feature_indices[0])
            else:
                # Multiple features with same mean - keep highest STD
                stds = [feature_stds[i] for i in feature_indices]
                max_std_idx = np.argmax(stds)
                selected_feature = feature_indices[max_std_idx]
                features_to_keep.append(selected_feature)
                
                # Track removed features
                for i in feature_indices:
                    if i != selected_feature:
                        features_removed.append({
                            'feature_idx': i,
                            'pair': self.unique_pairs[i],
                            'mean': feature_means[i],
                            'std': feature_stds[i]
                        })
        
        print(f"\nFound {len(mean_groups)} mean groups")
        print(f"Features with duplicate means: {len(features_removed)}")
        
        if remove and features_removed:
            # Apply filtering
            features_to_keep = sorted(features_to_keep)
            self.feature_matrix = self.feature_matrix[:, features_to_keep]
            self.unique_pairs = [self.unique_pairs[i] for i in features_to_keep]
            
            print(f"\n✂️  Removed {len(features_removed)} features with duplicate means")
            print(f"   Features remaining: {len(self.unique_pairs)}")
        
        return {
            'n_groups': len(mean_groups),
            'n_removed': len(features_removed),
            'removed_features': features_removed
        }
    
    def filter_anova_features(self, n_features: int = 300, 
                             plot_scores: bool = False,
                             remove: bool = True) -> np.ndarray:
        """
        Filter features using ANOVA F-value, keeping top N features that best separate classes.
        
        Uses sklearn's f_classif to compute F-statistic for each feature.
        Features with higher F-values better distinguish between classes.
        
        Args:
            n_features: Number of top features to keep (default 300)
            plot_scores: Whether to plot F-value distribution (default False)
            remove: If True, apply filtering to the feature matrix
            
        Returns:
            Array of selected feature indices
        """
        if self.feature_matrix is None or not self.unique_pairs:
            raise ValueError("Build feature matrix first")
        
        if self.labels is None:
            raise ValueError("Labels required for ANOVA F-value selection")
        
        from sklearn.feature_selection import f_classif
        
        print("\n" + "="*60)
        print("ANOVA F-VALUE FEATURE SELECTION")
        print("="*60)
        
        n_features_before = len(self.unique_pairs)
        print(f"\nFeatures before filtering: {n_features_before}")
        print(f"Target features to keep: {n_features}")
        
        # Compute F-statistics for each feature
        print("Computing ANOVA F-values...")
        
        # Handle NaN values: compute F-statistic only on valid samples
        f_scores = np.zeros(n_features_before)
        p_values = np.ones(n_features_before)
        
        for i in tqdm(range(n_features_before), desc="Computing F-values"):
            # Get valid samples (non-NaN) for this feature
            valid_mask = ~np.isnan(self.feature_matrix[:, i])
            
            if np.sum(valid_mask) > 2:  # Need at least 3 samples
                feature_data = self.feature_matrix[valid_mask, i].reshape(-1, 1)
                feature_labels = self.labels[valid_mask]
                
                # Check if we have both classes
                unique_labels = np.unique(feature_labels)
                if len(unique_labels) > 1:
                    f_stat, p_val = f_classif(feature_data, feature_labels)
                    f_scores[i] = f_stat[0]
                    p_values[i] = p_val[0]
        
        # Select top N features by F-score
        n_to_select = min(n_features, n_features_before)
        top_indices = np.argsort(f_scores)[::-1][:n_to_select]
        top_indices = np.sort(top_indices)  # Keep original order
        
        # Plot F-values if requested
        if plot_scores:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(f_scores[f_scores > 0], bins=50, edgecolor='black', color='skyblue')
            plt.axvline(f_scores[top_indices[-1]], color='red', linestyle='--', 
                       label=f'Selection threshold (F={f_scores[top_indices[-1]]:.2f})')
            plt.xlabel('F-score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of ANOVA F-scores', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            sorted_f_scores = np.sort(f_scores)[::-1]
            plt.plot(range(len(sorted_f_scores)), sorted_f_scores, linewidth=2)
            plt.axvline(n_to_select, color='red', linestyle='--', 
                       label=f'Top {n_to_select} features')
            plt.axhline(f_scores[top_indices[-1]], color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Feature rank', fontsize=12)
            plt.ylabel('F-score', fontsize=12)
            plt.title('Ranked F-scores', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plt.show()
        
        n_removed = n_features_before - n_to_select
        
        print(f"\n📊 ANOVA F-value selection results:")
        print(f"   Features removed: {n_removed}")
        print(f"   Features remaining: {n_to_select}")
        print(f"   F-score range: {f_scores[top_indices].min():.2f} - {f_scores[top_indices].max():.2f}")
        
        if remove and n_removed > 0:
            # Apply filtering
            self.feature_matrix = self.feature_matrix[:, top_indices]
            self.unique_pairs = [self.unique_pairs[i] for i in top_indices]
            
            print(f"\n✂️  Applied ANOVA F-value filtering")
            print(f"   New matrix shape: {self.feature_matrix.shape}")
        
        return top_indices
    
    def filter_low_variance_features(self, variance_threshold: float = 0.1,
                                    remove: bool = True) -> np.ndarray:
        """
        Filter features with low variance using sklearn's VarianceThreshold.
        
        Args:
            variance_threshold: Minimum variance threshold (default 0.1)
            remove: If True, apply filtering to the feature matrix
            
        Returns:
            Array of selected feature indices
        """
        if self.feature_matrix is None or not self.unique_pairs:
            raise ValueError("Build feature matrix first")
        
        from sklearn.feature_selection import VarianceThreshold
        
        print("\n" + "="*60)
        print("FILTERING LOW VARIANCE FEATURES")
        print("="*60)
        
        n_features_before = len(self.unique_pairs)
        print(f"\nFeatures before filtering: {n_features_before}")
        print(f"Variance threshold: {variance_threshold}")
        
        # Handle NaN values by computing variance manually
        has_nan = np.any(np.isnan(self.feature_matrix))
        if has_nan:
            print("Computing variance with NaN handling...")
            variances = np.nanvar(self.feature_matrix, axis=0)
            features_to_keep = variances >= variance_threshold
            selected_indices = np.where(features_to_keep)[0]
        else:
            # Use sklearn for clean data
            selector = VarianceThreshold(threshold=variance_threshold)
            selector.fit(self.feature_matrix)
            selected_indices = selector.get_support(indices=True)
        
        n_features_after = len(selected_indices)
        n_removed = n_features_before - n_features_after
        
        print(f"\n📊 Variance filtering results:")
        print(f"   Features removed: {n_removed}")
        print(f"   Features remaining: {n_features_after}")
        
        if remove and n_removed > 0:
            # Apply filtering
            self.feature_matrix = self.feature_matrix[:, selected_indices]
            self.unique_pairs = [self.unique_pairs[i] for i in selected_indices]
            
            print(f"\n✂️  Applied variance filtering")
            print(f"   New matrix shape: {self.feature_matrix.shape}")
        
        return selected_indices
    
    def filter_consecutive_residues(self, remove: bool = True) -> List[Tuple[int, int]]:
        """
        Find and optionally remove features where residues are consecutive (e.g., 130-131).
        
        Consecutive residues are often not informative for conformational analysis
        since they are always close together in the protein structure.
        
        Args:
            remove: If True, remove these features from the matrix
            
        Returns:
            List of consecutive residue pairs that were found/removed
        """
        if self.feature_matrix is None or not self.unique_pairs:
            raise ValueError("Build feature matrix first")
        
        consecutive_pairs = []
        features_to_keep = []
        
        for idx, (res1, res2) in enumerate(self.unique_pairs):
            # Check if residues are consecutive
            if abs(res2 - res1) == 1:
                consecutive_pairs.append((res1, res2))
                features_to_keep.append(False)
            else:
                features_to_keep.append(True)
        
        print(f"\n🔍 Found {len(consecutive_pairs)} features with consecutive residue indices:")
        if consecutive_pairs:
            for res1, res2 in consecutive_pairs[:10]:  # Show first 10
                print(f"   {res1}-{res2}")
            if len(consecutive_pairs) > 10:
                print(f"   ... and {len(consecutive_pairs) - 10} more")
        
        if remove and consecutive_pairs:
            # Remove consecutive features
            features_to_keep = np.array(features_to_keep)
            self.feature_matrix = self.feature_matrix[:, features_to_keep]
            self.unique_pairs = [pair for pair, keep in zip(self.unique_pairs, features_to_keep) if keep]
            
            print(f"✂️  Removed {len(consecutive_pairs)} consecutive residue features")
            print(f"   New feature count: {len(self.unique_pairs)}")
        
        return consecutive_pairs
    
    def drop_nan_features(self, max_nan_fraction: float = 1.0) -> Tuple[int, int]:
        """
        Drop features (residue pairs) and structures with too many NaN values.
        
        Args:
            max_nan_fraction: Maximum fraction of NaN values allowed.
                             1.0 = only drop if ALL values are NaN
                             0.5 = drop if more than 50% are NaN
                             
        Returns:
            Tuple of (features_dropped, structures_dropped)
        """
        if self.feature_matrix is None:
            raise ValueError("Build feature matrix first")
        
        original_shape = self.feature_matrix.shape
        
        # Drop features (columns) with too many NaN
        nan_fraction_per_feature = np.isnan(self.feature_matrix).sum(axis=0) / self.feature_matrix.shape[0]
        features_to_keep = nan_fraction_per_feature <= max_nan_fraction
        
        self.feature_matrix = self.feature_matrix[:, features_to_keep]
        self.unique_pairs = [pair for pair, keep in zip(self.unique_pairs, features_to_keep) if keep]
        
        # Drop structures (rows) with too many NaN
        nan_fraction_per_structure = np.isnan(self.feature_matrix).sum(axis=1) / self.feature_matrix.shape[1]
        structures_to_keep = nan_fraction_per_structure <= max_nan_fraction
        
        self.feature_matrix = self.feature_matrix[structures_to_keep, :]
        self.structure_names = [name for name, keep in zip(self.structure_names, structures_to_keep) if keep]
        if self.labels is not None:
            self.labels = self.labels[structures_to_keep]
        
        new_shape = self.feature_matrix.shape
        features_dropped = original_shape[1] - new_shape[1]
        structures_dropped = original_shape[0] - new_shape[0]
        
        print(f"\n🗑️  Dropped:")
        print(f"   Features (residue pairs): {features_dropped}")
        print(f"   Structures: {structures_dropped}")
        print(f"   New matrix shape: {new_shape[0]} structures × {new_shape[1]} features")
        
        return features_dropped, structures_dropped
    
    def find_outlier_distances(self, threshold: float = 70.0, set_to_nan: bool = True) -> pd.DataFrame:
        """
        Find distances exceeding a threshold and optionally set them to NaN.
        
        This filters individual outlier distance values rather than removing entire 
        structures or features. Outlier values are replaced with NaN in the feature matrix.
        
        Args:
            threshold: Distance threshold in Angstroms
            set_to_nan: If True, set outlier values to NaN in feature_matrix
            
        Returns:
            DataFrame of outlier distances before filtering
        """
        if self.feature_matrix is None:
            raise ValueError("Build feature matrix first")
        
        # Find outliers
        large_indices = np.where(self.feature_matrix > threshold)
        results = []
        
        for row_idx, col_idx in zip(large_indices[0], large_indices[1]):
            structure_name = self.structure_names[row_idx]
            feature_pair = self.unique_pairs[col_idx]
            distance = self.feature_matrix[row_idx, col_idx]
            
            results.append({
                'structure': structure_name,
                'residue_pair': f"{feature_pair[0]}-{feature_pair[1]}",
                'distance': distance,
                'structure_idx': row_idx,
                'feature_idx': col_idx
            })
        
        if results:
            outliers_df = pd.DataFrame(results).sort_values('distance', ascending=False)
            
            # Set outliers to NaN if requested
            if set_to_nan:
                for _, row in outliers_df.iterrows():
                    self.feature_matrix[row['structure_idx'], row['feature_idx']] = np.nan
                
                # Calculate statistics
                affected_features = outliers_df.groupby('residue_pair').size().sort_values(ascending=False)
                
                print(f"\n🔍 Found {len(outliers_df)} outlier distances > {threshold}Å")
                print(f"✂️  Set {len(outliers_df)} values to NaN in feature matrix")
                print(f"\n📊 Affected features: {len(affected_features)} residue pairs")
                print(f"\nTop 10 most affected features:")
                for pair, count in affected_features.head(10).items():
                    print(f"  {pair}: {count} outlier value(s)")
                
                # Calculate how many valid values remain per feature
                n_structures = len(self.structure_names)
                print(f"\n📈 Valid measurements per affected feature:")
                for pair in affected_features.head(10).index:
                    feature_idx = outliers_df[outliers_df['residue_pair'] == pair]['feature_idx'].iloc[0]
                    n_valid = np.sum(~np.isnan(self.feature_matrix[:, feature_idx]))
                    print(f"  {pair}: {n_valid}/{n_structures} structures have valid data")
            else:
                print(f"Found {len(outliers_df)} distances > {threshold} Å")
            
            # Return DataFrame without internal indices
            return outliers_df.drop(columns=['structure_idx', 'feature_idx'])
        else:
            print(f"✅ No distances > {threshold} Å found")
            return pd.DataFrame(columns=['structure', 'residue_pair', 'distance'])

    def filter_outlier_distances_and_drop_nan(
        self,
        threshold: float = 50.0,
        *,
        set_to_nan: bool = True,
        max_nan_fraction: float = 1.0,
        verbose: bool = True,
        outliers_csv_path: Optional[str] = "outlier_distances.csv",
        print_top_n: int = 10,
    ) -> Dict:
        """
        Convenience pipeline step:
        1) Replace distance outliers (> threshold Å) with NaN
        2) Drop all-NaN (or high-NaN) features and structures

        This is equivalent to the long notebook snippet used for initial cleanup.

        Args:
            threshold: Distance threshold in Angstroms for outlier detection
            set_to_nan: If True, outlier values are replaced with NaN in feature_matrix
            max_nan_fraction: Maximum NaN fraction allowed in a feature/structure.
                              1.0 = drop only if ALL values are NaN
            verbose: If True, print before/after summaries
            outliers_csv_path: If provided, save the outlier table to this CSV path
            print_top_n: If saving outliers, print the top-N largest outliers to stdout

        Returns:
            dict with summary statistics and the outliers dataframe
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not built yet. Build feature matrix first.")

        total_entries = int(self.feature_matrix.size)
        n_valid_before = int(np.sum(~np.isnan(self.feature_matrix)))
        shape_before = tuple(self.feature_matrix.shape)

        if verbose:
            print("=" * 60)
            print(f"FILTERING OUTLIER DISTANCES (>{threshold}Å)")
            print("=" * 60)
            print("\n📊 Before filtering:")
            print(f"   Matrix shape: {shape_before[0]:,} structures × {shape_before[1]:,} residue pairs")
            print(f"   Total entries: {total_entries:,}")
            print(f"   Valid measurements: {n_valid_before:,}")
            print(f"   NaN values: {total_entries - n_valid_before:,}")
            print(f"\n🔍 Searching for outliers >{threshold}Å...")

        outliers_df = self.find_outlier_distances(threshold=threshold, set_to_nan=set_to_nan)

        n_valid_after_outliers = int(np.sum(~np.isnan(self.feature_matrix)))
        outlier_values_removed = n_valid_before - n_valid_after_outliers

        if verbose:
            print(f"\n📊 After filtering outliers:")
            print(f"   Valid measurements: {n_valid_after_outliers:,}")
            print(f"   NaN values: {total_entries - n_valid_after_outliers:,}")
            print(f"   Measurements removed: {outlier_values_removed:,}")
            print(f"\n🗑️  Dropping NaN features and structures (max_nan_fraction={max_nan_fraction})...")

        features_dropped, structures_dropped = self.drop_nan_features(max_nan_fraction=max_nan_fraction)

        shape_after = tuple(self.feature_matrix.shape)
        n_valid_final = int(np.sum(~np.isnan(self.feature_matrix)))

        # Final statistics + optional outlier export (mirrors notebook behavior)
        if verbose:
            final_total = int(self.feature_matrix.size)
            print(f"\n📊 Final feature matrix:")
            print(f"   Matrix shape: {shape_after[0]:,} structures × {shape_after[1]:,} residue pairs")
            print(f"   Total entries: {final_total:,}")
            print(f"   Valid measurements: {n_valid_final:,}")
            print(f"   NaN values: {final_total - n_valid_final:,}")

        if outliers_csv_path and hasattr(outliers_df, "empty") and not outliers_df.empty:
            outliers_df.to_csv(outliers_csv_path, index=False)
            if verbose:
                print(f"\n💾 Saved outlier details to: {outliers_csv_path}")
                if print_top_n and print_top_n > 0:
                    print(f"\nTop {min(print_top_n, len(outliers_df))} largest outlier distances (before filtering):")
                    print(outliers_df.head(print_top_n).to_string(index=False))

        return {
            "threshold": threshold,
            "set_to_nan": set_to_nan,
            "max_nan_fraction": max_nan_fraction,
            "shape_before": shape_before,
            "shape_after": shape_after,
            "total_entries_before": total_entries,
            "valid_measurements_before": n_valid_before,
            "valid_measurements_after_outliers": n_valid_after_outliers,
            "valid_measurements_after_drop": n_valid_final,
            "measurements_removed_as_outliers": outlier_values_removed,
            "features_dropped": features_dropped,
            "structures_dropped": structures_dropped,
            "outliers": outliers_df,
        }
    
    def impute_remaining_nan(self, strategy: str = 'median') -> Tuple[int, int]:
        """
        Impute any remaining NaN values in the feature matrix.
        
        Args:
            strategy: Imputation strategy ('median', 'mean', or 'zero')
            
        Returns:
            Tuple of (number of NaN values imputed, total matrix size)
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not yet built")
        
        # Count NaN values before imputation
        nan_count = np.sum(np.isnan(self.feature_matrix))
        total_size = self.feature_matrix.size
        
        if nan_count == 0:
            print("✅ No NaN values to impute")
            return 0, total_size
        
        print(f"\n{'='*60}")
        print(f"IMPUTING REMAINING NaN VALUES")
        print(f"{'='*60}")
        print(f"📊 Before imputation:")
        print(f"   NaN values: {nan_count:,} ({100*nan_count/total_size:.2f}%)")
        print(f"   Valid values: {total_size - nan_count:,} ({100*(total_size-nan_count)/total_size:.2f}%)")
        
        # Impute column-wise (per feature)
        if strategy == 'median':
            col_impute_values = np.nanmedian(self.feature_matrix, axis=0)
        elif strategy == 'mean':
            col_impute_values = np.nanmean(self.feature_matrix, axis=0)
        elif strategy == 'zero':
            col_impute_values = np.zeros(self.feature_matrix.shape[1])
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'median', 'mean', or 'zero'")
        
        # Replace NaN values with imputed values
        for col_idx in range(self.feature_matrix.shape[1]):
            nan_mask = np.isnan(self.feature_matrix[:, col_idx])
            if np.any(nan_mask):
                self.feature_matrix[nan_mask, col_idx] = col_impute_values[col_idx]
        
        # Verify no NaN values remain
        remaining_nan = np.sum(np.isnan(self.feature_matrix))
        
        print(f"\n📈 After imputation:")
        print(f"   Strategy: {strategy}")
        print(f"   Imputed values: {nan_count:,}")
        print(f"   Remaining NaN: {remaining_nan:,}")
        
        if remaining_nan > 0:
            print(f"\n⚠️  Warning: {remaining_nan} NaN values could not be imputed")
            print("   (likely entire columns with all NaN values)")
        else:
            print(f"\n✅ All NaN values successfully imputed!")
        
        return nan_count, total_size
    
    def save_results(self, output_prefix: str = "") -> None:
        """
        Save all computed results to files.
        
        Args:
            output_prefix: Prefix for output filenames
        """
        if self.intra_structure_df is not None:
            filename = f"{output_prefix}intra_structure_distances.csv"
            self.intra_structure_df.to_csv(filename, index=False)
            print(f"Saved distance data to '{filename}'")
        
        if self.feature_matrix is not None:
            feature_names = [f"{p[0]}-{p[1]}" for p in self.unique_pairs]
            feature_df = pd.DataFrame(self.feature_matrix,
                                     index=self.structure_names,
                                     columns=feature_names)
            filename = f"{output_prefix}feature_matrix.csv"
            feature_df.to_csv(filename)
            print(f"Saved feature matrix to '{filename}'")
        
        if self.labels is not None:
            labels_df = pd.DataFrame({
                'structure': self.structure_names,
                'label': self.labels
            })
            filename = f"{output_prefix}labels.csv"
            labels_df.to_csv(filename, index=False)
            print(f"Saved labels to '{filename}'")
        
        # Save reference data
        reference_data = {
            'fully_conserved': self.fully_conserved,
            'dfg_index': self.dfg_index,
            'ape_index': self.ape_index,
            'unique_pairs': self.unique_pairs,
            'median_distances': self.median_distances,
            'structures': self.structures
        }
        filename = f"{output_prefix}reference_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(reference_data, f)
        print(f"Saved reference data to '{filename}'")
    
    def load_results(self, reference_file: str) -> None:
        """
        Load previously computed results.
        
        Args:
            reference_file: Path to pickled reference data file
        """
        with open(reference_file, 'rb') as f:
            data = pickle.load(f)
            self.fully_conserved = data['fully_conserved']
            self.dfg_index = data['dfg_index']
            self.ape_index = data['ape_index']
            self.unique_pairs = data.get('unique_pairs', [])
            self.median_distances = data.get('median_distances', {})
            self.structures = data.get('structures', {})
        print(f"Loaded reference data from '{reference_file}'")
    
    def plot_distance_heatmaps(self, 
                              n_examples: int = 4,
                              save_dir: Optional[str] = None) -> None:
        """
        Generate distance heatmaps for structures.
        
        Args:
            n_examples: Number of example heatmaps to display
            save_dir: If provided, save individual heatmaps to this directory
        """
        if self.intra_structure_df is None:
            raise ValueError("Calculate distances first")
        
        all_positions = [pos for pos, _ in self.fully_conserved]
        n_positions = len(all_positions)
        pos_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}
        
        all_distances = self.intra_structure_df['distance'].values
        vmin = np.min(all_distances)
        vmax = np.percentile(all_distances, 95)
        
        unique_structures = self.intra_structure_df['structure'].unique()
        
        # Plot examples
        n_examples = min(n_examples, len(unique_structures))
        fig, axes = plt.subplots(1, n_examples, figsize=(5*n_examples, 5))
        if n_examples == 1:
            axes = [axes]
        
        for i, struct_name in enumerate(unique_structures[:n_examples]):
            dist_matrix = np.full((n_positions, n_positions), np.nan)
            np.fill_diagonal(dist_matrix, 0)
            
            struct_data = self.intra_structure_df[
                self.intra_structure_df['structure'] == struct_name
            ]
            
            for _, row in struct_data.iterrows():
                i_idx = pos_to_idx[row['residue1_position']]
                j_idx = pos_to_idx[row['residue2_position']]
                dist_matrix[i_idx, j_idx] = row['distance']
                dist_matrix[j_idx, i_idx] = row['distance']
            
            mask = np.isnan(dist_matrix)
            sns.heatmap(dist_matrix, ax=axes[i], cmap='viridis', mask=mask,
                       vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False,
                       cbar_kws={'label': 'Distance (Å)'})
            axes[i].set_title(f"{struct_name[:8]}\n{np.sum(~np.isnan(dist_matrix[0]))} residues")
        
        plt.tight_layout()
        plt.suptitle('Intra-Structure Distance Matrices', fontsize=16, y=1.02)
        plt.show()
        
        # Save individual heatmaps if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"\nSaving heatmaps to {save_dir}...")
            
            for struct_name in tqdm(unique_structures, desc="Generating heatmaps"):
                dist_matrix = np.full((n_positions, n_positions), np.nan)
                np.fill_diagonal(dist_matrix, 0)
                
                struct_data = self.intra_structure_df[
                    self.intra_structure_df['structure'] == struct_name
                ]
                
                for _, row in struct_data.iterrows():
                    i_idx = pos_to_idx[row['residue1_position']]
                    j_idx = pos_to_idx[row['residue2_position']]
                    dist_matrix[i_idx, j_idx] = row['distance']
                    dist_matrix[j_idx, i_idx] = row['distance']
                
                mask = np.isnan(dist_matrix)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(dist_matrix, cmap='viridis', mask=mask,
                           vmin=vmin, vmax=vmax,
                           cbar_kws={'label': 'Distance (Å)'})
                plt.title(f"Distance Matrix: {struct_name}")
                plt.savefig(f"{save_dir}/{struct_name[:8]}_heatmap.png", 
                           dpi=200, bbox_inches='tight')
                plt.close()


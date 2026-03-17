"""
Feature Classification and Analysis

This module provides functionality for training machine learning models,
evaluating performance, and analyzing feature importance.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from scipy import stats
import time


class FeatureClassification:
    """
    Machine learning classification and feature importance analysis.
    
    This class handles:
    - Train/test splitting
    - Random Forest model training
    - Model evaluation and confusion matrices
    - Feature importance analysis (MDI and permutation)
    - SHAP value computation and visualization
    """
    
    def __init__(self, feature_matrix, labels, unique_pairs, fully_conserved=None, structure_names=None):
        """
        Initialize FeatureClassification.
        
        Args:
            feature_matrix: Feature matrix (n_structures x n_features)
            labels: Class labels for each structure
            unique_pairs: List of residue pairs (tuples) for each feature
            fully_conserved: List of (position, residue_name) tuples (optional)
            structure_names: List of structure names (optional)
        """
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.unique_pairs = unique_pairs
        self.fully_conserved = fully_conserved or []
        self.structure_names = structure_names or []
        
        # Create feature names and position mapping
        self.feature_names = [f"{p[0]}-{p[1]}" for p in unique_pairs]
        self.position_to_residue = {pos: name for pos, name in self.fully_conserved}
        
        # Model and results storage
        self.model = None
        self.train_set = None
        self.test_set = None
        self.train_class = None
        self.test_class = None
        self.predictions = None
        self.feature_importances = None
        self.feature_importances_std = None
        self.feature_importances_sem = None
        self.permutation_result = None
        self.shap_values = None
        
    def split_data(self, train_size=0.9, random_state=42, groups=None):
        """
        Split data into training and test sets.
        
        Args:
            train_size: Fraction of data for training (default 0.9)
            random_state: Random seed for reproducibility (default 42)
            groups: Optional group labels (length n_samples). If provided, performs a group-aware
                    split so that no group appears in both train and test sets.
        """
        print("="*60)
        print("SPLITTING DATA")
        print("="*60)

        if groups is None:
            self.train_set, self.test_set, self.train_class, self.test_class = train_test_split(
                self.feature_matrix, self.labels, train_size=train_size, random_state=random_state
            )
        else:
            groups = np.asarray(groups)
            if groups.shape[0] != len(self.labels):
                raise ValueError(f"groups must have length n_samples={len(self.labels)}; got {groups.shape[0]}")
            splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(self.feature_matrix, self.labels, groups=groups))
            self.train_set = self.feature_matrix[train_idx]
            self.test_set = self.feature_matrix[test_idx]
            self.train_class = np.asarray(self.labels)[train_idx]
            self.test_class = np.asarray(self.labels)[test_idx]
        
        print(f"\nTrain set: {self.train_set.shape[0]} samples")
        print(f"Test set: {self.test_set.shape[0]} samples")
        print(f"Features: {self.train_set.shape[1]}")
        
        # Show class distribution
        train_unique, train_counts = np.unique(self.train_class, return_counts=True)
        test_unique, test_counts = np.unique(self.test_class, return_counts=True)
        
        print(f"\nTrain class distribution: {dict(zip(train_unique, train_counts))}")
        print(f"Test class distribution: {dict(zip(test_unique, test_counts))}")
        
    def train_model(self, n_estimators=10, random_state=42, **kwargs):
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest (default 10)
            random_state: Random seed for reproducibility (default 42)
            **kwargs: Additional parameters for RandomForestClassifier
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        if self.train_set is None:
            raise ValueError("Split data first using split_data()")
        
        print(f"\nTraining with {n_estimators} trees...")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
        self.model.fit(self.train_set, self.train_class)
        
        print("✅ Model training complete!")
        
    def evaluate_model(self, show_metrics=True):
        """
        Evaluate model on test set.
        
        Args:
            show_metrics: Whether to print detailed metrics (default True)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Train model first using train_model()")
        
        # Make predictions
        self.predictions = self.model.predict(self.test_set)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_class, self.predictions)
        # When a class has no predicted samples, sklearn warns and sets that class precision to 0.
        # We set zero_division=0 explicitly to avoid noisy warnings during parameter scans.
        precision = precision_score(self.test_class, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.test_class, self.predictions, average='weighted', zero_division=0)
        
        success = np.sum((self.predictions - self.test_class) == 0)
        percent = float(success) / len(self.test_class) * 100
        
        if show_metrics:
            print(f"\n✅ Test Set Accuracy: {percent:.2f}%")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   Correct predictions: {success}/{len(self.test_class)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'percent_correct': percent,
            'n_correct': success,
            'n_total': len(self.test_class)
        }
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            figsize: Figure size (default (8, 6))
        """
        if self.predictions is None:
            raise ValueError("Evaluate model first using evaluate_model()")
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        
        cm = confusion_matrix(self.test_class, self.predictions)
        
        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def compute_feature_importances(self):
        """
        Compute feature importances using Mean Decrease in Impurity (MDI).
        
        Returns:
            Tuple of (importances, importances_std, importances_sem)
        """
        print("\n" + "="*60)
        print("COMPUTING FEATURE IMPORTANCES (MDI)")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Train model first using train_model()")
        
        self.feature_importances = self.model.feature_importances_
        self.feature_importances_std = np.std(
            [tree.feature_importances_ for tree in self.model.estimators_], axis=0
        )
        self.feature_importances_sem = stats.sem(
            [tree.feature_importances_ for tree in self.model.estimators_], axis=0
        )
        
        print(f"✅ Computed importances for {len(self.feature_importances)} features")
        print(f"   Importance range: {self.feature_importances.min():.6f} - {self.feature_importances.max():.6f}")
        
        return self.feature_importances, self.feature_importances_std, self.feature_importances_sem
    
    def compute_permutation_importances(self, n_repeats=10, random_state=42, n_jobs=2):
        """
        Compute permutation importances.
        
        Args:
            n_repeats: Number of times to permute each feature (default 10)
            random_state: Random seed (default 42)
            n_jobs: Number of parallel jobs (default 2)
            
        Returns:
            Permutation importance result
        """
        print("\n" + "="*60)
        print("COMPUTING PERMUTATION IMPORTANCES")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Train model first using train_model()")
        
        print(f"Computing with {n_repeats} repeats...")
        start_time = time.time()
        
        self.permutation_result = permutation_importance(
            self.model, self.test_set, self.test_class, 
            n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Elapsed time: {elapsed_time:.3f} seconds")
        
        return self.permutation_result
    
    def print_top_features(self, n_top=20, use_permutation=False):
        """
        Print top N features by importance.
        
        Args:
            n_top: Number of top features to show (default 20)
            use_permutation: Use permutation importances instead of MDI (default False)
        """
        print("\n" + "="*60)
        print(f"TOP {n_top} FEATURES BY IMPORTANCE")
        print("="*60)
        
        if use_permutation:
            if self.permutation_result is None:
                raise ValueError("Compute permutation importances first")
            importances = self.permutation_result.importances_mean
            title = "Permutation Importance"
        else:
            if self.feature_importances is None:
                raise ValueError("Compute feature importances first")
            importances = self.feature_importances
            title = "MDI Importance"
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\nFeature ranking ({title}):\n")
        for f in range(min(n_top, len(indices))):
            feature_idx = indices[f]
            importance = importances[feature_idx]
            
            # Get the residue positions
            pos1, pos2 = self.unique_pairs[feature_idx]
            
            # Get the residue names
            res1 = self.position_to_residue.get(pos1, f"Unknown-{pos1}")
            res2 = self.position_to_residue.get(pos2, f"Unknown-{pos2}")
            
            print(f"{f+1:3d}. Distance {res1}({pos1}) - {res2}({pos2}) | Importance: {importance:.6f}")
        
        # Print top important residues
        print(f"\n{'='*60}")
        print("TOP IMPORTANT RESIDUE POSITIONS")
        print("="*60)
        
        important_positions = set()
        for f in range(min(10, len(indices))):
            pos1, pos2 = self.unique_pairs[indices[f]]
            important_positions.add(pos1)
            important_positions.add(pos2)
        
        print("\nTop residue positions to investigate:")
        for pos in sorted(important_positions):
            res_name = self.position_to_residue.get(pos, f"Unknown-{pos}")
            print(f"   Position {pos}: {res_name}")
        
        return indices
    
    def plot_feature_ranking(self, n_top=20, use_permutation=False, figsize=(16, 12)):
        """
        Plot horizontal bar chart of top N features.
        
        Args:
            n_top: Number of top features to show (default 20)
            use_permutation: Use permutation importances instead of MDI (default False)
            figsize: Figure size (default (16, 12))
        """
        print("\n" + "="*60)
        print("PLOTTING FEATURE RANKING")
        print("="*60)
        
        if use_permutation:
            if self.permutation_result is None:
                raise ValueError("Compute permutation importances first")
            importances = self.permutation_result.importances_mean
            errors = self.permutation_result.importances_std
            title = "Feature Importance (Permutation)"
        else:
            if self.feature_importances is None:
                raise ValueError("Compute feature importances first")
            importances = self.feature_importances
            errors = self.feature_importances_sem
            title = "Feature Importance (MDI)"
        
        # Sort and get top N
        sorted_indices = np.argsort(importances)[::-1]
        top_n = min(n_top, len(sorted_indices))
        top_indices = sorted_indices[:top_n]
        top_importances = [importances[i] for i in top_indices]
        top_errors = [errors[i] for i in top_indices]
        
        # Remove zero importance features
        non_zero = [(i, imp, err) for i, imp, err in zip(top_indices, top_importances, top_errors) if imp > 0]
        if non_zero:
            top_indices, top_importances, top_errors = zip(*non_zero)
        else:
            top_indices = top_indices[:min(5, len(top_indices))]
            top_importances = top_importances[:min(5, len(top_importances))]
            top_errors = top_errors[:min(5, len(top_errors))]
        
        # Create labels
        feature_labels = []
        for idx in top_indices:
            pos1, pos2 = self.unique_pairs[idx]
            res1 = self.position_to_residue.get(pos1, f"Unk-{pos1}")
            res2 = self.position_to_residue.get(pos2, f"Unk-{pos2}")
            feature_labels.append(f"{res1}({pos1})-{res2}({pos2})")
        
        # Plot
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(feature_labels))
        
        bars = plt.barh(y_pos, top_importances, align='center', alpha=0.7, color='steelblue')
        plt.errorbar(top_importances, y_pos, xerr=top_errors, fmt='none', 
                    capsize=5, ecolor='black', elinewidth=1)
        
        plt.yticks(y_pos, feature_labels)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{title} - Top {len(feature_labels)} Features', fontsize=14)
        
        # Set x-axis limits
        max_error_extent = max([imp + 2*err for imp, err in zip(top_importances, top_errors)])
        plt.xlim(left=0, right=max_error_extent * 1.2)
        
        # Add values as text
        for i, v in enumerate(top_importances):
            plt.text(v + top_errors[i] + 0.002, i, f"{v:.4f}", va='center', fontsize=9)
        
        plt.tight_layout()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()
        
        print(f"✅ Plotted top {len(feature_labels)} features")
    
    def compute_shap_values(self, feature_perturbation='interventional'):
        """
        Compute SHAP values for feature importance interpretation.
        
        Args:
            feature_perturbation: Perturbation method for TreeExplainer (default 'interventional')
            
        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package required. Install with: pip install shap")
        
        print("\n" + "="*60)
        print("COMPUTING SHAP VALUES")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Train model first using train_model()")
        
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(self.model, feature_perturbation=feature_perturbation)
        
        print("Computing SHAP values...")
        self.shap_values = explainer.shap_values(self.feature_matrix)
        
        print(f"✅ SHAP values computed")
        print(f"   Shape: {np.shape(self.shap_values)}")
        
        return self.shap_values
    
    def plot_shap_summary(self, class_idx=1, max_display=20, figsize=(10, 8)):
        """
        Plot SHAP summary plot for a specific class.
        
        Args:
            class_idx: Class index to plot (0 or 1, default 1)
            max_display: Maximum number of features to display (default 20)
            figsize: Figure size (default (10, 8))
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package required. Install with: pip install shap")
        
        if self.shap_values is None:
            raise ValueError("Compute SHAP values first using compute_shap_values()")
        
        print(f"\n{'='*60}")
        print(f"SHAP SUMMARY PLOT - CLASS {class_idx}")
        print("="*60)
        
        # Extract SHAP values for specific class
        class_shap = self.shap_values[:, :, class_idx]
        
        plt.figure(figsize=figsize)
        shap.summary_plot(
            class_shap,
            self.feature_matrix,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Summary - Class {class_idx}")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, n_top=20, class_idx=1, figsize=(15, 40)):
        """
        Plot feature value distributions for top SHAP features.
        
        Args:
            n_top: Number of top features to plot (default 20)
            class_idx: Class index for SHAP ranking (default 1)
            figsize: Figure size (default (15, 40))
        """
        if self.shap_values is None:
            raise ValueError("Compute SHAP values first using compute_shap_values()")
        
        print(f"\n{'='*60}")
        print(f"FEATURE DISTRIBUTIONS - TOP {n_top} BY SHAP")
        print("="*60)
        
        # Get top features by mean absolute SHAP
        class_shap = self.shap_values[:, :, class_idx]
        mean_shap_values = np.abs(class_shap).mean(0)
        top_indices = np.argsort(mean_shap_values)[-n_top:]
        
        # Create plots
        plt.figure(figsize=figsize)
        n_rows = (n_top + 1) // 2
        
        for i, idx in enumerate(top_indices):
            plt.subplot(n_rows, 2, i+1)
            
            # Get feature values for each class
            cluster0_values = self.feature_matrix[self.labels == 0, idx]
            cluster1_values = self.feature_matrix[self.labels == 1, idx]
            
            # Calculate statistics
            mean_cluster0 = np.mean(cluster0_values)
            mean_cluster1 = np.mean(cluster1_values)
            std_cluster0 = np.std(cluster0_values)
            std_cluster1 = np.std(cluster1_values)
            
            # Plot KDE
            sns.kdeplot(cluster0_values, fill=True, alpha=0.5, label='Class 0')
            sns.kdeplot(cluster1_values, fill=True, alpha=0.5, label='Class 1')
            
            # Add mean lines
            plt.axvline(x=mean_cluster0, color='red', linestyle=':', linewidth=2,
                       label=f'Class 0: μ={mean_cluster0:.2f}, σ={std_cluster0:.2f}')
            plt.axvline(x=mean_cluster1, color='red', linestyle='--', linewidth=2,
                       label=f'Class 1: μ={mean_cluster1:.2f}, σ={std_cluster1:.2f}')
            
            plt.title(f"Feature: {self.feature_names[idx]}", fontsize=10)
            plt.xlabel("Feature Value", fontsize=9)
            plt.ylabel("Density", fontsize=9)
            plt.legend(loc='best', fontsize='small')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Plotted {n_top} feature distributions")

    @staticmethod
    def run_anova_sum_parameter_scan_with_guide_tree(
        *,
        reference_pickle: str = "corr_filtered_reference_data.pkl",
        feature_matrix_csv: str = "corr_filtered_feature_matrix.csv",
        labels_csv: str = "corr_filtered_labels.csv",
        newick_path: str = "Results/activation_segments/multi_aligned_foldmason/msa.nw",
        height_step: int = 2,
        height_min: int = 2,
        height_max: int = 0,
        k_values: Optional[list] = None,
        n_repeats: int = 3,
        repeat_seeds: Optional[list] = None,
        train_size: float = 0.9,
        n_estimators: int = 100,
        perm_repeats: int = 100,
        show_plots: bool = True,
    ) -> dict:
        """
        Scan (height, k) for ANOVA-selected features with group-aware splitting by guide-tree clusters.
        """
        from sklearn.feature_selection import f_classif
        from sklearn.metrics import confusion_matrix
        from workflow.feature_selection import FeatureSelection
        from workflow.guide_tree_clusters import parse_newick, cluster_leaves_by_height_cut

        if repeat_seeds is None:
            repeat_seeds = [42 + i for i in range(int(n_repeats))]
        if k_values is None:
            k_values = list(range(300, 2001, 200))
            if 2000 not in k_values:
                k_values.append(2000)

        fs_scan = FeatureSelection(dfg_index=145, ape_index=174, conservation_threshold=0.97)
        fs_scan.load_results(reference_pickle)

        X_df = pd.read_csv(feature_matrix_csv, index_col=0)
        y_df = pd.read_csv(labels_csv)
        if "structure" in y_df.columns:
            y_df = y_df.set_index("structure")
            y_df = y_df.loc[X_df.index]
        y = y_df["label"].values
        X = X_df.values

        fs_scan.feature_matrix = X
        fs_scan.structure_names = list(X_df.index)
        fs_scan.labels = y

        if len(getattr(fs_scan, "unique_pairs", [])) != X.shape[1]:
            parsed_pairs = []
            bad = []
            for c in X_df.columns:
                s = str(c).strip()
                try:
                    a, b = s.split("-")
                    a, b = int(a), int(b)
                    parsed_pairs.append((min(a, b), max(a, b)))
                except Exception:
                    bad.append(s)
            if bad:
                raise ValueError("Could not parse some feature columns as 'i-j': " + ", ".join(bad[:10]))
            fs_scan.unique_pairs = parsed_pairs

        X_imp = X.astype(float, copy=True)
        med = np.nanmedian(X_imp, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        nan_mask = np.isnan(X_imp)
        if np.any(nan_mask):
            X_imp[nan_mask] = med[np.where(nan_mask)[1]]

        f_scores, _ = f_classif(X_imp, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(f_scores)[::-1]

        k_values = sorted({int(k) for k in k_values if 1 <= int(k) <= X_imp.shape[1]})
        labels_all = np.unique(y)
        position_to_residue = {pos: name for pos, name in (fs_scan.fully_conserved or [])}

        def _feature_labels_for_pairs(pairs):
            out = []
            for (pos1, pos2) in pairs:
                res1 = position_to_residue.get(pos1, f"Unk-{pos1}")
                res2 = position_to_residue.get(pos2, f"Unk-{pos2}")
                out.append(f"{res1}({pos1})-{res2}({pos2})")
            return out

        with open(newick_path, "r", encoding="utf-8", errors="ignore") as f:
            newick_str = f.read().strip()
        tree_root = parse_newick(newick_str)
        _, max_h = cluster_leaves_by_height_cut(tree_root, cut_height=1e9)

        if int(height_max) > 0:
            h_max = int(height_max)
            if float(h_max) >= float(max_h):
                h_max = max(1, int(np.floor(max_h - 1e-9)))
        else:
            h_max = max(1, int(np.floor(max_h - 1e-9)))

        candidate_heights = list(range(int(height_min), h_max + 1, int(height_step)))
        structure_names = list(X_df.index)

        height_to_leaf_to_cluster = {}
        heights = []
        for h in candidate_heights:
            leaf_to_cluster, _ = cluster_leaves_by_height_cut(tree_root, cut_height=float(h))
            height_to_leaf_to_cluster[int(h)] = leaf_to_cluster
            next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
            groups_h = []
            for nm in structure_names:
                g = leaf_to_cluster.get(nm)
                if g is None:
                    g = next_missing
                    next_missing += 1
                groups_h.append(g)
            if len(set(groups_h)) >= 2:
                heights.append(int(h))
        if not heights:
            raise ValueError("No valid guide-tree cut heights produced >=2 clusters.")

        metric_rows = []
        for h in heights:
            leaf_to_cluster = height_to_leaf_to_cluster[int(h)]
            next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
            groups = []
            for nm in structure_names:
                g = leaf_to_cluster.get(nm)
                if g is None:
                    g = next_missing
                    next_missing += 1
                groups.append(g)

            for k in k_values:
                idx = np.sort(rank[:k])
                Xk = X_imp[:, idx]
                pairs_k = [fs_scan.unique_pairs[i] for i in idx]
                accs = []
                precs = []
                for seed in repeat_seeds:
                    clf = FeatureClassification(
                        feature_matrix=Xk,
                        labels=y,
                        unique_pairs=pairs_k,
                        fully_conserved=fs_scan.fully_conserved,
                        structure_names=fs_scan.structure_names,
                    )
                    clf.split_data(train_size=train_size, random_state=seed, groups=groups)
                    clf.train_model(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
                    m = clf.evaluate_model(show_metrics=False)
                    accs.append(float(m["accuracy"]))
                    precs.append(float(m["precision"]))

                metric_rows.append(
                    {
                        "height": int(h),
                        "k": int(k),
                        "accuracy_mean": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                        "precision_mean": float(np.mean(precs)),
                        "precision_std": float(np.std(precs, ddof=1)) if len(precs) > 1 else 0.0,
                    }
                )

        res_df = pd.DataFrame(metric_rows).sort_values(["height", "k"])
        try:
            from IPython.display import display
            display(res_df)
        except Exception:
            print(res_df.head().to_string(index=False))

        if show_plots:
            pivot_acc = res_df.pivot(index="height", columns="k", values="accuracy_mean")
            fig, ax = plt.subplots(figsize=(10.5, 7.5))
            im = ax.imshow(pivot_acc.values, aspect="auto", origin="lower")
            ax.set_title("Accuracy mean (cluster-aware split using FoldMason msa.nw)")
            ax.set_xlabel("k (#features)")
            ax.set_ylabel("Newick cut height")
            ax.set_xticks(np.arange(len(pivot_acc.columns)))
            ax.set_xticklabels([str(x) for x in pivot_acc.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot_acc.index)))
            ax.set_yticklabels([str(x) for x in pivot_acc.index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

        best = res_df.sort_values(["accuracy_mean", "precision_mean"], ascending=False).iloc[0]
        best_h = int(best["height"])
        best_k = int(best["k"])
        print(f"\nBest by accuracy_mean: height={best_h}, k={best_k}")
        print(best)

        leaf_to_cluster, _ = cluster_leaves_by_height_cut(tree_root, cut_height=float(best_h))
        next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
        best_groups = []
        for nm in structure_names:
            g = leaf_to_cluster.get(nm)
            if g is None:
                g = next_missing
                next_missing += 1
            best_groups.append(g)

        idx = np.sort(rank[:best_k])
        Xk = X_imp[:, idx]
        pairs_k = [fs_scan.unique_pairs[i] for i in idx]
        feature_labels_all = _feature_labels_for_pairs(pairs_k)

        cms = []
        gini_vals_reps = []
        gini_err_reps = []
        perm_vals_reps = []
        perm_err_reps = []
        for seed in repeat_seeds:
            clf = FeatureClassification(
                feature_matrix=Xk,
                labels=y,
                unique_pairs=pairs_k,
                fully_conserved=fs_scan.fully_conserved,
                structure_names=fs_scan.structure_names,
            )
            clf.split_data(train_size=train_size, random_state=seed, groups=best_groups)
            clf.train_model(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
            _ = clf.evaluate_model(show_metrics=False)
            y_pred = clf.model.predict(clf.test_set)
            cms.append(confusion_matrix(clf.test_class, y_pred, labels=labels_all))
            clf.compute_feature_importances()
            gini_vals_reps.append(np.asarray(clf.feature_importances, dtype=float))
            gini_err_reps.append(np.asarray(clf.feature_importances_sem, dtype=float))
            perm_res = clf.compute_permutation_importances(
                n_repeats=perm_repeats,
                random_state=seed,
                n_jobs=-1,
            )
            perm_vals_reps.append(np.asarray(perm_res.importances_mean, dtype=float))
            perm_err_reps.append(np.asarray(perm_res.importances_std, dtype=float))

        cms_arr = np.stack(cms, axis=0).astype(float)
        cm_mean = np.mean(cms_arr, axis=0)
        cm_std = np.std(cms_arr, axis=0, ddof=1) if cms_arr.shape[0] > 1 else np.zeros_like(cm_mean)

        if show_plots:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            im = ax.imshow(cm_mean, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean count")
            ax.set_xticks(np.arange(len(labels_all)))
            ax.set_yticks(np.arange(len(labels_all)))
            ax.set_xticklabels(labels_all)
            ax.set_yticklabels(labels_all)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"Best confusion matrix (mean ± std), height={best_h}, k={best_k}")
            thresh = cm_mean.max() / 2.0 if cm_mean.size else 0.0
            for i in range(cm_mean.shape[0]):
                for j in range(cm_mean.shape[1]):
                    m = cm_mean[i, j]
                    s = cm_std[i, j]
                    txt_color = "white" if m > thresh else "black"
                    ax.text(j, i, f"{m:.1f}\n±{s:.1f}", ha="center", va="center", color=txt_color, fontsize=10)
            plt.tight_layout()
            plt.show()

        gini_vals_reps = np.stack(gini_vals_reps, axis=0)
        gini_err_reps = np.stack(gini_err_reps, axis=0)
        gini_mean = np.mean(gini_vals_reps, axis=0)
        gini_err_prop = np.sqrt(np.sum(gini_err_reps ** 2, axis=0)) / float(n_repeats)

        perm_vals_reps = np.stack(perm_vals_reps, axis=0)
        perm_err_reps = np.stack(perm_err_reps, axis=0)
        perm_mean = np.mean(perm_vals_reps, axis=0)
        perm_err_prop = np.sqrt(np.sum(perm_err_reps ** 2, axis=0)) / float(n_repeats)

        def _plot_importance_barh(title, labels, mean_vals, err_vals, n_top=20, figsize=(12, 10)):
            order = np.argsort(np.asarray(mean_vals))[::-1][: int(min(n_top, len(mean_vals)))]
            lbl = [labels[i] for i in order][::-1]
            vals = np.asarray(mean_vals)[order][::-1]
            errs = np.asarray(err_vals)[order][::-1]
            fig, ax = plt.subplots(figsize=figsize)
            y_pos = np.arange(len(lbl))
            ax.barh(y_pos, vals, xerr=errs, color="steelblue", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(lbl, fontsize=8)
            ax.set_xlabel("Importance")
            ax.set_title(title)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            plt.tight_layout()
            plt.show()

        if show_plots:
            _plot_importance_barh(
                title=f"Best Gini/MDI importance (mean ± propagated error), height={best_h}, k={best_k}",
                labels=feature_labels_all,
                mean_vals=gini_mean,
                err_vals=gini_err_prop,
                n_top=20,
            )
            _plot_importance_barh(
                title=f"Best Permutation importance (mean ± propagated error), height={best_h}, k={best_k}",
                labels=feature_labels_all,
                mean_vals=perm_mean,
                err_vals=perm_err_prop,
                n_top=20,
            )

        return {
            "res_df": res_df,
            "best_h": best_h,
            "best_k": best_k,
            "X_df": X_df,
            "y": y,
            "Xk": Xk,
            "feature_labels_all": feature_labels_all,
            "gini_mean": gini_mean,
            "perm_mean": perm_mean,
            "cm_mean": cm_mean,
            "cm_std": cm_std,
        }

    @staticmethod
    def load_anova_scan_results_for_distributions(
        *,
        checkpoint_dir: str = "Results/anova_sum_scan_perm/checkpoints",
        best_combinations_csv: str = "Results/anova_sum_scan_perm/best_combinations.csv",
        reference_pickle: str = "corr_filtered_reference_data.pkl",
        feature_matrix_csv: str = "corr_filtered_feature_matrix.csv",
        labels_csv: str = "corr_filtered_labels.csv",
    ) -> dict:
        """
        Load best scan params and checkpoint importances, preparing objects for distribution plots.
        """
        from sklearn.feature_selection import f_classif
        from workflow.feature_selection import FeatureSelection

        best_df = pd.read_csv(best_combinations_csv)
        best_row = best_df.iloc[0]
        best_h = int(best_row["height"])
        best_k = int(best_row["k"])

        fs_scan = FeatureSelection(dfg_index=145, ape_index=174, conservation_threshold=0.97)
        fs_scan.load_results(reference_pickle)

        X_df = pd.read_csv(feature_matrix_csv, index_col=0)
        y_df = pd.read_csv(labels_csv)
        if "structure" in y_df.columns:
            y_df = y_df.set_index("structure")
            y_df = y_df.loc[X_df.index]
        y = y_df["label"].values
        X = X_df.values

        fs_scan.feature_matrix = X
        fs_scan.structure_names = list(X_df.index)
        fs_scan.labels = y

        if len(getattr(fs_scan, "unique_pairs", [])) != X.shape[1]:
            parsed_pairs = []
            bad = []
            for c in X_df.columns:
                s = str(c).strip()
                try:
                    a, b = s.split("-")
                    parsed_pairs.append((min(int(a), int(b)), max(int(a), int(b))))
                except Exception:
                    bad.append(s)
            if bad:
                raise ValueError("Could not parse some feature columns as 'i-j': " + ", ".join(bad[:10]))
            fs_scan.unique_pairs = parsed_pairs

        X_imp = X.astype(float, copy=True)
        med = np.nanmedian(X_imp, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        nan_mask = np.isnan(X_imp)
        if np.any(nan_mask):
            X_imp[nan_mask] = med[np.where(nan_mask)[1]]

        f_scores, _ = f_classif(X_imp, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(f_scores)[::-1]

        idx = np.sort(rank[:best_k])
        Xk = X_imp[:, idx]
        pairs_k = [fs_scan.unique_pairs[i] for i in idx]

        pos_to_residue = {pos: name for pos, name in (fs_scan.fully_conserved or [])}
        feature_labels_all = [
            f"{pos_to_residue.get(pos1, f'Unk-{pos1}')}({pos1})-{pos_to_residue.get(pos2, f'Unk-{pos2}')}({pos2})"
            for (pos1, pos2) in pairs_k
        ]

        checkpoint_file = os.path.join(checkpoint_dir, f"h{best_h}_k{best_k}.json")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)

        mdi_vals_reps = [np.array(v) for v in checkpoint_data["mdi_vals_reps"]]
        perm_vals_reps = [np.array(v) for v in checkpoint_data["perm_vals_reps"]]
        gini_mean = np.mean(np.stack(mdi_vals_reps, axis=0), axis=0)
        perm_mean = np.mean(np.stack(perm_vals_reps, axis=0), axis=0)

        return {
            "best_h": best_h,
            "best_k": best_k,
            "X_df": X_df,
            "y": y,
            "Xk": Xk,
            "feature_labels_all": feature_labels_all,
            "gini_mean": gini_mean,
            "perm_mean": perm_mean,
        }

    @staticmethod
    def plot_top_feature_distributions_by_label_and_cluster(
        *,
        X_df: pd.DataFrame,
        Xk: np.ndarray,
        feature_labels_all: list,
        gini_mean: np.ndarray,
        perm_mean: np.ndarray,
        best_h: int,
        best_k: int,
        biological_labels_csv: str = "corr_filtered_labels.csv",
        pca_cluster_labels_file: str = "cluster_labels_my_analysis_hierarchical.txt",
        n_top: int = 20,
    ) -> dict:
        """
        Plot top-feature violin distributions split by biological labels and PCA clusters.
        """
        sns.set_style("whitegrid")

        bio_labels_df = pd.read_csv(biological_labels_csv)
        if "structure" in bio_labels_df.columns:
            bio_labels_df = bio_labels_df.set_index("structure")
            bio_labels_df = bio_labels_df.loc[X_df.index]
        bio_labels = bio_labels_df["label"].values

        pca_labels_df = pd.read_csv(
            pca_cluster_labels_file,
            comment="#",
            names=["ClusterLabel", "PDBCode", "FullName"],
        )
        pca_labels_df = pca_labels_df.set_index("FullName").reindex(X_df.index)
        cluster_binary = pca_labels_df["ClusterLabel"].values
        valid_mask = ~pd.isna(cluster_binary)

        cluster_label_matrix = []
        for cid in np.unique(cluster_binary[valid_mask]):
            mask = (cluster_binary == cid) & valid_mask
            n_total = np.sum(mask)
            n_active = np.sum(bio_labels[mask] == 1)
            n_inactive = np.sum(bio_labels[mask] == 0)
            cluster_label_matrix.append(
                {
                    "pca_cluster_id": int(cid),
                    "n_total": int(n_total),
                    "n_active": int(n_active),
                    "n_inactive": int(n_inactive),
                    "pct_active": (100 * n_active / n_total) if n_total > 0 else 0.0,
                }
            )
        cluster_stats = pd.DataFrame(cluster_label_matrix)
        print("\nPCA Cluster composition:")
        print(cluster_stats)

        label_names = {0: "Inactive", 1: "Active"}
        if len(np.unique(bio_labels[valid_mask])) == 2:
            y_named = np.array([label_names.get(val, f"Label_{val}") for val in bio_labels])
        else:
            y_named = bio_labels.astype(str)

        def plot_feature_distributions(feature_indices, importance_type, split_by="biological", n_top_local=20):
            top_n = min(n_top_local, len(feature_indices))
            top_indices = feature_indices[:top_n]
            n_cols = 5
            n_rows = int(np.ceil(top_n / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
            axes = axes.flatten() if top_n > 1 else [axes]

            if split_by == "biological":
                group_labels = y_named[valid_mask]
                group_name = "Biological Label (Active/Inactive)"
                palette = ["#2ca02c", "#d62728"]
            else:
                group_labels = np.array([f"Cluster {int(c)}" for c in cluster_binary[valid_mask]])
                group_name = "PCA Cluster (0/1)"
                palette = ["#ff7f0e", "#1f77b4"]

            for plot_idx, feat_idx in enumerate(top_indices):
                ax = axes[plot_idx]
                feat_values = Xk[:, feat_idx]
                plot_data = pd.DataFrame({"value": feat_values[valid_mask], "group": group_labels})
                sns.violinplot(data=plot_data, x="group", y="value", ax=ax, palette=palette)
                feature_label = feature_labels_all[feat_idx]
                ax.set_title(f"{feature_label}", fontsize=9, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Distance (Å)", fontsize=8)
                ax.tick_params(axis="x", labelsize=8, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", alpha=0.3)

            for idx2 in range(top_n, len(axes)):
                axes[idx2].axis("off")
            fig.suptitle(
                f"Top {top_n} Features by {importance_type} - Distribution by {group_name}\n"
                f"(Best params: height={best_h}, k={best_k})",
                fontsize=14,
                fontweight="bold",
                y=1.002,
            )
            plt.tight_layout()
            plt.show()

        gini_order = np.argsort(gini_mean)[::-1]
        perm_order = np.argsort(perm_mean)[::-1]
        plot_feature_distributions(gini_order, "MDI (Gini) Importance", split_by="biological", n_top_local=n_top)
        plot_feature_distributions(gini_order, "MDI (Gini) Importance", split_by="pca_cluster", n_top_local=n_top)
        plot_feature_distributions(perm_order, "Permutation Importance", split_by="biological", n_top_local=n_top)
        plot_feature_distributions(perm_order, "Permutation Importance", split_by="pca_cluster", n_top_local=n_top)
        print("\n✅ ALL 4 DISTRIBUTION PLOTS COMPLETE")

        return {
            "cluster_stats": cluster_stats,
            "valid_mask": valid_mask,
            "cluster_binary": cluster_binary,
            "bio_labels": bio_labels,
        }
    
    def run_full_analysis(self, train_size=0.9, n_estimators=100, random_state=42,
                         n_top=20, compute_shap=True, plot_all=True):
        """
        Run complete classification analysis pipeline.
        
        Args:
            train_size: Train/test split ratio (default 0.9)
            n_estimators: Number of trees for Random Forest (default 100)
            random_state: Random seed (default 42)
            n_top: Number of top features to display (default 20)
            compute_shap: Whether to compute SHAP values (default True)
            plot_all: Whether to generate all plots (default True)
            
        Returns:
            Dictionary with all results
        """
        # Split data
        self.split_data(train_size=train_size, random_state=random_state)
        
        # Train model
        self.train_model(n_estimators=n_estimators, random_state=random_state)
        
        # Evaluate
        metrics = self.evaluate_model()
        
        # Confusion matrix
        if plot_all:
            cm = self.plot_confusion_matrix()
        
        # Feature importances
        self.compute_feature_importances()
        indices = self.print_top_features(n_top=n_top)
        
        if plot_all:
            self.plot_feature_ranking(n_top=n_top)
        
        # Permutation importances
        self.compute_permutation_importances()
        
        # SHAP analysis
        if compute_shap:
            try:
                self.compute_shap_values()
                if plot_all:
                    self.plot_shap_summary(class_idx=0, max_display=n_top)
                    self.plot_shap_summary(class_idx=1, max_display=n_top)
                    self.plot_feature_distributions(n_top=n_top)
            except ImportError:
                print("\n⚠️  SHAP not available. Install with: pip install shap")
        
        return {
            'metrics': metrics,
            'model': self.model,
            'feature_importances': self.feature_importances,
            'top_feature_indices': indices
        }

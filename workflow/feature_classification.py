"""
Feature Classification and Analysis

This module provides functionality for training machine learning models,
evaluating performance, and analyzing feature importance.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import wasserstein_distance
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
        self.train_idx = None
        self.test_idx = None
        self.split_labels = None
        self.predictions = None
        self.feature_importances = None
        self.feature_importances_std = None
        self.feature_importances_sem = None
        self.permutation_result = None
        self.shap_values = None

    @staticmethod
    def _parse_pair_columns(columns) -> list:
        """Parse feature columns named ``i-j`` into ``(i, j)`` pairs."""
        pairs = []
        for col in columns:
            parts = str(col).split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                pairs.append((int(parts[0]), int(parts[1])))
            else:
                return []
        return pairs

    @classmethod
    def resolve_feature_export_paths(
        cls,
        base_dir: str = ".",
        prefer: Optional[Tuple[str, ...]] = None,
    ) -> dict:
        """
        Pick the most-filtered feature-selection export trio available on disk.

        Preference (first complete match wins):
          1. ANOVA-filtered: filtered_feature_matrix.csv + filtered_labels.csv + filtered_reference_data.pkl
          2. Correlation-filtered: corr_filtered_*
          3. Unfiltered: feature_matrix.csv + labels.csv + reference_data.pkl

        Returns:
            dict with keys matrix_csv, labels_csv, reference_pkl, stage
        """
        base_dir = os.path.abspath(base_dir)
        stages = prefer or ("anova", "corr", "raw")
        catalog = {
            "anova": (
                "filtered_feature_matrix.csv",
                "filtered_labels.csv",
                "filtered_reference_data.pkl",
                "ANOVA-filtered",
            ),
            "corr": (
                "corr_filtered_feature_matrix.csv",
                "corr_filtered_labels.csv",
                "corr_filtered_reference_data.pkl",
                "correlation-filtered",
            ),
            "raw": (
                "feature_matrix.csv",
                "labels.csv",
                "reference_data.pkl",
                "unfiltered",
            ),
        }
        for key in stages:
            if key not in catalog:
                raise ValueError(f"Unknown export stage '{key}'. Expected one of {list(catalog)}")
            matrix_name, labels_name, ref_name, stage = catalog[key]
            matrix_csv = os.path.join(base_dir, matrix_name)
            labels_csv = os.path.join(base_dir, labels_name)
            reference_pkl = os.path.join(base_dir, ref_name)
            # Reference pickle is preferred but optional for RF training
            if os.path.isfile(matrix_csv) and os.path.isfile(labels_csv):
                if not os.path.isfile(reference_pkl):
                    reference_pkl = None
                return {
                    "matrix_csv": matrix_csv,
                    "labels_csv": labels_csv,
                    "reference_pkl": reference_pkl,
                    "stage": stage,
                }
        raise FileNotFoundError(
            f"No feature-selection exports found under '{base_dir}'. "
            "Expected filtered_*, corr_filtered_*, or feature_matrix.csv + labels.csv "
            "(from notebooks 09/10)."
        )

    @classmethod
    def from_feature_exports(
        cls,
        base_dir: str = ".",
        matrix_csv: Optional[str] = None,
        labels_csv: Optional[str] = None,
        reference_pkl: Optional[str] = None,
        prefer: Optional[Tuple[str, ...]] = None,
    ):
        """
        Build a FeatureClassification from feature-selection CSV/pickle exports.

        Makes notebook 11a independent of a live ``fs`` object from notebooks 09/10.
        """
        if matrix_csv is None or labels_csv is None:
            paths = cls.resolve_feature_export_paths(base_dir=base_dir, prefer=prefer)
            matrix_csv = matrix_csv or paths["matrix_csv"]
            labels_csv = labels_csv or paths["labels_csv"]
            if reference_pkl is None:
                reference_pkl = paths["reference_pkl"]
            stage = paths["stage"]
        else:
            stage = "custom"

        feature_df = pd.read_csv(matrix_csv, index_col=0)
        structure_names = list(feature_df.index)
        feature_matrix = feature_df.values.astype(float)

        unique_pairs = cls._parse_pair_columns(feature_df.columns)
        fully_conserved = []
        if reference_pkl and os.path.isfile(reference_pkl):
            with open(reference_pkl, "rb") as f:
                ref = pickle.load(f)
            fully_conserved = ref.get("fully_conserved", []) or []
            if not unique_pairs:
                ref_pairs = ref.get("unique_pairs", []) or []
                if len(ref_pairs) == feature_matrix.shape[1]:
                    unique_pairs = list(ref_pairs)
        if not unique_pairs:
            # Last resort: keep column labels as pseudo-pairs so feature_names stay usable
            unique_pairs = [(c, c) for c in feature_df.columns]

        labels_df = pd.read_csv(labels_csv)
        if "label" not in labels_df.columns:
            raise KeyError(f"{labels_csv} must contain a 'label' column")
        if "structure" in labels_df.columns:
            label_map = dict(zip(labels_df["structure"].astype(str), labels_df["label"].values))
            missing = [s for s in structure_names if s not in label_map]
            if missing:
                raise KeyError(
                    f"{len(missing)} structures in {matrix_csv} missing from {labels_csv}. "
                    f"Examples: {missing[:5]}"
                )
            labels = np.array([label_map[s] for s in structure_names])
        else:
            labels = np.asarray(labels_df["label"].values)
            if len(labels) != len(structure_names):
                raise ValueError(
                    f"Label count ({len(labels)}) does not match matrix rows ({len(structure_names)})"
                )

        obj = cls(
            feature_matrix=feature_matrix,
            labels=labels,
            unique_pairs=unique_pairs,
            fully_conserved=fully_conserved,
            structure_names=structure_names,
        )
        obj.export_stage = stage
        obj.export_paths = {
            "matrix_csv": matrix_csv,
            "labels_csv": labels_csv,
            "reference_pkl": reference_pkl,
        }
        print(f"Loaded {stage} feature exports:")
        print(f"  matrix: {matrix_csv}")
        print(f"  labels: {labels_csv}")
        print(f"  reference: {reference_pkl or '(none)'}")
        return obj
        
    def split_data(self, train_size=0.9, random_state=42, groups=None):
        """
        Split data into training and test sets.
        
        Also stores ``train_idx`` / ``test_idx`` and a per-sample
        ``split_labels`` array (``"training"`` / ``"validation"``) for
        distribution and Wasserstein/KL analyses.
        
        Args:
            train_size: Fraction of data for training (default 0.9)
            random_state: Random seed for reproducibility (default 42)
            groups: Optional group labels (length n_samples). If provided, uses
                ``GroupShuffleSplit`` so no group appears in both train and test.
        """
        print("="*60)
        print("SPLITTING DATA")
        print("="*60)

        n = len(self.labels)
        if groups is None:
            indices = np.arange(n)
            (
                self.train_set,
                self.test_set,
                self.train_class,
                self.test_class,
                self.train_idx,
                self.test_idx,
            ) = train_test_split(
                self.feature_matrix,
                self.labels,
                indices,
                train_size=train_size,
                random_state=random_state,
            )
        else:
            groups = np.asarray(groups)
            if groups.shape[0] != n:
                raise ValueError(
                    f"groups must have length n_samples={n}; got {groups.shape[0]}"
                )
            splitter = GroupShuffleSplit(
                n_splits=1, train_size=train_size, random_state=random_state
            )
            self.train_idx, self.test_idx = next(
                splitter.split(self.feature_matrix, self.labels, groups=groups)
            )
            self.train_set = self.feature_matrix[self.train_idx]
            self.test_set = self.feature_matrix[self.test_idx]
            self.train_class = np.asarray(self.labels)[self.train_idx]
            self.test_class = np.asarray(self.labels)[self.test_idx]

        self.split_labels = np.empty(n, dtype=object)
        self.split_labels[self.train_idx] = "training"
        self.split_labels[self.test_idx] = "validation"
        
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
        balanced_accuracy = balanced_accuracy_score(self.test_class, self.predictions)
        precision = precision_score(self.test_class, self.predictions, average='weighted')
        recall = recall_score(self.test_class, self.predictions, average='weighted')
        
        success = np.sum((self.predictions - self.test_class) == 0)
        percent = float(success) / len(self.test_class) * 100
        
        if show_metrics:
            print(f"\n✅ Test Set Accuracy: {percent:.2f}%")
            print(f"   Balanced accuracy: {balanced_accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   Correct predictions: {success}/{len(self.test_class)}")
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
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

    # ------------------------------------------------------------------
    # Split-violin distributions and Wasserstein / KL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_half_histogram_violin(ax, xc, widths, edges, color, sign):
        """One side of a histogram-density violin (no KDE). sign=-1 left, +1 right."""
        widths = np.asarray(widths, dtype=float)
        n = len(widths)
        if n == 0 or not np.any(np.isfinite(widths)):
            return
        xs = [xc]
        ys = [float(edges[0])]
        for i in range(n):
            w = float(widths[i])
            xs.extend([xc + sign * w, xc + sign * w])
            ys.extend([float(edges[i]), float(edges[i + 1])])
        xs.append(xc)
        ys.append(float(edges[n]))
        ax.fill(xs, ys, color=color, alpha=0.78, linewidth=0.6, edgecolor=color)

    @staticmethod
    def _quartile_lines_split(ax, xc, vals, side, halfwidth, palette_train, palette_val, hue_order):
        """Quartile reference lines confined to train (left) or validation (right) half."""
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            return
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])
        if side == hue_order[0]:
            x0, x1 = xc - float(halfwidth), xc
            qc = palette_train
        else:
            x0, x1 = xc, xc + float(halfwidth)
            qc = palette_val
        ax.plot([x0, x1], [q2, q2], color=qc, ls="-", lw=1.1, zorder=3)
        ax.plot([x0, x1], [q1, q1], color=qc, ls=":", lw=0.85, alpha=0.9, zorder=3)
        ax.plot([x0, x1], [q3, q3], color=qc, ls=":", lw=0.85, alpha=0.9, zorder=3)

    @staticmethod
    def _plot_split_histogram_violins(
        ax,
        *,
        cluster_order,
        feat_values,
        cluster_col,
        split_labels,
        palette,
        hue_order,
        n_bins=24,
        max_halfwidth=0.38,
        show_quartiles=True,
    ):
        """Split histogram-density violins per category on x (no KDE)."""
        feat_values = np.asarray(feat_values, dtype=float)
        ax.set_xticks(np.arange(len(cluster_order)))
        ax.set_xticklabels(cluster_order)
        c_tr, c_va = palette[hue_order[0]], palette[hue_order[1]]

        for ci, clabel in enumerate(cluster_order):
            xc = float(ci)
            m_c = cluster_col == clabel
            vt = feat_values[m_c & (split_labels == hue_order[0])]
            vv = feat_values[m_c & (split_labels == hue_order[1])]
            vt = vt[np.isfinite(vt)]
            vv = vv[np.isfinite(vv)]
            parts = [p for p in (vt, vv) if len(p) > 0]
            if not parts:
                continue
            lo = float(min(p.min() for p in parts))
            hi = float(max(p.max() for p in parts))
            if hi <= lo:
                hi = lo + 1e-9
            edges = np.linspace(lo, hi, int(n_bins) + 1)
            ht, _ = np.histogram(vt, bins=edges) if len(vt) else (np.zeros(n_bins), edges)
            hv, _ = np.histogram(vv, bins=edges) if len(vv) else (np.zeros(n_bins), edges)
            scale = float(max(ht.max(), hv.max(), 1))
            w_tr = ht.astype(float) / scale * max_halfwidth
            w_va = hv.astype(float) / scale * max_halfwidth
            FeatureClassification._fill_half_histogram_violin(ax, xc, w_tr, edges, c_tr, -1)
            FeatureClassification._fill_half_histogram_violin(ax, xc, w_va, edges, c_va, +1)
            if show_quartiles:
                FeatureClassification._quartile_lines_split(
                    ax, xc, vt, hue_order[0], max_halfwidth, c_tr, c_va, hue_order
                )
                FeatureClassification._quartile_lines_split(
                    ax, xc, vv, hue_order[1], max_halfwidth, c_tr, c_va, hue_order
                )

        ax.set_xlim(-0.5, len(cluster_order) - 0.5)
        ax.grid(axis="y", alpha=0.3)

    @staticmethod
    def _symmetric_kl_histogram(a: np.ndarray, b: np.ndarray, n_bins: int = 31) -> float:
        """Symmetric KL divergence from histogram density estimates (smoothed)."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0
        bins = np.linspace(lo, hi, n_bins + 1)
        ca, _ = np.histogram(a, bins=bins)
        cb, _ = np.histogram(b, bins=bins)
        eps = 1e-9
        pa = ca.astype(float) + eps
        pb = cb.astype(float) + eps
        pa /= pa.sum()
        pb /= pb.sum()
        kl_pq = float(np.sum(pa * np.log(pa / pb)))
        kl_qp = float(np.sum(pb * np.log(pb / pa)))
        return 0.5 * (kl_pq + kl_qp)

    @staticmethod
    def _impute_feature_matrix(X_df: pd.DataFrame) -> np.ndarray:
        X_imp = X_df.values.astype(float, copy=True)
        med = np.nanmedian(X_imp, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        nan_mask = np.isnan(X_imp)
        if np.any(nan_mask):
            X_imp[nan_mask] = med[np.where(nan_mask)[1]]
        return X_imp

    @staticmethod
    def _normalize_structure_name(name: str) -> str:
        name = str(name)
        for suffix in ("_reconstructed_aligned", "_aligned"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    @staticmethod
    def _load_pca_cluster_labels(pca_cluster_labels_file: str, structure_index) -> np.ndarray:
        pca_labels_df = pd.read_csv(
            pca_cluster_labels_file,
            comment="#",
            names=["ClusterLabel", "PDBCode", "FullName"],
        )
        pca_labels_df["structure_key"] = pca_labels_df["FullName"].map(
            FeatureClassification._normalize_structure_name
        )
        # Also try matching the raw index strings after normalization
        index_keys = pd.Index(
            [FeatureClassification._normalize_structure_name(s) for s in structure_index]
        )
        pca_by_structure = (
            pca_labels_df.drop_duplicates("structure_key", keep="first")
            .set_index("structure_key")
        )
        return pca_by_structure.reindex(index_keys)["ClusterLabel"].values

    @staticmethod
    def _resolve_split_labels(
        *,
        n_samples: int,
        split_labels=None,
        train_idx=None,
        test_idx=None,
    ):
        """Build split_labels from explicit arrays or train/test indices."""
        if split_labels is not None:
            split_labels = np.asarray(split_labels)
            if len(split_labels) != n_samples:
                raise ValueError(
                    f"split_labels length {len(split_labels)} != n_samples {n_samples}"
                )
            train_idx = np.where(split_labels == "training")[0] if train_idx is None else train_idx
            test_idx = np.where(split_labels == "validation")[0] if test_idx is None else test_idx
            return split_labels, np.asarray(train_idx), np.asarray(test_idx)
        if train_idx is None or test_idx is None:
            raise ValueError(
                "Provide split_labels, or both train_idx and test_idx "
                "(from FeatureClassification.split_data)."
            )
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        split_labels = np.empty(n_samples, dtype=object)
        split_labels[train_idx] = "training"
        split_labels[test_idx] = "validation"
        return split_labels, train_idx, test_idx

    @staticmethod
    def plot_top_feature_distributions_by_label_and_cluster(
        *,
        X_df: pd.DataFrame,
        Xk: np.ndarray,
        feature_labels_all: list,
        gini_mean: np.ndarray,
        perm_mean: np.ndarray,
        best_h: int = 0,
        best_k: int = 0,
        biological_labels_csv: str = "kincore_bio_labels.csv",
        pca_cluster_labels_file: str = "cluster_labels_my_analysis_hierarchical.txt",
        split_labels=None,
        train_idx=None,
        test_idx=None,
        n_top: int = 20,
        n_hist_bins: int = 24,
        plot_violins: bool = True,
        title_suffix: str = "",
    ) -> dict:
        """Plot top-feature split histogram violins grouped by PCA cluster (x-axis).

        Pass ``split_labels`` (or ``train_idx``/``test_idx``) from a random
        ``train_test_split``; Newick guide-tree splitting is not used.
        """
        sns.set_style("whitegrid")

        bio_labels_df = pd.read_csv(biological_labels_csv)
        if "structure" in bio_labels_df.columns:
            bio_labels_df = bio_labels_df.set_index("structure")
            bio_labels_df = bio_labels_df.reindex(X_df.index)
        bio_labels = bio_labels_df["label"].values

        cluster_binary = FeatureClassification._load_pca_cluster_labels(
            pca_cluster_labels_file, X_df.index
        )
        valid_mask = ~pd.isna(cluster_binary)

        cluster_label_matrix = []
        for cid in np.unique(cluster_binary[valid_mask]):
            mask = (cluster_binary == cid) & valid_mask
            n_total = int(np.sum(mask))
            n_active = int(np.sum(bio_labels[mask] == 1))
            n_inactive = int(np.sum(bio_labels[mask] == 0))
            cluster_label_matrix.append(
                {
                    "pca_cluster_id": int(cid),
                    "n_total": n_total,
                    "n_active": n_active,
                    "n_inactive": n_inactive,
                    "pct_active": (100 * n_active / n_total) if n_total > 0 else 0.0,
                }
            )
        cluster_stats = pd.DataFrame(cluster_label_matrix)
        print("\nPCA Cluster composition:")
        print(cluster_stats)

        split_labels, train_idx, test_idx = FeatureClassification._resolve_split_labels(
            n_samples=Xk.shape[0],
            split_labels=split_labels,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        palette = {"training": "#66c2a5", "validation": "#fc8d62"}
        hue_order = ["training", "validation"]
        cluster_col = np.array([f"Cluster {int(c)}" for c in cluster_binary[valid_mask]])
        cluster_order = sorted(np.unique(cluster_col).tolist())
        title_extra = title_suffix or f"(n_features={best_k})"

        def plot_feature_split_violin(feature_indices, importance_type):
            top_n = min(n_top, len(feature_indices))
            top_indices = feature_indices[:top_n]
            for rank_pos, feat_idx in enumerate(top_indices):
                feat_values = Xk[:, feat_idx]
                fig, ax = plt.subplots(figsize=(5, 4))
                FeatureClassification._plot_split_histogram_violins(
                    ax,
                    cluster_order=cluster_order,
                    feat_values=feat_values[valid_mask],
                    cluster_col=cluster_col,
                    split_labels=split_labels[valid_mask],
                    palette=palette,
                    hue_order=hue_order,
                    n_bins=n_hist_bins,
                    max_halfwidth=0.38,
                    show_quartiles=True,
                )
                feature_label = feature_labels_all[feat_idx]
                ax.set_title(
                    f"Rank {rank_pos + 1}  |  {feature_label}\n"
                    f"{importance_type}  {title_extra}",
                    fontsize=9,
                    fontweight="bold",
                )
                ax.set_xlabel("PCA Cluster", fontsize=9)
                ax.set_ylabel("Distance (Å)", fontsize=9)
                ax.tick_params(axis="both", labelsize=8)
                leg_handles = [
                    Patch(facecolor=palette["training"], edgecolor=palette["training"], alpha=0.78, label="training"),
                    Patch(facecolor=palette["validation"], edgecolor=palette["validation"], alpha=0.78, label="validation"),
                ]
                ax.legend(handles=leg_handles, title="Data split", title_fontsize=8, fontsize=8, loc="upper right")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()

        if plot_violins:
            gini_order = np.argsort(gini_mean)[::-1]
            perm_order = np.argsort(perm_mean)[::-1]
            print(f"\nPlotting top {n_top} features by MDI (Gini) Importance …")
            plot_feature_split_violin(gini_order, "MDI (Gini) Importance")
            print(f"\nPlotting top {n_top} features by Permutation Importance …")
            plot_feature_split_violin(perm_order, "Permutation Importance")
            print("\n✅ ALL SPLIT-VIOLIN DISTRIBUTION PLOTS COMPLETE (PCA cluster grouping)")

        return {
            "cluster_stats": cluster_stats,
            "valid_mask": valid_mask,
            "cluster_binary": cluster_binary,
            "bio_labels": bio_labels,
            "split_labels": split_labels,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    @staticmethod
    def plot_top_feature_distributions_by_activation(
        *,
        X_df: pd.DataFrame,
        Xk: np.ndarray,
        feature_labels_all: list,
        gini_mean: np.ndarray,
        perm_mean: np.ndarray,
        best_h: int = 0,
        best_k: int = 0,
        biological_labels_csv: str = "kincore_bio_labels.csv",
        split_labels=None,
        train_idx=None,
        test_idx=None,
        n_top: int = 20,
        n_hist_bins: int = 24,
        plot_violins: bool = True,
        title_suffix: str = "",
    ) -> dict:
        """Plot top-feature split histogram violins grouped by KinCore active/inactive."""
        sns.set_style("whitegrid")

        bio_labels_df = pd.read_csv(biological_labels_csv)
        if "structure" in bio_labels_df.columns:
            bio_labels_df = bio_labels_df.set_index("structure")
            bio_labels_df = bio_labels_df.reindex(X_df.index)
        bio_labels = bio_labels_df["label"].values.astype(float)
        valid_mask = np.isfinite(bio_labels) & np.isin(bio_labels, [0.0, 1.0])

        split_labels, train_idx, test_idx = FeatureClassification._resolve_split_labels(
            n_samples=Xk.shape[0],
            split_labels=split_labels,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        palette = {"training": "#66c2a5", "validation": "#fc8d62"}
        hue_order = ["training", "validation"]
        activation_col = np.array(
            ["Inactive" if int(v) == 0 else "Active" for v in bio_labels[valid_mask]]
        )
        activation_order = ["Inactive", "Active"]
        title_extra = title_suffix or f"(n_features={best_k})"

        def plot_feature_split_violin(feature_indices, importance_type):
            top_n = min(n_top, len(feature_indices))
            top_indices = feature_indices[:top_n]
            for rank_pos, feat_idx in enumerate(top_indices):
                feat_values = Xk[:, feat_idx]
                fig, ax = plt.subplots(figsize=(5, 4))
                FeatureClassification._plot_split_histogram_violins(
                    ax,
                    cluster_order=activation_order,
                    feat_values=feat_values[valid_mask],
                    cluster_col=activation_col,
                    split_labels=split_labels[valid_mask],
                    palette=palette,
                    hue_order=hue_order,
                    n_bins=n_hist_bins,
                    max_halfwidth=0.38,
                    show_quartiles=True,
                )
                feature_label = feature_labels_all[feat_idx]
                ax.set_title(
                    f"Rank {rank_pos + 1}  |  {feature_label}\n"
                    f"{importance_type}  {title_extra}",
                    fontsize=9,
                    fontweight="bold",
                )
                ax.set_xlabel("KinCore activation state", fontsize=9)
                ax.set_ylabel("Distance (Å)", fontsize=9)
                ax.tick_params(axis="both", labelsize=8)
                leg_handles = [
                    Patch(facecolor=palette["training"], edgecolor=palette["training"], alpha=0.78, label="training"),
                    Patch(facecolor=palette["validation"], edgecolor=palette["validation"], alpha=0.78, label="validation"),
                ]
                ax.legend(handles=leg_handles, title="Data split", title_fontsize=8, fontsize=8, loc="upper right")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()

        if plot_violins:
            gini_order = np.argsort(gini_mean)[::-1]
            perm_order = np.argsort(perm_mean)[::-1]
            print(f"\nPlotting top {n_top} features by MDI (Gini) — active vs inactive …")
            plot_feature_split_violin(gini_order, "MDI (Gini) Importance")
            print(f"\nPlotting top {n_top} features by Permutation Importance — active vs inactive …")
            plot_feature_split_violin(perm_order, "Permutation Importance")
            print("\n✅ ALL SPLIT-VIOLIN DISTRIBUTION PLOTS COMPLETE (activation grouping)")

        return {
            "valid_mask": valid_mask,
            "bio_labels": bio_labels,
            "split_labels": split_labels,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    @staticmethod
    def _wasserstein_kl_rows_for_binary_groups(
        X_imp: np.ndarray,
        X_df: pd.DataFrame,
        ordered_cols: np.ndarray,
        rank_col: str,
        split_labels: np.ndarray,
        valid_mask: np.ndarray,
        group_binary: np.ndarray,
        class0: int,
        class1: int,
        n_bins_kl: int,
    ) -> pd.DataFrame:
        is_train = split_labels == "training"
        is_val = split_labels == "validation"
        rows = []
        x_rank = np.arange(1, len(ordered_cols) + 1)
        for j_local, feat_col in enumerate(ordered_cols):
            col = X_imp[:, feat_col]
            row = {
                rank_col: x_rank[j_local],
                "matrix_column_idx": int(feat_col),
                "feature_name": str(X_df.columns[feat_col]),
            }
            for split_name, split_m in ("train", is_train), ("val", is_val):
                m = split_m & valid_mask
                m0 = m & (group_binary == class0)
                m1 = m & (group_binary == class1)
                v0 = col[m0]
                v1 = col[m1]
                if len(v0) < 2 or len(v1) < 2:
                    row[f"wasserstein_{split_name}"] = float("nan")
                    row[f"kl_symm_{split_name}"] = float("nan")
                else:
                    row[f"wasserstein_{split_name}"] = wasserstein_distance(v0, v1)
                    row[f"kl_symm_{split_name}"] = FeatureClassification._symmetric_kl_histogram(
                        v0, v1, n_bins=n_bins_kl
                    )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def compute_cluster0_vs_cluster1_wasserstein_kl_for_rf_features(
        *,
        X_df: pd.DataFrame,
        distribution_plot_results: dict,
        importance: np.ndarray,
        rank_col: str = "mdi_rank",
        n_features: int = None,
        n_bins_kl: int = 31,
    ) -> pd.DataFrame:
        """Compare PCA cluster 0 vs 1 for RF-importance-ranked features."""
        split_labels = np.asarray(distribution_plot_results["split_labels"])
        valid_mask = np.asarray(distribution_plot_results["valid_mask"], dtype=bool)
        cluster_binary = np.asarray(distribution_plot_results["cluster_binary"])

        X_imp = FeatureClassification._impute_feature_matrix(X_df)
        importance = np.nan_to_num(np.asarray(importance, dtype=float), nan=0.0)
        rank_desc = np.argsort(importance)[::-1]
        n_use = len(rank_desc) if n_features is None else min(int(n_features), len(rank_desc))
        ordered_cols = rank_desc[:n_use]

        c_valid = cluster_binary[valid_mask]
        cluster_ids = np.sort(np.unique(c_valid[~pd.isna(c_valid)]).astype(int))
        if len(cluster_ids) < 2:
            raise ValueError(f"Need at least two PCA cluster IDs; found {cluster_ids!r}")
        c0, c1 = int(cluster_ids[0]), int(cluster_ids[1])

        return FeatureClassification._wasserstein_kl_rows_for_binary_groups(
            X_imp, X_df, ordered_cols, rank_col, split_labels, valid_mask,
            cluster_binary, c0, c1, n_bins_kl,
        )

    @staticmethod
    def compute_active_vs_inactive_wasserstein_kl_for_rf_features(
        *,
        X_df: pd.DataFrame,
        distribution_plot_results: dict,
        importance: np.ndarray,
        rank_col: str = "mdi_rank",
        n_features: int = None,
        n_bins_kl: int = 31,
    ) -> pd.DataFrame:
        """Compare KinCore inactive (0) vs active (1) for RF-importance-ranked features."""
        split_labels = np.asarray(distribution_plot_results["split_labels"])
        valid_mask = np.asarray(distribution_plot_results["valid_mask"], dtype=bool)
        bio_labels = np.asarray(distribution_plot_results["bio_labels"], dtype=float)

        X_imp = FeatureClassification._impute_feature_matrix(X_df)
        importance = np.nan_to_num(np.asarray(importance, dtype=float), nan=0.0)
        rank_desc = np.argsort(importance)[::-1]
        n_use = len(rank_desc) if n_features is None else min(int(n_features), len(rank_desc))
        ordered_cols = rank_desc[:n_use]

        act_valid = valid_mask & np.isfinite(bio_labels) & np.isin(bio_labels, [0.0, 1.0])
        if np.sum(bio_labels[act_valid] == 0) < 2 or np.sum(bio_labels[act_valid] == 1) < 2:
            raise ValueError("Need at least two structures per KinCore class (active/inactive).")

        return FeatureClassification._wasserstein_kl_rows_for_binary_groups(
            X_imp, X_df, ordered_cols, rank_col, split_labels, act_valid,
            bio_labels, 0, 1, n_bins_kl,
        )

    @staticmethod
    def plot_cluster0_vs_cluster1_wasserstein_kl(
        metrics_df: pd.DataFrame,
        *,
        best_h: int = 0,
        best_k: int = 0,
        n_pool_features: int = 300,
        rank_col: str = "mdi_rank",
        rank_xlabel: str = "RF feature rank (1 = highest importance)",
        pool_caption: str = "RF-ranked",
        comparison_label: str = "cluster 0 vs 1",
        split_caption: str = "random train/test split",
        save_path=None,
    ) -> None:
        """Line plots: binary-class separation metrics vs feature rank (train vs val)."""
        if rank_col not in metrics_df.columns:
            raise KeyError(f"metrics_df missing column {rank_col!r}")
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        x = metrics_df[rank_col].values

        ax = axes[0]
        ax.plot(x, metrics_df["wasserstein_train"], label="Train", color="#66c2a5", linewidth=1.2, alpha=0.9)
        ax.plot(x, metrics_df["wasserstein_val"], label="Validation", color="#fc8d62", linewidth=1.2, alpha=0.9)
        ax.set_ylabel(f"Wasserstein distance\n({comparison_label})", fontsize=10)
        ax.set_title(
            f"Distribution separation ({comparison_label}) — top {n_pool_features} {pool_caption} features "
            f"({split_caption}; classifier uses {best_k} features)",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x, metrics_df["kl_symm_train"], label="Train", color="#66c2a5", linewidth=1.2, alpha=0.9)
        ax.plot(x, metrics_df["kl_symm_val"], label="Validation", color="#fc8d62", linewidth=1.2, alpha=0.9)
        ax.set_xlabel(rank_xlabel, fontsize=10)
        ax.set_ylabel(f"Sym. KL (histogram)\n({comparison_label})", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def _normalize_pair_key(feature_name):
        """Canonical residue-pair key ``\"i-j\"`` (sorted ints), or ``None`` if not a pair."""
        s = str(feature_name).strip()
        for sep in ("-", "_"):
            if sep not in s:
                continue
            parts = s.split(sep)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                a, b = int(parts[0]), int(parts[1])
                lo, hi = (a, b) if a <= b else (b, a)
                return f"{lo}-{hi}"
        return None

    @staticmethod
    def plot_sc_vs_ca_wkl_overlay_by_rank(
        metrics_sc: pd.DataFrame,
        metrics_ca: pd.DataFrame,
        *,
        rank_col: str = "mdi_rank",
        comparison_label: str = "cluster 0 vs 1",
        rank_xlabel: str = "RF feature rank (1 = highest importance)",
        n_top=None,
        save_path=None,
    ) -> None:
        """Overlay SC and Cα W/KL vs each modality's own RF rank (train + val)."""
        for name, df in ("side-chain", metrics_sc), ("Cα", metrics_ca):
            if rank_col not in df.columns:
                raise KeyError(f"{name} metrics missing column {rank_col!r}")

        sc = metrics_sc.sort_values(rank_col).reset_index(drop=True)
        ca = metrics_ca.sort_values(rank_col).reset_index(drop=True)
        if n_top is not None:
            n_use = int(n_top)
            sc = sc.iloc[:n_use]
            ca = ca.iloc[:n_use]

        color_sc, color_ca = "#1b9e77", "#7570b3"
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        def _plot_metric(ax, y_sc_train, y_sc_val, y_ca_train, y_ca_val, ylabel):
            ax.plot(sc[rank_col], y_sc_train, color=color_sc, linestyle="-", linewidth=1.3,
                    alpha=0.9, label="SC train")
            ax.plot(sc[rank_col], y_sc_val, color=color_sc, linestyle="--", linewidth=1.3,
                    alpha=0.9, label="SC validation")
            ax.plot(ca[rank_col], y_ca_train, color=color_ca, linestyle="-", linewidth=1.3,
                    alpha=0.9, label="Cα train")
            ax.plot(ca[rank_col], y_ca_val, color=color_ca, linestyle="--", linewidth=1.3,
                    alpha=0.9, label="Cα validation")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        _plot_metric(
            axes[0],
            sc["wasserstein_train"], sc["wasserstein_val"],
            ca["wasserstein_train"], ca["wasserstein_val"],
            f"Wasserstein distance\n({comparison_label})",
        )
        axes[0].set_title(
            f"SC vs Cα separation vs RF rank ({comparison_label})",
            fontsize=11,
            fontweight="bold",
        )
        _plot_metric(
            axes[1],
            sc["kl_symm_train"], sc["kl_symm_val"],
            ca["kl_symm_train"], ca["kl_symm_val"],
            f"Sym. KL (histogram)\n({comparison_label})",
        )
        axes[1].set_xlabel(rank_xlabel, fontsize=10)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

    @staticmethod
    def align_shared_pair_wkl(
        metrics_sc: pd.DataFrame,
        metrics_ca: pd.DataFrame,
        *,
        rank_col: str = "mdi_rank",
    ) -> pd.DataFrame:
        """Inner-join SC and Cα W/KL rows on canonical residue-pair keys."""
        sc = metrics_sc.copy()
        ca = metrics_ca.copy()
        sc["pair"] = sc["feature_name"].map(FeatureClassification._normalize_pair_key)
        ca["pair"] = ca["feature_name"].map(FeatureClassification._normalize_pair_key)
        sc = sc.dropna(subset=["pair"]).drop_duplicates("pair", keep="first")
        ca = ca.dropna(subset=["pair"]).drop_duplicates("pair", keep="first")

        metric_cols = [
            "wasserstein_train", "wasserstein_val",
            "kl_symm_train", "kl_symm_val",
        ]
        sc_keep = ["pair", "feature_name", rank_col] + [
            c for c in metric_cols if c in sc.columns
        ]
        ca_keep = ["pair", "feature_name", rank_col] + [
            c for c in metric_cols if c in ca.columns
        ]
        sc_sub = sc[sc_keep].rename(
            columns={
                "feature_name": "feature_name_sc",
                rank_col: f"{rank_col}_sc",
                **{c: f"{c}_sc" for c in metric_cols if c in sc.columns},
            }
        )
        ca_sub = ca[ca_keep].rename(
            columns={
                "feature_name": "feature_name_ca",
                rank_col: f"{rank_col}_ca",
                **{c: f"{c}_ca" for c in metric_cols if c in ca.columns},
            }
        )
        aligned = sc_sub.merge(ca_sub, on="pair", how="inner")
        if aligned.empty:
            return aligned

        rank_sc = aligned[f"{rank_col}_sc"].astype(float)
        rank_ca = aligned[f"{rank_col}_ca"].astype(float)
        aligned["_rank_mean"] = (rank_sc + rank_ca) / 2.0
        aligned = aligned.sort_values("_rank_mean", ascending=True).reset_index(drop=True)
        aligned = aligned.drop(columns=["_rank_mean"])
        return aligned

    @staticmethod
    def plot_sc_vs_ca_wkl_shared_pairs(
        aligned_df: pd.DataFrame,
        *,
        comparison_label: str = "cluster 0 vs 1",
        save_path=None,
    ) -> None:
        """Overlay SC vs Cα W/KL for shared residue pairs (order from align)."""
        if aligned_df is None or len(aligned_df) == 0:
            print(
                f"⚠️  No shared residue pairs for {comparison_label}; "
                "skipping shared-pair W/KL plot."
            )
            return

        x = np.arange(1, len(aligned_df) + 1)
        color_sc, color_ca = "#1b9e77", "#7570b3"
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        def _plot_metric(ax, col_prefix, ylabel):
            ax.plot(
                x, aligned_df[f"{col_prefix}_train_sc"], color=color_sc, linestyle="-",
                linewidth=1.3, alpha=0.9, label="SC train",
            )
            ax.plot(
                x, aligned_df[f"{col_prefix}_val_sc"], color=color_sc, linestyle="--",
                linewidth=1.3, alpha=0.9, label="SC validation",
            )
            ax.plot(
                x, aligned_df[f"{col_prefix}_train_ca"], color=color_ca, linestyle="-",
                linewidth=1.3, alpha=0.9, label="Cα train",
            )
            ax.plot(
                x, aligned_df[f"{col_prefix}_val_ca"], color=color_ca, linestyle="--",
                linewidth=1.3, alpha=0.9, label="Cα validation",
            )
            ax.set_ylabel(ylabel, fontsize=10)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        _plot_metric(
            axes[0], "wasserstein",
            f"Wasserstein distance\n({comparison_label})",
        )
        axes[0].set_title(
            f"SC vs Cα on shared residue pairs ({comparison_label}; n={len(aligned_df)})",
            fontsize=11,
            fontweight="bold",
        )
        _plot_metric(
            axes[1], "kl_symm",
            f"Sym. KL (histogram)\n({comparison_label})",
        )
        axes[1].set_xlabel(
            "Shared pair index (sorted by mean RF rank)",
            fontsize=10,
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


    @staticmethod
    def run_anova_sum_parameter_scan_with_guide_tree(
        *,
        reference_pickle: Optional[str] = None,
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
        output_dir: str = "Results/anova_scan",
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
        if reference_pickle and os.path.isfile(reference_pickle):
            fs_scan.load_results(reference_pickle)
        elif reference_pickle:
            print(f"⚠️  reference_pickle not found ({reference_pickle}); parsing pairs from columns")

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

        if len(getattr(fs_scan, "unique_pairs", []) or []) != X.shape[1]:
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

        y_arr = np.asarray(y)
        label_valid = ~pd.isna(y_arr)
        f_scores, _ = f_classif(X_imp[label_valid], y_arr[label_valid])
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

        # ── Save results ──────────────────────────────────────────────────────
        from pathlib import Path as _Path
        out_path = _Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_path / "metrics_by_height_k.csv", index=False)

        best_df = res_df.sort_values(["accuracy_mean", "precision_mean"], ascending=False).head(10)
        best_df.to_csv(out_path / "best_combinations.csv", index=False)

        ckpt_dir = out_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "best_h": best_h,
            "best_k": best_k,
            "mdi_vals_reps": [v.tolist() for v in gini_vals_reps],
            "perm_vals_reps": [v.tolist() for v in perm_vals_reps],
            "feature_labels_all": feature_labels_all,
        }
        with open(ckpt_dir / f"h{best_h}_k{best_k}.json", "w") as fh:
            json.dump(checkpoint_data, fh)
        print(f"\nResults saved to {output_dir}/")

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
        checkpoint_dir: str = "Results/anova_scan/checkpoints",
        best_combinations_csv: str = "Results/anova_scan/best_combinations.csv",
        reference_pickle: Optional[str] = None,
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
        if reference_pickle and os.path.isfile(reference_pickle):
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

        if len(getattr(fs_scan, "unique_pairs", []) or []) != X.shape[1]:
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

        y_arr = np.asarray(y)
        label_valid = ~pd.isna(y_arr)
        f_scores, _ = f_classif(X_imp[label_valid], y_arr[label_valid])
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

        metrics_csv = os.path.join(os.path.dirname(best_combinations_csv), "metrics_by_height_k.csv")
        res_df = pd.read_csv(metrics_csv) if os.path.isfile(metrics_csv) else None

        return {
            "best_h": best_h,
            "best_k": best_k,
            "X_df": X_df,
            "y": y,
            "Xk": Xk,
            "feature_labels_all": feature_labels_all,
            "gini_mean": gini_mean,
            "perm_mean": perm_mean,
            "res_df": res_df,
        }

    @staticmethod
    def compute_cluster0_vs_cluster1_wasserstein_kl_for_anova_features(
        *,
        X_df: pd.DataFrame,
        y: np.ndarray,
        distribution_plot_results: dict,
        n_anova_features: int = 300,
        n_bins_kl: int = 31,
    ) -> pd.DataFrame:
        """
        For each of the top ``n_anova_features`` columns (ANOVA F-score rank on the full matrix),
        compare PCA cluster 0 vs cluster 1 using Wasserstein distance and symmetric histogram KL.

        Train and validation metrics are computed separately using ``split_labels`` from
        ``distribution_plot_results`` (same guide-tree split as the violin plots).

        Requires at least two distinct cluster IDs among structures with valid PCA labels.

        Returns:
            DataFrame with ANOVA rank, column name, Wasserstein/KL for train and validation.
        """
        from sklearn.feature_selection import f_classif

        split_labels = np.asarray(distribution_plot_results["split_labels"])
        valid_mask = np.asarray(distribution_plot_results["valid_mask"], dtype=bool)
        cluster_binary = np.asarray(distribution_plot_results["cluster_binary"])

        X_imp = FeatureClassification._impute_feature_matrix(X_df)

        y_arr = np.asarray(y)
        label_valid = ~pd.isna(y_arr)
        f_scores, _ = f_classif(X_imp[label_valid], y_arr[label_valid])
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank_desc = np.argsort(f_scores)[::-1]
        n_use = min(n_anova_features, X_imp.shape[1])
        ordered_cols = rank_desc[:n_use]

        is_train = split_labels == "training"
        is_val = split_labels == "validation"

        c_valid = cluster_binary[valid_mask]
        cluster_ids = np.unique(c_valid[~pd.isna(c_valid)])
        cluster_ids = np.sort(cluster_ids.astype(int))
        if len(cluster_ids) < 2:
            raise ValueError(
                "Need at least two PCA cluster IDs among valid structures; "
                f"found {cluster_ids!r}"
            )
        c0, c1 = int(cluster_ids[0]), int(cluster_ids[1])

        rows = []
        x_rank = np.arange(1, n_use + 1)
        for j_local, feat_col in enumerate(ordered_cols):
            col = X_imp[:, feat_col]
            row = {
                "anova_rank": x_rank[j_local],
                "matrix_column_idx": int(feat_col),
                "feature_name": str(X_df.columns[feat_col]),
            }
            for split_name, split_m in ("train", is_train), ("val", is_val):
                m = split_m & valid_mask
                m0 = m & (cluster_binary == c0)
                m1 = m & (cluster_binary == c1)
                v0 = col[m0]
                v1 = col[m1]
                if len(v0) < 2 or len(v1) < 2:
                    row[f"wasserstein_{split_name}"] = float("nan")
                    row[f"kl_symm_{split_name}"] = float("nan")
                else:
                    row[f"wasserstein_{split_name}"] = wasserstein_distance(v0, v1)
                    row[f"kl_symm_{split_name}"] = FeatureClassification._symmetric_kl_histogram(
                        v0, v1, n_bins=n_bins_kl
                    )
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def add_balanced_accuracy_to_mi_scan_metrics(
        metrics_df: pd.DataFrame,
        *,
        reference_pickle: Optional[str] = None,
        feature_matrix_csv: str,
        labels_csv: str,
        newick_path: str,
        n_repeats: int = 3,
        repeat_seeds: Optional[list] = None,
        train_size: float = 0.9,
        n_estimators: int = 100,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Add balanced_accuracy_mean/std columns to an MI scan metrics DataFrame.

        Re-runs the classifier for each (height, n_features) row using the same
        MI ranking, guide-tree groups, and random seeds as the original scan.
        """
        from pathlib import Path
        from sklearn.feature_selection import mutual_info_classif
        from workflow.feature_selection import FeatureSelection
        from workflow.guide_tree_clusters import parse_newick, cluster_leaves_by_height_cut

        if "balanced_accuracy_mean" in metrics_df.columns:
            return metrics_df.copy()

        if repeat_seeds is None:
            repeat_seeds = [42 + i for i in range(int(n_repeats))]

        fs_scan = FeatureSelection(dfg_index=145, ape_index=174, conservation_threshold=0.97)
        if reference_pickle and os.path.isfile(reference_pickle):
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

        if len(getattr(fs_scan, "unique_pairs", []) or []) != X.shape[1]:
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

        if verbose:
            print("Computing Mutual Information scores…")
        mi_scores = mutual_info_classif(X_imp, y, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(mi_scores)[::-1]

        tree_root = parse_newick(Path(newick_path).read_text())
        structure_names = list(X_df.index)
        heights_needed = sorted({int(h) for h in metrics_df["height"].unique()})

        height_to_leaf_to_cluster = {}
        for h in heights_needed:
            leaf_to_cluster, _ = cluster_leaves_by_height_cut(tree_root, cut_height=float(h))
            height_to_leaf_to_cluster[int(h)] = leaf_to_cluster

        combos = metrics_df[["height", "n_features"]].drop_duplicates()
        bacc_by_key = {}
        n_combos = len(combos)
        for i, row in enumerate(combos.itertuples(index=False), start=1):
            h = int(row.height)
            n = int(row.n_features)
            leaf_to_cluster = height_to_leaf_to_cluster[h]
            next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
            groups = []
            for nm in structure_names:
                g = leaf_to_cluster.get(nm)
                if g is None:
                    g = next_missing
                    next_missing += 1
                groups.append(g)

            idx = np.sort(rank[:n])
            Xn = X_imp[:, idx]
            pairs_n = [fs_scan.unique_pairs[i] for i in idx]
            baccs = []
            for seed in repeat_seeds:
                clf = FeatureClassification(
                    feature_matrix=Xn,
                    labels=y,
                    unique_pairs=pairs_n,
                    fully_conserved=fs_scan.fully_conserved,
                    structure_names=fs_scan.structure_names,
                )
                clf.split_data(train_size=train_size, random_state=seed, groups=groups)
                clf.train_model(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
                m = clf.evaluate_model(show_metrics=False)
                baccs.append(float(m["balanced_accuracy"]))

            bacc_by_key[(h, n)] = (
                float(np.mean(baccs)),
                float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
            )
            if verbose and (i == 1 or i % 10 == 0 or i == n_combos):
                print(f"  [{i}/{n_combos}] height={h}, n_features={n} → balanced_accuracy={bacc_by_key[(h, n)][0]:.4f}")

        out = metrics_df.copy()
        out["balanced_accuracy_mean"] = out.apply(
            lambda r: bacc_by_key[(int(r["height"]), int(r["n_features"]))][0], axis=1
        )
        out["balanced_accuracy_std"] = out.apply(
            lambda r: bacc_by_key[(int(r["height"]), int(r["n_features"]))][1], axis=1
        )
        return out

    @staticmethod
    def run_mi_parameter_scan_with_guide_tree(
        *,
        reference_pickle: Optional[str] = None,
        feature_matrix_csv: str = "corr_filtered_feature_matrix.csv",
        labels_csv: str = "corr_filtered_labels.csv",
        newick_path: str = "Results/activation_segments/multi_aligned_foldmason/msa.nw",
        height_step: int = 2,
        height_min: int = 2,
        height_max: int = 0,
        n_features_values: Optional[list] = None,
        n_repeats: int = 3,
        repeat_seeds: Optional[list] = None,
        train_size: float = 0.9,
        n_estimators: int = 100,
        perm_repeats: int = 100,
        show_plots: bool = True,
        output_dir: str = "Results/mi_scan",
    ) -> dict:
        """
        Scan (height, n_features) using Mutual Information-ranked features with
        group-aware splitting by guide-tree clusters.

        Structurally identical to run_anova_sum_parameter_scan_with_guide_tree but
        replaces f_classif with mutual_info_classif for feature ranking.  MI makes
        no distributional assumptions and captures non-linear class separation,
        unlike the ANOVA F-test which assumes normality and equal variance.

        Args:
            reference_pickle:    Path to pickled FeatureSelection reference data.
            feature_matrix_csv:  Path to the correlation-filtered feature matrix CSV.
            labels_csv:          Path to the labels CSV.
            newick_path:         Path to the FoldMason guide-tree Newick file.
            height_step:         Step size for scanning Newick cut heights.
            height_min:          Minimum cut height to scan.
            height_max:          Maximum cut height (0 = auto from tree).
            n_features_values:   List of feature counts to scan.
                                 Defaults to [300, 500, …, 2000] matching k_values default.
            n_repeats:           Number of random seeds to average over.
            repeat_seeds:        Explicit seed list (overrides n_repeats).
            train_size:          Fraction of data used for training.
            n_estimators:        Number of trees in each RandomForest.
            perm_repeats:        Permutation-importance repetitions for the best model.
            show_plots:          Whether to display plots inline.
            output_dir:          Directory for saving results CSV and checkpoint JSON.

        Returns:
            dict with keys: res_df, best_h, best_n, X_df, y, Xk,
                            feature_labels_all, gini_mean, perm_mean, cm_mean, cm_std.
        """
        from pathlib import Path
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.metrics import confusion_matrix as _confusion_matrix
        from workflow.feature_selection import FeatureSelection
        from workflow.guide_tree_clusters import parse_newick, cluster_leaves_by_height_cut

        if repeat_seeds is None:
            repeat_seeds = [42 + i for i in range(int(n_repeats))]
        if n_features_values is None:
            n_features_values = list(range(300, 2001, 200))
            if 2000 not in n_features_values:
                n_features_values.append(2000)

        # ── Load data ────────────────────────────────────────────────────────
        fs_scan = FeatureSelection(dfg_index=145, ape_index=174, conservation_threshold=0.97)
        if reference_pickle and os.path.isfile(reference_pickle):
            fs_scan.load_results(reference_pickle)
        elif reference_pickle:
            print(f"⚠️  reference_pickle not found ({reference_pickle}); parsing pairs from columns")

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

        if len(getattr(fs_scan, "unique_pairs", []) or []) != X.shape[1]:
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

        # ── Impute NaN ───────────────────────────────────────────────────────
        X_imp = X.astype(float, copy=True)
        med = np.nanmedian(X_imp, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        nan_mask = np.isnan(X_imp)
        if np.any(nan_mask):
            X_imp[nan_mask] = med[np.where(nan_mask)[1]]

        # ── MI ranking (replaces f_classif) ──────────────────────────────────
        print("Computing Mutual Information scores…")
        mi_scores = mutual_info_classif(X_imp, y, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(mi_scores)[::-1]

        n_features_values = sorted({int(n) for n in n_features_values if 1 <= int(n) <= X_imp.shape[1]})
        labels_all = np.unique(y)
        position_to_residue = {pos: name for pos, name in (fs_scan.fully_conserved or [])}

        def _feature_labels_for_pairs(pairs):
            out = []
            for (pos1, pos2) in pairs:
                res1 = position_to_residue.get(pos1, f"Unk-{pos1}")
                res2 = position_to_residue.get(pos2, f"Unk-{pos2}")
                out.append(f"{res1}({pos1})-{res2}({pos2})")
            return out

        # ── Guide-tree heights ────────────────────────────────────────────────
        with open(newick_path, "r", encoding="utf-8", errors="ignore") as fh:
            newick_str = fh.read().strip()
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

        # ── Parameter scan ────────────────────────────────────────────────────
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

            for n in n_features_values:
                idx = np.sort(rank[:n])
                Xn = X_imp[:, idx]
                pairs_n = [fs_scan.unique_pairs[i] for i in idx]
                accs = []
                precs = []
                baccs = []
                for seed in repeat_seeds:
                    clf = FeatureClassification(
                        feature_matrix=Xn,
                        labels=y,
                        unique_pairs=pairs_n,
                        fully_conserved=fs_scan.fully_conserved,
                        structure_names=fs_scan.structure_names,
                    )
                    clf.split_data(train_size=train_size, random_state=seed, groups=groups)
                    clf.train_model(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
                    m = clf.evaluate_model(show_metrics=False)
                    accs.append(float(m["accuracy"]))
                    precs.append(float(m["precision"]))
                    baccs.append(float(m["balanced_accuracy"]))

                metric_rows.append(
                    {
                        "height": int(h),
                        "n_features": int(n),
                        "accuracy_mean": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                        "balanced_accuracy_mean": float(np.mean(baccs)),
                        "balanced_accuracy_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
                        "precision_mean": float(np.mean(precs)),
                        "precision_std": float(np.std(precs, ddof=1)) if len(precs) > 1 else 0.0,
                    }
                )

        res_df = pd.DataFrame(metric_rows).sort_values(["height", "n_features"])
        try:
            from IPython.display import display
            display(res_df)
        except Exception:
            print(res_df.head().to_string(index=False))

        if show_plots:
            pivot_acc = res_df.pivot(index="height", columns="n_features", values="accuracy_mean")
            fig, ax = plt.subplots(figsize=(10.5, 7.5))
            im = ax.imshow(pivot_acc.values, aspect="auto", origin="lower")
            ax.set_title("Accuracy mean — MI feature ranking + hierarchical-tree split")
            ax.set_xlabel("n_features (top-N by MI)")
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
        best_n = int(best["n_features"])
        print(f"\nBest by accuracy_mean: height={best_h}, n_features={best_n}")
        print(best)

        # ── Final model at best (h, n) ────────────────────────────────────────
        leaf_to_cluster, _ = cluster_leaves_by_height_cut(tree_root, cut_height=float(best_h))
        next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
        best_groups = []
        for nm in structure_names:
            g = leaf_to_cluster.get(nm)
            if g is None:
                g = next_missing
                next_missing += 1
            best_groups.append(g)

        idx = np.sort(rank[:best_n])
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
            cms.append(_confusion_matrix(clf.test_class, y_pred, labels=labels_all))
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
            ax.set_title(f"Best confusion matrix (mean ± std), height={best_h}, n_features={best_n}")
            thresh = cm_mean.max() / 2.0 if cm_mean.size else 0.0
            for i in range(cm_mean.shape[0]):
                for j in range(cm_mean.shape[1]):
                    m_val = cm_mean[i, j]
                    s_val = cm_std[i, j]
                    txt_color = "white" if m_val > thresh else "black"
                    ax.text(j, i, f"{m_val:.1f}\n±{s_val:.1f}", ha="center", va="center",
                            color=txt_color, fontsize=10)
            plt.tight_layout()
            plt.show()

        gini_vals_reps = np.stack(gini_vals_reps, axis=0)
        gini_err_reps = np.stack(gini_err_reps, axis=0)
        gini_mean = np.mean(gini_vals_reps, axis=0)
        gini_err_prop = np.sqrt(np.sum(gini_err_reps ** 2, axis=0)) / float(len(repeat_seeds))

        perm_vals_reps = np.stack(perm_vals_reps, axis=0)
        perm_err_reps = np.stack(perm_err_reps, axis=0)
        perm_mean = np.mean(perm_vals_reps, axis=0)
        perm_err_prop = np.sqrt(np.sum(perm_err_reps ** 2, axis=0)) / float(len(repeat_seeds))

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
                title=f"Best Gini/MDI importance (mean ± propagated error), height={best_h}, n_features={best_n}",
                labels=feature_labels_all,
                mean_vals=gini_mean,
                err_vals=gini_err_prop,
                n_top=20,
            )
            _plot_importance_barh(
                title=f"Best Permutation importance (mean ± propagated error), height={best_h}, n_features={best_n}",
                labels=feature_labels_all,
                mean_vals=perm_mean,
                err_vals=perm_err_prop,
                n_top=20,
            )

        # ── Save results ──────────────────────────────────────────────────────
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_path / "metrics_by_height_n.csv", index=False)

        best_df = res_df.sort_values(["accuracy_mean", "precision_mean"], ascending=False).head(10)
        best_df.to_csv(out_path / "best_combinations.csv", index=False)

        ckpt_dir = out_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "best_h": best_h,
            "best_n": best_n,
            "mdi_vals_reps": [v.tolist() for v in gini_vals_reps],
            "perm_vals_reps": [v.tolist() for v in perm_vals_reps],
            "feature_labels_all": feature_labels_all,
        }
        with open(ckpt_dir / f"h{best_h}_n{best_n}.json", "w") as fh:
            json.dump(checkpoint_data, fh)
        print(f"\nResults saved to {output_dir}/")

        return {
            "res_df": res_df,
            "best_h": best_h,
            "best_n": best_n,
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
    def load_mi_scan_results_for_distributions(
        *,
        checkpoint_dir: str = "Results/mi_scan/checkpoints",
        best_combinations_csv: str = "Results/mi_scan/best_combinations.csv",
        reference_pickle: Optional[str] = None,
        feature_matrix_csv: str = "corr_filtered_feature_matrix.csv",
        labels_csv: str = "corr_filtered_labels.csv",
    ) -> dict:
        """
        Load best MI scan (height, n_features), checkpoint importances, and feature matrix slice for plots.
        """
        from sklearn.feature_selection import mutual_info_classif
        from workflow.feature_selection import FeatureSelection

        best_df = pd.read_csv(best_combinations_csv)
        best_row = best_df.iloc[0]
        best_h = int(best_row["height"])
        best_n = int(best_row["n_features"])

        fs_scan = FeatureSelection(dfg_index=145, ape_index=174, conservation_threshold=0.97)
        if reference_pickle and os.path.isfile(reference_pickle):
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

        if len(getattr(fs_scan, "unique_pairs", []) or []) != X.shape[1]:
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

        mi_scores = mutual_info_classif(X_imp, y, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(mi_scores)[::-1]

        idx = np.sort(rank[:best_n])
        Xk = X_imp[:, idx]
        pairs_k = [fs_scan.unique_pairs[i] for i in idx]

        pos_to_residue = {pos: name for pos, name in (fs_scan.fully_conserved or [])}
        feature_labels_all = [
            f"{pos_to_residue.get(pos1, f'Unk-{pos1}')}({pos1})-{pos_to_residue.get(pos2, f'Unk-{pos2}')}({pos2})"
            for (pos1, pos2) in pairs_k
        ]

        checkpoint_file = os.path.join(checkpoint_dir, f"h{best_h}_n{best_n}.json")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)

        mdi_vals_reps = [np.array(v) for v in checkpoint_data["mdi_vals_reps"]]
        perm_vals_reps = [np.array(v) for v in checkpoint_data["perm_vals_reps"]]
        gini_mean = np.mean(np.stack(mdi_vals_reps, axis=0), axis=0)
        perm_mean = np.mean(np.stack(perm_vals_reps, axis=0), axis=0)

        metrics_csv = os.path.join(os.path.dirname(best_combinations_csv), "metrics_by_height_n.csv")
        res_df = pd.read_csv(metrics_csv) if os.path.isfile(metrics_csv) else None

        return {
            "best_h": best_h,
            "best_n": best_n,
            "X_df": X_df,
            "y": y,
            "Xk": Xk,
            "feature_labels_all": feature_labels_all,
            "gini_mean": gini_mean,
            "perm_mean": perm_mean,
            "res_df": res_df,
        }

    @staticmethod
    def compute_cluster0_vs_cluster1_wasserstein_kl_for_mi_features(
        *,
        X_df: pd.DataFrame,
        y: np.ndarray,
        distribution_plot_results: dict,
        n_mi_features: int = 300,
        n_bins_kl: int = 31,
    ) -> pd.DataFrame:
        """
        Same as ``compute_cluster0_vs_cluster1_wasserstein_kl_for_anova_features`` but ranks
        columns by Mutual Information (``mutual_info_classif``, ``random_state=42``).
        """
        from sklearn.feature_selection import mutual_info_classif

        split_labels = np.asarray(distribution_plot_results["split_labels"])
        valid_mask = np.asarray(distribution_plot_results["valid_mask"], dtype=bool)
        cluster_binary = np.asarray(distribution_plot_results["cluster_binary"])

        X_imp = FeatureClassification._impute_feature_matrix(X_df)

        mi_scores = mutual_info_classif(X_imp, y, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        rank_desc = np.argsort(mi_scores)[::-1]
        n_use = min(n_mi_features, X_imp.shape[1])
        ordered_cols = rank_desc[:n_use]

        is_train = split_labels == "training"
        is_val = split_labels == "validation"

        c_valid = cluster_binary[valid_mask]
        cluster_ids = np.unique(c_valid[~pd.isna(c_valid)])
        cluster_ids = np.sort(cluster_ids.astype(int))
        if len(cluster_ids) < 2:
            raise ValueError(
                "Need at least two PCA cluster IDs among valid structures; "
                f"found {cluster_ids!r}"
            )
        c0, c1 = int(cluster_ids[0]), int(cluster_ids[1])

        rows = []
        x_rank = np.arange(1, n_use + 1)
        for j_local, feat_col in enumerate(ordered_cols):
            col = X_imp[:, feat_col]
            row = {
                "mi_rank": x_rank[j_local],
                "matrix_column_idx": int(feat_col),
                "feature_name": str(X_df.columns[feat_col]),
            }
            for split_name, split_m in ("train", is_train), ("val", is_val):
                m = split_m & valid_mask
                m0 = m & (cluster_binary == c0)
                m1 = m & (cluster_binary == c1)
                v0 = col[m0]
                v1 = col[m1]
                if len(v0) < 2 or len(v1) < 2:
                    row[f"wasserstein_{split_name}"] = float("nan")
                    row[f"kl_symm_{split_name}"] = float("nan")
                else:
                    row[f"wasserstein_{split_name}"] = wasserstein_distance(v0, v1)
                    row[f"kl_symm_{split_name}"] = FeatureClassification._symmetric_kl_histogram(
                        v0, v1, n_bins=n_bins_kl
                    )
            rows.append(row)

        return pd.DataFrame(rows)


    @staticmethod
    def split_labels_from_newick_guide_tree(
        *,
        structure_names,
        newick_path: str,
        height: float,
        train_size: float = 0.9,
        random_state: int = 42,
        labels=None,
        feature_matrix=None,
    ):
        """Build training/validation ``split_labels`` from a Newick cut + GroupShuffleSplit.

        Returns ``(split_labels, train_idx, test_idx, groups)``.
        """
        from workflow.guide_tree_clusters import parse_newick, cluster_leaves_by_height_cut

        structure_names = list(structure_names)
        n = len(structure_names)
        with open(newick_path, "r", encoding="utf-8", errors="ignore") as fh:
            newick_str = fh.read().strip()
        tree_root = parse_newick(newick_str)
        leaf_to_cluster, _ = cluster_leaves_by_height_cut(tree_root, cut_height=float(height))
        next_missing = (max(leaf_to_cluster.values()) + 1) if leaf_to_cluster else 1
        groups = []
        for nm in structure_names:
            g = leaf_to_cluster.get(nm)
            if g is None:
                g = next_missing
                next_missing += 1
            groups.append(g)
        groups = np.asarray(groups)

        if feature_matrix is None:
            feature_matrix = np.zeros((n, 1))
        if labels is None:
            labels = np.zeros(n, dtype=int)

        splitter = GroupShuffleSplit(
            n_splits=1, train_size=train_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(feature_matrix, labels, groups=groups))
        split_labels = np.empty(n, dtype=object)
        split_labels[train_idx] = "training"
        split_labels[test_idx] = "validation"
        return split_labels, train_idx, test_idx, groups


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

    @staticmethod
    def undersample_majority_train(X_train, y_train, random_state=42):
        """
        Undersample the majority class in the training set (no replacement).

        Returns:
            keep indices into ``X_train`` / ``y_train`` (minority kept in full;
            majority randomly reduced to minority count).
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        unique_cls, cls_counts = np.unique(y_train, return_counts=True)
        if len(unique_cls) < 2:
            raise ValueError("Need at least two classes to undersample majority")
        maj_cls = unique_cls[np.argmax(cls_counts)]
        min_count = int(np.min(cls_counts))
        rng = np.random.RandomState(random_state)
        maj_idx = np.where(y_train == maj_cls)[0]
        min_idx = np.where(y_train != maj_cls)[0]
        sampled_maj = rng.choice(maj_idx, size=min_count, replace=False)
        return np.concatenate([sampled_maj, min_idx])

    @staticmethod
    def bootstrap_majority_ensemble_predict(
        X_train,
        y_train,
        X_test,
        n_bootstrap=30,
        n_estimators=100,
        random_state=42,
    ):
        """
        Bootstrap-subsample the majority class on train (with replacement),
        train one RF per round, return majority-vote predictions on ``X_test``.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        unique_cls, cls_counts = np.unique(y_train, return_counts=True)
        if len(unique_cls) < 2:
            raise ValueError("Need at least two classes for bootstrap majority ensemble")
        maj_cls = unique_cls[np.argmax(cls_counts)]
        min_count = int(np.min(cls_counts))
        maj_idx = np.where(y_train == maj_cls)[0]
        min_idx = np.where(y_train != maj_cls)[0]

        # Match March MajoritySubsampling §7: round seeds are 0..n_bootstrap-1
        # for both majority sampling RNG and RF ``random_state``.
        all_preds = []
        for seed in range(n_bootstrap):
            rng_b = np.random.RandomState(seed)
            sampled_maj_b = rng_b.choice(maj_idx, size=min_count, replace=True)
            keep_b = np.concatenate([sampled_maj_b, min_idx])
            rf_b = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
            rf_b.fit(X_train[keep_b], y_train[keep_b])
            all_preds.append(rf_b.predict(X_test))

        all_preds_arr = np.array(all_preds)
        return np.apply_along_axis(
            lambda col: np.bincount(col.astype(int)).argmax(),
            axis=0,
            arr=all_preds_arr,
        )

    @staticmethod
    def _majority_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    @staticmethod
    def compare_majority_subsampling_models(
        baseline_clf,
        *,
        n_bootstrap=30,
        n_estimators=100,
        random_state=42,
        save_dir=None,
        show_plot=True,
    ):
        """
        Compare baseline RF vs majority undersampling vs bootstrap ensemble
        on the shared held-out test set from ``baseline_clf``.

        Undersample / bootstrap are applied to the **training** set only.
        ``baseline_clf`` must already have ``split_data`` (+ preferably
        ``train_model`` / ``evaluate_model``) run so ``train_set``,
        ``train_class``, ``test_set``, ``test_class`` (and optionally
        ``predictions``) are available.

        Returns:
            dict with metrics/confusion matrices for baseline, undersample,
            bootstrap, plus a comparison DataFrame.
        """
        if getattr(baseline_clf, "train_set", None) is None:
            raise ValueError("baseline_clf must have split_data() run first")
        if getattr(baseline_clf, "test_set", None) is None:
            raise ValueError("baseline_clf must have a test_set from split_data()")

        train_X = baseline_clf.train_set
        train_y = baseline_clf.train_class
        test_X = baseline_clf.test_set
        test_y = baseline_clf.test_class

        # Baseline predictions (reuse evaluate if available)
        if getattr(baseline_clf, "predictions", None) is None:
            if baseline_clf.model is None:
                baseline_clf.train_model(
                    n_estimators=n_estimators, random_state=random_state
                )
            baseline_clf.predictions = baseline_clf.model.predict(test_X)
        preds_baseline = baseline_clf.predictions
        metrics_baseline = FeatureClassification._majority_metrics(test_y, preds_baseline)
        cm_baseline = confusion_matrix(test_y, preds_baseline)

        keep = FeatureClassification.undersample_majority_train(
            train_X, train_y, random_state=random_state
        )
        rf_under = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        rf_under.fit(train_X[keep], train_y[keep])
        preds_under = rf_under.predict(test_X)
        metrics_under = FeatureClassification._majority_metrics(test_y, preds_under)
        cm_under = confusion_matrix(test_y, preds_under)

        preds_boot = FeatureClassification.bootstrap_majority_ensemble_predict(
            train_X,
            train_y,
            test_X,
            n_bootstrap=n_bootstrap,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        metrics_boot = FeatureClassification._majority_metrics(test_y, preds_boot)
        cm_boot = confusion_matrix(test_y, preds_boot)

        metrics_table = pd.DataFrame(
            {
                "Model": [
                    "No subsampling",
                    "Random undersampling",
                    f"Bootstrap ensemble ({n_bootstrap}×)",
                ],
                "Accuracy": [
                    metrics_baseline["accuracy"],
                    metrics_under["accuracy"],
                    metrics_boot["accuracy"],
                ],
                "BalancedAccuracy": [
                    metrics_baseline["balanced_accuracy"],
                    metrics_under["balanced_accuracy"],
                    metrics_boot["balanced_accuracy"],
                ],
                "Precision": [
                    metrics_baseline["precision"],
                    metrics_under["precision"],
                    metrics_boot["precision"],
                ],
                "Recall": [
                    metrics_baseline["recall"],
                    metrics_under["recall"],
                    metrics_boot["recall"],
                ],
                "F1": [
                    metrics_baseline["f1"],
                    metrics_under["f1"],
                    metrics_boot["f1"],
                ],
            }
        ).round(4)

        out = {
            "metrics_baseline": metrics_baseline,
            "metrics_undersample": metrics_under,
            "metrics_bootstrap": metrics_boot,
            "cm_baseline": cm_baseline,
            "cm_undersample": cm_under,
            "cm_bootstrap": cm_boot,
            "preds_baseline": preds_baseline,
            "preds_undersample": preds_under,
            "preds_bootstrap": preds_boot,
            "metrics_table": metrics_table,
            "n_bootstrap": n_bootstrap,
        }

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            label_names = ["Active", "Inactive"]
            titles = [
                "No subsampling\n(baseline)",
                "Random undersampling",
                f"Bootstrap ensemble\n({n_bootstrap} rounds)",
            ]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, cm_mat, title in zip(
                axes, [cm_baseline, cm_under, cm_boot], titles
            ):
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm_mat, display_labels=label_names
                )
                disp.plot(ax=ax, colorbar=False)
                ax.set_title(title, fontsize=12)
            plt.suptitle(
                "Confusion matrix comparison — Active vs Inactive (0=Active, 1=Inactive)",
                y=1.02,
            )
            plt.tight_layout()
            cm_png = os.path.join(save_dir, "subsampling_confusion_matrices.png")
            plt.savefig(cm_png, dpi=200, bbox_inches="tight")
            if show_plot:
                plt.show()
            else:
                plt.close()

            # Metric bar chart
            plot_df = metrics_table.set_index("Model")[
                ["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1"]
            ]
            ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=20)
            ax.set_ylabel("Score")
            ax.set_title("Majority-subsampling model comparison")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right")
            plt.tight_layout()
            bar_png = os.path.join(save_dir, "subsampling_metrics_bar.png")
            plt.savefig(bar_png, dpi=200, bbox_inches="tight")
            if show_plot:
                plt.show()
            else:
                plt.close()

            csv_path = os.path.join(save_dir, "subsampling_metrics_comparison.csv")
            metrics_table.to_csv(csv_path, index=False)
            out["save_dir"] = save_dir
            out["cm_png"] = cm_png
            out["bar_png"] = bar_png
            out["metrics_csv"] = csv_path
            print(f"Saved confusion matrices → {cm_png}")
            print(f"Saved metrics bar chart  → {bar_png}")
            print(f"Saved metrics table      → {csv_path}")

        try:
            from IPython.display import display

            display(metrics_table)
        except Exception:
            print(metrics_table.to_string(index=False))

        return out


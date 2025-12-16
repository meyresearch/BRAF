"""
Feature Classification and Analysis

This module provides functionality for training machine learning models,
evaluating performance, and analyzing feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
        
    def split_data(self, train_size=0.9, random_state=42):
        """
        Split data into training and test sets.
        
        Args:
            train_size: Fraction of data for training (default 0.9)
            random_state: Random seed for reproducibility (default 42)
        """
        print("="*60)
        print("SPLITTING DATA")
        print("="*60)
        
        self.train_set, self.test_set, self.train_class, self.test_class = train_test_split(
            self.feature_matrix, self.labels, train_size=train_size, random_state=random_state
        )
        
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
        precision = precision_score(self.test_class, self.predictions, average='weighted')
        recall = recall_score(self.test_class, self.predictions, average='weighted')
        
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

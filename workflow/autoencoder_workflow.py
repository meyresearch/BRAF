"""
Autoencoder Workflow for Protein Structure Analysis

This module provides a class-based workflow for training and analyzing
protein structures using an autoencoder model.
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), 'src'))
from molearn.data import PDBData
from molearn.trainers import Trainer
from molearn.models.small_foldingnet import Small_AutoEncoder
from molearn.analysis.analyser import MolearnAnalysis
from molearn.analysis import MolearnGUI

from utilities import ifnotmake


class AutoencoderWorkflow:
    """
    A workflow class for training and analyzing protein structures using autoencoders.
    """
    
    def __init__(self, folder_name, output_base_dir, manual_seed=25, batch_size=8, 
                 validation_split=0.1, device=None, processes=4):
        """
        Initialize the AutoencoderWorkflow.
        
        Args:
            folder_name (str): Path to folder containing PDB files
            output_base_dir (str): Base directory for all output files
            manual_seed (int): Random seed for reproducibility
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
            device (str or torch.device): Device to use ('cuda' or 'cpu')
            processes (int): Number of processes for parallel processing
        """
        self.folder_name = folder_name
        self.output_base_dir = output_base_dir
        self.manual_seed = manual_seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processes = processes
        
        # Setup paths
        self.combined_file_path = os.path.join(folder_name, 'combined.pdb')
        self.checkpoint_dir = output_base_dir  # Checkpoints are in the base directory
        self.log_dir = os.path.join(output_base_dir, 'xbb_foldingnet_checkpoints')
        self.get_dataset_dir = os.path.join(output_base_dir, 'getDatasetTrial')
        self.decoded_train_dir = os.path.join(output_base_dir, 'decoded_train')
        self.decoded_valid_dir = os.path.join(output_base_dir, 'decoded_valid')
        self.labeled_train_dir = os.path.join(output_base_dir, 'hdbscan_labels_train')
        self.labeled_valid_dir = os.path.join(output_base_dir, 'hdbscan_labels_valid')
        
        # Initialize attributes
        self.data = None
        self.trainer = None
        self.net = None
        self.MA = None
        self.data_train = None
        self.data_valid = None
        self.train_indices = None
        self.valid_indices = None
        self.labels_train = None
        self.labels_valid = None
        
    def prepare_data(self, atom_selection=['CA', 'C', 'N', 'CB', 'O']):
        """
        Prepare and combine PDB files into a single combined.pdb file.
        
        Args:
            atom_selection (list): List of atom names to select
        """
        # Get sorted list of all files excluding combined.pdb
        files = sorted([
            f for f in os.listdir(self.folder_name) 
            if os.path.isfile(os.path.join(self.folder_name, f)) and f != 'combined.pdb'
        ])
        
        # Create combined.pdb file
        with open(self.combined_file_path, 'w') as combined_file:
            for i, filename in enumerate(files):
                file_path = os.path.join(self.folder_name, filename)
                
                # Read content while filtering out lines starting with "MODEL" or "END"
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    lines = [line for line in lines if not line.startswith(("MODEL", "END"))]
                
                # Write MODEL, lines, and ENDMDL
                combined_file.write(f'MODEL {i}\n')
                combined_file.writelines(lines)
                combined_file.write('ENDMDL\n')
            
            combined_file.write('END\n')
        
        # Import combined.pdb
        self.data = PDBData()
        self.data.import_pdb(filename=self.combined_file_path)
        self.data.fix_terminal()
        self.data.atomselect(atoms=atom_selection)
        self.data.prepare_dataset()
        
        print(f"Loaded {len(self.data._mol.trajectory)} structures")
        
    def train(self, network_class=Small_AutoEncoder, max_epochs=32, patience=32):
        """
        Train the autoencoder model.
        
        Args:
            network_class: Network class to use (default: Small_AutoEncoder)
            max_epochs (int): Maximum number of epochs per training cycle
            patience (int): Number of epochs without improvement before stopping
        """
        self.trainer = Trainer(device=self.device)
        self.trainer.set_data(self.data, batch_size=self.batch_size, 
                              validation_split=self.validation_split, 
                              manual_seed=self.manual_seed)
        self.trainer.set_autoencoder(network_class, out_points=self.data.dataset.shape[-1])
        self.trainer.prepare_optimiser()
        
        # Training loop with patience
        runkwargs = dict(
            log_filename='log_file.dat',
            log_folder=self.log_dir,
            checkpoint_folder=self.checkpoint_dir,
        )
        
        ifnotmake(self.log_dir)
        ifnotmake(self.checkpoint_dir)
        
        best = 1e24
        while True:
            self.trainer.run(max_epochs=max_epochs + self.trainer.epoch, **runkwargs)
            if not best > self.trainer.best:
                break
            best = self.trainer.best
        
        print(f'Training complete. Best loss: {self.trainer.best}, Best file: {self.trainer.best_name}')
        
    def load_checkpoint(self, checkpoint_pattern=None):
        """
        Load a trained checkpoint.
        
        Args:
            checkpoint_pattern (str): Glob pattern for checkpoint files (default: all .ckpt files)
        """
        if checkpoint_pattern is None:
            checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.ckpt')
        
        matching_files = sorted(glob.glob(checkpoint_pattern))
        
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No files matched the pattern: {checkpoint_pattern}")
        
        networkfile = matching_files[0]
        checkpoint = torch.load(networkfile, map_location=torch.device('cpu'))
        
        self.net = Small_AutoEncoder(**checkpoint['network_kwargs'])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded checkpoint from: {networkfile}")
        print(f"Network kwargs: {checkpoint['network_kwargs']}")
        
    def setup_analysis(self, atom_selection=['CA', 'C', 'N', 'CB', 'O']):
        """
        Setup the MolearnAnalysis object with training and validation datasets.
        Automatically prepares data if not already loaded.
        
        Args:
            atom_selection (list): List of atom names to select for data preparation
        """
        if self.data is None:
            self.prepare_data(atom_selection=atom_selection)
        
        if self.net is None:
            raise ValueError("Model not loaded. Call load_checkpoint() first.")
        
        self.MA = MolearnAnalysis()
        self.MA.set_network(self.net)
        
        # Get training and validation splits (PDBData objects for MA)
        data_train_pdb, data_valid_pdb = self.data.split(manual_seed=self.manual_seed)
        self.MA.set_dataset("training", data_train_pdb)
        self.MA.set_dataset("validation", data_valid_pdb)
        
        # Get the actual tensor datasets for computing indices
        data_train, data_valid = self.data.get_datasets(manual_seed=self.manual_seed)
        
        # Store as class attributes
        self.data_train, self.data_valid = data_train, data_valid
        
        # Get indices
        indices = self.data.indices.numpy()
        n_train = data_train.shape[0]
        self.train_indices = indices[:n_train]
        self.valid_indices = indices[n_train:]
        
        # Set batch size and processes
        self.MA.batch_size = self.batch_size
        self.MA.processes = self.processes
        
    def extract_dataset(self):
        """
        Extract dataset structures to individual PDB files.
        """
        ifnotmake(self.get_dataset_dir)
        
        for i, index in enumerate(self.train_indices):
            self.data._mol.trajectory[index]
            self.data._mol.select_atoms("name CA").write(
                os.path.join(self.get_dataset_dir, f's{i}.pdb')
            )
            
    def decode_structures(self):
        """
        Encode and decode training and validation structures.
        """
        # Decode training set
        ifnotmake(self.decoded_train_dir)
        latent_coords_train = self.MA.get_encoded('training')
        self.MA.generate(latent_coords_train.numpy().reshape(
            1, len(latent_coords_train), 2), self.decoded_train_dir, relax=False)
        
        # Decode validation set
        ifnotmake(self.decoded_valid_dir)
        latent_coords_valid = self.MA.get_encoded('validation')
        self.MA.generate(latent_coords_valid.numpy().reshape(
            1, len(latent_coords_valid), 2), self.decoded_valid_dir, relax=False)
        
    def calculate_errors(self, save_prefix='_foldingnet_checkpoint'):
        """
        Calculate reconstruction errors and save to CSV files.
        
        Args:
            save_prefix (str): Prefix for output files
        """
        # Get errors
        err_train = list(self.MA.get_error('training', align=False))
        err_valid = list(self.MA.get_error('validation', align=False))
        
        # Save to CSV
        df_err_train = pd.DataFrame(err_train, columns=['err_train'])
        df_err_train.to_csv(
            os.path.join(self.output_base_dir, f'err_train_{save_prefix}.csv'), 
            index=False
        )
        
        df_err_valid = pd.DataFrame(err_valid, columns=['err_test'])
        df_err_valid.to_csv(
            os.path.join(self.output_base_dir, f'err_valid_{save_prefix}.csv'), 
            index=False
        )
        
        return err_train, err_valid
        
    def rename_files(self):
        """
        Rename generic s{i}.pdb files to original filenames and create mapping CSVs.
        """
        files = sorted([
            f for f in os.listdir(self.folder_name)
            if os.path.isfile(os.path.join(self.folder_name, f)) and f != 'combined.pdb'
        ])
        
        # Training set
        mapping_path = os.path.join(self.output_base_dir, 'train_index_mapping.csv')
        with open(mapping_path, 'w') as f:
            f.write("loop_index,train_index,pdb_filename\n")
            for i, index in enumerate(self.train_indices):
                f.write(f"{i},{index},{files[index]}\n")
        
        # Validation set
        mapping_path = os.path.join(self.output_base_dir, 'valid_index_mapping.csv')
        with open(mapping_path, 'w') as f:
            f.write("loop_index,valid_index,pdb_filename\n")
            for i, index in enumerate(self.valid_indices):
                f.write(f"{i},{index},{files[index]}\n")
        
        # Rename files
        self._rename_files_set('train', files, self.get_dataset_dir, 
                               self.decoded_train_dir)
        self._rename_files_set('valid', files, self.get_dataset_dir, 
                               self.decoded_valid_dir)
        
    def _rename_files_set(self, set_type, files, source_dir, decoded_dir):
        """Helper method to rename files for a given set."""
        indices = self.train_indices if set_type == 'train' else self.valid_indices
        
        for i, index in enumerate(indices):
            original_filename = files[index]
            
            # Rename in source directory
            old_file = os.path.join(source_dir, f's{i}.pdb')
            new_file = os.path.join(source_dir, original_filename)
            if os.path.exists(old_file):
                os.rename(old_file, new_file)
            
            # Rename in decoded directory
            old_file_decoded = os.path.join(decoded_dir, f's{i}.pdb')
            new_file_decoded = os.path.join(decoded_dir, original_filename)
            if os.path.exists(old_file_decoded):
                os.rename(old_file_decoded, new_file_decoded)
        
    def scan_error_landscape(self, grid_size=30, save_prefix='_foldingnet_checkpoint'):
        """
        Scan the latent space to build an error landscape.
        
        Args:
            grid_size (int): Size of the grid (grid_size x grid_size)
            save_prefix (str): Prefix for output files
        """
        self.MA.setup_grid(grid_size)
        landscape_err_latent, landscape_err_3d, xaxis, yaxis = self.MA.scan_error()
        
        # Save results
        pd.DataFrame(landscape_err_latent).to_csv(
            os.path.join(self.output_base_dir, f'landscape_err_latent_{save_prefix}.csv'), 
            index=False
        )
        pd.DataFrame(landscape_err_3d).to_csv(
            os.path.join(self.output_base_dir, f'landscape_err_3d_{save_prefix}.csv'), 
            index=False
        )
        pd.DataFrame(xaxis).to_csv(
            os.path.join(self.output_base_dir, f'landscape_err_xaxis_{save_prefix}.csv'), 
            index=False
        )
        pd.DataFrame(yaxis).to_csv(
            os.path.join(self.output_base_dir, f'landscape_err_yaxis_{save_prefix}.csv'), 
            index=False
        )
        
    def extract_encoded_coordinates(self, save_prefix='_foldingnet_checkpoint'):
        """
        Extract encoded latent coordinates for training and validation sets.
        
        Args:
            save_prefix (str): Prefix for output files
        """
        with torch.no_grad():
            z_train = self.net.encode(self.data_train.float())
            z_valid = self.net.encode(self.data_valid.float())
        
        z_train_np = z_train.data.cpu().numpy()[:, :, 0]
        z_valid_np = z_valid.data.cpu().numpy()[:, :, 0]
        
        pd.DataFrame(z_train_np).to_csv(
            os.path.join(self.output_base_dir, 
                        'landscape_encoded_train_coordinates.csv'), 
            index=False
        )
        pd.DataFrame(z_valid_np).to_csv(
            os.path.join(self.output_base_dir, 
                        'landscape_encoded_valid_coordinates.csv'), 
            index=False
        )
        
    def perform_clustering(self, min_cluster_size_list=[2, 5, 10, 20, 30, 35, 40],
                          min_samples_list=[None, 1, 5, 10, 20, 30, 35, 40],
                          best_min_cluster_size=2, best_min_samples=2):
        """
        Perform HDBSCAN clustering on encoded coordinates.
        
        Args:
            min_cluster_size_list (list): List of min_cluster_size values to try
            min_samples_list (list): List of min_samples values to try
            best_min_cluster_size (int): Selected min_cluster_size for final clustering
            best_min_samples (int): Selected min_samples for final clustering
        """
        # Load encoded coordinates
        train_coords_file = os.path.join(self.output_base_dir, 
                                         'landscape_encoded_train_coordinates.csv')
        valid_coords_file = os.path.join(self.output_base_dir, 
                                         'landscape_encoded_valid_coordinates.csv')
        
        X_train = pd.read_csv(train_coords_file, header=0).to_numpy()
        X_valid = pd.read_csv(valid_coords_file, header=0).to_numpy()
        
        # Parameter search
        best_score = -1
        best_params = (None, None)
        results = []
        
        for min_cluster_size in min_cluster_size_list:
            for min_samples in min_samples_list:
                hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                labels_train = hdb.fit_predict(X_train)
                
                unique_labels = set(labels_train)
                if len(unique_labels) < 2:
                    score = float('nan')
                    n_clusters = 0
                else:
                    score = silhouette_score(X_train, labels_train)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    
                    if score > best_score:
                        best_score = score
                        best_params = (min_cluster_size, min_samples)
                
                results.append((min_cluster_size, min_samples, score, n_clusters))
        
        # Print results
        df_results = pd.DataFrame(results, columns=[
            'min_cluster_size', 'min_samples', 'silhouette_score', 'num_clusters'
        ])
        print(df_results)
        print(f"\nBest Silhouette Score: {best_score:0.3f}")
        print(f"Best Parameters: min_cluster_size={best_params[0]}, min_samples={best_params[1]}")
        
        # Perform final clustering with best parameters
        hdb = HDBSCAN(min_cluster_size=best_min_cluster_size, 
                     min_samples=best_min_samples)
        self.labels_train = hdb.fit_predict(X_train)
        self.labels_valid = hdb.fit_predict(X_valid)
        
    def organize_by_clusters(self):
        """
        Organize structures into subdirectories based on cluster labels.
        """
        files = sorted([
            f for f in os.listdir(self.folder_name)
            if os.path.isfile(os.path.join(self.folder_name, f)) and f != 'combined.pdb'
        ])
        
        # Organize training set
        train_mapping_file = os.path.join(self.output_base_dir, 'train_index_mapping.csv')
        self._organize_set(train_mapping_file, self.labels_train, 
                          self.labeled_train_dir, files)
        
        # Organize validation set
        valid_mapping_file = os.path.join(self.output_base_dir, 'valid_index_mapping.csv')
        self._organize_set(valid_mapping_file, self.labels_valid, 
                          self.labeled_valid_dir, files)
        
    def _organize_set(self, mapping_file, labels, output_dir, files):
        """Helper method to organize files for a given set."""
        ifnotmake(output_dir)
        
        import shutil
        
        with open(mapping_file, 'r') as f:
            header = next(f).strip()
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                loop_index = int(parts[0])
                pdb_filename = parts[2]
                
                # Extract PDB code (first 4 letters)
                pdb_code_4 = pdb_filename[:4]
                
                # Get cluster label
                label = labels[loop_index]
                
                # Determine subfolder
                subfolder = "noise" if label == -1 else f"cluster_{label}"
                label_folder = os.path.join(output_dir, subfolder)
                ifnotmake(label_folder)
                
                # Find and copy matching file
                for fname in os.listdir(self.get_dataset_dir):
                    if fname.startswith(pdb_code_4):
                        src = os.path.join(self.get_dataset_dir, fname)
                        dst = os.path.join(label_folder, fname)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
    
    def load_external_labels(self, pca_labels_file, label_column='ClusterLabel', 
                            filename_column='FullName'):
        """
        Load cluster labels from external source (e.g., PCA clustering results).
        
        Args:
            pca_labels_file (str): Path to CSV file with cluster labels
            label_column (str): Name of column containing cluster labels
            filename_column (str): Name of column containing PDB filenames
            
        Returns:
            tuple: (labels_train, labels_valid) arrays of cluster labels
        """
        # Read PCA labels - skip comment lines and read header
        with open(pca_labels_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (starts with #)
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('#'):
                header_line = line[1:].strip()  # Remove # and whitespace
                data_start = i + 1
                break
        
        if header_line:
            # Read CSV with proper header
            df_pca = pd.read_csv(pca_labels_file, skiprows=data_start, header=None, 
                                names=header_line.split(','))
        else:
            # No header found, read normally
            df_pca = pd.read_csv(pca_labels_file)
        
        print(f"Loaded PCA labels file with columns: {list(df_pca.columns)}")
        
        # Get list of files in order
        files = sorted([
            f for f in os.listdir(self.folder_name)
            if os.path.isfile(os.path.join(self.folder_name, f)) and f != 'combined.pdb'
        ])
        
        # Create mapping from filename to label
        filename_to_label = {}
        for _, row in df_pca.iterrows():
            filename = row[filename_column]
            # Ensure .pdb extension
            if not filename.endswith('.pdb'):
                filename = filename + '.pdb'
            filename_to_label[filename] = int(row[label_column])
        
        # Map labels to train and validation indices
        labels_train = []
        missing_train = []
        for idx in self.train_indices:
            fname = files[idx]
            if fname in filename_to_label:
                labels_train.append(filename_to_label[fname])
            else:
                labels_train.append(-1)
                missing_train.append(fname)
        
        labels_valid = []
        missing_valid = []
        for idx in self.valid_indices:
            fname = files[idx]
            if fname in filename_to_label:
                labels_valid.append(filename_to_label[fname])
            else:
                labels_valid.append(-1)
                missing_valid.append(fname)
        
        labels_train = np.array(labels_train)
        labels_valid = np.array(labels_valid)
        
        # Report missing files
        if missing_train:
            print(f"WARNING: {len(missing_train)} training files not found in PCA labels:")
            for fname in missing_train[:5]:  # Show first 5
                print(f"  - {fname}")
            if len(missing_train) > 5:
                print(f"  ... and {len(missing_train) - 5} more")
        
        if missing_valid:
            print(f"WARNING: {len(missing_valid)} validation files not found in PCA labels:")
            for fname in missing_valid[:5]:  # Show first 5
                print(f"  - {fname}")
            if len(missing_valid) > 5:
                print(f"  ... and {len(missing_valid) - 5} more")
        
        # Store labels
        self.labels_train = labels_train
        self.labels_valid = labels_valid
        
        print(f"Loaded external labels from {pca_labels_file}")
        print(f"Training set: {len(labels_train)} structures")
        print(f"Validation set: {len(labels_valid)} structures")
        print(f"Unique labels in training: {np.unique(labels_train)}")
        print(f"Unique labels in validation: {np.unique(labels_valid)}")
        
        return labels_train, labels_valid
    
    def load_kincore_labels(self, kincore_file='Results/dunbrack_assignments/kinase_conformation_assignments.csv'):
        """
        Load activation state labels from KinCore classification file.
        
        Args:
            kincore_file (str): Path to the KinCore classification CSV file
            
        Returns:
            tuple: (labels_train, labels_valid) arrays of activation state labels
                   0 = Inactive, 1 = Active
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
        
        # Get list of files from folder
        files = sorted(glob.glob(os.path.join(self.folder_name, '*.pdb')))
        files = [os.path.basename(f).replace('.pdb', '') for f in files]
        
        # Map to activation labels
        labels_train = []
        for idx in self.train_indices:
            fname = files[idx]
            if fname in activation_map:
                labels_train.append(activation_map[fname])
            elif fname + '.pdb' in activation_map:
                labels_train.append(activation_map[fname + '.pdb'])
            else:
                labels_train.append(0)  # Default to inactive
        
        labels_valid = []
        for idx in self.valid_indices:
            fname = files[idx]
            if fname in activation_map:
                labels_valid.append(activation_map[fname])
            elif fname + '.pdb' in activation_map:
                labels_valid.append(activation_map[fname + '.pdb'])
            else:
                labels_valid.append(0)  # Default to inactive
        
        labels_train = np.array(labels_train)
        labels_valid = np.array(labels_valid)
        
        # Store as activation labels (separate from cluster labels)
        self.activation_labels_train = labels_train
        self.activation_labels_valid = labels_valid
        
        print(f"Loaded KinCore activation labels from {kincore_file}")
        print(f"Training set: Active={sum(labels_train == 1)}, Inactive={sum(labels_train == 0)}")
        print(f"Validation set: Active={sum(labels_valid == 1)}, Inactive={sum(labels_valid == 0)}")
        
        return labels_train, labels_valid
    
    def plot_latent_space(self, labels=None, title="Latent Space Projection",
                         output_file=None, show_train=True, show_valid=True,
                         colormap='tab10', alpha=0.7, s=50,
                         save_prefix='_foldingnet_checkpoint', vmin=0., vmax=10.,
                         activation_colors=None):
        """
        Plot the latent space projection colored by cluster labels with RMSD landscape background.
        
        Args:
            labels (tuple or None): Tuple of (labels_train, labels_valid). 
                                   If None, uses self.labels_train and self.labels_valid
            title (str): Plot title
            output_file (str): Path to save figure. If None, displays plot
            show_train (bool): Whether to show training data
            show_valid (bool): Whether to show validation data
            colormap (str): Matplotlib colormap name for scatter points
            alpha (float): Point transparency
            s (float): Point size
            save_prefix (str): Prefix used when saving landscape data
            vmin (float): Minimum value for RMSD colorbar
            vmax (float): Maximum value for RMSD colorbar
            activation_colors (list or None): List of colors for activation states 
                                             [inactive_color, active_color]. If provided,
                                             creates a custom colormap.
            
        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects
        """
        # Load encoded coordinates
        train_coords_file = os.path.join(self.output_base_dir, 
                                         'landscape_encoded_train_coordinates.csv')
        valid_coords_file = os.path.join(self.output_base_dir, 
                                         'landscape_encoded_valid_coordinates.csv')
        
        X_train = pd.read_csv(train_coords_file, header=0).to_numpy()
        X_valid = pd.read_csv(valid_coords_file, header=0).to_numpy()
        
        x_encoded_train = X_train[:, 0]
        y_encoded_train = X_train[:, 1]
        x_encoded_valid = X_valid[:, 0]
        y_encoded_valid = X_valid[:, 1]
        
        # Load RMSD landscape data
        df_z = pd.read_csv(os.path.join(self.output_base_dir, 
                          f'landscape_err_3d_{save_prefix}.csv'), header=0).to_numpy()
        df_x = pd.read_csv(os.path.join(self.output_base_dir, 
                          f'landscape_err_xaxis_{save_prefix}.csv'), header=0).to_numpy().flatten()
        df_y = pd.read_csv(os.path.join(self.output_base_dir, 
                          f'landscape_err_yaxis_{save_prefix}.csv'), header=0).to_numpy().flatten()
        
        # Get labels
        if labels is None:
            if self.labels_train is None or self.labels_valid is None:
                raise ValueError("No labels provided and no labels stored in workflow.")
            labels_train = self.labels_train
            labels_valid = self.labels_valid
        else:
            labels_train, labels_valid = labels
        
        # Verify sizes match
        print(f"Data sizes - Train: coords={len(X_train)}, labels={len(labels_train)}")
        print(f"Data sizes - Valid: coords={len(X_valid)}, labels={len(labels_valid)}")
        
        if len(X_train) != len(labels_train):
            raise ValueError(
                f"Size mismatch for training data: {len(X_train)} coordinates "
                f"but {len(labels_train)} labels"
            )
        if len(X_valid) != len(labels_valid):
            raise ValueError(
                f"Size mismatch for validation data: {len(X_valid)} coordinates "
                f"but {len(labels_valid)} labels"
            )
        
        # Create figure with 2 square subplots
        fig = plt.figure(figsize=(8, 8))
        
        # Prepare discrete color boundaries (same as PCA)
        n_clusters = len(np.unique(np.concatenate([labels_train, labels_valid])))
        bounds = list(range(n_clusters + 1))
        norm_bound = BoundaryNorm(bounds, ncolors=n_clusters, clip=True)
        
        # Use custom colormap for activation states if provided
        if activation_colors is not None:
            scatter_cmap = ListedColormap(activation_colors)
        else:
            scatter_cmap = colormap
        
        # Create 2 square subplots - specify the position
        ax1 = fig.add_axes([0.0, 0.1, 0.35, 0.35])
        ax1.imshow(df_z, cmap='viridis', vmin=vmin, vmax=vmax, 
                   extent=[np.min(df_x), np.max(df_x), np.min(df_y), np.max(df_y)])
        if show_train:
            ax1.scatter(x_encoded_train, y_encoded_train, c=labels_train, 
                       marker='.', cmap=scatter_cmap, norm=norm_bound)
        
        ax2 = fig.add_axes([0.38, 0.1, 0.35, 0.35])
        im = ax2.imshow(df_z, cmap='viridis', vmin=vmin, vmax=vmax, 
                       extent=[np.min(df_x), np.max(df_x), np.min(df_y), np.max(df_y)])
        if show_valid:
            ax2.scatter(x_encoded_valid, y_encoded_valid, c=labels_valid, 
                       marker='.', cmap=scatter_cmap, norm=norm_bound)
        
        # Create colorbar axis for RMSD
        cbar_ax = fig.add_axes([0.755, 0.1, 0.02, 0.35])
        cbar_ax.tick_params(left=False, labelleft=False, right=True, labelright=True, 
                           labelbottom=False, bottom=False)
        
        # Create colorbar
        cbar = fig.colorbar(im, cax=cbar_ax, label='RMSD [$\AA$]')
        cbar_ticks = np.linspace(vmin, vmax, 7)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{tick:.0f}' for tick in cbar_ticks])
        
        # Set axes properties
        ax1.tick_params(direction='inout', labelbottom=True, top=False, bottom=True)
        ax2.tick_params(direction='inout', labelbottom=True, top=False, bottom=True, 
                       left=False, labelleft=False)
        
        # Set labels
        ax1.set_xlabel('Latent vector 1')
        ax1.set_ylabel('Latent vector 2')
        ax2.set_xlabel('Latent vector 1')
        
        # Set titles
        ax1.set_title('Training dataset')
        ax2.set_title('Validation dataset')
        
        # Save or show
        if output_file:
            plt.savefig(os.path.join(self.output_base_dir, output_file), dpi=300, bbox_inches='tight')
            print(f"Saved latent space plot to {os.path.join(self.output_base_dir, output_file)}")
        else:
            plt.show()
        
        return fig, (ax1, ax2)


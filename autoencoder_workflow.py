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

from molearn.data import PDBData
from molearn.trainers import Trainer
from molearn.models.small_foldingnet import Small_AutoEncoder
from molearn.models.CNN2d_AE import AutoEncoder as CNN2d_AutoEncoder
from wrCNN2D import AutoEncoder as wrCNN2D_AutoEncoder
from wrTrainer import WritheTrainer
from molearn.analysis.analyser import MolearnAnalysis
from molearn.analysis import MolearnGUI

try:
    from .utilities import ifnotmake
except ImportError:  # pragma: no cover
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
        self.output_root_dir = output_base_dir
        self.output_base_dir = output_base_dir
        self.manual_seed = manual_seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this workflow, but no CUDA device is available.")
        self.device = torch.device('cuda')
        self.processes = processes
        
        # Setup paths
        self.combined_file_path = os.path.join(folder_name, 'combined.pdb')
        self.output_subfolder = None
        self._refresh_output_paths()
        
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
        self.network_class = Small_AutoEncoder
        self.network_kwargs = {}
        self.last_checkpoint_path = None
        self.last_checkpoint_network_class = None

    def _refresh_output_paths(self):
        """
        Refresh all output-related paths according to current output_base_dir.
        """
        self.checkpoint_dir = self.output_base_dir  # Checkpoints are in the base directory
        self.log_dir = os.path.join(self.output_base_dir, 'xbb_foldingnet_checkpoints')
        self.get_dataset_dir = os.path.join(self.output_base_dir, 'getDatasetTrial')
        self.decoded_train_dir = os.path.join(self.output_base_dir, 'decoded_train')
        self.decoded_valid_dir = os.path.join(self.output_base_dir, 'decoded_valid')
        self.labeled_train_dir = os.path.join(self.output_base_dir, 'hdbscan_labels_train')
        self.labeled_valid_dir = os.path.join(self.output_base_dir, 'hdbscan_labels_valid')

    def set_output_subfolder(self, subfolder=None, create=True):
        """
        Route outputs to a dedicated subfolder under output_root_dir.

        Args:
            subfolder (str or None): Subfolder name (e.g. 'cnn2d_ae'). If None,
                                     reset to output_root_dir.
            create (bool): Create output folder if it doesn't exist.
        """
        self.output_subfolder = subfolder
        if subfolder:
            self.output_base_dir = os.path.join(self.output_root_dir, subfolder)
        else:
            self.output_base_dir = self.output_root_dir
        self._refresh_output_paths()
        if create:
            ifnotmake(self.output_base_dir)
        self._print_status("output")

    def _print_status(self, stage):
        """
        Minimal one-line status message for key workflow stages.
        """
        model_name = (
            self.network_class.__name__
            if hasattr(self.network_class, '__name__')
            else str(self.network_class)
        )
        print(f"[{stage}] model={model_name} output={self.output_base_dir}")
        
    def prepare_data(self, atom_selection=['CA', 'C', 'N', 'CB', 'O']):
        """
        Prepare and combine PDB files into a single combined.pdb file.
        
        Args:
            atom_selection (list): List of atom names to select
        """
        selected_atoms = list(atom_selection)

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
        self.data.atomselect(atoms=selected_atoms)
        if len(selected_atoms) == 1 and selected_atoms[0] == 'CA':
            # CA-only trajectories are not supported by molearn_latest prepare_dataset(),
            # so build a compatible dataset/metadata bundle manually.
            coords = np.asarray(
                [self.data._mol.atoms.positions.astype(float) for _ in self.data._mol.trajectory]
            )
            self.data.mean = float(coords.mean())
            self.data.std = float(coords.std())
            if self.data.std == 0.0:
                self.data.std = 1.0
            self.data.standardize = True
            self.data.dataset = torch.from_numpy(
                (coords - self.data.mean) / self.data.std
            ).float()
            self.data._atom_names = list(np.unique(self.data._mol.atoms.names))

            n_atoms = self.data.dataset.shape[1]
            ca_idx = torch.arange(n_atoms, dtype=torch.long)
            self.data.indices = {
                "N": ca_idx.clone(),
                "CA": ca_idx.clone(),
                "C": ca_idx.clone(),
                "O": ca_idx.clone(),
                "CB": torch.full((n_atoms,), -1, dtype=torch.long),
            }
            self.data.cb_valid_idx = self.data.indices["CB"][self.data.indices["CB"] >= 0]

            print(f"Dataset shape: {self.data.dataset.shape}")
            print(f"mean: {self.data.mean}\n std: {self.data.std}")
        else:
            self.data.prepare_dataset()
        
        print(f"Loaded {len(self.data._mol.trajectory)} structures")

    def _ensure_dataset_layout_bn3(self):
        """
        Ensure dataset layout is [frames, n_atoms, 3].
        """
        if self.data is None or not hasattr(self.data, "dataset"):
            raise ValueError("Dataset is not prepared yet. Call prepare_data() first.")

        dataset = self.data.dataset
        if dataset.ndim != 3:
            raise ValueError(
                f"Expected 3D dataset, got shape {tuple(dataset.shape)}"
            )

        if dataset.shape[-1] == 3:
            return
        if dataset.shape[1] == 3:
            self.data.dataset = dataset.permute(0, 2, 1).contiguous()
            return

        raise ValueError(
            f"Cannot convert dataset shape {tuple(dataset.shape)} to [frames, n_atoms, 3]"
        )

    def _num_atoms(self):
        """
        Return atom count from [frames, n_atoms, 3].
        """
        self._ensure_dataset_layout_bn3()
        return self.data.dataset.shape[1]
        
    def _resolve_network_class(self, network_class):
        """
        Resolve a network class from class object or known string aliases.
        """
        if network_class is None:
            return self.network_class or Small_AutoEncoder

        if isinstance(network_class, str):
            key = network_class.strip().lower()
            if key in ('small', 'small_autoencoder', 'small_foldingnet'):
                return Small_AutoEncoder
            if key in ('cnn2d', 'cnn2d_ae', '2d_ae', 'autoencoder2d'):
                return CNN2d_AutoEncoder
            if key in ('wrcnn2d', 'wrcnn2d_ae', 'writhe_ae', 'writhe'):
                return wrCNN2D_AutoEncoder
            raise ValueError(
                f"Unknown network alias '{network_class}'. "
                "Use Small_AutoEncoder, CNN2d_AutoEncoder, wrCNN2D_AutoEncoder, or supported aliases."
            )

        return network_class

    def _prepare_network_kwargs(self, network_class, network_kwargs=None):
        """
        Build model kwargs with sensible defaults based on selected network.
        """
        kwargs = dict(network_kwargs or {})
        n_points = self._num_atoms()

        # Small foldingnet requires out_points
        if network_class is Small_AutoEncoder and 'out_points' not in kwargs:
            kwargs['out_points'] = n_points

        # CNN2d_AE AutoEncoder expects dm_dim
        if CNN2d_AutoEncoder is not None and network_class is CNN2d_AutoEncoder:
            kwargs.setdefault('dm_dim', n_points)
            kwargs.setdefault('latent_dim', 2)

        if wrCNN2D_AutoEncoder is not None and network_class is wrCNN2D_AutoEncoder:
            kwargs.setdefault('n_atoms', n_points)
            kwargs.setdefault('latent_dim', 2)

        return kwargs

    def train(self, network_class=Small_AutoEncoder, network_kwargs=None,
              max_epochs=32, patience=32):
        """
        Train the autoencoder model.
        
        Args:
            network_class: Network class or alias (default: Small_AutoEncoder)
            network_kwargs (dict or None): kwargs used to initialize the model
            max_epochs (int): Maximum number of epochs per training cycle
            patience (int): Number of epochs without improvement before stopping
        """
        self._ensure_dataset_layout_bn3()
        network_class = self._resolve_network_class(network_class)
        network_kwargs = self._prepare_network_kwargs(network_class, network_kwargs)

        if network_class is wrCNN2D_AutoEncoder:
            self.trainer = WritheTrainer(device=self.device)
        else:
            self.trainer = Trainer(device=self.device)
        self.trainer.set_autoencoder(network_class, **network_kwargs)
        self.trainer.set_data(
            self.data,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            manual_seed=self.manual_seed,
        )
        self.trainer.prepare_optimiser()
        self.network_class = network_class
        self.network_kwargs = dict(network_kwargs)
        self.net = self.trainer.autoencoder

        ifnotmake(self.log_dir)
        ifnotmake(self.checkpoint_dir)
        self._print_status("train")

        # Old workflow behavior: train in chunks and continue while best improves.
        best = float("inf")
        final_fit_result = None
        while True:
            fit_result = self.trainer.run(
                epochs=max_epochs,
                log_filename='log_file.dat',
                log_folder=self.log_dir,
                checkpoint_folder=self.checkpoint_dir,
            )
            final_fit_result = fit_result
            if fit_result.best_loss is None or not (fit_result.best_loss < best):
                break
            best = fit_result.best_loss

        self.last_checkpoint_path = (
            final_fit_result.best_checkpoint if final_fit_result is not None else None
        )
        self.last_checkpoint_network_class = network_class
        print(
            f"Training complete. Best loss: {self.trainer.progress.best_loss}, "
            f"Best file: {self.trainer.progress.best_checkpoint}"
        )

    def train_cnn2d_ae(self, max_epochs=32, patience=32, latent_dim=2,
                       init_c=32, m=2, min_size=9, output_subfolder='cnn2d_ae'):
        """
        Convenience wrapper to train the 2D CNN autoencoder.
        """
        self.set_output_subfolder(output_subfolder, create=True)

        network_kwargs = dict(
            dm_dim=self._num_atoms(),
            latent_dim=latent_dim,
            init_c=init_c,
            m=m,
            min_size=min_size,
        )
        self.train(
            network_class=CNN2d_AutoEncoder,
            network_kwargs=network_kwargs,
            max_epochs=max_epochs,
            patience=patience,
        )

    def train_wrCNN2D_ae(self, max_epochs=32, patience=32, latent_dim=2,
                         init_c=32, m=2, min_size=9, output_subfolder='wrCNN2D_ae'):
        """
        Convenience wrapper to train the writhe-space 2D CNN autoencoder.
        """
        self.set_output_subfolder(output_subfolder, create=True)
        network_kwargs = dict(
            n_atoms=self._num_atoms(),
            latent_dim=latent_dim,
            init_c=init_c,
            m=m,
            min_size=min_size,
        )
        self.train(
            network_class=wrCNN2D_AutoEncoder,
            network_kwargs=network_kwargs,
            max_epochs=max_epochs,
            patience=patience,
        )
        
    def load_checkpoint(self, checkpoint_pattern=None, network_class=None):
        """
        Load a trained checkpoint.
        
        Args:
            checkpoint_pattern (str): Glob pattern for checkpoint files (default: all .ckpt files)
            network_class: Optional network class/alias override for checkpoint loading
        """
        requested_network_class = self._resolve_network_class(network_class) if network_class is not None else None

        networkfile = None
        if checkpoint_pattern is None and self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
            if (
                requested_network_class is None
                or self.last_checkpoint_network_class is None
                or requested_network_class is self.last_checkpoint_network_class
            ):
                networkfile = self.last_checkpoint_path

        if networkfile is None:
            if checkpoint_pattern is None:
                checkpoint_pattern = os.path.join(self.checkpoint_dir, 'checkpoint_*.ckpt')
            matching_files = sorted(glob.glob(checkpoint_pattern))
            if len(matching_files) == 0:
                raise FileNotFoundError(f"No files matched the pattern: {checkpoint_pattern}")
            # Use most recent matching checkpoint by filename sort.
            networkfile = matching_files[-1]

        checkpoint = torch.load(
            networkfile,
            map_location=torch.device('cpu'),
            weights_only=False,
        )

        ckpt_kwargs = checkpoint.get('network_kwargs', {})
        if requested_network_class is None:
            # Heuristic: dm_dim => CNN2d_AE checkpoint, else Small_AutoEncoder
            if 'dm_dim' in ckpt_kwargs and CNN2d_AutoEncoder is not None:
                requested_network_class = CNN2d_AutoEncoder
            elif 'n_atoms' in ckpt_kwargs and wrCNN2D_AutoEncoder is not None:
                requested_network_class = wrCNN2D_AutoEncoder
            else:
                requested_network_class = self.network_class or Small_AutoEncoder

        network_class = requested_network_class
        self.net = network_class(**ckpt_kwargs)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net = self.net.to(self.device)
        self.network_class = network_class
        self.network_kwargs = dict(ckpt_kwargs)
        self.last_checkpoint_path = networkfile
        self.last_checkpoint_network_class = network_class
        self._print_status("load_checkpoint")
        
        print(f"Loaded checkpoint from: {networkfile}")
        print(f"Network class: {network_class.__name__}")
        print(f"Network kwargs: {checkpoint['network_kwargs']}")

    def _build_analysis_pdb(self, dataset_tensor):
        """
        Build a lightweight PDBData object for analysis without deepcopy().
        """
        subset = PDBData()
        subset._mol = self.data._mol
        subset.dataset = dataset_tensor
        subset.std = self.data.std
        subset.mean = self.data.mean
        subset.standardize = getattr(self.data, "standardize", True)
        subset.indices = self.data.indices
        subset.cb_valid_idx = getattr(self.data, "cb_valid_idx", None)
        if hasattr(self.data, "_atom_names"):
            subset._atom_names = self.data._atom_names
        return subset
        
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
        self._print_status("analysis")
        self._ensure_dataset_layout_bn3()
        
        self.MA = MolearnAnalysis()
        self.MA.set_network(self.net)

        # Build train/valid tensors with the same split used by training.
        data_train, data_valid = self.data.get_datasets(
            validation_split=self.validation_split,
            manual_seed=self.manual_seed,
        )
        
        # Store as class attributes
        self.data_train, self.data_valid = data_train, data_valid

        # Reuse split indices generated by PDBData.get_datasets/get_dataloader.
        if not hasattr(self.data, "train_indices") or not hasattr(self.data, "valid_indices"):
            raise ValueError("Training/validation indices are unavailable on PDBData.")
        self.train_indices = self.data.train_indices.numpy()
        self.valid_indices = self.data.valid_indices.numpy()

        # Build analysis PDBData views without deepcopying MDAnalysis internals.
        data_train_pdb = self._build_analysis_pdb(data_train)
        data_valid_pdb = self._build_analysis_pdb(data_valid)
        self.MA.set_dataset("training", data_train_pdb)
        self.MA.set_dataset("validation", data_valid_pdb)
        
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
        self.MA.generate(latent_coords_train.numpy(), self.decoded_train_dir, relax=False)
        
        # Decode validation set
        ifnotmake(self.decoded_valid_dir)
        latent_coords_valid = self.MA.get_encoded('validation')
        self.MA.generate(latent_coords_valid.numpy(), self.decoded_valid_dir, relax=False)
        
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
        # scan_error() returns (rmsd_surface, z_drift_surface, xaxis, yaxis)
        landscape_err_3d, landscape_err_latent, xaxis, yaxis = self.MA.scan_error()
        
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
        model_device = next(self.net.parameters()).device
        with torch.no_grad():
            z_train = self.net.encode(self.data_train.float().to(model_device)).cpu()
            z_valid = self.net.encode(self.data_valid.float().to(model_device)).cpu()

        z_train_np = self._to_2d_latent_array(z_train)
        z_valid_np = self._to_2d_latent_array(z_valid)
        
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

    @staticmethod
    def _to_2d_latent_array(z_tensor):
        """
        Convert encoder output to Nx2 numpy array for downstream analysis.
        """
        z_np = z_tensor.detach().cpu().numpy()
        if z_np.ndim > 2:
            z_np = z_np.reshape(z_np.shape[0], -1)
        if z_np.ndim != 2:
            raise ValueError(f"Unexpected latent shape: {z_np.shape}")
        if z_np.shape[1] < 2:
            raise ValueError(
                f"Latent dimension must be at least 2 for landscape analysis, got {z_np.shape[1]}"
            )
        return z_np[:, :2]

    def run_standard_analysis(self, save_prefix='_foldingnet_checkpoint', grid_size=30,
                              decode=True, rename=True):
        """
        Run the standard post-training latent-space analysis pipeline.
        """
        if self.MA is None:
            self.setup_analysis()
        self._print_status("post_analysis")
        if decode:
            self.decode_structures()
        if rename:
            self.rename_files()
        err_train, err_valid = self.calculate_errors(save_prefix=save_prefix)
        self.scan_error_landscape(grid_size=grid_size, save_prefix=save_prefix)
        self.extract_encoded_coordinates(save_prefix=save_prefix)
        return err_train, err_valid

    def run_cnn2d_analysis(self, checkpoint_pattern=None, save_prefix='_cnn2d_ae_checkpoint',
                           grid_size=30, atom_selection=('CA',),
                           output_subfolder='cnn2d_ae'):
        """
        Convenience entrypoint to run analysis for a CNN2d_AE checkpoint.
        """
        self.set_output_subfolder(output_subfolder, create=True)
        if self.net is None:
            self.load_checkpoint(
                checkpoint_pattern=checkpoint_pattern,
                network_class='cnn2d_ae',
            )
        if self.MA is None:
            self.setup_analysis(atom_selection=list(atom_selection))
        return self.run_standard_analysis(
            save_prefix=save_prefix,
            grid_size=grid_size,
            decode=True,
            rename=True,
        )

    def run_repeats(self, model='small', n_repeats=5, atom_selection=('CA',),
                    max_epochs=32, patience=32, latent_dim=2, grid_size=30):
        """
        Run repeated train + core analysis cycles for a selected model.

        Returns:
            list[str]: output directories for each repeat.
        """
        run_dirs = []
        atom_selection = list(atom_selection)

        for i in range(n_repeats):
            seed = self.manual_seed + i
            wf = AutoencoderWorkflow(
                folder_name=self.folder_name,
                output_base_dir=self.output_root_dir,
                manual_seed=seed,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                device=self.device,
                processes=self.processes,
            )
            wf.prepare_data(atom_selection=atom_selection)

            if model == 'small':
                wf.set_output_subfolder(f'small_repeat_{i + 1}', create=True)
                wf.train(max_epochs=max_epochs, patience=patience)
            elif model == 'cnn2d':
                wf.train_cnn2d_ae(
                    max_epochs=max_epochs,
                    patience=patience,
                    latent_dim=latent_dim,
                    output_subfolder=f'cnn2d_repeat_{i + 1}',
                )
            else:
                raise ValueError("model must be 'small' or 'cnn2d'")

            wf.setup_analysis(atom_selection=atom_selection)
            prefix = f'_{model}_checkpoint'
            wf.calculate_errors(save_prefix=prefix)
            wf.scan_error_landscape(grid_size=grid_size, save_prefix=prefix)
            wf.extract_encoded_coordinates(save_prefix=prefix)
            run_dirs.append(wf.output_base_dir)

        return run_dirs

    @staticmethod
    def plot_latent_spaces_across_runs(run_dirs, output_file='LatentSpaces_5Runs.png'):
        """
        Plot train/validation latent spaces for all runs side-by-side.
        """
        fig, axes = plt.subplots(2, len(run_dirs), figsize=(4 * len(run_dirs), 8),
                                 sharex=False, sharey=False)
        if len(run_dirs) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for i, run_dir in enumerate(run_dirs):
            train_csv = os.path.join(run_dir, 'landscape_encoded_train_coordinates.csv')
            valid_csv = os.path.join(run_dir, 'landscape_encoded_valid_coordinates.csv')
            if not os.path.exists(train_csv) or not os.path.exists(valid_csv):
                axes[0, i].set_title(f'run_{i + 1} missing')
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                continue

            train_xy = pd.read_csv(train_csv).to_numpy()
            valid_xy = pd.read_csv(valid_csv).to_numpy()

            axes[0, i].scatter(train_xy[:, 0], train_xy[:, 1], s=5, alpha=0.7)
            axes[1, i].scatter(valid_xy[:, 0], valid_xy[:, 1], s=5, alpha=0.7, color='tab:orange')

            axes[0, i].set_title(f'run_{i + 1}')
            axes[0, i].set_ylabel('train' if i == 0 else '')
            axes[1, i].set_ylabel('valid' if i == 0 else '')
            axes[1, i].set_xlabel('latent dim 1')

        fig.suptitle('Latent Spaces Across Repeats', y=1.02)
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        return fig, axes

    @staticmethod
    def plot_violin_rmsd_across_runs(run_dirs, output_file='ViolinRMSD_5Runs.png', ymax=10):
        """
        Plot train/validation RMSD violin distributions across run directories.
        """
        from violinPlotRMSD import plot_violin_rmsd
        return plot_violin_rmsd(run_dirs, output_file=output_file, ymax=ymax)

    @staticmethod
    def plot_training_histories_across_runs(run_dirs, output_prefix='TrainingHistory'):
        """
        Create one training-history plot per run directory.

        Returns:
            list[str]: saved figure paths.
        """
        saved = []
        for i, run_dir in enumerate(run_dirs):
            log_file = os.path.join(run_dir, 'xbb_foldingnet_checkpoints', 'log_file.dat')
            if not os.path.exists(log_file):
                continue

            log_data = pd.read_csv(log_file)
            required = {'epoch', 'train_loss', 'valid_loss'}
            if not required.issubset(set(log_data.columns)):
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(log_data['epoch'], log_data['train_loss'], label='Training Loss', linewidth=2)
            ax.plot(log_data['epoch'], log_data['valid_loss'], label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Run {i + 1}: Training/Validation Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()

            out_path = os.path.join(run_dir, f'{output_prefix}_run{i + 1}.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved.append(out_path)

        return saved
        
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
            vmin (float): Minimum value for RMSD colorbar.
            vmax (float): Maximum value for RMSD colorbar.
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

    def plot_rmsd_landscape_projection(self, labels=None, output_file=None,
                                      save_prefix='_foldingnet_checkpoint',
                                      vmin=0., vmax=10., point_cmap='Paired'):
        """
        Plot training and validation latent projections on top of RMSD landscape.

        This reproduces the classic two-panel view:
        - left: training latent coordinates on RMSD landscape
        - right: validation latent coordinates on RMSD landscape

        Args:
            labels (tuple or None): Tuple (labels_train, labels_valid) used to color points.
                                   If None, stored workflow labels are used.
            output_file (str or None): Optional output filename inside output_base_dir.
            save_prefix (str): Prefix used when loading landscape CSV files.
            vmin (float): Minimum RMSD value for landscape colormap.
            vmax (float): Maximum RMSD value for landscape colormap.
            point_cmap (str): Colormap for scatter points.

        Returns:
            tuple: (fig, (ax1, ax2)) matplotlib figure and axes objects.
        """
        # Load encoded coordinates
        train_coords_file = os.path.join(
            self.output_base_dir, 'landscape_encoded_train_coordinates.csv'
        )
        valid_coords_file = os.path.join(
            self.output_base_dir, 'landscape_encoded_valid_coordinates.csv'
        )
        X_train = pd.read_csv(train_coords_file, header=0).to_numpy()
        X_valid = pd.read_csv(valid_coords_file, header=0).to_numpy()
        x_encoded_train = X_train[:, 0]
        y_encoded_train = X_train[:, 1]
        x_encoded_valid = X_valid[:, 0]
        y_encoded_valid = X_valid[:, 1]

        # Load RMSD landscape
        df_z = pd.read_csv(os.path.join(
            self.output_base_dir, f'landscape_err_3d_{save_prefix}.csv'
        ), header=0).to_numpy()
        df_x = pd.read_csv(os.path.join(
            self.output_base_dir, f'landscape_err_xaxis_{save_prefix}.csv'
        ), header=0).to_numpy().flatten()
        df_y = pd.read_csv(os.path.join(
            self.output_base_dir, f'landscape_err_yaxis_{save_prefix}.csv'
        ), header=0).to_numpy().flatten()


        # Decide labels used for point coloring
        if labels is None:
            if self.labels_train is None or self.labels_valid is None:
                labels_train = np.zeros(len(X_train))
                labels_valid = np.zeros(len(X_valid))
            else:
                labels_train = self.labels_train
                labels_valid = self.labels_valid
        else:
            labels_train, labels_valid = labels

        # Validate label sizes
        if len(labels_train) != len(X_train):
            raise ValueError(
                f"Training size mismatch: {len(X_train)} latent points but "
                f"{len(labels_train)} labels"
            )
        if len(labels_valid) != len(X_valid):
            raise ValueError(
                f"Validation size mismatch: {len(X_valid)} latent points but "
                f"{len(labels_valid)} labels"
            )

        fig = plt.figure(figsize=(8, 8))

        # Two square subplot axes
        ax1 = fig.add_axes([0.0, 0.1, 0.35, 0.35])
        ax1.imshow(
            df_z, cmap='viridis', vmin=vmin, vmax=vmax,
            extent=[np.min(df_x), np.max(df_x), np.min(df_y), np.max(df_y)]
        )
        ax1.scatter(
            x_encoded_train, y_encoded_train, c=labels_train,
            marker='.', cmap=point_cmap
        )

        ax2 = fig.add_axes([0.38, 0.1, 0.35, 0.35])
        im = ax2.imshow(
            df_z, cmap='viridis', vmin=vmin, vmax=vmax,
            extent=[np.min(df_x), np.max(df_x), np.min(df_y), np.max(df_y)]
        )
        ax2.scatter(
            x_encoded_valid, y_encoded_valid, c=labels_valid,
            marker='.', cmap=point_cmap
        )

        # RMSD colorbar for the landscape
        cbar_ax = fig.add_axes([0.755, 0.1, 0.02, 0.35])
        cbar_ax.tick_params(
            left=False, labelleft=False, right=True, labelright=True,
            labelbottom=False, bottom=False
        )
        cbar = fig.colorbar(im, cax=cbar_ax, label='RMSD [$\\AA$]')
        cbar_ticks = np.linspace(vmin, vmax, 7)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{tick:.0f}' for tick in cbar_ticks])

        # Axes styling
        ax1.tick_params(direction='inout', labelbottom=True, top=False, bottom=True)
        ax2.tick_params(
            direction='inout', labelbottom=True, top=False, bottom=True,
            left=False, labelleft=False
        )

        ax1.set_xlabel('Latent vector 1')
        ax1.set_ylabel('Latent vector 2')
        ax2.set_xlabel('Latent vector 1')

        ax1.set_title('Training dataset')
        ax2.set_title('Validation dataset')

        if output_file:
            save_path = os.path.join(self.output_base_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved RMSD landscape projection to {save_path}")

        plt.show()
        return fig, (ax1, ax2)

    def plot_error_violins(self, output_file=None, save_prefix='_foldingnet_checkpoint',
                           train_color='#4C78A8', valid_color='#F58518'):
        """
        Plot train/validation reconstruction errors as two violins on one axis.

        Args:
            output_file (str or None): Optional output filename inside output_base_dir.
            save_prefix (str): Prefix used when loading error CSV files.
            train_color (str): Fill color for training violin.
            valid_color (str): Fill color for validation violin.

        Returns:
            tuple: (fig, ax, err_train, err_valid)
        """
        err_train_file = os.path.join(
            self.output_base_dir, f'err_train_{save_prefix}.csv'
        )
        err_valid_file = os.path.join(
            self.output_base_dir, f'err_valid_{save_prefix}.csv'
        )

        if not os.path.exists(err_train_file):
            raise FileNotFoundError(
                f"Training error file not found: {err_train_file}. "
                "Run calculate_errors() first."
            )
        if not os.path.exists(err_valid_file):
            raise FileNotFoundError(
                f"Validation error file not found: {err_valid_file}. "
                "Run calculate_errors() first."
            )

        err_train = pd.read_csv(err_train_file).iloc[:, 0].to_numpy()
        err_valid = pd.read_csv(err_valid_file).iloc[:, 0].to_numpy()

        fig, ax = plt.subplots(figsize=(7, 5))
        violin = ax.violinplot(
            [err_train, err_valid],
            positions=[1, 2],
            widths=0.8,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )

        for i, body in enumerate(violin['bodies']):
            body.set_alpha(0.7)
            body.set_facecolor(train_color if i == 0 else valid_color)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)

        # Keep mean/median/extrema visible
        for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            if key in violin:
                violin[key].set_color('black')
                violin[key].set_linewidth(1.0)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Training', 'Validation'])
        ax.set_ylabel('RMSD [$\\AA$]')
        ax.set_title('Train vs Validation Reconstruction Error')
        ax.grid(axis='y', alpha=0.25)

        if output_file:
            save_path = os.path.join(self.output_base_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved error violin plot to {save_path}")

        plt.show()
        return fig, ax, err_train, err_valid

    def plot_training_history(self, log_filename='log.dat', output_file=None,
                              title='Training and Validation Loss Over Time',
                              latest_only=True):
        """
        Plot training/validation loss over epochs from the training log file.

        Args:
            log_filename (str): Log filename inside self.log_dir.
            output_file (str or None): Optional output filename inside output_base_dir.
            title (str): Figure title.
            latest_only (bool): If True, plot only the latest run segment when
                                log file contains multiple appended runs.

        Returns:
            tuple: (fig, ax, log_data)
        """
        log_file = os.path.join(self.log_dir, log_filename)
        if not os.path.exists(log_file):
            raise FileNotFoundError(
                f"Log file not found: {log_file}. "
                "Train the model first or provide a valid log_filename."
            )

        log_data = pd.read_csv(log_file)
        required_cols = {'epoch', 'train_loss', 'valid_loss'}
        missing = required_cols.difference(set(log_data.columns))
        if missing:
            raise ValueError(
                f"Missing required columns in {log_file}: {sorted(missing)}"
            )

        if latest_only and len(log_data) > 1:
            # Split appended runs by epoch reset (e.g., ...31 then 0)
            epoch_vals = log_data['epoch'].to_numpy()
            reset_points = np.where(np.diff(epoch_vals) < 0)[0]
            if len(reset_points) > 0:
                start_idx = int(reset_points[-1] + 1)
                log_data = log_data.iloc[start_idx:].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(log_data['epoch'], log_data['train_loss'],
                label='Training Loss', linewidth=2)
        ax.plot(log_data['epoch'], log_data['valid_loss'],
                label='Validation Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_file:
            save_path = os.path.join(self.output_base_dir, output_file)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history plot to {save_path}")

        plt.show()

        final_train = log_data['train_loss'].iloc[-1]
        final_valid = log_data['valid_loss'].iloc[-1]
        best_valid_idx = log_data['valid_loss'].idxmin()
        best_valid = log_data.loc[best_valid_idx, 'valid_loss']
        best_epoch = log_data.loc[best_valid_idx, 'epoch']

        print(f"Final training loss: {final_train:.6f}")
        print(f"Final validation loss: {final_valid:.6f}")
        print(f"Best validation loss: {best_valid:.6f} (epoch {best_epoch:.0f})")

        return fig, ax, log_data


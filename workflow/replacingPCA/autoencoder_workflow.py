"""
CNN2d / Small / writheCH2 autoencoder workflow for BRAF CG activation-loop segments.

Slimmed from the JoshCollab wr2DCNNAE AutoencoderWorkflow: PDB-folder prepare,
molearn CNN2d_AE / Small FoldingNet / writheCH2 training, analysis setup,
optional decode, and Kabsch-aligned reconstruction metrics via ae_aligned_export.
"""

from __future__ import annotations

import glob
import os
import subprocess

import MDAnalysis as mda
import numpy as np
import torch
from molearn.analysis.analyser import MolearnAnalysis
from molearn.data import PDBData
from molearn.models.CNN2d_AE import AutoEncoder as CNN2d_AutoEncoder
from molearn.trainers import Trainer
from tqdm import tqdm

from workflow.utilities import ifnotmake

from .small_foldingnet_latent import Small_AutoEncoder
from .wrCNN2D_ch2 import AutoEncoder as writheCH2_AutoEncoder
from .wrTrainer import WritheCH2Trainer


def _gpu_free_memory_mib():
    """Return free MiB per GPU via nvidia-smi (no CUDA context init)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    free = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line:
            free.append(int(line))
    return free or None


def _select_cuda_device(device=None):
    """Resolve a CUDA device; default to the GPU with the most free memory."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this workflow, but no CUDA device is available."
        )
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got {resolved}")
        torch.cuda.set_device(resolved)
        return resolved

    free_mib = _gpu_free_memory_mib()
    if free_mib is None:
        resolved = torch.device("cuda:0")
    else:
        best_idx = int(max(range(len(free_mib)), key=lambda i: free_mib[i]))
        resolved = torch.device(f"cuda:{best_idx}")
    torch.cuda.set_device(resolved)
    return resolved


def _device_memory_summary(device):
    """Human-readable free/total MiB for ``device`` without probing other GPUs."""
    free_mib = _gpu_free_memory_mib()
    idx = device.index if device.index is not None else 0
    if free_mib is not None and 0 <= idx < len(free_mib):
        return f"{free_mib[idx]} MiB free (nvidia-smi)"
    try:
        free, total = torch.cuda.mem_get_info(device)
        return f"{free / 1024**2:.0f} / {total / 1024**2:.0f} MiB free"
    except Exception:
        return "memory info unavailable"


class AutoencoderWorkflow:
    """Train and evaluate molearn CNN2d_AE on a folder of per-structure PDBs."""

    def __init__(
        self,
        folder_name,
        output_base_dir,
        manual_seed=25,
        batch_size=8,
        validation_split=0.1,
        device=None,
        processes=4,
    ):
        self.folder_name = folder_name
        self.output_root_dir = output_base_dir
        self.output_base_dir = output_base_dir
        self.manual_seed = manual_seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.device = _select_cuda_device(device)
        print(f"Using {self.device} ({_device_memory_summary(self.device)})")
        self.processes = processes

        self.combined_file_path = os.path.join(folder_name, "combined.pdb")
        self.output_subfolder = None
        self._refresh_output_paths()

        self.data = None
        self.trainer = None
        self.net = None
        self.MA = None
        self.data_train = None
        self.data_valid = None
        self.train_indices = None
        self.valid_indices = None
        self.network_class = CNN2d_AutoEncoder
        self.network_kwargs = {}
        self.last_checkpoint_path = None
        self.last_checkpoint_network_class = None

    def _refresh_output_paths(self):
        self.checkpoint_dir = self.output_base_dir
        self.log_dir = os.path.join(self.output_base_dir, "xbb_foldingnet_checkpoints")
        self.get_dataset_dir = os.path.join(self.output_base_dir, "getDatasetTrial")
        self.decoded_train_dir = os.path.join(self.output_base_dir, "decoded_train")
        self.decoded_valid_dir = os.path.join(self.output_base_dir, "decoded_valid")

    def set_output_subfolder(self, subfolder=None, create=True):
        self.output_subfolder = subfolder
        if subfolder:
            self.output_base_dir = os.path.join(self.output_root_dir, subfolder)
        else:
            self.output_base_dir = self.output_root_dir
        self._refresh_output_paths()
        if create:
            ifnotmake(self.output_base_dir)
        self._print_status("output")

    def export_kabsch_aligned_datasets(self, **kwargs):
        from .ae_aligned_export import export_kabsch_aligned_datasets as _export

        self._print_status("export_aligned")
        return _export(self, **kwargs)

    def plot_rmsd_comparison(self, aligned_export=None, show=True, **kwargs):
        from .ae_aligned_export import plot_rmsd_comparison as _plot

        self._print_status("plot_rmsd")
        return _plot(self, aligned_export=aligned_export, show=show, **kwargs)

    def plot_rg_comparison(self, aligned_export=None, show=True, **kwargs):
        from .ae_aligned_export import plot_rg_comparison as _plot

        self._print_status("plot_rg")
        return _plot(self, aligned_export=aligned_export, show=show, **kwargs)

    def plot_rmsf_comparison(self, aligned_export=None, show=True, **kwargs):
        from .ae_aligned_export import plot_rmsf_comparison as _plot

        self._print_status("plot_rmsf")
        return _plot(self, aligned_export=aligned_export, show=show, **kwargs)

    def plot_ca_bondlength_comparison(self, aligned_export=None, show=True, **kwargs):
        from .ae_aligned_export import plot_ca_bondlength_comparison as _plot

        self._print_status("plot_ca_bondlength")
        return _plot(self, aligned_export=aligned_export, show=show, **kwargs)

    def plot_ca_angle_comparison(self, aligned_export=None, show=True, **kwargs):
        from .ae_aligned_export import plot_ca_angle_comparison as _plot

        self._print_status("plot_ca_angle")
        return _plot(self, aligned_export=aligned_export, show=show, **kwargs)

    def _print_status(self, stage):
        model_name = (
            self.network_class.__name__
            if hasattr(self.network_class, "__name__")
            else str(self.network_class)
        )
        print(f"[{stage}] model={model_name} output={self.output_base_dir}")

    def _clean_run_outputs(self):
        removed_ckpts = []
        if os.path.isdir(self.checkpoint_dir):
            for fname in sorted(os.listdir(self.checkpoint_dir)):
                if fname.endswith(".ckpt"):
                    fp = os.path.join(self.checkpoint_dir, fname)
                    try:
                        os.remove(fp)
                        removed_ckpts.append(fname)
                    except OSError as exc:
                        print(f"[train] WARNING: could not remove {fp}: {exc}")

        removed_logs = []
        log_basenames = ("log.dat", "log_file.dat")
        if os.path.isdir(self.log_dir):
            for fname in sorted(os.listdir(self.log_dir)):
                is_log = fname in log_basenames or any(
                    fname.startswith(b + "_") for b in log_basenames
                )
                if is_log:
                    fp = os.path.join(self.log_dir, fname)
                    try:
                        os.remove(fp)
                        removed_logs.append(fname)
                    except OSError as exc:
                        print(f"[train] WARNING: could not remove {fp}: {exc}")

        if removed_ckpts or removed_logs:
            parts = []
            if removed_ckpts:
                parts.append(
                    f"{len(removed_ckpts)} checkpoint(s) in {self.checkpoint_dir}"
                )
            if removed_logs:
                parts.append(f"{len(removed_logs)} log file(s) in {self.log_dir}")
            print(f"[train] Cleaned previous run artifacts: {'; '.join(parts)}.")

    def prepare_data(self, atom_selection=("CA",)):
        """Combine per-structure PDBs into combined.pdb and load as PDBData."""
        selected_atoms = list(atom_selection)
        files = sorted(
            [
                f
                for f in os.listdir(self.folder_name)
                if f.endswith(".pdb")
                and f != "combined.pdb"
                and os.path.isfile(os.path.join(self.folder_name, f))
            ]
        )
        if not files:
            raise FileNotFoundError(f"No PDB files found in {self.folder_name}")

        with open(self.combined_file_path, "w") as combined_file:
            for i, filename in enumerate(files):
                file_path = os.path.join(self.folder_name, filename)
                with open(file_path, "r") as file:
                    lines = [
                        line
                        for line in file.readlines()
                        if not line.startswith(("MODEL", "END"))
                    ]
                combined_file.write(f"MODEL {i}\n")
                combined_file.writelines(lines)
                combined_file.write("ENDMDL\n")
            combined_file.write("END\n")

        self._load_pdb_dataset(self.combined_file_path, selected_atoms)
        print(f"Combined {len(files)} PDBs into {self.combined_file_path}")

    def _load_pdb_dataset(self, pdb_path, selected_atoms):
        self.data = PDBData()
        self.data.import_pdb(filename=pdb_path)
        self._standardize_loaded_data(selected_atoms)

    @staticmethod
    def _ensure_mdanalysis_pdb_metadata(mol):
        n_atoms = mol.atoms.n_atoms
        for attr, values in (
            ("record_type", ["ATOM"] * n_atoms),
            ("chainID", ["A"] * n_atoms),
        ):
            try:
                getattr(mol.atoms[0], attr)
            except mda.exceptions.NoDataError:
                mol.add_TopologyAttr(attr, values)

    def _standardize_loaded_data(self, selected_atoms):
        self._ensure_mdanalysis_pdb_metadata(self.data._mol)
        self.data.fix_terminal()
        self.data.atomselect(atoms=selected_atoms)
        if len(selected_atoms) == 1 and selected_atoms[0] == "CA":
            coords = np.asarray(
                [
                    self.data._mol.atoms.positions.astype(float)
                    for _ in self.data._mol.trajectory
                ]
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

    def _real_input_coords_from_indices(self, indices):
        if self.data is None or not hasattr(self.data, "dataset"):
            raise ValueError("Dataset not prepared. Call prepare_data() first.")
        coords_std = self.data.dataset[np.asarray(indices)].cpu().numpy()
        return (coords_std * self.data.std + self.data.mean).astype(np.float32)

    @staticmethod
    def _kabsch_align_pair(P, Q):
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        cP = P.mean(axis=0)
        cQ = Q.mean(axis=0)
        H = (P - cP).T @ (Q - cQ)
        U, _, Vt = np.linalg.svd(H)
        V = Vt.T
        d = np.sign(np.linalg.det(V @ U.T))
        if d == 0:
            d = 1.0
        R = V @ np.diag([1.0, 1.0, d]) @ U.T
        return ((P - cP) @ R.T + cQ).astype(np.float32)

    @classmethod
    def _kabsch_align_batch(cls, P_batch, Q_batch):
        if len(P_batch) != len(Q_batch):
            raise ValueError(
                f"frame count mismatch: {len(P_batch)} vs {len(Q_batch)}"
            )
        return np.stack(
            [
                cls._kabsch_align_pair(P_batch[i], Q_batch[i])
                for i in range(len(P_batch))
            ],
            axis=0,
        )

    def _write_multimodel_pdb(self, coords_array, path):
        coords_array = np.asarray(coords_array, dtype=np.float32)
        if coords_array.ndim != 3 or coords_array.shape[2] != 3:
            raise ValueError(
                f"expected [n_frames, n_atoms, 3], got {coords_array.shape}"
            )

        if self.data is None or not hasattr(self.data, "_mol"):
            self._write_multimodel_ca_pdb(coords_array, path)
            return

        ref_atoms = self.data._mol.atoms
        if coords_array.shape[1] != len(ref_atoms):
            raise ValueError(
                f"atom count mismatch: coords has {coords_array.shape[1]} "
                f"atoms but self.data._mol carries {len(ref_atoms)}"
            )

        from MDAnalysis.coordinates.memory import MemoryReader

        new_u = mda.Merge(ref_atoms)
        new_u.load_new(coords_array, format=MemoryReader)
        with mda.Writer(path, n_atoms=len(ref_atoms), multiframe=True) as W:
            for _ in new_u.trajectory:
                W.write(new_u.atoms)

    @staticmethod
    def _write_multimodel_ca_pdb(coords_array, path, chain="A", resname="ALA"):
        coords_array = np.asarray(coords_array, dtype=np.float32)
        if coords_array.ndim != 3 or coords_array.shape[2] != 3:
            raise ValueError(
                f"expected [n_frames, n_atoms, 3], got {coords_array.shape}"
            )
        with open(path, "w") as f:
            for model_idx, coords in enumerate(coords_array, start=1):
                f.write(f"MODEL    {model_idx:5d}\n")
                for i, (x, y, z) in enumerate(coords, start=1):
                    f.write(
                        f"ATOM  {i:5d}  CA  {resname} {chain}{i:4d}    "
                        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                        f"  1.00  0.00           C\n"
                    )
                f.write("ENDMDL\n")
            f.write("END\n")

    @staticmethod
    def _purge_per_frame_pdbs(directory):
        if not os.path.isdir(directory):
            return 0
        removed = 0
        for fname in os.listdir(directory):
            if not fname.endswith(".pdb"):
                continue
            if fname.startswith("s") and (
                fname[1:2].isdigit() or fname.startswith("s_")
            ):
                fp = os.path.join(directory, fname)
                try:
                    os.remove(fp)
                    removed += 1
                except OSError as exc:
                    print(f"WARNING: could not remove {fp}: {exc}")
        return removed

    def _ensure_dataset_layout_bn3(self):
        if self.data is None or not hasattr(self.data, "dataset"):
            raise ValueError("Dataset is not prepared yet. Call prepare_data() first.")

        dataset = self.data.dataset
        if dataset.ndim != 3:
            raise ValueError(f"Expected 3D dataset, got shape {tuple(dataset.shape)}")

        if dataset.shape[-1] == 3:
            return
        if dataset.shape[1] == 3:
            self.data.dataset = dataset.permute(0, 2, 1).contiguous()
            return

        raise ValueError(
            f"Cannot convert dataset shape {tuple(dataset.shape)} to [frames, n_atoms, 3]"
        )

    def _num_atoms(self):
        self._ensure_dataset_layout_bn3()
        return self.data.dataset.shape[1]

    def _prepare_network_kwargs(self, network_class, network_kwargs=None):
        """Fill model-specific defaults from the loaded dataset."""
        kwargs = dict(network_kwargs or {})
        n_points = self._num_atoms()
        if network_class is CNN2d_AutoEncoder:
            kwargs.setdefault("dm_dim", n_points)
            kwargs.setdefault("latent_dim", 2)
        elif network_class is Small_AutoEncoder:
            kwargs.setdefault("out_points", n_points)
            kwargs.setdefault("latent_dimension", 2)
        elif network_class is writheCH2_AutoEncoder:
            kwargs.setdefault("n_atoms", n_points)
            kwargs.setdefault("latent_dim", 2)
            kwargs.setdefault("in_channels", 2)
        return kwargs

    @staticmethod
    def _infer_network_class(ckpt_kwargs, fallback=None):
        if "dm_dim" in ckpt_kwargs:
            return CNN2d_AutoEncoder
        if "out_points" in ckpt_kwargs or "latent_dimension" in ckpt_kwargs:
            return Small_AutoEncoder
        if "n_atoms" in ckpt_kwargs:
            return writheCH2_AutoEncoder
        return fallback or CNN2d_AutoEncoder

    def train(
        self,
        network_class=CNN2d_AutoEncoder,
        network_kwargs=None,
        max_epochs=32,
        patience=32,
        trainer_kwargs=None,
        trainer_class=None,
    ):
        """Train with molearn Trainer or WritheCH2Trainer. ``patience`` kept for API parity."""
        del patience  # outer loop stops when best loss no longer improves
        self._ensure_dataset_layout_bn3()
        network_kwargs = self._prepare_network_kwargs(network_class, network_kwargs)
        trainer_kwargs = dict(trainer_kwargs or {})

        # Drop stale CUDA allocations from prior notebook cells / failed runs.
        if self.trainer is not None:
            self.trainer = None
        if self.net is not None:
            self.net = None
        if self.MA is not None:
            self.MA = None
        import gc

        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        print(
            f"[train] {self.device} before model init: "
            f"{_device_memory_summary(self.device)}"
        )

        if trainer_class is not None:
            self.trainer = trainer_class(device=self.device, **trainer_kwargs)
        elif network_class is writheCH2_AutoEncoder:
            self.trainer = WritheCH2Trainer(device=self.device, **trainer_kwargs)
        else:
            if trainer_kwargs:
                raise ValueError(
                    f"trainer_kwargs={trainer_kwargs} are only supported for "
                    "writheCH2_AutoEncoder / custom trainer_class; "
                    "the base Trainer accepts none."
                )
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
        self._clean_run_outputs()
        self._print_status("train")

        best = float("inf")
        final_fit_result = None
        while True:
            fit_result = self.trainer.run(
                epochs=max_epochs,
                log_filename="log_file.dat",
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

    def train_cnn2d_ae(
        self,
        max_epochs=32,
        patience=32,
        latent_dim=2,
        init_c=32,
        m=2,
        min_size=9,
        output_subfolder="cnn2d_ae",
    ):
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

    def train_small_ae(
        self,
        max_epochs=32,
        patience=32,
        latent_dimension=2,
        output_subfolder="small_ae",
    ):
        """Train the Small FoldingNet AE (December-workflow model family)."""
        self.set_output_subfolder(output_subfolder, create=True)
        network_kwargs = dict(
            out_points=self._num_atoms(),
            latent_dimension=latent_dimension,
        )
        self.train(
            network_class=Small_AutoEncoder,
            network_kwargs=network_kwargs,
            max_epochs=max_epochs,
            patience=patience,
        )

    def train_writheCH2_ae(
        self,
        max_epochs=32,
        patience=32,
        latent_dim=2,
        init_c=32,
        m=2,
        min_size=9,
        beta=0.0275,
        output_subfolder="writheCH2_ae",
    ):
        """Train the 2-channel writhe + reciprocal-DM autoencoder (wr2DCNN / writheCH2).

        Encoder input is ``[writhe, R=1/D]``; decoder outputs coordinates.
        Loss: ``L = MSE_offdiag(W_hat - W) + beta * MSE(R_hat - R)``.
        """
        self.set_output_subfolder(output_subfolder, create=True)
        network_kwargs = dict(
            n_atoms=self._num_atoms(),
            latent_dim=latent_dim,
            init_c=init_c,
            m=m,
            min_size=min_size,
            in_channels=2,
        )
        self.train(
            network_class=writheCH2_AutoEncoder,
            network_kwargs=network_kwargs,
            max_epochs=max_epochs,
            patience=patience,
            trainer_kwargs=dict(beta=beta),
        )

    def load_checkpoint(self, checkpoint_pattern=None, network_class=None):
        networkfile = None
        if (
            checkpoint_pattern is None
            and self.last_checkpoint_path
            and os.path.exists(self.last_checkpoint_path)
        ):
            networkfile = self.last_checkpoint_path

        if networkfile is None:
            if checkpoint_pattern is None:
                checkpoint_pattern = os.path.join(
                    self.checkpoint_dir, "checkpoint_*.ckpt"
                )
            matching_files = sorted(glob.glob(checkpoint_pattern))
            if len(matching_files) == 0:
                raise FileNotFoundError(
                    f"No files matched the pattern: {checkpoint_pattern}"
                )
            networkfile = matching_files[-1]

        checkpoint = torch.load(
            networkfile,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        ckpt_kwargs = checkpoint.get("network_kwargs", {})
        if network_class is None:
            network_class = self._infer_network_class(
                ckpt_kwargs, fallback=self.network_class or CNN2d_AutoEncoder
            )

        self.net = network_class(**ckpt_kwargs)
        self.net.load_state_dict(checkpoint["model_state_dict"])
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

    def setup_analysis(self, atom_selection=("CA",)):
        if self.data is None:
            self.prepare_data(atom_selection=atom_selection)

        if self.net is None:
            raise ValueError(
                "Model not loaded. Call train_cnn2d_ae()/train_small_ae()/"
                "train_writheCH2_ae() or load_checkpoint() first."
            )
        self._print_status("analysis")
        self._ensure_dataset_layout_bn3()

        self.MA = MolearnAnalysis()
        self.MA.set_network(self.net)

        data_train, data_valid = self.data.get_datasets(
            validation_split=self.validation_split,
            manual_seed=self.manual_seed,
        )
        self.data_train, self.data_valid = data_train, data_valid

        if not hasattr(self.data, "train_indices") or not hasattr(
            self.data, "valid_indices"
        ):
            raise ValueError("Training/validation indices are unavailable on PDBData.")
        self.train_indices = self.data.train_indices.numpy()
        self.valid_indices = self.data.valid_indices.numpy()

        self.MA.set_dataset("training", self._build_analysis_pdb(data_train))
        self.MA.set_dataset("validation", self._build_analysis_pdb(data_valid))
        self.MA.batch_size = self.batch_size
        self.MA.processes = self.processes

    def decode_structures(self, align_to_input=True):
        ifnotmake(self.decoded_train_dir)
        ifnotmake(self.decoded_valid_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not align_to_input:
            for key, out_dir in (
                ("training", self.decoded_train_dir),
                ("validation", self.decoded_valid_dir),
            ):
                decoded_std = self.MA.get_decoded(key, update=True)
                gen_coords = (
                    decoded_std * self.MA.stdval + self.MA.meanval
                ).cpu().numpy()
                for i, coord in enumerate(
                    tqdm(gen_coords, desc=f"Generating pdb files ({key})")
                ):
                    self.MA._pdb_file(coord, os.path.join(out_dir, f"s_{i}.pdb"))
            return

        decoded_train = (
            self.MA.get_decoded("training", update=True) * self.MA.stdval
            + self.MA.meanval
        ).cpu().numpy()
        decoded_valid = (
            self.MA.get_decoded("validation", update=True) * self.MA.stdval
            + self.MA.meanval
        ).cpu().numpy()

        input_train = self._real_input_coords_from_indices(self.train_indices)
        input_valid = self._real_input_coords_from_indices(self.valid_indices)
        decoded_train_aligned = self._kabsch_align_batch(decoded_train, input_train)
        decoded_valid_aligned = self._kabsch_align_batch(decoded_valid, input_valid)

        for d in (self.decoded_train_dir, self.decoded_valid_dir):
            n_purged = self._purge_per_frame_pdbs(d)
            if n_purged:
                print(
                    f"[decode_structures] purged {n_purged} stale per-frame PDB(s) in {d}"
                )

        train_path = os.path.join(self.decoded_train_dir, "decoded_train.pdb")
        valid_path = os.path.join(self.decoded_valid_dir, "decoded_valid.pdb")
        self._write_multimodel_pdb(decoded_train_aligned, train_path)
        self._write_multimodel_pdb(decoded_valid_aligned, valid_path)
        print("[decode_structures] wrote Kabsch-aligned multi-MODEL decodes:")
        print(f"  {train_path}  ({decoded_train_aligned.shape[0]} frames)")
        print(f"  {valid_path}  ({decoded_valid_aligned.shape[0]} frames)")

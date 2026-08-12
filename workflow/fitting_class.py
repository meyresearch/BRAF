#!/usr/bin/env python3
"""
Protein Structure Fitting Class

This module provides a class-based approach for fitting protein structures using
cubic interpolation along the backbone. It works with already CA-stripped structures.
"""

import os
import shutil
import tempfile
import contextlib
import io
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import Bio.PDB as PDB
import mdtraj as md
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import copy


class Fitting:
    """
    A class for fitting protein structures using cubic interpolation along the backbone.
    Assumes input structures are already CA-stripped.
    """
    
    # Default template metadata extracted from the historical `template.pdb` (CA-only, chain A, resid 593..619).
    # Template length can be overridden at runtime via ``n_ca_template``.
    _TEMPLATE_CHAIN_ID = "A"
    _TEMPLATE_START_RESSEQ = 593
    _TEMPLATE_DEFAULT_RESNAMES = [
        "ASP",
        "PHE",
        "GLY",
        "LEU",
        "ALA",
        "THR",
        "VAL",
        "LYS",
        "SER",
        "ARG",
        "TRP",
        "SER",
        "GLY",
        "SER",
        "HIS",
        "GLN",
        "PHE",
        "GLU",
        "GLN",
        "LEU",
        "SER",
        "GLY",
        "SER",
        "ILE",
        "LEU",
        "TRP",
        "MET",
    ]

    def __init__(
        self,
        template_path: str | None = None,
        n_ca_template: int | None = None,
        copy_exact_length: bool = True,
    ):
        """
        Initialize the Fitting class.

        Parameters:
        -----------
        template_path : str | None
            Optional path to the template PDB file for CA atom configuration.
            If None, an in-memory template is built from hardcoded metadata.
        n_ca_template : int | None
            If provided, build an in-memory CA-only template with exactly this many CA atoms.
        copy_exact_length : bool, default=True
            If True, structures whose CA count already equals ``n_ca_template`` / ``Nnew``
            are written with their original coordinates (mapped onto the template) instead
            of spline fitting.
        """
        self.template_path = template_path
        self.n_ca_template = n_ca_template
        self.copy_exact_length = copy_exact_length
        self.template_model = None
        self.Nnew = 0
        self._init_template()

    def _init_template(self):
        """
        Initialize the template model.

        If ``n_ca_template`` is set, build an in-memory template of that length.
        Else if ``template_path`` exists, load it. Otherwise build the default
        in-memory template.
        """
        if self.n_ca_template is not None:
            self._build_template_in_memory(n_ca=int(self.n_ca_template))
            return

        if self.template_path and os.path.exists(self.template_path):
            self._load_template_from_file(self.template_path)
        else:
            self._build_template_in_memory()

    def _load_template_from_file(self, template_path: str):
        """Load the template structure from a PDB file and count CA atoms."""
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("template", template_path)
            self.template_model = structure[0]
            self.Nnew = len([atom for atom in self.template_model.get_atoms() if atom.get_id() == "CA"])
            print(f"Template loaded from file with {self.Nnew} CA atoms: {template_path}")
        except Exception as e:
            print(f"Error loading template from file: {e}")
            raise

    def _build_template_in_memory(self, n_ca: int | None = None):
        """
        Build a minimal Bio.PDB Model containing only CA atoms with desired residue numbering.

        If n_ca is provided, build a template with that many CA atoms and sequential residue IDs
        starting from `_TEMPLATE_START_RESSEQ`.
        """
        from Bio.PDB.Structure import Structure
        from Bio.PDB.Model import Model
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Residue import Residue
        from Bio.PDB.Atom import Atom

        if n_ca is None:
            n_ca = len(self._TEMPLATE_DEFAULT_RESNAMES)
        try:
            n_ca = int(n_ca)
        except Exception:
            raise ValueError(f"n_ca must be an integer; got {n_ca!r}")
        if n_ca <= 0:
            raise ValueError(f"n_ca must be > 0; got {n_ca}")

        structure = Structure("template")
        model = Model(0)
        chain = Chain(self._TEMPLATE_CHAIN_ID)

        serial_number = 1
        base = list(self._TEMPLATE_DEFAULT_RESNAMES)
        if n_ca <= len(base):
            resnames = base[:n_ca]
        else:
            resnames = base + (["ALA"] * (n_ca - len(base)))

        for i, resname in enumerate(resnames):
            resseq = int(self._TEMPLATE_START_RESSEQ) + i
            residue_id = (" ", int(resseq), " ")
            residue = Residue(residue_id, resname, " ")
            atom = Atom(
                "CA",
                np.array([0.0, 0.0, 0.0], dtype=float),
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=" CA ",
                serial_number=serial_number,
                element="C",
            )
            serial_number += 1
            residue.add(atom)
            chain.add(residue)

        model.add(chain)
        structure.add(model)

        self.template_model = model
        self.Nnew = len(resnames)
        print(f"In-memory template initialized with {self.Nnew} CA atoms (no template.pdb needed)")
    
    def read_structure(self, input_data):
        """
        Read PDB file or trajectory object and return the first model.
        
        Parameters:
        -----------
        input_data : str or md.Trajectory
            Either a file path to a PDB file or a trajectory object
            
        Returns:
        --------
        Bio.PDB.Model
            The first model from the structure
        """
        if isinstance(input_data, str):
            # If input is a string, treat it as a file path
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('structure', input_data)
        elif isinstance(input_data, md.Trajectory):
            # If input is a trajectory, save to temp PDB and read
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmpfile:
                input_data.save(tmpfile.name)
                tmpfile.close()
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure('structure', tmpfile.name)
            os.unlink(tmpfile.name)
        else:
            raise ValueError("Unsupported input type. Provide a file path or md.Trajectory.")
        return structure[0]

    def _extract_ca_coordinates(self, model):
        """
        Extract CA atom coordinates from a model.
        
        Parameters:
        -----------
        model : Bio.PDB.Model
            The protein model
            
        Returns:
        --------
        numpy.ndarray
            Array of CA atom coordinates
        """
        atom_list = [atom for atom in model.get_atoms() if atom.get_id() == 'CA']
        return np.array([atom.coord for atom in atom_list])
    
    def _fit_cubic_interpolation(self, coordinates):
        """
        Fit cubic interpolation for each axis (x, y, z).
        
        Parameters:
        -----------
        coordinates : numpy.ndarray
            Array of CA atom coordinates
            
        Returns:
        --------
        dict
            Dictionary containing interpolation functions for each dimension
        """
        n = len(coordinates)
        dims = ['x', 'y', 'z']
        fits = {}
        
        for j, dim in enumerate(dims):
            fits[dim] = interp1d(np.arange(n), coordinates[:, j], kind='cubic', fill_value='extrapolate')
        
        return fits
    
    def _calculate_arc_length_parameterization(self, fits, n):
        """
        Calculate arc length parameterization for even spacing.
        
        Parameters:
        -----------
        fits : dict
            Dictionary of interpolation functions
        n : int
            Number of original points
            
        Returns:
        --------
        tuple
            (X, pt) where X is the fine grid and pt are indices for evenly spaced points
        """
        dims = ['x', 'y', 'z']
        
        # Create a finer grid of points (X) for interpolation
        X = np.arange(0, n - 1, 0.1)
        
        # Gradient in each dimension
        dYdX = {dim: np.gradient(fits[dim](X)) for dim in dims}
        
        # Speed along path (magnitude of the gradient)
        Y = np.sqrt(sum(np.square(dYdX[dim]) for dim in dims))
        
        # Total arc length (area under the speed curve)
        L = np.trapz(Y, X)
        
        # Create an evenly spaced set of arc lengths (Li)
        Li = np.linspace(0, L, self.Nnew)
        
        # Precompute partial arc length at each step in X
        flen = np.array([np.trapz(Y[:ibig], X[:ibig]) for ibig in range(1, len(X))])
        
        # For each required point (Nnew), find the corresponding index in X
        pt = np.zeros(self.Nnew, dtype=int)
        for i in range(self.Nnew):
            pt[i] = np.argmin(np.abs(flen - Li[i]))
        
        return X, pt
    
    def _interpolate_coordinates(self, fits, X, pt):
        """
        Interpolate 3D coordinates for evenly spaced points.
        
        Parameters:
        -----------
        fits : dict
            Dictionary of interpolation functions
        X : numpy.ndarray
            Fine grid of points
        pt : numpy.ndarray
            Indices for evenly spaced points
            
        Returns:
        --------
        numpy.ndarray
            Array of interpolated 3D coordinates
        """
        dims = ['x', 'y', 'z']
        new_coords = np.array([[fits[dim](X[pt[i]]) for dim in dims] for i in range(self.Nnew)])
        return new_coords
    
    def _create_fitted_model(self, new_coords):
        """
        Create a fitted model with interpolated coordinates without modifying the template.
        
        Parameters:
        -----------
        new_coords : numpy.ndarray
            Array of new coordinates
            
        Returns:
        --------
        Bio.PDB.Model
            A copy of the template model with updated coordinates
        """
        # Create a deep copy of the template model
        fitted_model = copy.deepcopy(self.template_model)
        
        # Update the copy with new coordinates
        ca_index = 0
        for atom in fitted_model.get_atoms():
            if atom.get_id() == 'CA':
                atom.set_coord(new_coords[ca_index])
                ca_index += 1
        
        return fitted_model
    
    def _save_structure(self, fitted_model, save_path):
        """
        Save the fitted structure to a file.
        
        Parameters:
        -----------
        fitted_model : Bio.PDB.Model
            The fitted model to save
        save_path : str
            Path to save the structure
        """
        try:
            with open(save_path, "w") as file:
                io = PDB.PDBIO()
                io.set_structure(fitted_model)
                io.save(file)
            print(f'Successfully saved the structure to {save_path}')
        except Exception as e:
            print(f"Error during file save: {e}")
            raise
    
    def plot_comparison(self, original_pdb_path, fitted_pdb_path, save_plot_path=None, 
                        show_interpolation=True):
        """
        Create a 3D plot comparing original and fitted structures.
        
        Parameters:
        -----------
        original_pdb_path : str
            Path to the original PDB file
        fitted_pdb_path : str
            Path to the fitted PDB file
        save_plot_path : str, optional
            Path to save the plot image
        show_interpolation : bool, optional
            Whether to show the interpolated path (default True)
        """
        try:
            # Load original structure
            xyz = md.load(original_pdb_path)
            
            # Extract CA atom indices
            atoms = sum([[atom.index for atom in res.atoms if atom.name == "CA"] 
                        for res in xyz.top._residues[:]], [])
            
            # Extract coordinates from original structure (convert from nm to Angstrom for consistency)
            coords = xyz.xyz[0, atoms].T * 10  # nm to Angstrom
            x = coords[0]
            y = coords[1]
            z = coords[2]
            
            # Load fitted structure
            new_coords = md.load(fitted_pdb_path)
            
            # Extract CA atom indices from fitted structure
            atoms_fitted = sum([[atom.index for atom in res.atoms if atom.name == "CA"] 
                               for res in new_coords.top._residues[:]], [])
            
            # Extract coordinates from fitted structure
            new_coords_xyz = new_coords.xyz[0, atoms_fitted].T * 10  # nm to Angstrom
            xp = new_coords_xyz[0]
            yp = new_coords_xyz[1]
            zp = new_coords_xyz[2]
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            
            # Plot original structure with markers
            ax.plot3D(x, y, z, 'blue', marker="o", label="Original CA atoms", linewidth=2, markersize=4)
            
            # Plot interpolated path if requested
            if show_interpolation:
                # Compute the cubic interpolation path from original coordinates
                original_coords = np.column_stack([x, y, z])
                n = len(original_coords)
                
                # Fit cubic interpolation
                fits = self._fit_cubic_interpolation(original_coords)
                
                # Create a fine grid for the interpolation curve
                X_fine = np.linspace(0, n - 1, 500)
                x_interp = fits['x'](X_fine)
                y_interp = fits['y'](X_fine)
                z_interp = fits['z'](X_fine)
                
                # Plot the interpolated path
                ax.plot3D(x_interp, y_interp, z_interp, 'green', linewidth=1.5, 
                         alpha=0.7, label="Interpolated path")
            
            # Plot fitted structure
            ax.plot3D(xp, yp, zp, 'red', marker=".", label="Fitted CA atoms", linewidth=2, markersize=3)
            
            # Customize plot
            plt.tick_params(bottom=False, top=False, labelbottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.legend(loc='upper left', fontsize=10)
            
            # Add title
            structure_name = os.path.basename(original_pdb_path).split('.')[0]
            ax.set_title(f"Structure Fitting Comparison: {structure_name}", fontsize=12)
            
            # Save plot if path provided
            if save_plot_path:
                plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plot for {original_pdb_path}: {e}")
    
    def fit_structure(self, fp_or_traj, save_path):
        """
        Fit a structure using cubic interpolation and save the result.
        Assumes input structure is already CA-stripped.

        When ``copy_exact_length`` is enabled and the input already has
        ``self.Nnew`` CA atoms, the original coordinates are copied onto the
        template without spline reparameterisation.

        Parameters:
        -----------
        fp_or_traj : str or md.Trajectory
            Input PDB file path or trajectory object (should be CA-stripped)
        save_path : str
            Path to save the fitted structure

        Returns
        -------
        bool
            True if spline fitting was applied, False if coordinates were copied as-is.
        """
        try:
            # Read input structure (assuming it's already CA-stripped)
            my_model = self.read_structure(fp_or_traj)

            # Extract CA coordinates
            coordinates = self._extract_ca_coordinates(my_model)
            n = len(coordinates)

            copied = self.copy_exact_length and n == self.Nnew
            if copied:
                new_coords = coordinates
            else:
                fits = self._fit_cubic_interpolation(coordinates)
                X, pt = self._calculate_arc_length_parameterization(fits, n)
                new_coords = self._interpolate_coordinates(fits, X, pt)

            # Create fitted model (without modifying template)
            fitted_model = self._create_fitted_model(new_coords)

            # Save the fitted structure
            self._save_structure(fitted_model, save_path)
            return not copied

        except Exception as e:
            print(f"Error during fitting: {e}")
            raise

    def process_directory(self, input_dir, output_dir, create_plots=True, plot_dir=None):
        """
        Process all PDB files in a directory with fitting and optional plotting.
        Assumes input PDB files are already CA-stripped.
        
        Parameters:
        -----------
        input_dir : str
            Input directory containing CA-stripped PDB files
        output_dir : str
            Output directory for fitted structures
        create_plots : bool, optional
            Whether to create comparison plots (default: True)
        plot_dir : str, optional
            Directory to save plots (default: output_dir/plots)
        """
        # Clear output so re-runs overwrite previous results
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Set up plot directory
        if create_plots:
            if plot_dir is None:
                plot_dir = os.path.join(output_dir, "plots")
            if os.path.exists(plot_dir):
                shutil.rmtree(plot_dir)
            os.makedirs(plot_dir)

        print(f"\n{'#'*80}")
        print(f"PROCESSING CA-STRIPPED STRUCTURES WITH FITTING")
        print(f"{'#'*80}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        if create_plots:
            print(f"Plots:  {plot_dir}")

        # Find all PDB files in the input directory
        pdb_files = glob(os.path.join(input_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files to process")

        if not pdb_files:
            print("No PDB files found in input directory!")
            return

        # Process each PDB file
        successful_count = 0
        for pdb_file in tqdm(pdb_files, desc="Fitting structures"):
            try:
                file_name = os.path.basename(pdb_file)
                output_file_path = os.path.join(output_dir, file_name)

                print(f"Processing: {file_name}")

                # Fit the structure
                self.fit_structure(pdb_file, output_file_path)

                # Create comparison plot
                if create_plots:
                    plot_filename = os.path.splitext(file_name)[0] + "_comparison.png"
                    plot_path = os.path.join(plot_dir, plot_filename)
                    self.plot_comparison(pdb_file, output_file_path, plot_path)

                successful_count += 1

            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
                continue

        print(f"\n{'#'*80}")
        print(f"PROCESSING COMPLETE")
        print(f"{'#'*80}")
        print(f"Successfully processed {successful_count}/{len(pdb_files)} structures")
        if create_plots:
            print(f"Plots saved to: {plot_dir}")

    def process_chains_directory(
        self,
        input_dir,
        output_dir,
        motifs=None,
        basename_allowlist: Optional[Sequence[str]] = None,
        quiet_strip: bool = False,
    ):
        """
        Coarse-grain full-chain PDB files.

        For each PDB in ``input_dir``:

        1. Detect the DFG/APE activation-loop region via ``CAStripper`` and CA-strip it
           in a temporary directory.
        2. Fit a cubic spline to the loop Cα coordinates and resample to ``self.Nnew``
           equidistant points (or copy unchanged when length already matches).
        3. Save the coarse-grained structure to ``output_dir``.

        The output directory is cleared on each run so re-running the notebook cell
        overwrites previous results.

        Parameters
        ----------
        input_dir : str
            Directory containing full-chain PDB files.
        output_dir : str
            Directory to save coarse-grained PDB files.
        motifs : list of str, optional
            Motifs used to locate the activation loop (default: ``['DFG', 'APE']``).
        basename_allowlist : sequence of str, optional
            If set, only process these basenames (with or without ``.pdb``).
        quiet_strip : bool, default=False
            If True, suppress verbose ``CAStripper`` stdout during stripping.
        """
        from workflow.ca_stripper import CAStripper

        if motifs is None:
            motifs = ["DFG", "APE"]

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        pdb_files = sorted(glob(os.path.join(input_dir, "*.pdb")))
        if basename_allowlist is not None:
            allow = {
                os.path.splitext(os.path.basename(str(b)))[0] for b in basename_allowlist
            }
            pdb_files = [
                p
                for p in pdb_files
                if os.path.splitext(os.path.basename(p))[0] in allow
            ]

        if not pdb_files:
            print("No PDB files found in input directory!")
            return

        print(f"\n{'#'*80}")
        print(f"COARSE-GRAINING FROM FULL-CHAIN STRUCTURES  (template: {self.Nnew} Cα)")
        print(f"{'#'*80}")
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir} (cleared)")
        print(f"Found {len(pdb_files)} PDB files\n")

        stripper = CAStripper(motifs=motifs)
        successful_count = 0
        copied_count = 0
        fitted_count = 0

        with tempfile.TemporaryDirectory(prefix="cg_chains_temp_") as temp_dir:
            for pdb_file in tqdm(pdb_files, desc=f"Coarse-graining (N={self.Nnew})"):
                filename = os.path.basename(pdb_file)
                output_file_path = os.path.join(output_dir, filename)

                if quiet_strip:
                    with contextlib.redirect_stdout(io.StringIO()):
                        success = stripper.process_single_structure(pdb_file, temp_dir)
                else:
                    success = stripper.process_single_structure(pdb_file, temp_dir)
                temp_out = os.path.join(temp_dir, filename)
                if not success or not os.path.exists(temp_out):
                    print(f"  Skipping {filename} (could not extract loop region)")
                    continue

                try:
                    was_fitted = self.fit_structure(temp_out, output_file_path)
                    successful_count += 1
                    if was_fitted:
                        fitted_count += 1
                    else:
                        copied_count += 1
                except Exception as exc:
                    print(f"  Error fitting {filename}: {exc}")

        print(f"\n{'#'*80}")
        print("COARSE-GRAINING COMPLETE")
        print(f"{'#'*80}")
        print(f"Successfully processed {successful_count}/{len(pdb_files)} structures")
        if self.copy_exact_length:
            print(
                f"  Copied unchanged ({self.Nnew} CAs) : {copied_count}\n"
                f"  Spline-fitted (≠ {self.Nnew} CAs)  : {fitted_count}"
            )

    @staticmethod
    def list_nonexact_loop_basenames(
        input_dir: str,
        n_ca: int = 27,
        motifs: Optional[Sequence[str]] = None,
        *,
        quiet: bool = True,
    ) -> Dict[str, List]:
        """
        CA-strip DFG→APE loops and return basenames whose CA count ≠ ``n_ca``.

        Returns
        -------
        dict
            ``nonexact`` (list of basenames), ``exact`` (list), ``failed`` (list),
            ``ca_counts`` (dict basename → int).
        """
        from workflow.ca_stripper import CAStripper

        if motifs is None:
            motifs = ["DFG", "APE"]

        pdb_files = sorted(glob(os.path.join(input_dir, "*.pdb")))
        stripper = CAStripper(motifs=list(motifs))
        nonexact: List[str] = []
        exact: List[str] = []
        failed: List[str] = []
        ca_counts: Dict[str, int] = {}

        with tempfile.TemporaryDirectory(prefix="loop_ca_count_") as temp_dir:
            for pdb_file in tqdm(pdb_files, desc=f"Counting loop CAs (≠{n_ca} cohort)"):
                filename = os.path.basename(pdb_file)
                stem = os.path.splitext(filename)[0]
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ok = stripper.process_single_structure(pdb_file, temp_dir)
                else:
                    ok = stripper.process_single_structure(pdb_file, temp_dir)
                temp_out = os.path.join(temp_dir, filename)
                if not ok or not os.path.isfile(temp_out):
                    failed.append(stem)
                    continue
                try:
                    traj = md.load(temp_out)
                    n = int(traj.n_atoms)  # CA-only stripped
                    ca_counts[stem] = n
                    if n == int(n_ca):
                        exact.append(stem)
                    else:
                        nonexact.append(stem)
                except Exception:
                    failed.append(stem)
                finally:
                    if os.path.isfile(temp_out):
                        os.remove(temp_out)

        print(
            f"Loop CA counts in {input_dir}: "
            f"nonexact(≠{n_ca})={len(nonexact)}, exact={len(exact)}, failed={len(failed)}"
        )
        return {
            "nonexact": nonexact,
            "exact": exact,
            "failed": failed,
            "ca_counts": ca_counts,
        }

    @staticmethod
    def ca_distance_matrix(pdb_path: str) -> np.ndarray:
        """Return the full pairwise Euclidean CA distance matrix (Å)."""
        traj = md.load(pdb_path)
        # Prefer CA selection if heavy atoms present; CA-stripped loops are all-CA
        try:
            ca_idx = traj.topology.select("name CA")
            if ca_idx is not None and len(ca_idx) > 0:
                xyz = traj.xyz[0, ca_idx] * 10.0  # nm → Å
            else:
                xyz = traj.xyz[0] * 10.0
        except Exception:
            xyz = traj.xyz[0] * 10.0
        diff = xyz[:, None, :] - xyz[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    @staticmethod
    def compare_fitted_distance_matrices(
        dir_a: str,
        dir_b: str,
        *,
        label_a: str = "no_modeller",
        label_b: str = "with_modeller",
    ) -> pd.DataFrame:
        """
        Per-structure MSE between full CA distance matrices in two fitted dirs.

        MSE is the mean of squared element-wise differences over the **full**
        NxN matrix (including the zero diagonal), for basenames present in both
        directories with matching matrix shape.
        """
        files_a = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(dir_a, "*.pdb"))
        }
        files_b = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob(os.path.join(dir_b, "*.pdb"))
        }
        common = sorted(set(files_a) & set(files_b))
        rows = []
        skipped = 0
        for stem in tqdm(common, desc="Distance-matrix MSE"):
            try:
                Da = Fitting.ca_distance_matrix(files_a[stem])
                Db = Fitting.ca_distance_matrix(files_b[stem])
                if Da.shape != Db.shape:
                    skipped += 1
                    continue
                diff = Da - Db
                mse = float(np.mean(diff * diff))
                fro = float(np.linalg.norm(diff, ord="fro"))
                rows.append(
                    {
                        "basename": stem,
                        "mse": mse,
                        "frobenius": fro,
                        "n_ca": int(Da.shape[0]),
                        f"path_{label_a}": files_a[stem],
                        f"path_{label_b}": files_b[stem],
                        "label_a": label_a,
                        "label_b": label_b,
                    }
                )
            except Exception as exc:
                skipped += 1
                print(f"  Skip {stem}: {exc}")

        df = pd.DataFrame(rows)
        print(
            f"Compared {len(df)} structures "
            f"(intersection={len(common)}, skipped={skipped})"
        )
        if len(df):
            print(
                f"MSE: median={df['mse'].median():.4f}, "
                f"mean={df['mse'].mean():.4f}, max={df['mse'].max():.4f}"
            )
        return df

    @staticmethod
    def plot_modeller_fit_mse(
        mse_df: pd.DataFrame,
        save_dir: str,
        *,
        path_col_a: str = "path_no_modeller",
        path_col_b: str = "path_with_modeller",
        n_examples: int = 3,
        show: bool = True,
    ) -> Dict[str, str]:
        """
        Histogram + box/strip of per-structure MSE, and heatmaps for low / median / high MSE.

        Returns dict of output file paths.
        """
        os.makedirs(save_dir, exist_ok=True)
        if mse_df is None or len(mse_df) == 0:
            raise ValueError("mse_df is empty — nothing to plot")

        df = mse_df.sort_values("mse").reset_index(drop=True)
        out: Dict[str, str] = {}

        # Histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["mse"], bins=40, color="steelblue", edgecolor="white")
        ax.axvline(df["mse"].median(), color="black", linestyle="--", label=f"median={df['mse'].median():.4f}")
        ax.set_xlabel(r"MSE$(D_{\mathrm{noMod}} - D_{\mathrm{Mod}})$")
        ax.set_ylabel("Count")
        ax.set_title("Fitted-loop distance-matrix MSE\n(no MODELLER vs with MODELLER)")
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(save_dir, "mse_histogram.png")
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        out["histogram"] = hist_path

        # Box + strip
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.boxplot(df["mse"].values, vert=True, widths=0.4)
        rng = np.random.RandomState(0)
        x = 1 + (rng.rand(len(df)) - 0.5) * 0.12
        ax.scatter(x, df["mse"].values, alpha=0.35, s=12, color="indianred")
        ax.set_xticks([1])
        ax.set_xticklabels(["cohort"])
        ax.set_ylabel(r"MSE$(D_{\mathrm{noMod}} - D_{\mathrm{Mod}})$")
        ax.set_title("Per-structure MSE distribution")
        plt.tight_layout()
        box_path = os.path.join(save_dir, "mse_boxplot.png")
        fig.savefig(box_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        out["boxplot"] = box_path

        # Example heatmaps: low / median / high
        n = len(df)
        if n_examples < 1:
            return out
        idxs = sorted(
            {
                0,
                n // 2,
                n - 1,
            }
        )
        # If n_examples > 3, spread evenly
        if n_examples > 3 and n > 3:
            idxs = sorted(
                set(int(i) for i in np.linspace(0, n - 1, n_examples))
            )

        for rank, idx in enumerate(idxs):
            row = df.iloc[idx]
            if path_col_a not in row or path_col_b not in row:
                # fall back to columns written by compare_fitted_distance_matrices
                pa = row.get("path_no_modeller") or row.filter(like="path_").iloc[0]
                pb = row.get("path_with_modeller") or row.filter(like="path_").iloc[1]
            else:
                pa, pb = row[path_col_a], row[path_col_b]
            Da = Fitting.ca_distance_matrix(pa)
            Db = Fitting.ca_distance_matrix(pb)
            Dabs = np.abs(Da - Db)
            vmax = max(float(Da.max()), float(Db.max()), 1e-6)

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            for ax, M, title in zip(
                axes,
                [Da, Db, Dabs],
                [
                    r"$D_{\mathrm{noMod}}$",
                    r"$D_{\mathrm{Mod}}$",
                    r"$|D_{\mathrm{noMod}}-D_{\mathrm{Mod}}|$",
                ],
            ):
                im = ax.imshow(
                    M,
                    cmap="viridis" if title != r"$|D_{\mathrm{noMod}}-D_{\mathrm{Mod}}|$" else "magma",
                    vmin=0,
                    vmax=vmax if "abs" not in title.lower() and "|" not in title else max(float(Dabs.max()), 1e-6),
                )
                ax.set_title(title)
                ax.set_xlabel("CA index")
                ax.set_ylabel("CA index")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"{row['basename']}  MSE={row['mse']:.4f}  (rank {idx + 1}/{n})",
                y=1.02,
            )
            plt.tight_layout()
            tag = {0: "low", n // 2: "median", n - 1: "high"}.get(idx, f"rank{idx}")
            hpath = os.path.join(save_dir, f"heatmap_{tag}_{row['basename']}.png")
            fig.savefig(hpath, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            out[f"heatmap_{tag}"] = hpath

        print(f"Saved MSE plots under {save_dir}")
        return out



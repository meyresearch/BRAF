#!/usr/bin/env python3
"""
Protein Structure Fitting Class

This module provides a class-based approach for fitting protein structures using
cubic interpolation along the backbone. It works with already CA-stripped structures.
"""

import os
import re
import io
import sys
import contextlib
import tempfile
import numpy as np
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
    # We keep this for backwards-compatibility, but the template length can now be overridden at runtime
    # (e.g., set to the median CA count of a cleaned dataset).
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

    def __init__(self, template_path: str | None = None, n_ca_template: int | None = None):
        """
        Initialize the Fitting class.
        
        Parameters:
        -----------
        template_path : str | None
            Optional path to the template PDB file for CA atom configuration.
            If None, an in-memory template is built from hardcoded metadata.
        n_ca_template : int | None
            If provided, build an in-memory CA-only template with exactly this many CA atoms.
            This overrides the default 27-CA template (and also overrides template_path unless you pass
            a template file with the same CA count).
        """
        self.template_path = template_path
        self.n_ca_template = n_ca_template
        self.template_model = None
        self.Nnew = 0
        self._init_template()
    
    def _init_template(self):
        """
        Initialize the template model.

        If template_path is provided and exists, load it. Otherwise build an in-memory
        CA-only template from hardcoded residue metadata.
        """
        # If a target CA count is requested, prefer building the in-memory template of that size.
        # This avoids accidentally loading a mismatched template from disk.
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
        # Use the historical residue-name series as a prefix:
        # - If n_ca < 27: take the first n_ca names
        # - If n_ca == 27: identical to the historical template
        # - If n_ca > 27: extend by padding with ALA
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
        
        Parameters:
        -----------
        fp_or_traj : str or md.Trajectory
            Input PDB file path or trajectory object (should be CA-stripped)
        save_path : str
            Path to save the fitted structure
        """
        try:
            # Read input structure (assuming it's already CA-stripped)
            my_model = self.read_structure(fp_or_traj)
            
            # Extract CA coordinates
            coordinates = self._extract_ca_coordinates(my_model)
            n = len(coordinates)
            
            print(f"Processing structure with {n} CA atoms")
            
            # Fit cubic interpolation
            fits = self._fit_cubic_interpolation(coordinates)
            
            # Calculate arc length parameterization
            X, pt = self._calculate_arc_length_parameterization(fits, n)
            
            # Interpolate coordinates
            new_coords = self._interpolate_coordinates(fits, X, pt)
            
            # Create fitted model (without modifying template)
            fitted_model = self._create_fitted_model(new_coords)
            
            # Save the fitted structure
            self._save_structure(fitted_model, save_path)
            
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
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plot directory
        if create_plots:
            if plot_dir is None:
                plot_dir = os.path.join(output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
        
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

    @staticmethod
    def _discover_anchored_n_values(base_dir="Results/activation_segments"):
        pattern = os.path.join(base_dir, "CA_segments_anchored_*")
        n_values = []
        for d in glob(pattern):
            m = re.search(r"CA_segments_anchored_(\d+)$", d)
            if m:
                n_values.append(int(m.group(1)))
        n_values = sorted(set(n_values))
        if not n_values:
            raise NameError(
                "No anchored datasets found. Expected folders like "
                "'Results/activation_segments/CA_segments_anchored_<N>/'."
            )
        return n_values

    @staticmethod
    def _median_ca_from_outlier_results(final_results_by_N, N):
        if final_results_by_N is None:
            return None
        if N not in final_results_by_N:
            return None
        res = final_results_by_N[N]
        df = res.get("final_clean_df") if isinstance(res, dict) else None
        if df is None or "ca_count" not in df.columns or len(df) == 0:
            return None
        return int(round(float(df["ca_count"].median())))

    @staticmethod
    def _median_ca_from_csv(base_dir, N):
        ca_dir = os.path.join(base_dir, f"CA_segments_anchored_{N}")
        candidates = sorted(glob(os.path.join(ca_dir, "final_clean_dataset_k*.csv")))
        if not candidates:
            return None
        import pandas as pd
        df = pd.read_csv(candidates[-1])
        if "ca_count" not in df.columns or len(df) == 0:
            return None
        return int(round(float(df["ca_count"].median())))

    @classmethod
    def fit_anchored_datasets(
        cls,
        *,
        base_dir="Results/activation_segments",
        n_values=None,
        final_results_by_N=None,
        default_n_ca_template=27,
        suppress_output=True,
    ):
        """
        Fit structures for each anchored dataset N using median CA count as template size.

        Returns:
            dict[int, str]: mapping N -> fitted output directory
        """
        if n_values is None:
            n_values = cls._discover_anchored_n_values(base_dir=base_dir)
        else:
            n_values = sorted(set(int(x) for x in n_values))

        devnull = io.StringIO()
        fitted_dirs_by_N = {}

        for N in n_values:
            input_directory = os.path.join(
                base_dir,
                f"CA_segments_anchored_{N}",
                f"CA_segments_final_cleaned_anchored_{N}",
            ) + "/"
            output_directory = os.path.join(base_dir, f"fitted_anchored_{N}") + "/"
            os.makedirs(output_directory, exist_ok=True)

            pdb_files = sorted(glob(os.path.join(input_directory, "*.pdb")))

            n_ca_template = cls._median_ca_from_outlier_results(final_results_by_N, N)
            if n_ca_template is None:
                n_ca_template = cls._median_ca_from_csv(base_dir, N)
            if n_ca_template is None:
                n_ca_template = int(default_n_ca_template)

            if suppress_output:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    fitter = cls(n_ca_template=n_ca_template)
            else:
                fitter = cls(n_ca_template=n_ca_template)

            with tqdm(
                total=len(pdb_files),
                desc=f"Fitting N={N} (template CA={n_ca_template})",
                leave=True,
                file=sys.stdout,
            ) as pbar:
                for pdb_file in pdb_files:
                    file_name = os.path.basename(pdb_file)
                    output_file_path = os.path.join(output_directory, file_name)

                    if os.path.exists(output_file_path):
                        pbar.update(1)
                        continue

                    if suppress_output:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            fitter.fit_structure(pdb_file, output_file_path)
                    else:
                        fitter.fit_structure(pdb_file, output_file_path)
                    pbar.update(1)

            fitted_dirs_by_N[int(N)] = output_directory

        return fitted_dirs_by_N



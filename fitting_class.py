#!/usr/bin/env python3
"""
Protein Structure Fitting Class

This module provides a class-based approach for fitting protein structures using
cubic interpolation along the backbone. It works with already CA-stripped structures.
"""

import os
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
    
    def __init__(self, template_path='template.pdb'):
        """
        Initialize the Fitting class.
        
        Parameters:
        -----------
        template_path : str
            Path to the template PDB file for CA atoms configuration
        """
        self.template_path = template_path
        self.template_model = None
        self.Nnew = 0
        self._load_template()
    
    def _load_template(self):
        """Load the template structure and count CA atoms."""
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('template', self.template_path)
            self.template_model = structure[0]
            self.Nnew = len([atom for atom in self.template_model.get_atoms() if atom.get_id() == 'CA'])
            print(f"Template loaded with {self.Nnew} CA atoms")
        except Exception as e:
            print(f"Error loading template: {e}")
            raise
    
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
    
    def plot_comparison(self, original_pdb_path, fitted_pdb_path, save_plot_path=None):
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
        """
        try:
            # Load original structure
            xyz = md.load(original_pdb_path)
            
            # Extract CA atom indices
            atoms = sum([[atom.index for atom in res.atoms if atom.name == "CA"] 
                        for res in xyz.top._residues[:]], [])
            
            # Extract coordinates from original structure
            coords = xyz.xyz[0, atoms].T
            x = coords[0]
            y = coords[1]
            z = coords[2]
            
            # Load fitted structure
            new_coords = md.load(fitted_pdb_path)
            
            # Extract CA atom indices from fitted structure
            atoms_fitted = sum([[atom.index for atom in res.atoms if atom.name == "CA"] 
                               for res in new_coords.top._residues[:]], [])
            
            # Extract coordinates from fitted structure
            new_coords_xyz = new_coords.xyz[0, atoms_fitted].T
            xp = new_coords_xyz[0]
            yp = new_coords_xyz[1]
            zp = new_coords_xyz[2]
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.plot3D(x, y, z, 'blue', marker="o", label="Original", linewidth=2, markersize=4)
            ax.plot3D(xp, yp, zp, 'red', label="Fitted", linewidth=2)
            
            # Customize plot
            plt.tick_params(bottom=False, top=False, labelbottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.legend()
            
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



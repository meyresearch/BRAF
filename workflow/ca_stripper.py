#!/usr/bin/env python3
"""
CA Stripper Class

This module provides a class-based approach for stripping protein structures to CA atoms
with specific focus on DFG-APE motif regions.
"""

import os
import pickle
import MDAnalysis as mda
from Bio.PDB import PDBParser, PPBuilder
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil


class CAStripper:
    """
    A class for stripping protein structures to CA atoms with motif-based region extraction.
    """
    
    def __init__(self, motifs=None):
        """
        Initialize the CAStripper class.
        
        Parameters:
        -----------
        motifs : list of str, optional
            List of motifs to search for (default: ['DFG', 'APE'])
        """
        self.motifs = motifs or ['DFG', 'APE']
        print(f"CAStripper initialized with motifs: {self.motifs}")
    
    def extract_sequence_and_residue_numbers(self, pdb_file):
        """
        Extract sequence and corresponding PDB residue numbers from atomic coordinates 
        for the first chain found in the PDB file.
        
        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file
            
        Returns:
        --------
        tuple
            (sequence, residue_numbers) where sequence is str and residue_numbers is list
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB', pdb_file)
        
        for model in structure:
            for chain in model:
                ppb = PPBuilder()
                sequence = ''
                residue_numbers = []
                
                for pp in ppb.build_peptides(chain):
                    sequence += pp.get_sequence()
                    # Get the actual residue numbers from the PDB
                    for residue in pp:
                        residue_numbers.append(residue.get_id()[1])  # residue.get_id()[1] gives the residue number
                return str(sequence), residue_numbers
        return None, None
    
    def find_motif_indices(self, seq, motif):
        """
        Find the indices of a motif in a sequence.
        
        Parameters:
        -----------
        seq : str
            Protein sequence
        motif : str
            Motif to search for
            
        Returns:
        --------
        tuple or None
            (start_index, end_index) if motif found, None otherwise
        """
        index = seq.find(motif)
        if index == -1:
            return None
        return index, index + len(motif)
    
    def find_all_motifs(self, seq):
        """
        Find all configured motifs in a sequence.
        
        Parameters:
        -----------
        seq : str
            Protein sequence
            
        Returns:
        --------
        dict
            Dictionary with motif names as keys and (start, end) tuples as values
        """
        motif_indices = {}
        for motif in self.motifs:
            indices = self.find_motif_indices(seq, motif)
            if indices:
                motif_indices[motif] = indices
        return motif_indices
    
    def get_motif_region_bounds(self, motif_indices):
        """
        Get the start and end bounds that encompass all found motifs.
        
        Parameters:
        -----------
        motif_indices : dict
            Dictionary with motif names as keys and (start, end) tuples as values
            
        Returns:
        --------
        tuple
            (start_residue, end_residue) encompassing all motifs
        """
        if not motif_indices:
            return None, None
        
        all_starts = [indices[0] for indices in motif_indices.values()]
        all_ends = [indices[1] for indices in motif_indices.values()]
        
        start_residue = min(all_starts)
        end_residue = max(all_ends)
        
        return start_residue, end_residue
    
    def strip_to_ca_atoms(self, pdb_path, start_residue, end_residue, residue_numbers):
        """
        Load PDB, extract CA atoms for residues in the specified slice,
        and return an MDAnalysis Universe object with just those atoms.
        
        Parameters:
        -----------
        pdb_path : str
            Path to the PDB file
        start_residue : int
            Starting residue index (0-based, sequence position)
        end_residue : int
            Ending residue index (0-based, sequence position, exclusive)
        residue_numbers : list
            List of actual PDB residue numbers corresponding to sequence positions
            
        Returns:
        --------
        mda.Universe
            Universe object with only CA atoms from the specified range
        """
        u = mda.Universe(pdb_path)
        print(f"Loaded PDB: {pdb_path}")

        # Map sequence indices to actual PDB residue numbers
        # start_residue and end_residue are sequence indices
        # residue_numbers[i] gives the actual PDB residue number for sequence position i
        start_pdb_resid = residue_numbers[start_residue]
        end_pdb_resid = residue_numbers[end_residue - 1]  # end_residue is exclusive, so -1
        
        print(f"Sequence indices: {start_residue} to {end_residue-1}")
        print(f"PDB residue numbers: {start_pdb_resid} to {end_pdb_resid}")

        # Extract CA atoms using actual PDB residue numbers
        # Build selection string that includes all residues in the range
        ca_atoms = u.select_atoms(f"name CA and resid {start_pdb_resid}:{end_pdb_resid}")
        print(f"Selected {len(ca_atoms)} CA atoms from PDB residues {start_pdb_resid} to {end_pdb_resid}")

        # Create a new universe with only the selected CA atoms
        ca_universe = mda.Merge(ca_atoms)
        return ca_universe
    
    def process_single_structure(self, pdb_file, target_dir):
        """
        Process a single PDB file to extract motif regions and strip to CA atoms.
        
        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file to process
        target_dir : str
            Directory to save stripped PDB files
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Extract sequence and residue numbers
            seq, residue_numbers = self.extract_sequence_and_residue_numbers(pdb_file)
            if seq is None or residue_numbers is None:
                print(f"Skipping {pdb_file} due to inability to extract sequence and residue numbers.")
                return False
            
            print(f"Sequence for {os.path.basename(pdb_file)}: {seq}")
            
            # Find all motifs in the sequence
            motif_indices = self.find_all_motifs(seq)
            print(f"Found motifs: {motif_indices}")

            if not motif_indices:
                print(f"Skipping {pdb_file} due to missing required motifs.")
                return False

            # Determine start and end residues (using residue indices, not PDB numbers)
            start_residue, end_residue = self.get_motif_region_bounds(motif_indices)
            
            if start_residue is None or end_residue is None:
                print(f"Skipping {pdb_file} due to invalid motif bounds.")
                return False
            
            print(f"Start residue index: {start_residue}, End residue index: {end_residue}")
            
            # Strip to CA atoms (passing residue_numbers to map sequence indices to PDB residue numbers)
            stripped = self.strip_to_ca_atoms(pdb_file, start_residue, end_residue, residue_numbers)
            
            # Save stripped PDB
            output_file = os.path.join(target_dir, os.path.basename(pdb_file))
            stripped.atoms.write(output_file)
            print(f"Saved stripped PDB to: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all PDB files in the input directory to extract motif regions and strip to CA atoms.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing PDB files to process
        output_dir : str
            Directory to save stripped PDB files
            
        Returns:
        --------
        str
            Path to the output directory
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'#'*80}")
        print(f"PROCESSING STRUCTURES FOR CA STRIPPING")
        print(f"{'#'*80}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Motifs: {self.motifs}")
        
        # Find all PDB files in input directory
        pdb_files = glob(os.path.join(input_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files to process")
        
        if not pdb_files:
            print("No PDB files found in input directory!")
            return output_dir
        
        # Process each PDB file
        successful_count = 0
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            print(f"\nProcessing: {os.path.basename(pdb_file)}")
            if self.process_single_structure(pdb_file, output_dir):
                successful_count += 1
        
        print(f"\n{'#'*80}")
        print(f"STRIPPING COMPLETE")
        print(f"{'#'*80}")
        print(f"Successfully processed {successful_count}/{len(pdb_files)} structures")
        print(f"Results saved to: {output_dir}")
        
        return output_dir
    
    def strip_to_ca(self, input_dir="Results/activation_segments/reconstructed_mustang_ends/", 
                    output_dir="Results/activation_segments/CA_segments/reconstructed_endsAlignment"):
        """
        Convenience method to process all PDB files in the input directory.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing PDB files to process
        output_dir : str
            Directory to save stripped PDB files
            
        Returns:
        --------
        str
            Path to the output directory
        """
        return self.process_directory(input_dir, output_dir)


class OutlierStripper:
    """
    A class for detecting and removing outliers from CA-stripped PDB files using Tukey's method.
    
    This class analyzes CA atom counts in already-stripped structures and removes outliers
    based on the Interquartile Range (IQR) method. Also supports distance-based filtering
    when a reference PDB is provided.
    """
    
    def __init__(self, k_factor=1.5, reference_pdb=None, ref_first_resid=None, ref_last_resid=None,
                 verbose=True):
        """
        Initialize the OutlierStripper class.
        
        Parameters:
        -----------
        k_factor : float, default=1.5
            The multiplier for the IQR in Tukey's method
        reference_pdb : str, optional
            Path to reference PDB for distance-based filtering
        ref_first_resid : int, optional
            0-based index of first CA in reference (for distance calculations)
        ref_last_resid : int, optional
            0-based index of last CA in reference (for distance calculations)
        verbose : bool, default=True
            If False, suppress routine initialization messages.
        """
        self.k_factor = k_factor
        self.results = None
        self.reference_pdb = reference_pdb
        self.ref_first_resid = ref_first_resid
        self.ref_last_resid = ref_last_resid
        self.ref_first_ca = None
        self.ref_last_ca = None
        self.distances_df = None
        self.coordinates_data = None  # Store actual 3D coordinates for plotting
        self.verbose = verbose
        
        if self.verbose:
            print(f"OutlierStripper initialized with k-factor: {k_factor}")
        
        # Load reference coordinates if provided
        if reference_pdb is not None:
            if ref_first_resid is None or ref_last_resid is None:
                if self.verbose:
                    print("Warning: ref_first_resid and ref_last_resid required for distance calculations")
            else:
                if self.verbose:
                    print(f"Loading reference structure: {reference_pdb}")
                self.ref_first_ca = self._get_ca_coordinates(reference_pdb, resid=ref_first_resid)
                self.ref_last_ca = self._get_ca_coordinates(reference_pdb, resid=ref_last_resid)
                
                if self.ref_first_ca is not None and self.ref_last_ca is not None:
                    if self.verbose:
                        print(f"Reference first CA (resid {ref_first_resid}): {self.ref_first_ca}")
                        print(f"Reference last CA (resid {ref_last_resid}): {self.ref_last_ca}")
                else:
                    if self.verbose:
                        print("Warning: Could not extract reference CA coordinates")
    
    def _get_ca_coordinates(self, pdb_file, resid=None, first_last=None):
        """
        Get CA atom coordinates from a PDB file using MDAnalysis.
        
        Parameters:
        -----------
        pdb_file : str
            Path to the PDB file
        resid : int, optional
            Specific residue ID to get CA from (0-based index)
        first_last : str, optional
            'first' or 'last' to get first/last CA atom
            
        Returns:
        --------
        np.ndarray or None
            3D coordinates [x, y, z] in Angstroms, or None if not found
        """
        try:
            # Load the PDB file
            u = mda.Universe(pdb_file)
            
            # Select CA atoms
            ca_atoms = u.select_atoms("name CA")
            
            if len(ca_atoms) == 0:
                print(f"Warning: No CA atoms found in {pdb_file}")
                return None
            
            if resid is not None:
                # Get CA from specific residue index
                if resid >= len(ca_atoms):
                    print(f"Warning: Residue index {resid} out of range in {pdb_file}")
                    return None
                # Return coordinates in Angstroms (MDAnalysis uses Angstroms by default)
                return ca_atoms[resid].position
            
            elif first_last == 'first':
                # Get first CA atom
                return ca_atoms[0].position
            
            elif first_last == 'last':
                # Get last CA atom
                return ca_atoms[-1].position
            
            else:
                print(f"Error: Must specify either resid or first_last parameter")
                return None
                
        except Exception as e:
            print(f"Error reading {pdb_file}: {e}")
            return None
    
    def tukey_outlier_detection(self, data, k=None):
        """
        Apply Tukey's method to detect outliers using the Interquartile Range (IQR).
        
        Parameters:
        -----------
        data : array-like
            The data to analyze for outliers
        k : float, optional
            The multiplier for the IQR. If None, uses self.k_factor
            
        Returns:
        --------
        tuple
            (outlier_indices, lower_bound, upper_bound, Q1, Q3, IQR)
        """
        if k is None:
            k = self.k_factor
            
        data = np.array(data)
        
        # Calculate quartiles
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        
        # Find outlier indices
        outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return outlier_indices, lower_bound, upper_bound, Q1, Q3, IQR
    
    def calculate_distances(self, dataset_dir):
        """
        Calculate Euclidean distances between terminal CA atoms of all structures 
        in a dataset directory and the reference structure.
        
        Parameters:
        -----------
        dataset_dir : str
            Directory containing dataset PDB files
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - pdb_file: filename
            - first_ca_distance: distance between first CA atoms (Å)
            - last_ca_distance: distance between last CA atoms (Å)
            - max_distance: maximum of the two distances
            
        Raises:
        ------
        ValueError
            If reference coordinates not loaded or no PDB files found
        """
        if self.ref_first_ca is None or self.ref_last_ca is None:
            raise ValueError("Reference CA coordinates not loaded. Initialize with reference_pdb.")
        
        # Get all PDB files in dataset directory
        pdb_files = glob(os.path.join(dataset_dir, "*.pdb"))
        
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in {dataset_dir}")
        
        print(f"\nCalculating terminal CA distances for {len(pdb_files)} structures...")
        
        # Store results
        results = []
        coordinates = []
        
        for pdb_file in tqdm(pdb_files, desc="Calculating distances"):
            filename = os.path.basename(pdb_file)
            
            # Get first and last CA coordinates
            first_ca = self._get_ca_coordinates(pdb_file, first_last='first')
            last_ca = self._get_ca_coordinates(pdb_file, first_last='last')
            
            if first_ca is None or last_ca is None:
                # Skip structures where we couldn't get coordinates
                results.append({
                    'pdb_file': filename,
                    'first_ca_distance': np.nan,
                    'last_ca_distance': np.nan,
                    'max_distance': np.nan
                })
                coordinates.append({
                    'pdb_file': filename,
                    'first_ca_coords': None,
                    'last_ca_coords': None
                })
                continue
            
            # Calculate Euclidean distances
            first_distance = np.linalg.norm(first_ca - self.ref_first_ca)
            last_distance = np.linalg.norm(last_ca - self.ref_last_ca)
            max_distance = max(first_distance, last_distance)
            
            results.append({
                'pdb_file': filename,
                'first_ca_distance': first_distance,
                'last_ca_distance': last_distance,
                'max_distance': max_distance
            })
            
            # Store actual coordinates
            coordinates.append({
                'pdb_file': filename,
                'first_ca_coords': first_ca.copy(),
                'last_ca_coords': last_ca.copy()
            })
        
        # Create DataFrame
        self.distances_df = pd.DataFrame(results)
        self.coordinates_data = coordinates
        
        # Print statistics
        print(f"\nTerminal CA Distance Statistics:")
        print(f"First CA - Mean: {self.distances_df['first_ca_distance'].mean():.2f} Å, "
              f"Median: {self.distances_df['first_ca_distance'].median():.2f} Å")
        print(f"Last CA - Mean: {self.distances_df['last_ca_distance'].mean():.2f} Å, "
              f"Median: {self.distances_df['last_ca_distance'].median():.2f} Å")
        
        return self.distances_df
    
    def filter_by_distance(self, max_distance=3.0):
        """
        Filter structures based on terminal CA distances.
        
        Parameters:
        -----------
        max_distance : float, default=3.0
            Maximum allowed distance (Å) for either terminal CA
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame with structures passing the distance cutoff
            
        Raises:
        ------
        ValueError
            If calculate_distances() hasn't been run yet
        """
        if self.distances_df is None:
            raise ValueError("No distance results available. Run calculate_distances() first.")
        
        # Filter structures where both first and last distances are <= max_distance
        filtered_df = self.distances_df[
            (self.distances_df['first_ca_distance'] <= max_distance) & 
            (self.distances_df['last_ca_distance'] <= max_distance)
        ].copy()
        
        n_total = len(self.distances_df)
        n_filtered = len(filtered_df)
        n_outliers = n_total - n_filtered
        pct_kept = (n_filtered / n_total * 100) if n_total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Distance-Based Filtering (cutoff: {max_distance:.1f} Å)")
        print(f"{'='*60}")
        print(f"Total structures: {n_total}")
        print(f"Structures kept: {n_filtered} ({pct_kept:.1f}%)")
        print(f"Structures removed: {n_outliers} ({100-pct_kept:.1f}%)")
        print(f"{'='*60}")
        
        return filtered_df
    
    def get_outliers(self, max_distance=3.0):
        """
        Get structures that exceed the distance cutoff.
        
        Parameters:
        -----------
        max_distance : float, default=3.0
            Distance threshold (Å)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with outlier structures, sorted by max_distance
            
        Raises:
        ------
        ValueError
            If calculate_distances() hasn't been run yet
        """
        if self.distances_df is None:
            raise ValueError("No distance results available. Run calculate_distances() first.")
        
        outliers = self.distances_df[
            (self.distances_df['first_ca_distance'] > max_distance) | 
            (self.distances_df['last_ca_distance'] > max_distance)
        ].copy()
        
        outliers = outliers.sort_values('max_distance', ascending=False)
        
        print(f"\nFound {len(outliers)} distance outliers (> {max_distance:.1f} Å)")
        print(f"Percentage: {len(outliers)/len(self.distances_df)*100:.1f}%")
        
        return outliers
    
    def create_plots(self, original_ca, clean_ca, output_dir, k_factor, ca_lower, ca_upper, ca_Q1, ca_Q3,
                     distance_data=None, distance_cutoff=3.0,
                     show_original=False, show_clean=True, show_boxplot=False):
        """
        Create plots showing original and clean CA distributions, and optionally distance distributions.
        
        Parameters:
        -----------
        original_ca : array-like
            Original CA atom counts
        clean_ca : array-like
            Clean CA atom counts after outlier removal
        output_dir : str
            Directory to save plots
        k_factor : float
            The k-factor used for outlier detection
        ca_lower : float
            Lower bound for outliers
        ca_upper : float
            Upper bound for outliers
        ca_Q1 : float
            First quartile
        ca_Q3 : float
            Third quartile
        distance_data : pd.DataFrame, optional
            DataFrame with distance information (first_ca_distance, last_ca_distance columns)
        distance_cutoff : float, default=3.0
            Distance cutoff for violin plots
        show_original : bool, default=False
            Whether to show original CA distribution with Tukey cutoffs
        show_clean : bool, default=True
            Whether to show clean CA distribution
        show_boxplot : bool, default=False
            Whether to show box plot comparison
        """
        # 1. Original CA distribution with Tukey cutoff lines
        if show_original:
            plt.figure(figsize=(12, 8))
            
            # Plot histogram
            n, bins, patches = plt.hist(original_ca, bins=20, alpha=0.7, color='lightblue', 
                                        edgecolor='black', label='Original Data')
            
            # Add vertical lines for Tukey bounds
            plt.axvline(x=ca_lower, color='red', linestyle='--', linewidth=2, 
                       label=f'Lower Bound: {ca_lower:.1f}')
            plt.axvline(x=ca_upper, color='red', linestyle='--', linewidth=2, 
                       label=f'Upper Bound: {ca_upper:.1f}')
            
            # Add quartile lines
            plt.axvline(x=ca_Q1, color='orange', linestyle=':', linewidth=2, label=f'Q1: {ca_Q1:.1f}')
            plt.axvline(x=ca_Q3, color='orange', linestyle=':', linewidth=2, label=f'Q3: {ca_Q3:.1f}')
            
            # Add median line
            median_ca = np.median(original_ca)
            plt.axvline(x=median_ca, color='green', linestyle='-', linewidth=2, 
                       label=f'Median: {median_ca:.1f}')
            
            # Shade outlier regions
            plt.axvspan(min(original_ca), ca_lower, alpha=0.2, color='red', label='Outlier Region')
            plt.axvspan(ca_upper, max(original_ca), alpha=0.2, color='red')
            
            plt.xlabel('Number of CA atoms')
            plt.ylabel('Frequency')
            plt.title(f'Original Dataset - CA Counts with Tukey Cutoffs (k={k_factor})\n(n={len(original_ca)})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'original_ca_distribution_with_cutoffs_k{k_factor}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Clean CA distribution
        if show_clean:
            plt.figure(figsize=(10, 6))
            
            n, bins, patches = plt.hist(clean_ca, bins=20, alpha=0.7, color='lightgreen', 
                                        edgecolor='black')
            
            # Add median line
            clean_median = np.median(clean_ca)
            plt.axvline(x=clean_median, color='darkgreen', linestyle='--', linewidth=2, 
                        label=f'Median: {clean_median:.1f}')
            
            plt.xlabel('Number of CA atoms')
            plt.ylabel('Frequency')
            plt.title(f'Clean Dataset - CA Counts (n={len(clean_ca)})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'clean_ca_distribution_k{k_factor}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Box plot comparison for CA counts
        if show_boxplot:
            plt.figure(figsize=(8, 6))
            
            box_data_ca = [original_ca, clean_ca]
            bp = plt.boxplot(box_data_ca, labels=['Original', 'Clean'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightgreen')
            plt.ylabel('Number of CA atoms')
            plt.title('CA Counts Distribution Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add outlier bounds as horizontal lines
            plt.axhline(y=ca_lower, color='red', linestyle='--', alpha=0.7, label='Tukey Bounds')
            plt.axhline(y=ca_upper, color='red', linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'ca_boxplot_comparison_k{k_factor}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Violin plots for terminal CA distances (if distance_data provided)
        if distance_data is not None:
            # Prepare data (remove NaN values)
            first_data = distance_data['first_ca_distance'].dropna()
            last_data = distance_data['last_ca_distance'].dropna()
            
            if len(first_data) > 0 and len(last_data) > 0:
                # Create figure with two subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: First CA distances
                parts1 = axes[0].violinplot([first_data], positions=[1], showmeans=True, showmedians=True)
                axes[0].axhline(y=distance_cutoff, color='red', linestyle=':', linewidth=2, 
                               label=f'{distance_cutoff} Å cutoff')
                axes[0].set_ylabel('Distance (Å)', fontsize=12)
                axes[0].set_title('First CA Distance\nvs Reference', fontsize=12, fontweight='bold')
                axes[0].set_xticks([1])
                axes[0].set_xticklabels(['First CA'])
                axes[0].set_ylim(0, max(8, first_data.max() + 1))
                axes[0].legend()
                axes[0].grid(axis='y', alpha=0.3)
                
                # Plot 2: Last CA distances
                parts2 = axes[1].violinplot([last_data], positions=[1], showmeans=True, showmedians=True)
                axes[1].axhline(y=distance_cutoff, color='red', linestyle=':', linewidth=2, 
                               label=f'{distance_cutoff} Å cutoff')
                axes[1].set_ylabel('Distance (Å)', fontsize=12)
                axes[1].set_title('Last CA Distance\nvs Reference', fontsize=12, fontweight='bold')
                axes[1].set_xticks([1])
                axes[1].set_xticklabels(['Last CA'])
                axes[1].set_ylim(0, max(8, last_data.max() + 1))
                axes[1].legend()
                axes[1].grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'terminal_ca_violin_distances.png'), 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"Distance violin plots saved to: {output_dir}/terminal_ca_violin_distances.png")
            
            # 5. 3D scatter plot of terminal CA positions
            if hasattr(self, 'coordinates_data') and self.coordinates_data is not None:
                print("\nCreating 3D scatter plot of terminal CA positions...")
                
                # Extract coordinates from stored data
                first_ca_positions = []
                last_ca_positions = []
                
                for coord_data in self.coordinates_data:
                    if coord_data['first_ca_coords'] is not None:
                        first_ca_positions.append(coord_data['first_ca_coords'])
                    if coord_data['last_ca_coords'] is not None:
                        last_ca_positions.append(coord_data['last_ca_coords'])
                
                if len(first_ca_positions) > 0 and len(last_ca_positions) > 0:
                    # Convert to numpy arrays
                    first_ca_positions = np.array(first_ca_positions)
                    last_ca_positions = np.array(last_ca_positions)
                    
                    # Create 3D scatter plot
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Plot all first CA positions in red
                    ax.scatter(first_ca_positions[:, 0], 
                              first_ca_positions[:, 1], 
                              first_ca_positions[:, 2],
                              c='red', marker='o', s=20, alpha=0.6, label='First CA (dataset)')
                    
                    # Plot all last CA positions in blue
                    ax.scatter(last_ca_positions[:, 0], 
                              last_ca_positions[:, 1], 
                              last_ca_positions[:, 2],
                              c='blue', marker='o', s=20, alpha=0.6, label='Last CA (dataset)')
                    
                    # Plot reference first CA in red (larger marker)
                    if self.ref_first_ca is not None:
                        ax.scatter(self.ref_first_ca[0], 
                                  self.ref_first_ca[1], 
                                  self.ref_first_ca[2],
                                  c='red', marker='*', s=300, 
                                  edgecolors='black', linewidths=2,
                                  label='First CA (reference)', zorder=10)
                    
                    # Plot reference last CA in blue (larger marker)
                    if self.ref_last_ca is not None:
                        ax.scatter(self.ref_last_ca[0], 
                                  self.ref_last_ca[1], 
                                  self.ref_last_ca[2],
                                  c='blue', marker='*', s=300, 
                                  edgecolors='black', linewidths=2,
                                  label='Last CA (reference)', zorder=10)
                    
                    # Set labels and title
                    ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
                    ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
                    ax.set_title('3D Distribution of Terminal CA Atoms\n' + 
                                f'(n={len(first_ca_positions)} structures)',
                                fontsize=14, fontweight='bold', pad=20)
                    
                    # Add legend
                    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3)
                    
                    # Adjust viewing angle for better visualization
                    ax.view_init(elev=20, azim=45)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'terminal_ca_3d_positions.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    print(f"3D scatter plot saved to: {output_dir}/terminal_ca_3d_positions.png")
                else:
                    print("Warning: No valid coordinates found for 3D plotting")
    
    def analyze(self, ca_segments_dir, create_plots=True, clean_dir_name=None,
                distance_cutoff=3.0, apply_distance_filter=None,
                apply_ca_count_filter=True,
                show_original=False, show_clean=False, show_boxplot=True):
        """
        Analyze CA-segment PDB files and optionally apply CA-count and distance filters.
        
        Parameters:
        -----------
        ca_segments_dir : str
            Directory containing the stripped PDB files with CA atoms
        create_plots : bool, default=True
            Whether to create visualization plots
        clean_dir_name : str, optional
            Name for the clean directory (default: adds '_cleaned' suffix to input directory name)
        distance_cutoff : float, default=3.0
            Distance cutoff (Å) for terminal CA filtering
        apply_distance_filter : bool, optional
            Whether to apply distance-based filtering. If None, applies if reference_pdb is set.
        apply_ca_count_filter : bool, default=True
            Whether to apply Tukey's method to CA counts. If False, all structures
            proceed directly to the optional terminal-distance filter.
        show_original : bool, default=False
            Whether to show original CA distribution with Tukey cutoffs
        show_clean : bool, default=False
            Whether to show clean CA distribution histogram
        show_boxplot : bool, default=True
            Whether to show boxplot comparison of original vs clean
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and file information
        """
        # Find all PDB files in the specified directory
        pdb_files = glob(os.path.join(ca_segments_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files in {ca_segments_dir}")
        
        if not pdb_files:
            # Fail fast with a clear message (returning None later causes confusing NoneType errors)
            raise ValueError(
                f"No PDB files found in '{ca_segments_dir}'. "
                f"Make sure this directory contains CA-stripped .pdb files (e.g., run CAStripper.strip_to_ca "
                f"with output_dir='{ca_segments_dir}', or point ca_segments_dir to the folder that actually "
                f"contains the CA segment PDBs)."
            )
        
        # Lists to store results
        file_info = []
        
        # Process each PDB file
        for pdb_file in tqdm(pdb_files, desc="Analyzing CA segments"):
            try:
                # Load the PDB file
                u = mda.Universe(pdb_file)
                
                # Count the number of CA atoms
                num_ca = len(u.select_atoms("name CA"))
                
                file_info.append({
                    'filename': os.path.basename(pdb_file),
                    'filepath': pdb_file,
                    'ca_count': num_ca
                })
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(file_info)
        
        # Extract data for outlier detection
        ca_counts = df['ca_count'].values
        
        print(f"\n{'='*60}")
        print("ORIGINAL DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total structures: {len(df)}")
        print(f"CA counts - Min: {ca_counts.min()}, Max: {ca_counts.max()}, "
              f"Mean: {ca_counts.mean():.2f}, Median: {np.median(ca_counts):.2f}")
        
        if apply_ca_count_filter:
            ca_outlier_indices, ca_lower, ca_upper, ca_Q1, ca_Q3, ca_IQR = (
                self.tukey_outlier_detection(ca_counts)
            )

            print(f"\n{'='*60}")
            print("TUKEY'S OUTLIER DETECTION - CA COUNTS")
            print(f"{'='*60}")
            print(f"Q1: {ca_Q1:.2f}, Q3: {ca_Q3:.2f}, IQR: {ca_IQR:.2f}")
            print(f"Lower bound: {ca_lower:.2f}, Upper bound: {ca_upper:.2f}")
            print(f"Outliers found: {len(ca_outlier_indices)} out of {len(ca_counts)} "
                  f"({len(ca_outlier_indices)/len(ca_counts)*100:.1f}%)")

            if len(ca_outlier_indices) > 0:
                print("CA count outliers:")
                for idx in ca_outlier_indices:
                    print(f"  - {df.iloc[idx]['filename']}: {df.iloc[idx]['ca_count']} CA atoms")

            clean_df = df.drop(df.index[ca_outlier_indices]).reset_index(drop=True)
        else:
            ca_outlier_indices = np.array([], dtype=int)
            ca_lower = ca_upper = ca_Q1 = ca_Q3 = ca_IQR = np.nan
            clean_df = df.copy().reset_index(drop=True)
            print("\nCA-count Tukey filtering disabled; retaining all structures.")

        clean_ca_counts = clean_df['ca_count'].values
        
        # Determine if distance filtering should be applied
        if apply_distance_filter is None:
            apply_distance_filter = (self.ref_first_ca is not None and self.ref_last_ca is not None)
        
        # Apply distance-based filtering if enabled
        distance_filtered_df = None
        if apply_distance_filter:
            print(f"\n{'='*60}")
            print("APPLYING DISTANCE-BASED FILTERING")
            print(f"{'='*60}")
            
            # Create a temporary directory with CA-filtered structures
            temp_ca_clean_dir = os.path.join(ca_segments_dir, "_temp_ca_filtered")
            os.makedirs(temp_ca_clean_dir, exist_ok=True)
            
            # Copy CA-filtered structures to temp directory
            for _, row in clean_df.iterrows():
                src = row['filepath']
                dst = os.path.join(temp_ca_clean_dir, row['filename'])
                shutil.copy2(src, dst)
            
            # Calculate distances for CA-filtered structures
            self.calculate_distances(temp_ca_clean_dir)
            
            # Filter by distance
            distance_filtered = self.filter_by_distance(max_distance=distance_cutoff)
            
            # Get filenames that passed distance filter
            distance_passed_files = set(distance_filtered['pdb_file'].values)
            
            # Update clean_df to only include structures that passed both filters
            distance_filtered_df = clean_df[clean_df['filename'].isin(distance_passed_files)].copy()
            
            print(f"\n{'='*60}")
            filter_description = (
                "after CA count + distance filtering"
                if apply_ca_count_filter
                else "after distance filtering"
            )
            print(f"FINAL CLEAN DATASET ({filter_description})")
            print(f"{'='*60}")
            print(f"Total structures: {len(distance_filtered_df)}")
            print(f"Removed by distance filter: {len(clean_df) - len(distance_filtered_df)}")
            
            # Clean up temp directory
            shutil.rmtree(temp_ca_clean_dir)
            
            # Use distance-filtered dataset as final clean dataset
            final_clean_df = distance_filtered_df
        else:
            final_clean_df = clean_df
            print("\nDistance-based filtering not applied (no reference PDB provided)")
        
        # Create CA distribution plots (including distance plots if available)
        if create_plots:
            self.create_plots(ca_counts, clean_ca_counts, ca_segments_dir, self.k_factor, 
                            ca_lower, ca_upper, ca_Q1, ca_Q3,
                            distance_data=self.distances_df if apply_distance_filter else None,
                            distance_cutoff=distance_cutoff,
                            show_original=show_original and apply_ca_count_filter,
                            show_clean=show_clean and apply_ca_count_filter,
                            show_boxplot=show_boxplot and apply_ca_count_filter)
        
        # Create clean directory with non-outlier files (using final filtered dataset)
        if clean_dir_name is None:
            # Auto-generate clean directory name by adding '_cleaned' suffix
            base_name = os.path.basename(ca_segments_dir.rstrip('/'))
            clean_dir_name = f"{base_name}_cleaned"
        clean_dir = os.path.join(os.path.dirname(ca_segments_dir), clean_dir_name)
        if os.path.exists(clean_dir):
            shutil.rmtree(clean_dir)
        os.makedirs(clean_dir)
        
        print(f"\nCopying final clean structures to: {clean_dir}")
        for _, row in tqdm(final_clean_df.iterrows(), total=len(final_clean_df), desc="Copying clean files"):
            src = row['filepath']
            dst = os.path.join(clean_dir, row['filename'])
            shutil.copy2(src, dst)
        
        print(f"Created clean directory with {len(final_clean_df)} structures")
        
        # Save results
        results = {
            'original_df': df,
            'ca_filtered_df': clean_df,
            'final_clean_df': final_clean_df,
            'apply_ca_count_filter': apply_ca_count_filter,
            'outlier_indices': ca_outlier_indices,
            'ca_outlier_info': {
                'indices': ca_outlier_indices,
                'bounds': (ca_lower, ca_upper),
                'quartiles': (ca_Q1, ca_Q3),
                'IQR': ca_IQR
            },
            'distance_df': self.distances_df if apply_distance_filter else None,
            'distance_cutoff': distance_cutoff if apply_distance_filter else None,
            'k_factor': self.k_factor,
            'clean_dir': clean_dir
        }
        
        # Save outlier information to CSV
        if apply_ca_count_filter and len(ca_outlier_indices) > 0:
            outlier_df = df.iloc[ca_outlier_indices].copy()
            outlier_df.to_csv(os.path.join(ca_segments_dir, f"ca_outliers_removed_k{self.k_factor}.csv"), 
                            index=False)
            print(f"CA count outlier information saved to: ca_outliers_removed_k{self.k_factor}.csv")
        
        if apply_ca_count_filter:
            clean_df.to_csv(
                os.path.join(ca_segments_dir, f"ca_filtered_dataset_k{self.k_factor}.csv"),
                index=False,
            )
            print(
                f"CA-filtered dataset information saved to: "
                f"ca_filtered_dataset_k{self.k_factor}.csv"
            )
        
        # Save final clean dataset information (after all filters)
        final_clean_df.to_csv(os.path.join(ca_segments_dir, f"final_clean_dataset_k{self.k_factor}.csv"), 
                             index=False)
        print(f"Final clean dataset information saved to: final_clean_dataset_k{self.k_factor}.csv")
        
        # Save distance information if available
        if apply_distance_filter and self.distances_df is not None:
            self.distances_df.to_csv(os.path.join(ca_segments_dir, f"terminal_ca_distances.csv"), 
                                    index=False)
            print(f"Distance information saved to: terminal_ca_distances.csv")
            
            # Save distance outliers
            distance_outliers = self.get_outliers(max_distance=distance_cutoff)
            if len(distance_outliers) > 0:
                distance_outliers.to_csv(os.path.join(ca_segments_dir, f"distance_outliers_cutoff{distance_cutoff}.csv"),
                                        index=False)
                print(f"Distance outliers saved to: distance_outliers_cutoff{distance_cutoff}.csv")
        
        # Store results in instance
        self.results = results
        
        return results

    def filter_chains_by_loop_termini(
        self,
        input_chains_dir,
        output_chains_dir,
        motifs=None,
        distance_cutoff=3.0,
        create_plots=True,
    ):
        """Filter aligned full chains using activation-loop terminal Cα distances.

        The DFG→APE CA segment is extracted temporarily for measurement. Passing
        full-chain PDB files are copied to ``output_chains_dir``. No CA-count or
        Tukey filtering is applied.
        """
        import tempfile

        if self.ref_first_ca is None or self.ref_last_ca is None:
            raise RuntimeError(
                "Reference Cα coordinates not loaded. Initialise OutlierStripper "
                "with reference_pdb, ref_first_resid, and ref_last_resid."
            )

        if motifs is None:
            motifs = ["DFG", "APE"]

        pdb_files = sorted(glob(os.path.join(input_chains_dir, "*.pdb")))
        if not pdb_files:
            raise ValueError(f"No PDB files found in '{input_chains_dir}'.")

        if os.path.exists(output_chains_dir):
            shutil.rmtree(output_chains_dir)
        os.makedirs(output_chains_dir)

        stripper = CAStripper(motifs=motifs)
        file_info = []

        with tempfile.TemporaryDirectory(prefix="extremity_filter_temp_") as temp_dir:
            for pdb_file in tqdm(pdb_files, desc="Measuring loop termini"):
                filename = os.path.basename(pdb_file)
                temp_out = os.path.join(temp_dir, filename)

                success = stripper.process_single_structure(pdb_file, temp_dir)
                if not success or not os.path.exists(temp_out):
                    continue

                first_ca = self._get_ca_coordinates(temp_out, first_last="first")
                last_ca = self._get_ca_coordinates(temp_out, first_last="last")
                if first_ca is None or last_ca is None:
                    continue

                first_distance = float(np.linalg.norm(first_ca - self.ref_first_ca))
                last_distance = float(np.linalg.norm(last_ca - self.ref_last_ca))
                file_info.append(
                    {
                        "filename": filename,
                        "filepath": pdb_file,
                        "first_ca_distance": first_distance,
                        "last_ca_distance": last_distance,
                        "max_distance": max(first_distance, last_distance),
                    }
                )

        if not file_info:
            raise RuntimeError("No activation-loop termini could be measured.")

        df = pd.DataFrame(file_info)
        self.distances_df = (
            df[
                [
                    "filename",
                    "first_ca_distance",
                    "last_ca_distance",
                    "max_distance",
                ]
            ]
            .rename(columns={"filename": "pdb_file"})
            .copy()
        )

        applied_cutoff = float(distance_cutoff)
        clean_df = df[df["max_distance"] <= applied_cutoff].copy().reset_index(drop=True)
        passed_filenames = set(clean_df["filename"])

        if create_plots:
            self.create_plots(
                np.array([]),
                np.array([]),
                output_chains_dir,
                self.k_factor,
                0,
                0,
                0,
                0,
                distance_data=self.distances_df,
                distance_cutoff=applied_cutoff,
                show_original=False,
                show_clean=False,
                show_boxplot=False,
            )

        self.distances_df.to_csv(
            os.path.join(output_chains_dir, "terminal_ca_distances.csv"), index=False
        )
        outliers_df = self.distances_df[
            ~self.distances_df["pdb_file"].isin(passed_filenames)
        ]
        if not outliers_df.empty:
            outliers_df.to_csv(
                os.path.join(
                    output_chains_dir,
                    f"extremity_outliers_cutoff{applied_cutoff:.2f}A.csv",
                ),
                index=False,
            )

        for _, row in tqdm(
            clean_df.iterrows(), total=len(clean_df), desc="Copying passing chains"
        ):
            shutil.copy2(
                row["filepath"], os.path.join(output_chains_dir, row["filename"])
            )

        print(f"Structures passing: {len(clean_df)} / {len(df)}")
        print(f"Full chains → {output_chains_dir}")

        results = {
            "original_df": df,
            "final_clean_df": clean_df,
            "distance_df": self.distances_df,
            "distance_cutoff": applied_cutoff,
            "output_chains_dir": output_chains_dir,
        }
        self.results = results
        return results

    @staticmethod
    def _extract_seqres_from_pdb(pdb_file: str) -> dict[str, str]:
        """Parse SEQRES records from a full PDB file into {chain_id: sequence}."""
        from Bio.SeqUtils import seq1

        seq_dict: dict[str, str] = {}
        current_chain = None
        current_seq: list[str] = []

        with open(pdb_file, "r") as file:
            for line in file:
                if not line.startswith("SEQRES"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                chain_id = parts[2]
                if chain_id != current_chain:
                    if current_chain is not None:
                        seq_dict[current_chain] = "".join(
                            seq1(residue) for residue in current_seq
                        )
                    current_chain = chain_id
                    current_seq = []
                current_seq.extend(parts[4:])

        if current_chain is not None:
            seq_dict[current_chain] = "".join(seq1(residue) for residue in current_seq)
        return seq_dict

    @staticmethod
    def _atom_loop_length(chain_pdb_path: str) -> int | None:
        """ATOM-only DFG→APE length for diagnostics; None if motifs not found."""
        from Bio.PDB import PDBParser, PPBuilder

        try:
            structure = PDBParser(QUIET=True).get_structure("PDB", chain_pdb_path)
            sequence = ""
            for model in structure:
                for chain in model:
                    for pp in PPBuilder().build_peptides(chain):
                        sequence += str(pp.get_sequence())
                    if sequence:
                        break
                if sequence:
                    break
            if not sequence:
                return None
            dfg = sequence.find("DFG")
            ape = sequence.find("APE")
            if dfg < 0 or ape < 0 or ape < dfg:
                return None
            return ape + 3 - dfg
        except Exception:
            return None

    @staticmethod
    def build_activation_loop_tsv(
        input_chains_dir,
        loop_tsv_path="Results/activation_loop_sequences.tsv",
        *,
        full_pdb_dir="Results/InterPro_PDBs",
        quiet=False,
    ) -> pd.DataFrame:
        """
        Build loop lengths from full-PDB SEQRES (includes missing residues).

        For each ``PDBID_CHAIN.pdb`` in ``input_chains_dir``, parse SEQRES for that
        chain from ``full_pdb_dir/{PDBID}.pdb``, take DFG through APE inclusive, and
        write ``pdb_basename``, ``loop_sequence``, ``loop_length`` (plus diagnostic
        ``atom_loop_length``).
        """
        pdb_files = sorted(glob(os.path.join(input_chains_dir, "*.pdb")))
        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in '{input_chains_dir}'.")

        rows = []
        skipped = []

        for pdb_path in tqdm(pdb_files, desc="Extracting SEQRES DFG–APE loops", disable=quiet):
            basename = os.path.splitext(os.path.basename(pdb_path))[0]
            parts = basename.split("_")
            if len(parts) < 2:
                skipped.append((basename, "unexpected filename"))
                continue
            pdb_id, chain_id = parts[0].upper(), parts[1]

            full_pdb_path = os.path.join(full_pdb_dir, f"{pdb_id}.pdb")
            if not os.path.isfile(full_pdb_path):
                # also try lowercase / .cif is not supported by SEQRES parser here
                alt = os.path.join(full_pdb_dir, f"{pdb_id.lower()}.pdb")
                if os.path.isfile(alt):
                    full_pdb_path = alt
                else:
                    skipped.append((basename, f"no SEQRES PDB in {full_pdb_dir}"))
                    continue

            try:
                seq_dict = OutlierStripper._extract_seqres_from_pdb(full_pdb_path)
                # SEQRES chain IDs can be case-sensitive; try exact then casefold match
                sequence = seq_dict.get(chain_id)
                if sequence is None:
                    for cid, seq in seq_dict.items():
                        if cid.upper() == chain_id.upper():
                            sequence = seq
                            break
                if not sequence:
                    skipped.append((basename, f"no SEQRES for chain {chain_id}"))
                    continue

                dfg = sequence.find("DFG")
                ape = sequence.find("APE")
                if dfg < 0 or ape < 0 or ape < dfg:
                    skipped.append((basename, "DFG/APE not found in SEQRES order"))
                    continue

                loop_seq = sequence[dfg : ape + 3]
                rows.append(
                    {
                        "pdb_basename": basename,
                        "loop_sequence": loop_seq,
                        "loop_length": len(loop_seq),
                        "atom_loop_length": OutlierStripper._atom_loop_length(pdb_path),
                    }
                )
            except Exception as exc:
                skipped.append((basename, str(exc)))

        if not rows:
            raise RuntimeError(
                f"Could not extract any SEQRES DFG–APE loops "
                f"({len(skipped)} skipped)."
            )

        loop_df = pd.DataFrame(rows)
        out_dir = os.path.dirname(os.path.abspath(loop_tsv_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        loop_df.to_csv(loop_tsv_path, sep="\t", index=False)

        if not quiet:
            print(
                f"Wrote {len(loop_df)} SEQRES loop sequences → {loop_tsv_path} "
                f"({len(skipped)} skipped)"
            )
        return loop_df

    def filter_chains_by_loop_length(
        self,
        loop_tsv_path="Results/activation_loop_sequences.tsv",
        input_chains_dir="Results/gap_filtered_chains/",
        output_chains_dir="Results/Bounds_CAfilter_chains/",
        plot_dir=None,
        *,
        full_pdb_dir="Results/InterPro_PDBs",
        length_lower=None,
        length_upper=None,
        display_plot=True,
        quiet=False,
    ):
        """
        Filter full-chain PDBs by activation-loop sequence length.

        Loop length is SEQRES DFG→APE inclusive (missing residues counted).
        Always rebuilds ``loop_tsv_path`` from ``full_pdb_dir`` so lengths stay correct.

        If ``length_lower`` and ``length_upper`` are both set, uses a fixed inclusive
        window. Otherwise applies Tukey's IQR method with ``self.k_factor``.
        """
        from workflow.utilities import clear_and_make

        quiet = quiet or (not getattr(self, "verbose", True))

        if not quiet:
            print(
                f"Rebuilding SEQRES loop lengths from {full_pdb_dir} "
                f"using chains in {input_chains_dir} …"
            )
        self.build_activation_loop_tsv(
            input_chains_dir,
            loop_tsv_path=loop_tsv_path,
            full_pdb_dir=full_pdb_dir,
            quiet=quiet,
        )

        if plot_dir is None:
            plot_dir = os.path.dirname(os.path.abspath(loop_tsv_path)) or "."

        loop_df = pd.read_csv(loop_tsv_path, sep="\t")
        required = {"pdb_basename", "loop_sequence"}
        missing = required - set(loop_df.columns)
        if missing:
            raise KeyError(
                f"{loop_tsv_path} missing columns {sorted(missing)}. "
                f"Available: {list(loop_df.columns)}"
            )

        loop_df = loop_df.copy()
        loop_df["loop_length"] = loop_df["loop_sequence"].astype(str).str.len()
        lengths = loop_df["loop_length"].to_numpy()
        names = loop_df["pdb_basename"].astype(str).to_numpy()

        use_fixed = length_lower is not None and length_upper is not None
        if use_fixed:
            lower_bound = float(length_lower)
            upper_bound = float(length_upper)
            Q1 = Q3 = IQR = float("nan")
            mask_pass = (lengths >= lower_bound) & (lengths <= upper_bound)
            outlier_idx = np.where(~mask_pass)[0]
        else:
            outlier_idx, lower_bound, upper_bound, Q1, Q3, IQR = (
                self.tukey_outlier_detection(lengths)
            )
            mask_pass = np.ones(len(lengths), dtype=bool)
            mask_pass[outlier_idx] = False

        passing_names = names[mask_pass]
        removed_names = names[~mask_pass]
        removed_lengths = lengths[~mask_pass]

        if not quiet:
            if use_fixed:
                print(f"Fixed bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
            else:
                print(f"k-factor : {self.k_factor}")
                print(f"Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}")
                print(f"Bounds   : [{lower_bound:.1f}, {upper_bound:.1f}]")
            print(f"Passing  : {int(mask_pass.sum())} / {len(lengths)}")
            print(f"Removed  : {int((~mask_pass).sum())}")

        clear_and_make(output_chains_dir)
        n_copied = 0
        for name in passing_names:
            base = name if str(name).lower().endswith(".pdb") else f"{name}.pdb"
            stem = os.path.splitext(os.path.basename(str(name)))[0]
            candidates = [
                os.path.join(input_chains_dir, base),
                os.path.join(input_chains_dir, f"{stem}.pdb"),
            ]
            src = next((p for p in candidates if os.path.isfile(p)), None)
            if src:
                shutil.copy2(src, output_chains_dir)
                n_copied += 1
        if not quiet:
            print(f"\nCopied {n_copied} structures → {output_chains_dir}")

        os.makedirs(plot_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(lengths, bins=50, color="steelblue", edgecolor="white")
        axes[0].set_title("Raw loop-length distribution")
        axes[0].set_xlabel("Activation loop length (residues)")
        axes[0].set_ylabel("Count")

        axes[1].hist(lengths, bins=50, color="steelblue", edgecolor="white")
        axes[1].axvline(
            lower_bound, color="red", linestyle="--", label=f"Lower = {lower_bound:.1f}"
        )
        axes[1].axvline(
            upper_bound, color="red", linestyle="--", label=f"Upper = {upper_bound:.1f}"
        )
        axes[1].set_title("Length cutoffs")
        axes[1].set_xlabel("Activation loop length (residues)")
        axes[1].legend()

        axes[2].hist(
            lengths[mask_pass], bins=50, color="mediumseagreen", edgecolor="white"
        )
        axes[2].set_title("After length filter")
        axes[2].set_xlabel("Activation loop length (residues)")

        plt.tight_layout()
        tag = (
            f"fixed_{int(lower_bound)}_{int(upper_bound)}"
            if use_fixed
            else f"tukey_k{self.k_factor}"
        )
        plot_path = os.path.join(plot_dir, f"loop_length_summary_{tag}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if display_plot:
            try:
                from IPython.display import Image, display

                display(Image(filename=plot_path))
            except Exception:
                pass

        below = [
            (n, int(l))
            for n, l in zip(removed_names, removed_lengths)
            if l < lower_bound
        ]
        above = [
            (n, int(l))
            for n, l in zip(removed_names, removed_lengths)
            if l > upper_bound
        ]

        if not quiet:
            if below:
                print(f"\nToo short  (length < {lower_bound:.1f})  [{len(below)}]:")
                for name, le in sorted(below, key=lambda x: x[1]):
                    print(f"  {name}  ({le} residues)")
            if above:
                print(f"\nToo long   (length > {upper_bound:.1f})  [{len(above)}]:")
                for name, le in sorted(above, key=lambda x: x[1]):
                    print(f"  {name}  ({le} residues)")

        results = {
            "loop_df": loop_df,
            "k_factor": self.k_factor,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "bounds": (lower_bound, upper_bound),
            "quartiles": (Q1, Q3),
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "outlier_indices": outlier_idx,
            "n_passing": int(mask_pass.sum()),
            "n_removed": int((~mask_pass).sum()),
            "n_copied": n_copied,
            "output_chains_dir": output_chains_dir,
            "plot_path": plot_path,
            "passing_basenames": [os.path.splitext(os.path.basename(str(n)))[0] for n in passing_names],
            "too_short": sorted(below, key=lambda x: x[1]),
            "too_long": sorted(above, key=lambda x: x[1]),
        }
        self.results = results
        return results

    @staticmethod
    def tukey_metrics_for_k(k, lengths, names):
        """
        Return one summary row for a given Tukey k-factor on loop lengths.

        Metrics only — no PDB I/O.
        """
        lengths = np.asarray(lengths)
        names = np.asarray(names)
        stripper = OutlierStripper(k_factor=float(k), verbose=False)
        outlier_idx, lower, upper, q1, q3, iqr = stripper.tukey_outlier_detection(
            lengths, k=k
        )

        mask_pass = np.ones(len(lengths), dtype=bool)
        mask_pass[outlier_idx] = False
        removed_lengths = lengths[outlier_idx]
        removed_names = names[outlier_idx]

        n_too_short = int((removed_lengths < lower).sum()) if len(outlier_idx) else 0
        n_too_long = int((removed_lengths > upper).sum()) if len(outlier_idx) else 0

        return {
            "k_factor": float(k),
            "method": "tukey",
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_total": len(lengths),
            "n_passing": int(mask_pass.sum()),
            "n_removed": int(len(outlier_idx)),
            "pct_passing": 100.0 * mask_pass.sum() / len(lengths),
            "pct_removed": 100.0 * len(outlier_idx) / len(lengths),
            "n_too_short": n_too_short,
            "n_too_long": n_too_long,
            "removed_basenames": list(removed_names),
            "passing_basenames": list(names[mask_pass]),
        }

    @staticmethod
    def fixed_bounds_metrics(lengths, names, length_lower=18, length_upper=32):
        """
        Return one summary row for a fixed inclusive length window (eyeball method).

        Schema matches ``tukey_metrics_for_k`` (metrics only — no PDB I/O).
        """
        lengths = np.asarray(lengths)
        names = np.asarray(names)
        lower = float(length_lower)
        upper = float(length_upper)
        if lower > upper:
            raise ValueError(
                f"length_lower ({lower}) must be <= length_upper ({upper})"
            )

        mask_pass = (lengths >= lower) & (lengths <= upper)
        outlier_idx = np.where(~mask_pass)[0]
        removed_lengths = lengths[outlier_idx]
        n_too_short = int((removed_lengths < lower).sum()) if len(outlier_idx) else 0
        n_too_long = int((removed_lengths > upper).sum()) if len(outlier_idx) else 0
        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1

        return {
            "k_factor": np.nan,
            "method": "fixed_bounds",
            "Q1": float(q1),
            "Q3": float(q3),
            "IQR": float(iqr),
            "lower_bound": lower,
            "upper_bound": upper,
            "n_total": len(lengths),
            "n_passing": int(mask_pass.sum()),
            "n_removed": int(len(outlier_idx)),
            "pct_passing": 100.0 * mask_pass.sum() / len(lengths),
            "pct_removed": 100.0 * len(outlier_idx) / len(lengths),
            "n_too_short": n_too_short,
            "n_too_long": n_too_long,
            "removed_basenames": list(names[outlier_idx]),
            "passing_basenames": list(names[mask_pass]),
            "length_lower": lower,
            "length_upper": upper,
        }

    @staticmethod
    def scan_tukey_k_factors(
        lengths,
        names,
        k_values,
        *,
        experiment_dir="Results/tukey_k_scan",
        scan_csv=None,
        scan_pkl=None,
        verbose=True,
    ):
        """
        Scan Tukey k-factors over loop lengths; save CSV/PKL summaries.

        Returns
        -------
        dict
            ``scan_df``, ``scan_rows``, ``scan_csv``, ``scan_pkl``
        """
        lengths = np.asarray(lengths)
        names = np.asarray(names)
        k_values = np.asarray(k_values, dtype=float)

        os.makedirs(experiment_dir, exist_ok=True)
        if scan_csv is None:
            scan_csv = os.path.join(experiment_dir, "k_factor_scan_summary.csv")
        if scan_pkl is None:
            scan_pkl = os.path.join(experiment_dir, "k_factor_scan_summary.pkl")

        scan_rows = []
        for k in k_values:
            row = OutlierStripper.tukey_metrics_for_k(k, lengths, names)
            scan_rows.append(row)
            if verbose:
                print(
                    f"k={k:4.1f}  pass={row['n_passing']:4d}  "
                    f"rem={row['n_removed']:4d}  "
                    f"bounds=[{row['lower_bound']:5.1f}, {row['upper_bound']:5.1f}]  "
                    f"short={row['n_too_short']:3d}  long={row['n_too_long']:3d}"
                )

        drop_keys = ("removed_basenames", "passing_basenames")
        scan_df = pd.DataFrame(
            [{kk: vv for kk, vv in r.items() if kk not in drop_keys} for r in scan_rows]
        )
        scan_df.to_csv(scan_csv, index=False)

        with open(scan_pkl, "wb") as f:
            pickle.dump(
                {
                    "scan_rows": scan_rows,
                    "k_values": k_values,
                    "lengths": lengths,
                    "names": names,
                },
                f,
            )

        if verbose:
            print(f"\nSaved summary → {scan_csv}")
            print(f"Saved details → {scan_pkl}")

        return {
            "scan_df": scan_df,
            "scan_rows": scan_rows,
            "scan_csv": scan_csv,
            "scan_pkl": scan_pkl,
        }

    @staticmethod
    def compare_fixed_bounds_vs_tukey(
        lengths,
        names,
        *,
        length_lower=18,
        length_upper=32,
        k_factor=0.8,
        save_dir=None,
        verbose=True,
    ):
        """
        Compare fixed-bounds (eyeball) retention vs Tukey at ``k_factor``.

        Returns overlap counts and basename sets; optionally writes a CSV.
        """
        lengths = np.asarray(lengths)
        names = np.asarray(names)

        eyeball = OutlierStripper.fixed_bounds_metrics(
            lengths, names, length_lower=length_lower, length_upper=length_upper
        )
        tukey = OutlierStripper.tukey_metrics_for_k(k_factor, lengths, names)

        pass_eye = set(eyeball["passing_basenames"])
        pass_tuk = set(tukey["passing_basenames"])
        rem_eye = set(eyeball["removed_basenames"])
        rem_tuk = set(tukey["removed_basenames"])

        overlap = {
            "n_total": len(lengths),
            "eyeball_bounds": (float(length_lower), float(length_upper)),
            "tukey_k": float(k_factor),
            "tukey_bounds": (float(tukey["lower_bound"]), float(tukey["upper_bound"])),
            "n_pass_eyeball": len(pass_eye),
            "n_pass_tukey": len(pass_tuk),
            "n_pass_both": len(pass_eye & pass_tuk),
            "n_pass_only_eyeball": len(pass_eye - pass_tuk),
            "n_pass_only_tukey": len(pass_tuk - pass_eye),
            "n_removed_eyeball": len(rem_eye),
            "n_removed_tukey": len(rem_tuk),
            "n_removed_both": len(rem_eye & rem_tuk),
            "n_removed_only_eyeball": len(rem_eye - rem_tuk),
            "n_removed_only_tukey": len(rem_tuk - rem_eye),
            "pass_only_eyeball": sorted(pass_eye - pass_tuk),
            "pass_only_tukey": sorted(pass_tuk - pass_eye),
            "removed_only_eyeball": sorted(rem_eye - rem_tuk),
            "removed_only_tukey": sorted(rem_tuk - rem_eye),
        }

        summary_df = pd.DataFrame(
            [
                {
                    "method": "eyeball_fixed",
                    "lower_bound": eyeball["lower_bound"],
                    "upper_bound": eyeball["upper_bound"],
                    "k_factor": np.nan,
                    "n_passing": eyeball["n_passing"],
                    "n_removed": eyeball["n_removed"],
                    "pct_passing": eyeball["pct_passing"],
                    "n_too_short": eyeball["n_too_short"],
                    "n_too_long": eyeball["n_too_long"],
                },
                {
                    "method": f"tukey_k{k_factor}",
                    "lower_bound": tukey["lower_bound"],
                    "upper_bound": tukey["upper_bound"],
                    "k_factor": float(k_factor),
                    "n_passing": tukey["n_passing"],
                    "n_removed": tukey["n_removed"],
                    "pct_passing": tukey["pct_passing"],
                    "n_too_short": tukey["n_too_short"],
                    "n_too_long": tukey["n_too_long"],
                },
            ]
        )

        overlap_df = pd.DataFrame(
            [
                {
                    "metric": "pass_both",
                    "count": overlap["n_pass_both"],
                },
                {
                    "metric": "pass_only_eyeball",
                    "count": overlap["n_pass_only_eyeball"],
                },
                {
                    "metric": "pass_only_tukey",
                    "count": overlap["n_pass_only_tukey"],
                },
                {
                    "metric": "removed_both",
                    "count": overlap["n_removed_both"],
                },
                {
                    "metric": "removed_only_eyeball",
                    "count": overlap["n_removed_only_eyeball"],
                },
                {
                    "metric": "removed_only_tukey",
                    "count": overlap["n_removed_only_tukey"],
                },
            ]
        )

        out = {
            "eyeball": eyeball,
            "tukey": tukey,
            "overlap": overlap,
            "summary_df": summary_df,
            "overlap_df": overlap_df,
        }

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            summary_csv = os.path.join(save_dir, "eyeball_vs_tukey_summary.csv")
            overlap_csv = os.path.join(save_dir, "eyeball_vs_tukey_overlap.csv")
            summary_df.to_csv(summary_csv, index=False)
            overlap_df.to_csv(overlap_csv, index=False)
            out["summary_csv"] = summary_csv
            out["overlap_csv"] = overlap_csv
            if verbose:
                print(f"Saved eyeball vs Tukey summary → {summary_csv}")
                print(f"Saved overlap counts          → {overlap_csv}")

        if verbose:
            print(
                f"Eyeball [{length_lower}, {length_upper}]: "
                f"pass={eyeball['n_passing']} rem={eyeball['n_removed']}"
            )
            print(
                f"Tukey k={k_factor}: "
                f"bounds=[{tukey['lower_bound']:.1f}, {tukey['upper_bound']:.1f}] "
                f"pass={tukey['n_passing']} rem={tukey['n_removed']}"
            )
            print(
                f"Pass overlap: both={overlap['n_pass_both']}  "
                f"only_eyeball={overlap['n_pass_only_eyeball']}  "
                f"only_tukey={overlap['n_pass_only_tukey']}"
            )

        return out


def strip_to_ca(input_dir="Results/activation_segments/reconstructed_mustang_ends/", 
                output_dir="Results/activation_segments/CA_segments/reconstructed_endsAlignment"):
    """
    Convenience function to create a CAStripper instance and process structures.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing PDB files to process
    output_dir : str
        Directory to save stripped PDB files
        
    Returns:
    --------
    str
        Path to the output directory
    """
    stripper = CAStripper()
    return stripper.strip_to_ca(input_dir, output_dir)



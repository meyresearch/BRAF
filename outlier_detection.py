import os
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from glob import glob
from tqdm import tqdm
import pandas as pd
from scipy import stats
import shutil
from Bio.PDB import PDBParser, PPBuilder


class PDBProcessor:
    """
    Class for processing PDB files, extracting motifs, and stripping to CA atoms.
    """
    
    def __init__(self, motifs=None):
        """
        Initialize the PDB processor.
        
        Parameters
        ----------
        motifs : list of str, optional
            List of motifs to search for (default: ['DFG', 'APE'])
        """
        self.motifs = motifs if motifs is not None else ['DFG', 'APE']
    
    def strip_and_save_ca_coordinates(self, input_dir, output_dir, overwrite=True):
        """
        Strip CA coordinates from PDB files and save them to a new directory.
        
        Parameters
        ----------
        input_dir : str
            Directory containing the input PDB files
        output_dir : str
            Directory to save the stripped CA coordinate files
        overwrite : bool, default=True
            Whether to overwrite the output directory if it exists
            
        Returns
        -------
        list
            List of dictionaries containing file information for each processed file
        """
        # Find all PDB files in the input directory
        pdb_files = glob(os.path.join(input_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files in {input_dir}")
        
        # Create output directory for CA coordinates
        if overwrite and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory for CA coordinates: {output_dir}")
        
        # Lists to store results
        file_info = []
        
        # Process each PDB file
        for pdb_file in tqdm(pdb_files, desc="Stripping and saving CA coordinates"):
            try:
                # Load the PDB file
                traj = md.load(pdb_file)
                
                # Count the number of CA atoms
                num_ca = traj.n_atoms
                
                # Save the CA coordinates to the output directory
                output_filename = os.path.basename(pdb_file)
                output_path = os.path.join(output_dir, output_filename)
                traj.save(output_path)
                
                file_info.append({
                    'filename': os.path.basename(pdb_file),
                    'filepath': pdb_file,
                    'ca_count': num_ca,
                    'ca_coordinates_path': output_path
                })
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
        
        print(f"Saved {len(file_info)} CA coordinate files to {output_dir}")
        return file_info

    def extract_sequence_and_residue_numbers(self, pdb_file):
        """
        Extract sequence and corresponding PDB residue numbers from atomic coordinates 
        for the first chain found in the PDB file.
        
        Parameters
        -----------
        pdb_file : str
            Path to the PDB file
            
        Returns
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
        
        Parameters
        -----------
        seq : str
            Protein sequence
        motif : str
            Motif to search for
            
        Returns
        --------
        tuple or None
            (start_index, end_index) if motif found, None otherwise
        """
        index = seq.find(motif)
        if index == -1:
            return None
        return index, index + len(motif)

    def find_all_motifs(self, seq, motifs=None):
        """
        Find all configured motifs in a sequence.
        
        Parameters
        -----------
        seq : str
            Protein sequence
        motifs : list of str, optional
            List of motifs to search for (default: uses self.motifs)
            
        Returns
        --------
        dict
            Dictionary with motif names as keys and (start, end) tuples as values
        """
        if motifs is None:
            motifs = self.motifs
        
        motif_indices = {}
        for motif in motifs:
            indices = self.find_motif_indices(seq, motif)
            if indices:
                motif_indices[motif] = indices
        return motif_indices

    def get_motif_region_bounds(self, motif_indices):
        """
        Get the start and end bounds that encompass all found motifs.
        
        Parameters
        -----------
        motif_indices : dict
            Dictionary with motif names as keys and (start, end) tuples as values
            
        Returns
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

    def strip_to_ca_atoms(self, pdb_path, start_residue, end_residue):
        """
        Load PDB, extract CA atoms for residues in the specified slice,
        and return an mdtraj.Trajectory object with just those atoms.
        
        Parameters
        -----------
        pdb_path : str
            Path to the PDB file
        start_residue : int
            Starting residue index (0-based)
        end_residue : int
            Ending residue index (0-based, exclusive)
            
        Returns
        --------
        md.Trajectory
            Trajectory object with only CA atoms from the specified range
        """
        pdb = md.load(pdb_path)
        print(f"Loaded PDB: {pdb_path}")

        # Extract CA atoms within the specified range of residues
        atom_indices = [
            atom.index 
            for res in pdb.top._residues[start_residue:end_residue] 
            for atom in res.atoms 
            if atom.name == "CA"
        ]
        print(f"Atom indices for CA: {atom_indices}")

        return pdb.atom_slice(atom_indices)

    def process_single_structure_with_motifs(self, pdb_file, target_dir, motifs=None):
        """
        Process a single PDB file to extract motif regions and strip to CA atoms.
        
        Parameters
        -----------
        pdb_file : str
            Path to the PDB file to process
        target_dir : str
            Directory to save stripped PDB files
        motifs : list of str, optional
            List of motifs to search for (default: uses self.motifs)
            
        Returns
        --------
        dict or None
            Dictionary with file information if successful, None otherwise
        """
        if motifs is None:
            motifs = self.motifs
        
        try:
            # Extract sequence and residue numbers
            seq, residue_numbers = self.extract_sequence_and_residue_numbers(pdb_file)
            if seq is None or residue_numbers is None:
                print(f"Skipping {pdb_file} due to inability to extract sequence and residue numbers.")
                return None
            
            print(f"Sequence for {os.path.basename(pdb_file)}: {seq}")
            
            # Find all motifs in the sequence
            motif_indices = self.find_all_motifs(seq, motifs)
            print(f"Found motifs: {motif_indices}")

            if not motif_indices:
                print(f"Skipping {pdb_file} due to missing required motifs.")
                return None

            # Determine start and end residues (using residue indices, not PDB numbers)
            start_residue, end_residue = self.get_motif_region_bounds(motif_indices)
            
            if start_residue is None or end_residue is None:
                print(f"Skipping {pdb_file} due to invalid motif bounds.")
                return None
            
            print(f"Start residue index: {start_residue}, End residue index: {end_residue}")
            
            # Strip to CA atoms
            stripped = self.strip_to_ca_atoms(pdb_file, start_residue, end_residue)
            
            # Save stripped PDB
            output_file = os.path.join(target_dir, os.path.basename(pdb_file))
            stripped.save(output_file)
            print(f"Saved stripped PDB to: {output_file}")
            
            return {
                'filename': os.path.basename(pdb_file),
                'filepath': pdb_file,
                'ca_count': stripped.n_atoms,
                'ca_coordinates_path': output_file,
                'motifs_found': motif_indices,
                'start_residue': start_residue,
                'end_residue': end_residue
            }
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            return None

    def process_directory_with_motifs(self, input_dir, output_dir, motifs=None, overwrite=True):
        """
        Process all PDB files in the input directory to extract motif regions and strip to CA atoms.
        
        Parameters
        -----------
        input_dir : str
            Directory containing PDB files to process
        output_dir : str
            Directory to save stripped PDB files
        motifs : list of str, optional
            List of motifs to search for (default: uses self.motifs)
        overwrite : bool, default=True
            Whether to overwrite the output directory if it exists
            
        Returns
        --------
        list
            List of dictionaries containing file information for each processed file
        """
        if motifs is None:
            motifs = self.motifs
        
        # Create output directory
        if overwrite and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'#'*80}")
        print(f"PROCESSING STRUCTURES FOR MOTIF-BASED CA STRIPPING")
        print(f"{'#'*80}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Motifs: {motifs}")
        
        # Find all PDB files in input directory
        pdb_files = glob(os.path.join(input_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files to process")
        
        if not pdb_files:
            print("No PDB files found in input directory!")
            return []
        
        # Process each PDB file
        file_info = []
        successful_count = 0
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            print(f"\nProcessing: {os.path.basename(pdb_file)}")
            result = self.process_single_structure_with_motifs(pdb_file, output_dir, motifs)
            if result:
                file_info.append(result)
                successful_count += 1
        
        print(f"\n{'#'*80}")
        print(f"MOTIF-BASED STRIPPING COMPLETE")
        print(f"{'#'*80}")
        print(f"Successfully processed {successful_count}/{len(pdb_files)} structures")
        print(f"Results saved to: {output_dir}")
        
        return file_info

    def strip_to_ca_with_motifs(self, input_dir="Results/activation_segments/reconstructed_mustang_ends/", 
                               output_dir="Results/activation_segments/CA_segments/reconstructed_endsAlignment",
                               motifs=None):
        """
        Convenience function to process all PDB files in the input directory with motif-based stripping.
        
        Parameters
        -----------
        input_dir : str
            Directory containing PDB files to process
        output_dir : str
            Directory to save stripped PDB files
        motifs : list of str, optional
            List of motifs to search for (default: uses self.motifs)
            
        Returns
        --------
        list
            List of dictionaries containing file information for each processed file
        """
        if motifs is None:
            motifs = self.motifs
        
        return self.process_directory_with_motifs(input_dir, output_dir, motifs)


class OutlierDetectionAnalyzer:
    """
    Class for analyzing PDB files and detecting outliers using Tukey's method.
    Focuses on CA atom counts for outlier detection.
    """
    
    def __init__(self, ca_segments_dir, k_factor=1.5):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        ca_segments_dir : str
            Directory containing the stripped PDB files with CA atoms
        k_factor : float, default=1.5
            The multiplier for the IQR in Tukey's method
        """
        self.ca_segments_dir = ca_segments_dir
        self.k_factor = k_factor
        self.results = None
    
    def tukey_outlier_detection(self, data, k=None):
        """
        Apply Tukey's method to detect outliers using the Interquartile Range (IQR).
        
        Parameters
        ----------
        data : array-like
            The data to analyze for outliers
        k : float, optional
            The multiplier for the IQR. If None, uses self.k_factor
            
        Returns
        -------
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
    
    def create_plots(self, original_ca, clean_ca, output_dir, k_factor, ca_lower, ca_upper, ca_Q1, ca_Q3):
        """Create plots showing original and clean CA distributions"""
        
        # 1. Original CA distribution with Tukey cutoff lines
        plt.figure(figsize=(12, 8))
        
        # Plot histogram
        n, bins, patches = plt.hist(original_ca, bins=20, alpha=0.7, color='lightblue', edgecolor='black', label='Original Data')
        
        # Add vertical lines for Tukey bounds
        plt.axvline(x=ca_lower, color='red', linestyle='--', linewidth=2, label=f'Lower Bound: {ca_lower:.1f}')
        plt.axvline(x=ca_upper, color='red', linestyle='--', linewidth=2, label=f'Upper Bound: {ca_upper:.1f}')
        
        # Add quartile lines
        plt.axvline(x=ca_Q1, color='orange', linestyle=':', linewidth=2, label=f'Q1: {ca_Q1:.1f}')
        plt.axvline(x=ca_Q3, color='orange', linestyle=':', linewidth=2, label=f'Q3: {ca_Q3:.1f}')
        
        # Add median line
        median_ca = np.median(original_ca)
        plt.axvline(x=median_ca, color='green', linestyle='-', linewidth=2, label=f'Median: {median_ca:.1f}')
        
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
        plt.figure(figsize=(10, 6))
        
        n, bins, patches = plt.hist(clean_ca, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        
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
    
    def analyze_ca_segments_with_outlier_removal(self, create_plots=True, clean_dir_name="reconstructed_endsAlignment_cleaned"):
        """
        Analyzes PDB files in the specified directory, applies Tukey's method to remove outliers based on CA counts only,
        and generates plots showing the cutoffs and cleaned dataset.
        
        Parameters
        ----------
        create_plots : bool, default=True
            Whether to create visualization plots
        clean_dir_name : str, default="reconstructed_endsAlignment_cleaned"
            Name for the clean directory
            
        Returns
        -------
        dict
            Dictionary containing analysis results and file information
        """
        # Find all PDB files in the specified directory
        pdb_files = glob(os.path.join(self.ca_segments_dir, "*.pdb"))
        print(f"Found {len(pdb_files)} PDB files in {self.ca_segments_dir}")
        
        # Lists to store results
        file_info = []
        
        # Process each PDB file
        for pdb_file in tqdm(pdb_files, desc="Analyzing CA segments"):
            try:
                # Load the PDB file
                traj = md.load(pdb_file)
                
                # Count the number of CA atoms
                num_ca = traj.n_atoms
                
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
        print(f"CA counts - Min: {ca_counts.min()}, Max: {ca_counts.max()}, Mean: {ca_counts.mean():.2f}, Median: {np.median(ca_counts):.2f}")
        
        # Apply Tukey's method to CA counts
        ca_outlier_indices, ca_lower, ca_upper, ca_Q1, ca_Q3, ca_IQR = self.tukey_outlier_detection(ca_counts)
        
        print(f"\n{'='*60}")
        print("TUKEY'S OUTLIER DETECTION - CA COUNTS")
        print(f"{'='*60}")
        print(f"Q1: {ca_Q1:.2f}, Q3: {ca_Q3:.2f}, IQR: {ca_IQR:.2f}")
        print(f"Lower bound: {ca_lower:.2f}, Upper bound: {ca_upper:.2f}")
        print(f"Outliers found: {len(ca_outlier_indices)} out of {len(ca_counts)} ({len(ca_outlier_indices)/len(ca_counts)*100:.1f}%)")
        
        if len(ca_outlier_indices) > 0:
            print("CA count outliers:")
            for idx in ca_outlier_indices:
                print(f"  - {df.iloc[idx]['filename']}: {df.iloc[idx]['ca_count']} CA atoms")
        
        # Create clean dataset based on CA counts
        clean_df = df.drop(df.index[ca_outlier_indices]).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print("OUTLIER REMOVAL BASED ON CA COUNTS")
        print(f"{'='*60}")
        print(f"Outliers to remove: {len(ca_outlier_indices)} out of {len(df)} ({len(ca_outlier_indices)/len(df)*100:.1f}%)")
        print(f"Clean dataset size: {len(clean_df)} structures")
        
        # Print outliers being removed
        if len(ca_outlier_indices) > 0:
            print("\nStructures being removed as outliers:")
            outlier_df = df.iloc[ca_outlier_indices]
            for _, row in outlier_df.iterrows():
                print(f"  - {row['filename']}: {row['ca_count']} CA atoms")
        
        # Clean dataset statistics
        clean_ca_counts = clean_df['ca_count'].values
        
        print(f"\n{'='*60}")
        print("CLEAN DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total structures: {len(clean_df)}")
        print(f"CA counts - Min: {clean_ca_counts.min()}, Max: {clean_ca_counts.max()}, Mean: {clean_ca_counts.mean():.2f}, Median: {np.median(clean_ca_counts):.2f}")
        
        # Create CA distribution plots
        if create_plots:
            self.create_plots(ca_counts, clean_ca_counts, self.ca_segments_dir, self.k_factor, ca_lower, ca_upper, ca_Q1, ca_Q3)
        
        # Create clean directory with non-outlier files
        clean_dir = os.path.join("Results/activation_segments/CA_segments", clean_dir_name)
        if os.path.exists(clean_dir):
            shutil.rmtree(clean_dir)
        os.makedirs(clean_dir)
        
        print(f"\nCopying clean structures to: {clean_dir}")
        for _, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Copying clean files"):
            src = row['filepath']
            dst = os.path.join(clean_dir, row['filename'])
            shutil.copy2(src, dst)
        
        print(f"Created clean directory with {len(clean_df)} structures")
        
        # Save results
        results = {
            'original_df': df,
            'clean_df': clean_df,
            'outlier_indices': ca_outlier_indices,
            'ca_outlier_info': {
                'indices': ca_outlier_indices,
                'bounds': (ca_lower, ca_upper),
                'quartiles': (ca_Q1, ca_Q3),
                'IQR': ca_IQR
            },
            'k_factor': self.k_factor,
            'clean_dir': clean_dir
        }
        
        # Save outlier information to CSV
        if len(ca_outlier_indices) > 0:
            outlier_df = df.iloc[ca_outlier_indices].copy()
            outlier_df.to_csv(os.path.join(self.ca_segments_dir, f"outliers_removed_k{self.k_factor}.csv"), index=False)
            print(f"Outlier information saved to: outliers_removed_k{self.k_factor}.csv")
        
        # Save clean dataset information
        clean_df.to_csv(os.path.join(self.ca_segments_dir, f"clean_dataset_k{self.k_factor}.csv"), index=False)
        print(f"Clean dataset information saved to: clean_dataset_k{self.k_factor}.csv")
        
        # Store results in instance
        self.results = results
        
        return results
    
    def get_summary_stats(self):
        """
        Get summary statistics from the last analysis.
        
        Returns
        -------
        dict
            Summary statistics or None if no analysis has been run
        """
        if self.results is None:
            print("No analysis has been run yet. Call analyze_ca_segments_with_outlier_removal() first.")
            return None
        
        return {
            'original_count': len(self.results['original_df']),
            'clean_count': len(self.results['clean_df']),
            'outliers_removed': len(self.results['outlier_indices']),
            'outlier_percentage': len(self.results['outlier_indices'])/len(self.results['original_df'])*100,
            'k_factor': self.results['k_factor'],
            'clean_directory': self.results['clean_dir']
        }
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        stats = self.get_summary_stats()
        if stats:
            print(f"\n{'='*60}")
            print("OUTLIER DETECTION SUMMARY")
            print(f"{'='*60}")
            print(f"Original dataset: {stats['original_count']} structures")
            print(f"Clean dataset: {stats['clean_count']} structures")
            print(f"Outliers removed: {stats['outliers_removed']} structures")
            print(f"Outlier percentage: {stats['outlier_percentage']:.1f}%")
            print(f"K-factor used: {stats['k_factor']}")
            print(f"Clean structures saved to: {stats['clean_directory']}") 
import os
import glob
import shutil
import pickle as p
import time
import requests
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder
import pandas as pd
import mdtraj as md

'''
This file contains utility functions for the pipeline.
IT WOULD BE VERY GOOD TO HAVE A GUI UTILITY TO VISUALISE MANIPULATIONS OF PROTEINS
'''

def find_pdbs(directory):
    """Find all PDB files in the given directory."""
    return glob.glob(os.path.join(directory, "*.pdb"))

def find_pdbs_recursive(directory):
    """Find all PDB files in the given directory and all subdirectories."""
    pdb_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))
    return pdb_files

def fname(path):
    """Extract filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]

def count_pdb_files(directory, recursive=False):
    """
    Count PDB files in a directory.
    
    Args:
        directory (str): Path to the directory
        recursive (bool): If True, count PDB files in subdirectories as well
        
    Returns:
        int: Number of PDB files
    """
    if recursive:
        # Count PDB files recursively in all subdirectories
        pdb_files = find_pdbs_recursive(directory)
        return len(pdb_files)
    else:
        # Use the existing find_pdbs function for non-recursive counting
        pdb_files = find_pdbs(directory)
        return len(pdb_files)

def create_nonreconstructed_folder(unaligned_dir, reconstructed_dir, target_dir):
    """
    Create a folder with unaligned structures that have matching entries in the reconstructed folder.
    
    Args:
        unaligned_dir (str): Path to the directory containing unaligned PDB files
        reconstructed_dir (str): Path to the directory containing reconstructed PDB files
        target_dir (str): Path to the target directory where matching files will be copied
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    # Get list of files in both directories using find_pdbs
    unaligned_files = [os.path.basename(f) for f in find_pdbs(unaligned_dir)]
    reconstructed_files = [os.path.basename(f) for f in find_pdbs(reconstructed_dir)]
    
    # Extract prefixes (first 6 characters) from reconstructed files
    reconstructed_prefixes = {f[:6] for f in reconstructed_files}
    
    # Counter for copied files
    copied_count = 0
    skipped_count = 0
    
    # For each unaligned file, check if its prefix exists in reconstructed files
    for unaligned_file in unaligned_files:
        file_prefix = unaligned_file[:6]
        
        if file_prefix in reconstructed_prefixes:
            source_path = os.path.join(unaligned_dir, unaligned_file)
            target_path = os.path.join(target_dir, unaligned_file)
            
            # Copy the file
            shutil.copy(source_path, target_path)
            copied_count += 1
        else:
            skipped_count += 1
    
    print(f"Copied {copied_count} files from '{unaligned_dir}' to '{target_dir}'")
    print(f"Skipped {skipped_count} files that don't have matching entries in '{reconstructed_dir}'")
    print(f"These files have matching entries in '{reconstructed_dir}' based on the first 6 characters")

def extract_sequence_from_pdb(pdb_file):
    """
    Extract the amino acid sequence from a PDB file as single-letter amino acid codes.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        str or None: Protein sequence (single-letter amino acids) if found, None otherwise
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB', pdb_file)
        
        # Extract sequence from first chain (matches ca_stripper.py implementation)
        for model in structure:
            for chain in model:
                ppb = PPBuilder()
                sequence = ''
                
                for pp in ppb.build_peptides(chain):
                    sequence += pp.get_sequence()
                
                if sequence:
                    return str(sequence)
        return None
    except Exception as e:
        print(f"Error extracting sequence from {pdb_file}: {e}")
        return None

def check_motifs_in_sequence(sequence, motifs=['DFG', 'APE']):
    """
    Check if all specified motifs are present in a sequence.
    
    Args:
        sequence (str): Protein sequence
        motifs (list): List of motifs to check for (default: ['DFG', 'APE'])
        
    Returns:
        bool: True if all motifs are present, False otherwise
    """
    if sequence is None:
        return False
    
    for motif in motifs:
        if motif not in sequence:
            return False
    return True

def check_motifs_in_pdb(pdb_file, motifs=['DFG', 'APE']):
    """
    Check if a PDB file contains all specified motifs in its sequence.
    Similar to filter_alignments_by_motifs but works directly on PDB files
    without requiring alignment information.
    
    Args:
        pdb_file (str): Path to the PDB file
        motifs (list): List of motifs to check for (default: ['DFG', 'APE'])
        
    Returns:
        bool: True if all motifs are present, False otherwise
    """
    sequence = extract_sequence_from_pdb(pdb_file)
    return check_motifs_in_sequence(sequence, motifs)

def filter_pdbs_by_motifs(pdb_files, motifs=['DFG', 'APE'], verbose=True):
    """
    Filter a list of PDB files to keep only those containing all specified motifs.
    Similar to filter_alignments_by_motifs but works directly on PDB files.
    
    Args:
        pdb_files (list): List of PDB file paths
        motifs (list): List of motifs to check for (default: ['DFG', 'APE'])
        verbose (bool): If True, print progress and statistics
        
    Returns:
        tuple: (valid_pdbs, invalid_pdbs) where valid_pdbs contains files with all motifs
               and invalid_pdbs contains files missing one or more motifs
    """
    valid_pdbs = []
    invalid_pdbs = []
    
    iterator = tqdm(pdb_files, desc="Filtering PDBs by motifs") if verbose else pdb_files
    
    for pdb_file in iterator:
        if check_motifs_in_pdb(pdb_file, motifs):
            valid_pdbs.append(pdb_file)
        else:
            invalid_pdbs.append(pdb_file)
    
    if verbose:
        print(f"\n{len(invalid_pdbs)} / {len(pdb_files)} structures don't have {' and '.join(motifs)} motifs.")
        print(f"Continuing with {len(valid_pdbs)} structures")
    
    return valid_pdbs, invalid_pdbs

def copy_filtered_pdbs(source_dir, target_dir, motifs=['DFG', 'APE']):
    """
    Copy PDB files that contain specified motifs from source to target directory.
    
    Args:
        source_dir (str): Directory containing source PDB files
        target_dir (str): Directory where filtered files will be copied
        motifs (list): List of motifs to check for (default: ['DFG', 'APE'])
        
    Returns:
        tuple: (valid_pdbs, invalid_pdbs) lists of file paths
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    # Get all PDB files in source directory
    pdb_files = find_pdbs(source_dir)
    
    # Filter by motifs
    valid_pdbs, invalid_pdbs = filter_pdbs_by_motifs(pdb_files, motifs, verbose=True)
    
    # Copy valid files
    print(f"Copying {len(valid_pdbs)} files to {target_dir}...")
    for pdb_file in tqdm(valid_pdbs, desc="Copying files"):
        filename = os.path.basename(pdb_file)
        target_path = os.path.join(target_dir, filename)
        shutil.copy(pdb_file, target_path)
    
    print(f"Successfully copied {len(valid_pdbs)} files containing {' and '.join(motifs)} motifs")
    
    return valid_pdbs, invalid_pdbs

def ifnotmake(dir_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path (str): Path to directory
        
    Returns:
        str: Path to directory
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path

def clear_and_make(dir_path):
    """
    Clear directory contents if it exists, then ensure it exists.
    Useful for ensuring clean state before copying files.
    
    Args:
        dir_path (str): Path to directory
        
    Returns:
        str: Path to directory
    """
    if os.path.exists(dir_path):
        # Remove all files in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        # Create directory if it doesn't exist
        os.makedirs(dir_path)
    return dir_path

def get_pdb_files(folder_path, exclude_combined=True, sorted_output=True):
    """
    Get list of PDB files from folder, optionally excluding 'combined' files.
    
    Args:
        folder_path (str): Path to folder
        exclude_combined (bool): Whether to exclude files with 'combined' in name (default: True)
        sorted_output (bool): Whether to sort the output list (default: True)
        
    Returns:
        list: List of PDB file paths
    """
    all_pdbs = find_pdbs(folder_path)
    if exclude_combined:
        all_pdbs = [fp for fp in all_pdbs if "combined" not in os.path.basename(fp)]
    if sorted_output:
        all_pdbs = sorted(all_pdbs)
    return all_pdbs

# def braf_res(folder):
#     """
#     Return a sorted list of unique residue names in all PDB files of a folder,
#     excluding any whose name contains 'combined'.
    
#     Args:
#         folder (str): Path to folder containing PDB files
        
#     Returns:
#         list: Sorted list of unique residue names
        
#     Raises:
#         IOError: If no PDB files found in folder
#     """
#     pdb_files = find_pdbs(folder)
#     # Skip files that contain 'combined' in their basename
#     filtered_pdb_files = [fp for fp in pdb_files if "combined" not in os.path.basename(fp)]
#     if not filtered_pdb_files:
#         raise IOError(f"No .pdb files found in '{folder}' (after excluding 'combined' files).")
    
#     all_residues = set()
#     for fp in filtered_pdb_files:
#         traj = md.load(fp)
#         top = traj.topology
#         for res in top.residues:
#             all_residues.add(res.name)
#     return sorted(list(all_residues))

def braf_res(folder=None):
    """
    Return residue names from a reference PDB file.
    
    Args:
        folder (str, optional): Path to folder containing PDB files. 
                                If provided, uses first non-combined PDB file.
                                If None, uses default reference file.
    
    Returns:
        list: List of formatted residue names (e.g., "ALA-123")
    """
    if folder is not None:
        # Get first PDB file from folder (excluding combined files)
        pdb_files = get_pdb_files(folder, exclude_combined=True)
        if pdb_files:
            fp = pdb_files[0]
        else:
            # Fall back to default if no files found
            fp = "./6UAN_chainD.pdb"
    else:
        fp = "./6UAN_chainD.pdb"
    
    top = md.load(fp).top
    return [res_namer(res) for res in top.residues]

def res_namer(res):
    return f"{res.name}-{res.resSeq}"

def make_seg(a):
    seq = [t for t in a.aligned if t[0] != "-"]
    return seq
    
def save_cluster_labels(cluster_labels, structure_names, output_filename):
    """
    Save cluster labels along with corresponding structure names to a CSV file.
    
    Args:
        cluster_labels (array-like): Array of cluster labels
        structure_names (list): List of structure names
        output_filename (str): Output CSV file path
    """
    # Create DataFrame with structure names and cluster labels
    df = pd.DataFrame({
        'structure': structure_names,
        'cluster': cluster_labels
    })
    
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Saved cluster labels to {output_filename}")


def plot_distance_distributions(feature_matrix, labels, unique_pairs, 
                                feature_indices=None, threshold=None,
                                max_plots=10, figsize=(8, 4),
                                cluster0_color='skyblue', cluster1_color='salmon'):
    """
    Plot histograms comparing distance distributions between two clusters.
    
    Args:
        feature_matrix (np.ndarray): Feature matrix (n_structures x n_features)
        labels (np.ndarray): Cluster labels for each structure
        unique_pairs (list): List of tuples representing residue pairs
        feature_indices (list, optional): Specific feature indices to plot. If None, plots all.
        threshold (float, optional): If provided, adds vertical line at this threshold
        max_plots (int): Maximum number of plots to generate
        figsize (tuple): Figure size for each plot
        cluster0_color (str): Color for cluster 0 histogram
        cluster1_color (str): Color for cluster 1 histogram
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # If no specific features provided, plot all (up to max_plots)
    if feature_indices is None:
        feature_indices = list(range(min(max_plots, len(unique_pairs))))
    else:
        feature_indices = list(feature_indices)[:max_plots]
    
    print(f"Plotting histograms for {len(feature_indices)} features...")
    
    for feature_idx in feature_indices:
        if feature_idx >= len(unique_pairs):
            continue
            
        pair = unique_pairs[feature_idx]
        feature_label = f"{pair[0]}-{pair[1]}"
        values = feature_matrix[:, feature_idx]
        
        # Get values for each cluster
        cluster0_values = values[labels == 0]
        cluster1_values = values[labels == 1]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot histograms
        plt.hist(cluster0_values, bins=30, alpha=0.6, label='Cluster 0 (Inactive)', 
                color=cluster0_color, edgecolor='black')
        plt.hist(cluster1_values, bins=30, alpha=0.6, label='Cluster 1 (Active)', 
                color=cluster1_color, edgecolor='black')
        
        # Add threshold line if provided
        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold: {threshold}Å', linewidth=2)
        
        # Add statistics
        mean0, std0 = np.mean(cluster0_values), np.std(cluster0_values)
        mean1, std1 = np.mean(cluster1_values), np.std(cluster1_values)
        
        stats_text = f"Cluster 0: μ={mean0:.1f}Å, σ={std0:.1f}Å (n={len(cluster0_values)})\n"
        stats_text += f"Cluster 1: μ={mean1:.1f}Å, σ={std1:.1f}Å (n={len(cluster1_values)})"
        
        plt.title(f"Distance Distribution for Residue Pair {feature_label}\n{stats_text}", 
                 fontsize=10)
        plt.xlabel("Distance (Å)", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.legend(loc='best')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print(f"Displayed {len(feature_indices)} histogram(s)")


class PDBDownloader:
    """
    A class to download PDB files from the RCSB database using multi-threading.
    
    This class implements a robust download system with retry mechanisms, 
    empty file detection, and parallel processing capabilities.
    """
    
    def __init__(self, base_url="https://files.rcsb.org/download", default_dir="Results/InterProPDBs"):
        """
        Initialize the PDBDownloader.
        
        Args:
            base_url (str): Base URL for PDB downloads
            default_dir (str): Default directory for storing downloaded files
        """
        self.base_url = base_url
        self.default_dir = default_dir
    
    def download_single(self, code, pdir=None, max_retries=3):
        """
        Download a single PDB file with retry mechanism and failure handling.
        
        Args:
            code (str): PDB code to download
            pdir (str): Directory to save the file
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str or None: Path to downloaded file if successful, None if failed
        """
        pdb_url = f"{self.base_url}/{code}.pdb"
        directory = pdir if pdir else self.default_dir
        f_p = os.path.join(directory, f"{code}.pdb")

        for attempt in range(max_retries):
            try:
                response = requests.get(pdb_url, stream=True, timeout=10)
                if response.status_code == 404:
                    print(f"{code}.pdb file does not exist (404 Not Found)")
                    return None
                response.raise_for_status()
                
                with open(f_p, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Check file size to prevent empty files
                if os.path.getsize(f_p) == 0:
                    print(f"{code}.pdb download failed (empty file), retrying {attempt+1}/{max_retries}...")
                    os.remove(f_p)
                    continue  # Retry

                print(f"{code}.pdb downloaded successfully")
                return f_p
            except requests.exceptions.RequestException as e:
                print(f"{code}.pdb download failed, retrying {attempt+1}/{max_retries}... Error: {e}")
                time.sleep(2)

        print(f"{code}.pdb download ultimately failed")
        return None

    def download_multiple(self, pdb_list, pdir=None):
        """
        Download multiple PDB files sequentially.
        
        Args:
            pdb_list (list): List of PDB codes to download
            pdir (str): Directory to save the files
        """
        directory = os.path.abspath(pdir if pdir else self.default_dir)
        os.makedirs(directory, exist_ok=True)

        # Get already downloaded PDB files to avoid duplicate downloads
        existing_files = {os.path.splitext(f)[0] for f in os.listdir(directory)}

        for code in pdb_list:
            if code not in existing_files:
                file_path = self.download_single(code, pdir=directory)
                if file_path:
                    print(f"{code}.pdb downloaded successfully")
                else:
                    print(f"{code}.pdb download failed")

    def parallel_download(self, pdb_list, pdir=None):
        """
        Download PDB files in parallel using multiple threads.
        
        Args:
            pdb_list (list): List of PDB codes to download
            pdir (str): Directory to save the files
        """
        max_workers = min(20, multiprocessing.cpu_count() * 2)
        chunk_size = max(10, len(pdb_list) // max_workers)
        splited_pdb_lists = [pdb_list[i:i+chunk_size] for i in range(0, len(pdb_list), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_multiple, chunk, pdir) for chunk in splited_pdb_lists]
            # Wait for all downloads to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()


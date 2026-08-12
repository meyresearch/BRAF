import os
import glob
import shutil
import pickle as p
import time
import threading
import requests
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder
import pandas as pd
import mdtraj as md

from workflow.chain_basenames import (
    load_excluded_pseudokinase_basenames,
    normalize_chain_basename,
)

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

def _load_excluded_pseudokinase_basenames(excluded_basenames_file: str | None) -> set[str]:
    """Backward-compatible wrapper; missing file → empty set."""
    if not excluded_basenames_file:
        return set()
    return load_excluded_pseudokinase_basenames(excluded_basenames_file, required=False)


def copy_filtered_pdbs(
    source_dir,
    target_dir,
    motifs=['DFG', 'APE'],
    *,
    excluded_basenames_file: str | None = None,
):
    """
    Copy PDB files that contain specified motifs from source to target directory.
    
    Args:
        source_dir (str): Directory containing source PDB files
        target_dir (str): Directory where filtered files will be copied
        motifs (list): List of motifs to check for (default: ['DFG', 'APE'])
        excluded_basenames_file: If set and the file exists, skip copying PDBs whose basename
            appears in this list (e.g. pseudokinase basenames written by
            ``annotate_dataset_chains_with_kinome`` for motif-filtered directories only).
            When ``None``, uses ``<dirname(target_dir)>/excluded_pseudokinase_basenames.txt``
            if that path exists.
        
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

    if excluded_basenames_file is None:
        candidate = os.path.join(
            os.path.dirname(os.path.normpath(target_dir)) or ".",
            "excluded_pseudokinase_basenames.txt",
        )
        excluded_basenames_file = candidate if os.path.isfile(candidate) else None

    skip = _load_excluded_pseudokinase_basenames(excluded_basenames_file)
    if skip:
        before = len(valid_pdbs)
        valid_pdbs = [
            p for p in valid_pdbs
            if normalize_chain_basename(os.path.basename(p)) not in skip
        ]
        n_skipped = before - len(valid_pdbs)
        if n_skipped:
            print(
                f"Skipped {n_skipped} pseudokinase chain PDB(s) (listed in {excluded_basenames_file!r})."
            )
    
    # Copy valid files
    print(f"Copying {len(valid_pdbs)} files to {target_dir}...")
    for pdb_file in tqdm(valid_pdbs, desc="Copying files"):
        filename = os.path.basename(pdb_file)
        target_path = os.path.join(target_dir, filename)
        shutil.copy(pdb_file, target_path)
    
    print(f"Successfully copied {len(valid_pdbs)} files containing {' and '.join(motifs)} motifs")
    
    return valid_pdbs, invalid_pdbs


def copy_motif_filtered_datasets(
    source_dir_protein: str = "Results/activation_segments/unaligned/",
    target_dir_protein: str = "Results/activation_segments/motif_filtered/",
    source_dir_ligands: str = "Results/activation_segments/unaligned+ligands/",
    target_dir_ligands: str = "Results/activation_segments/motif_filtered+ligands/",
    motifs: list = None,
    *,
    excluded_basenames_file: str | None = None,
    **kwargs,
):
    """
    Run motif filtering on protein-only chains, then mirror passing filenames
    into the paired ligand-chain directory (KinCore / ligand-screening input).

    The ligand-side source directory is often populated from a first-pass
    small-molecule HETATM extraction (e.g. ``.../InterPro_protein_small_molecules/``);
    pass that path as ``source_dir_ligands`` when it differs from the default layout.

    When ``excluded_basenames_file`` is omitted, ``copy_filtered_pdbs`` looks for
    ``<parent of target_dir_protein>/excluded_pseudokinase_basenames.txt`` (e.g. after
    ``annotate_dataset_chains_with_kinome(..., exclude_pseudokinase=True)``) and skips
    those chain PDB basenames after motif filtering. The main annotation CSV is unchanged
    and still lists pseudokinase rows; only ``motif_filtered_*`` copies omit them.

    Returns:
        tuple: (valid_pdbs, invalid_pdbs)

    Deprecated keyword arguments (via ``kwargs`` for compatibility):
        ``source_dir_small_molecules`` → ``source_dir_ligands``,
        ``target_dir_small_molecules`` → ``target_dir_ligands``.
    """
    if "source_dir_small_molecules" in kwargs:
        source_dir_ligands = kwargs.pop("source_dir_small_molecules")
    if "target_dir_small_molecules" in kwargs:
        target_dir_ligands = kwargs.pop("target_dir_small_molecules")
    if kwargs:
        raise TypeError(
            "copy_motif_filtered_datasets() got unexpected keyword arguments: "
            f"{sorted(kwargs.keys())}"
        )
    if motifs is None:
        motifs = ["DFG", "APE"]

    valid_pdbs, invalid_pdbs = copy_filtered_pdbs(
        source_dir=source_dir_protein,
        target_dir=target_dir_protein,
        motifs=motifs,
        excluded_basenames_file=excluded_basenames_file,
    )

    os.makedirs(target_dir_ligands, exist_ok=True)
    copied_ligand_files = 0
    missing_ligand_files = 0

    for p in valid_pdbs:
        fn = os.path.basename(p)
        src = os.path.join(source_dir_ligands, fn)
        dst = os.path.join(target_dir_ligands, fn)
        if not os.path.exists(src):
            missing_ligand_files += 1
            continue
        shutil.copy2(src, dst)
        copied_ligand_files += 1

    print(
        f"Copied {copied_ligand_files} ligand-chain files to {target_dir_ligands} "
        f"(missing: {missing_ligand_files})"
    )
    return valid_pdbs, invalid_pdbs


def chain_small_molecule_basename_id(filename: str) -> str:
    """Return ``PDBID_CHAIN`` shared by e.g. ``6G9D_A.pdb`` and ``6G9D_A_aligned.pdb``."""
    base, _ = os.path.splitext(os.path.basename(filename))
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base


def copy_cg_chain_small_molecules(
    small_molecules_src: str = "Results/motif_filtered_small_molecules/",
    small_molecules_dst: str = "Results/CG_chain_small_molecules/",
    cg_dir: str = "Results/activation_segments/fitted",
) -> dict:
    """
    Copy protein–small-molecule complexes whose ``PDBID_CHAIN`` is present in the CG/fitted set.

    Both fitted names (e.g. ``1A9U_A_aligned.pdb``) and small-molecule complex names
    (e.g. ``1A9U_A.pdb``) are normalized with :func:`chain_small_molecule_basename_id`
    so the ``_aligned`` suffix does not prevent matching.
    """
    clear_and_make(small_molecules_dst)

    if not os.path.isdir(cg_dir):
        raise FileNotFoundError(f"CG/fitted directory not found: {cg_dir}")
    if not os.path.isdir(small_molecules_src):
        raise FileNotFoundError(
            f"Small-molecule source directory not found: {small_molecules_src}"
        )

    cg_ids = {
        chain_small_molecule_basename_id(name)
        for name in os.listdir(cg_dir)
        if name.lower().endswith(".pdb")
    }

    n_copy = 0
    for name in os.listdir(small_molecules_src):
        if chain_small_molecule_basename_id(name) in cg_ids:
            shutil.copy2(
                os.path.join(small_molecules_src, name),
                os.path.join(small_molecules_dst, name),
            )
            n_copy += 1

    print(
        f"Copied {n_copy} files to {small_molecules_dst} "
        f"({len(cg_ids)} basenames in {cg_dir})."
    )
    return {
        "n_copied": n_copy,
        "n_cg_ids": len(cg_ids),
        "small_molecules_dst": small_molecules_dst,
        "cg_dir": cg_dir,
    }


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

def _default_reference_pdb():
    """Return path to the BRAF reference PDB bundled with the workflow package."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "6UAN_chainD.pdb")


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
            fp = _default_reference_pdb()
    else:
        fp = _default_reference_pdb()
    
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
    A class to download structure files from the RCSB file server using multi-threading.

    For each entry code, legacy ``.pdb`` is requested first; many newer depositions are
    mmCIF-only, so on 404 the downloader falls back to ``.cif`` (same ``files.rcsb.org`` tree).
    """

    def __init__(self, base_url="https://files.rcsb.org/download", default_dir="Results/InterPro_PDBs"):
        """
        Initialize the PDBDownloader.

        Args:
            base_url (str): Base URL for RCSB ``/download/{id}.pdb`` and ``/download/{id}.cif``
            default_dir (str): Default directory for storing downloaded files
        """
        self.base_url = base_url
        self.default_dir = default_dir

    def download_single(self, code, pdir=None, max_retries=3, quiet=False):
        """
        Download coordinates for one PDB ID: try ``{code}.pdb``, then ``{code}.cif``.

        Args:
            code (str): PDB identifier (4 characters; normalized to uppercase)
            pdir (str): Directory to save the file
            max_retries (int): Maximum retry attempts per format
            quiet (bool): If True, suppress progress and retry messages (caller may log failures)

        Returns:
            str or None: Path to the saved ``.pdb`` or ``.cif`` file if successful, else None
        """
        directory = pdir if pdir else self.default_dir
        os.makedirs(directory, exist_ok=True)
        code = str(code).strip().upper()
        f_pdb = os.path.join(directory, f"{code}.pdb")
        f_cif = os.path.join(directory, f"{code}.cif")

        if os.path.isfile(f_pdb) and os.path.getsize(f_pdb) > 0:
            return f_pdb
        if os.path.isfile(f_cif) and os.path.getsize(f_cif) > 0:
            return f_cif

        formats = (
            ("pdb", f"{code}.pdb", f_pdb, 30),
            ("cif", f"{code}.cif", f_cif, 120),
        )

        for fmt, fname, dest_path, timeout_s in formats:
            url = f"{self.base_url}/{fname}"
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, stream=True, timeout=timeout_s)
                    if response.status_code == 404:
                        if not quiet:
                            if fmt == "pdb":
                                print(f"{code}: no PDB-format file from RCSB (404); trying mmCIF...")
                            else:
                                print(f"{code}.cif file does not exist (404 Not Found)")
                        break
                    response.raise_for_status()

                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if os.path.getsize(dest_path) == 0:
                        if not quiet:
                            print(
                                f"{fname} download produced empty file, retrying {attempt + 1}/{max_retries}..."
                            )
                        os.remove(dest_path)
                        continue

                    if not quiet:
                        print(f"{fname} downloaded successfully")
                    return dest_path
                except requests.exceptions.RequestException as e:
                    if not quiet:
                        print(
                            f"{fname} download failed, retrying {attempt + 1}/{max_retries}... Error: {e}"
                        )
                    if os.path.isfile(dest_path):
                        try:
                            os.remove(dest_path)
                        except OSError:
                            pass
                    time.sleep(2)

        if not quiet:
            print(f"{code}: coordinate download ultimately failed (no usable .pdb or .cif)")
        return None

    def download_multiple(self, pdb_list, pdir=None, pbar=None, lock=None, quiet=False):
        """
        Download multiple structures sequentially (each ID: .pdb then .cif fallback).

        Args:
            pdb_list (list): List of PDB codes to download
            pdir (str): Directory to save the files
            pbar (tqdm.tqdm | None): Optional progress bar; one ``update`` per code in ``pdb_list``
            lock (threading.Lock | None): Lock for ``pbar.update`` when using threads
            quiet (bool): If True, suppress per-file success noise; print only failures when True
        """
        directory = os.path.abspath(pdir if pdir else self.default_dir)
        os.makedirs(directory, exist_ok=True)

        existing_files = set()
        for f in os.listdir(directory):
            if not os.path.isfile(os.path.join(directory, f)):
                continue
            if f.lower().endswith((".pdb", ".cif")):
                existing_files.add(os.path.splitext(f)[0].upper())

        def _tick():
            if pbar is not None:
                if lock is not None:
                    with lock:
                        pbar.update(1)
                else:
                    pbar.update(1)

        for code in pdb_list:
            cu = str(code).strip().upper()
            if cu not in existing_files:
                file_path = self.download_single(cu, pdir=directory, quiet=quiet)
                if file_path:
                    existing_files.add(cu)
                elif quiet:
                    print(f"Download failed: {cu}")
                # not quiet: download_single already prints failure details
            _tick()

    def fetch_interpro_structures(
        self,
        entry_id="IPR011009",
        page_size=200,
        show_progress=True,
        max_retries=3,
        cache_path=None,
    ):
        """
        Query the InterPro REST API for all PDB structures linked to an InterPro entry.

        Results can be cached to a TSV file so that subsequent calls load from disk
        instead of hitting the API.  Delete the cache file to force a fresh fetch.

        Args:
            entry_id (str): InterPro entry accession (e.g. 'IPR011009')
            page_size (int): Results per API page (max 200)
            show_progress (bool): Show a tqdm progress bar while paging
            max_retries (int): Retry attempts on transient HTTP errors
            cache_path (str | None): Optional path to a TSV cache file.
                If the file exists it is returned immediately; otherwise the
                API is queried and the result is written to this path.

        Returns:
            pd.DataFrame: One row per PDB accession, columns matching the
                InterPro structure-matching TSV export:
                Accession, Source Database, Name, Experiment Type,
                Resolution, Chains, Proteins, Protein Length,
                Structure Location, matches
        """
        COLUMNS = [
            "Accession", "Source Database", "Name", "Experiment Type",
            "Resolution", "Chains", "Proteins", "Protein Length",
            "Structure Location", "matches",
        ]

        if cache_path and os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
            print(f"Loading InterPro structures from cache: {cache_path}")
            df = pd.read_csv(cache_path, sep="\t", header=0, engine="python")
            df["Accession"] = df["Accession"].astype(str).str.upper()
            return df

        api_url = (
            f"https://www.ebi.ac.uk/interpro/api/structure/pdb"
            f"/entry/interpro/{entry_id}/?page_size={page_size}"
        )

        rows = []
        pbar = None
        url = api_url

        try:
            while url:
                for attempt in range(max_retries):
                    try:
                        resp = requests.get(url, timeout=60)
                        resp.raise_for_status()
                        break
                    except requests.exceptions.RequestException as exc:
                        if attempt == max_retries - 1:
                            raise
                        print(f"InterPro API error ({exc}); retrying {attempt + 1}/{max_retries}...")
                        time.sleep(2 ** attempt)

                data = resp.json()

                if pbar is None and show_progress:
                    total = data.get("count", 0)
                    pbar = tqdm(total=total, desc=f"InterPro {entry_id}", unit="struct")

                for result in data.get("results", []):
                    meta = result.get("metadata", {})
                    entries = result.get("entries", [])

                    accession = str(meta.get("accession", "")).upper()
                    name = meta.get("name", "")
                    source_db = meta.get("source_database", "pdb")
                    exp_type = meta.get("experiment_type", "")
                    resolution = meta.get("resolution", "")

                    chains, proteins, prot_lengths, struct_locs, match_locs = (
                        [], [], [], [], []
                    )

                    for entry in entries:
                        chains.append(str(entry.get("chain", "")))
                        proteins.append(str(entry.get("protein", "") or ""))
                        prot_lengths.append(str(entry.get("protein_length", "") or ""))

                        s_frags = [
                            f"{f.get('auth_start', f['start'])}..{f.get('auth_end', f['end'])}"
                            for loc in (entry.get("entry_structure_locations") or [])
                            for f in (loc.get("fragments") or [])
                        ]
                        struct_locs.append(";".join(s_frags))

                        p_frags = [
                            f"{f['start']}..{f['end']}"
                            for loc in (entry.get("entry_protein_locations") or [])
                            for f in (loc.get("fragments") or [])
                        ]
                        match_locs.append(";".join(p_frags))

                    rows.append({
                        "Accession": accession,
                        "Source Database": source_db,
                        "Name": name,
                        "Experiment Type": exp_type,
                        "Resolution": resolution,
                        "Chains": ";".join(chains),
                        "Proteins": ";".join(proteins),
                        "Protein Length": ";".join(prot_lengths),
                        "Structure Location": ";".join(struct_locs),
                        "matches": ";".join(match_locs),
                    })

                    if pbar is not None:
                        pbar.update(1)

                url = data.get("next")
        finally:
            if pbar is not None:
                pbar.close()

        df = pd.DataFrame(rows, columns=COLUMNS)

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            df.to_csv(cache_path, sep="\t", index=False)
            print(f"InterPro results cached to: {cache_path}")

        return df

    def parallel_download(self, pdb_list, pdir=None, show_progress=True):
        """
        Download structure files in parallel using multiple threads.

        Args:
            pdb_list (list): List of PDB codes to download
            pdir (str): Directory to save the files
            show_progress (bool): If True, show a tqdm bar and only print failed IDs
        """
        pdb_list = list(pdb_list)
        n = len(pdb_list)
        max_workers = min(20, multiprocessing.cpu_count() * 2)
        chunk_size = max(10, n // max_workers) if n else 1
        splited_pdb_lists = [pdb_list[i : i + chunk_size] for i in range(0, n, chunk_size)]

        lock = threading.Lock()
        pbar = (
            tqdm(total=n, desc="RCSB download", unit="id", smoothing=0.05)
            if show_progress and n
            else None
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.download_multiple, chunk, pdir, pbar, lock, show_progress
                    )
                    for chunk in splited_pdb_lists
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            if pbar is not None:
                pbar.close()


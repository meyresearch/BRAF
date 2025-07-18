import os
import glob
import shutil
import pickle as p
from tqdm import tqdm

'''
This file contains utility functions for the pipeline.
IT WOULD BE VERY GOOD TO HAVE A GUI UTILITY TO VISUALISE MANIPULATIONS OF PROTEINS
'''

def find_pdbs(directory):
    """Find all PDB files in the given directory."""
    return glob.glob(os.path.join(directory, "*.pdb"))

def fname(path):
    """Extract filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]

def count_pdb_files(directory):
    """
    Count PDB files in a directory.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        int: Number of PDB files
    """
    # Use the find_pdbs function
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


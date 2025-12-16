import os
import glob
import pickle as p
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import html

class analyse_alignment:
    """
    Comprehensive class for handling MUSTANG protein alignments.
    Currently only supports braf_monomers!
    """
    def __init__(self, name=None, seq1=None, seq2=None, alignment_dir="Results/activation_segments/nonReconstructed_mustang"):
        # Handle case where only alignment_dir is passed as first argument
        if name is not None and seq1 is None and seq2 is None and os.path.isdir(name):
            self.alignment_dir = name
        else:
            self.alignment_dir = alignment_dir
            
            # If name and sequences are provided, create individual alignment
            if name and seq1 and seq2:
                self.name = name
                self.seq1 = seq1
                self.seq2 = seq2
                self.aligned = self.find_aligned()
    
    def afasta_parse(self, file):
        """
        Parse mustang afasta format output file.
        Returns two sequences of equal length (for backward compatibility with pairwise alignments).
        """
        try:
            with open(file, "r") as f:
                lines = f.readlines()
            
            # Find sequence headers and their positions
            header_indices = []
            for i, line in enumerate(lines):
                if line.strip().startswith(">"):
                    header_indices.append(i)
            
            if len(header_indices) < 2:
                print(f"Warning: {file} does not contain exactly 2 sequences (found {len(header_indices)})")
                return None, None
            
            # Extract sequences
            sequences = []
            for i in range(len(header_indices)):
                start_idx = header_indices[i] + 1  # Start after header
                end_idx = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
                
                # Collect sequence lines (skip empty lines)
                seq_lines = []
                for j in range(start_idx, end_idx):
                    line = lines[j].strip()
                    if line and not line.startswith(">"):
                        seq_lines.append(line)
                
                # Join sequence lines
                sequence = "".join(seq_lines)
                sequences.append(sequence)
            
            # Return first two sequences
            if len(sequences) >= 2:
                return sequences[0], sequences[1]
            else:
                print(f"Warning: {file} only contains {len(sequences)} sequences")
                return None, None
                
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            return None, None
    
    def afasta_parse_multi(self, file):
        """
        Parse mustang afasta format output file for multi-structure alignment.
        Returns a list of (header, sequence) tuples.
        
        Returns:
            list: List of tuples [(header1, seq1), (header2, seq2), ...]
        """
        try:
            with open(file, "r") as f:
                lines = f.readlines()
            
            # Find sequence headers and their positions
            header_indices = []
            headers = []
            for i, line in enumerate(lines):
                if line.strip().startswith(">"):
                    header_indices.append(i)
                    headers.append(line.strip()[1:])  # Remove '>' prefix
            
            if len(header_indices) < 2:
                print(f"Warning: {file} does not contain at least 2 sequences (found {len(header_indices)})")
                return []
            
            # Extract sequences
            sequences = []
            for i in range(len(header_indices)):
                start_idx = header_indices[i] + 1  # Start after header
                end_idx = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
                
                # Collect sequence lines (skip empty lines)
                seq_lines = []
                for j in range(start_idx, end_idx):
                    line = lines[j].strip()
                    if line and not line.startswith(">"):
                        seq_lines.append(line)
                
                # Join sequence lines
                sequence = "".join(seq_lines)
                sequences.append((headers[i], sequence))
            
            return sequences
                
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            return []
    
    def make_align_pickle(self):
        """
        Create pickle file containing alignment objects from MUSTANG afasta files.
        """
        alignments = []
        failed_files = []

        # Iterate over directories in the alignment directory
        for directory_name in os.listdir(self.alignment_dir):
            directory_path = os.path.join(self.alignment_dir, directory_name)

            # Ensure we are working with directories
            if os.path.isdir(directory_path):
                fasta_files = tqdm(glob.glob(os.path.join(directory_path, "*.afasta")), desc=f"Processing {directory_name} .afasta files")
                
                for fasta_file in fasta_files:
                    name = os.path.splitext(os.path.basename(fasta_file))[0]
                    fasta_files.set_description(f"Working on {name}")

                    # Create alignment object
                    seq1, seq2 = self.afasta_parse(fasta_file)
                    
                    # Check if parsing was successful
                    if seq1 is not None and seq2 is not None:
                        aligned = analyse_alignment(name, seq1, seq2, self.alignment_dir)
                        alignments.append(aligned)
                    else:
                        failed_files.append(fasta_file)
                        print(f"Skipping {fasta_file} due to parsing error")

        # Report results
        print(f"\nSuccessfully processed {len(alignments)} files")
        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for failed_file in failed_files:
                print(f"  - {failed_file}")

        # Define a path for the output pickle file
        ppath = os.path.join(self.alignment_dir, "mustang_alignments.fasta")
        with open(ppath, "wb") as pickled:
            p.dump(alignments, pickled)
    
    def load_alignments(self, force_reload=False):
        """
        Load alignments from pickle file or create new ones if not found.
        
        Args:
            force_reload (bool): If True, regenerate the pickle even if it exists.
                                If False (default), check if the pickle is out of date
                                by comparing file counts.
        
        Returns:
            list: List of analyse_alignment objects
        """
        ppath = os.path.join(self.alignment_dir, "mustang_alignments.fasta")
        
        if os.path.isfile(ppath) and not force_reload:
            # Check if pickle needs regeneration by comparing file counts
            afasta_files = glob.glob(os.path.join(self.alignment_dir, "*", "*.afasta"))
            with open(ppath,"rb") as pickled:
                pickled_alignments = p.load(pickled)
            
            # If pickle has fewer alignments than actual files, regenerate
            if len(pickled_alignments) != len(afasta_files):
                print(f"Pickle file has {len(pickled_alignments)} entries but {len(afasta_files)} afasta files found.")
                print("Regenerating pickle...")
                self.make_align_pickle()
                return self.load_alignments(force_reload=False)
            else:
                return pickled_alignments
        else:
            # Pickle doesn't exist or force_reload is True
            self.make_align_pickle()
            return self.load_alignments(force_reload=False)
    
    def load_multi_alignment(self, afasta_file, reference_name=None):
        """
        Load a multi-structure alignment from a single afasta file.
        
        Args:
            afasta_file (str): Path to the multi-structure alignment afasta file
            reference_name (str): Optional name/substring to identify reference structure.
                                 If None, uses first sequence as reference.
        
        Returns:
            dict: Dictionary containing:
                - 'reference': Tuple of (header, sequence) for reference structure
                - 'structures': List of analyse_alignment objects, one for each structure vs reference
                - 'all_sequences': List of all (header, sequence) tuples
        """
        sequences = self.afasta_parse_multi(afasta_file)
        
        if not sequences or len(sequences) < 2:
            print(f"Error: Could not load multi-structure alignment from {afasta_file}")
            return None
        
        # Find reference sequence
        ref_header = None
        ref_seq = None
        ref_index = 0
        
        if reference_name:
            # Search for reference by name
            for i, (header, seq) in enumerate(sequences):
                if reference_name in header:
                    ref_header, ref_seq = header, seq
                    ref_index = i
                    print(f"Found reference structure: {header}")
                    break
            
            if ref_header is None:
                print(f"Warning: Reference '{reference_name}' not found. Using first sequence as reference.")
                ref_header, ref_seq = sequences[0]
        else:
            # Use first sequence as reference
            ref_header, ref_seq = sequences[0]
        
        # Create pairwise alignment objects for each structure vs reference
        alignments = []
        for i, (struct_header, struct_seq) in enumerate(sequences):
            # Skip the reference itself
            if i == ref_index:
                continue
                
            # Create a name from the structure header
            name = struct_header.split()[0] if ' ' in struct_header else struct_header
            
            # Create alignment object
            alignment = analyse_alignment(name, ref_seq, struct_seq, self.alignment_dir)
            alignments.append(alignment)
        
        print(f"Loaded multi-structure alignment: {len(alignments)} structures aligned to reference")
        
        return {
            'reference': (ref_header, ref_seq),
            'structures': alignments,
            'all_sequences': sequences,
            'n_structures': len(alignments)
        }
    
    def find_aligned(self):
        """
        Find aligned residue pairs (removes gaps from seq1).
        
        Returns:
            list: List of tuples containing aligned character pairs
        """
        aligned = []
        for char1, char2 in zip(self.seq1, self.seq2):
            if char1 != "-":
                aligned.append((char1, char2))
        return aligned

    def make_seg(self):
        """
        Create alignment segment (equivalent to make_seg function from notebook).
        
        Returns:
            list: List of tuples containing aligned character pairs without gaps in seq1
        """
        return self.find_aligned()

    def visualize_residue_conservation(self, filtered_alignments, reference_residues=None, 
                                     output_file="conservation_plot.png", 
                                     figsize=(15, 6), show_plot=True, tick_interval=10):
        """
        Visualize residue conservation across filtered alignments.
        
        Args:
            filtered_alignments (list): List of analyse_alignment objects
            reference_residues (list): List of residue names in "ALA-123" format (from braf_res()).
                                      If provided, x-axis shows "ALA123 0" format labels.
            output_file (str): Output filename for the plot
            figsize (tuple): Figure size
            show_plot (bool): Whether to display the plot
            tick_interval (int): Show x-axis label every N residues (default: 10)
            
        Returns:
            tuple: (conservation_array, highly_conserved_positions)
        """
        if not filtered_alignments:
            print("No alignments provided for conservation analysis.")
            return None, None
        
        # Calculate conservation
        seq1mag = len(filtered_alignments[0].seq1.replace("-", ""))
        counts = np.zeros(seq1mag)
        total_alignments = len(filtered_alignments)
        
        for alignment in filtered_alignments:
            segment = alignment.make_seg()
            for i, (b, c) in enumerate(segment):
                if c != "-":
                    counts[i] += 1
        
        # Convert to conservation percentage
        conservation = counts / total_alignments
        
        # Find highly conserved residues (100% conservation)
        highly_conserved = []
        for i, cons_value in enumerate(conservation):
            if cons_value == 1.0:
                highly_conserved.append((i, cons_value))
        
        # Helper function to format residue labels
        def format_residue_label(idx):
            """Format residue label as 'ALA123 (idx)' from 'ALA-123' format."""
            if reference_residues and idx < len(reference_residues):
                res = reference_residues[idx]  # e.g., "ALA-123"
                parts = res.split('-')
                if len(parts) == 2:
                    return f"{parts[0]}{parts[1]} ({idx})"
                return f"{res} ({idx})"
            return f"({idx})"
        
        # Create visualization
        fig, ax = plt.subplots(1, figsize=figsize)
        x = list(range(len(conservation)))
        
        # Create bars
        bars = ax.bar(x, conservation, linewidth=0.05, width=1, color='royalblue', 
                     alpha=0.5, edgecolor='steelblue', zorder=10)
        
        # Define important structural regions with colors
        color_regions = [
            (14, 21, 'blue', 'pLoop'),      # pLoop region
            (43, 58, 'green', 'alphaC'),    # alphaC region  
            (145, 168, 'hotpink', 'DFG-APE') # DFG-APE region
        ]
        
        # Add colored regions above bars
        for start, end, color, label in color_regions:
            for i in range(start, end+1):
                if i < len(conservation):
                    bar_height = conservation[i]
                    rect_height = 1.0 - bar_height
                    if rect_height > 0:
                        rect = patches.Rectangle((i-0.5, bar_height), 1, rect_height, 
                                              facecolor=color, alpha=0.7, zorder=5)
                        ax.add_patch(rect)
        
        # Set up x-axis labels with residue names and indices
        tick_positions = x[::tick_interval]
        tick_labels = [format_residue_label(i) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, fontsize=8, ha='right')
        
        # Formatting
        ax.set_xlim(-0.5, len(conservation)-0.5)
        ax.set_ylim(0, 1.0001)
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0, labelsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.xlabel("ResiduePDBindex (0-based index)", fontsize=12)
        plt.ylabel("Conservation Percentage", fontsize=12)
        plt.title(f"Residue Conservation across {total_alignments} Structures", fontsize=14)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Conservation plot saved to {output_file}")
        
        if show_plot:
            plt.show()
        
        # Print conservation statistics
        print(f"\nConservation Analysis Results:")
        print(f"Total structures analyzed: {total_alignments}")
        print(f"Highly conserved positions (100%): {len(highly_conserved)}")
        
        return conservation, highly_conserved

    def run_multi_alignment_conservation_analysis(
        self,
        *,
        alignment_file: str,
        reference_name: str = "6UAN_chainD",
        reference_residues: list = None,
        conservation_threshold: float = 0.70,
        output_plot: str = "Results/multi_alignment_foldMason_conservation.png",
        output_csv: str = "Results/conserved_residues_70percent.csv",
        show_plot: bool = True,
        tick_interval: int = 10,
        verbose: bool = True,
    ):
        """
        Notebook convenience wrapper:
        - loads a multi-structure alignment (e.g., FoldMason 3Di msa_3di.fa)
        - plots residue conservation with residue-labeled x-axis
        - prints conserved residues above a threshold
        - saves conserved residues table to CSV

        Returns:
            dict with keys:
              - 'multi_data', 'conservation', 'highly_conserved'
              - 'conserved_indices', 'conserved_df'
        """
        import pandas as pd

        multi_data = self.load_multi_alignment(alignment_file, reference_name=reference_name)
        if multi_data is None:
            raise ValueError(f"Could not load multi alignment from: {alignment_file}")

        conservation, highly_conserved = self.visualize_residue_conservation(
            filtered_alignments=multi_data["structures"],
            reference_residues=reference_residues,
            output_file=output_plot,
            show_plot=show_plot,
            tick_interval=tick_interval,
        )

        if verbose:
            print(f"Analyzed {len(multi_data['structures'])} structures (total: {multi_data['n_structures']})")

        conserved_indices = np.where(conservation >= conservation_threshold)[0]

        if verbose:
            print(f"\n=== Residues with ≥{conservation_threshold*100:.0f}% Conservation ===")
            print(f"Total conserved residues: {len(conserved_indices)}")
            print("\nPositions and residues:")
            if reference_residues is not None:
                for idx in conserved_indices:
                    if idx < len(reference_residues):
                        print(f"  Position {idx:3d}: {reference_residues[idx]:>10s} - {conservation[idx]*100:.1f}% conserved")
                    else:
                        print(f"  Position {idx:3d}: (no reference) - {conservation[idx]*100:.1f}% conserved")
            else:
                for idx in conserved_indices:
                    print(f"  Position {idx:3d}: {conservation[idx]*100:.1f}% conserved")

        # Save to CSV
        conserved_df = pd.DataFrame(
            {
                "position": conserved_indices,
                "residue": [
                    reference_residues[i] if (reference_residues is not None and i < len(reference_residues)) else "N/A"
                    for i in conserved_indices
                ],
                "conservation": [conservation[i] for i in conserved_indices],
            }
        )
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        conserved_df.to_csv(output_csv, index=False)

        if verbose:
            print(f"\nSaved conserved residues to: {output_csv}")

        return {
            "multi_data": multi_data,
            "conservation": conservation,
            "highly_conserved": highly_conserved,
            "conserved_indices": conserved_indices,
            "conserved_df": conserved_df,
        }

    def validate_dfg_ape_alignment(self):
        """
        Validate that DFG and APE motifs are not aligned to gaps.
        
        Returns:
            dict: Dictionary containing validation results with keys:
                - 'valid': Boolean indicating if both motifs are properly aligned
                - 'dfg_found': Boolean indicating if DFG motif was found in seq1
                - 'ape_found': Boolean indicating if APE motif was found in seq1
                - 'dfg_aligned': Boolean indicating if DFG is not aligned to gap
                - 'ape_aligned': Boolean indicating if APE is not aligned to gap
                - 'dfg_index': Index of DFG motif in seq1 (-1 if not found)
                - 'ape_index': Index of APE motif in seq1 (-1 if not found)
                - 'dfg_seq2': Corresponding sequence in seq2 for DFG motif
                - 'ape_seq2': Corresponding sequence in seq2 for APE motif
        """
        # Check if sequences are initialized
        if not hasattr(self, 'seq1') or not hasattr(self, 'seq2'):
            return {
                'valid': False,
                'dfg_found': False,
                'ape_found': False,
                'dfg_aligned': False,
                'ape_aligned': False,
                'dfg_index': -1,
                'ape_index': -1,
                'dfg_seq2': '',
                'ape_seq2': ''
            }
        
        # Find DFG and APE motifs in seq1
        dfg_index = self.seq1.find("DFG")
        ape_index = self.seq1.find("APE")
        
        dfg_found = dfg_index != -1
        ape_found = ape_index != -1
        
        dfg_aligned = False
        ape_aligned = False
        dfg_seq2 = ''
        ape_seq2 = ''
        
        # Check DFG alignment
        if dfg_found:
            if dfg_index + 2 < len(self.seq2):
                dfg_seq2 = self.seq2[dfg_index:dfg_index+3]
                # Check if any of the three positions (D, F, G) are aligned to gaps
                dfg_aligned = '-' not in dfg_seq2
        
        # Check APE alignment
        if ape_found:
            if ape_index + 2 < len(self.seq2):
                ape_seq2 = self.seq2[ape_index:ape_index+3]
                # Check if any of the three positions (A, P, E) are aligned to gaps
                ape_aligned = '-' not in ape_seq2
        
        valid = dfg_found and ape_found and dfg_aligned and ape_aligned
        
        return {
            'valid': valid,
            'dfg_found': dfg_found,
            'ape_found': ape_found,
            'dfg_aligned': dfg_aligned,
            'ape_aligned': ape_aligned,
            'dfg_index': dfg_index,
            'ape_index': ape_index,
            'dfg_seq2': dfg_seq2,
            'ape_seq2': ape_seq2
        }

    def filter_alignments_by_gaps(self, alignments):
        """
        Filter alignments to keep only those where DFG and APE are not aligned to gaps.
        
        Args:
            alignments (list): List of analyse_alignment objects
            
        Returns:
            tuple: (validated_alignments, invalid_alignments)
                - validated_alignments: List of valid alignment objects
                - invalid_alignments: List of alignment objects that were filtered out
        """
        validated_alignments = []
        invalid_alignments = []
        
        for alignment in alignments:
            validation_result = alignment.validate_dfg_ape_alignment()
            
            if validation_result['valid']:
                validated_alignments.append(alignment)
            else:
                invalid_alignments.append(alignment)
        
        print(f"{len(validated_alignments)} structures with proper DFG and APE alignment")
        print(f"{len(invalid_alignments)} structures filtered out")
        
        # Print filtered structure names
        if invalid_alignments:
            print("\nExcluded structures:")
            for alignment in invalid_alignments:
                print(f"  - {alignment.name}")
        
        # Save filtered results
        with open('filtered_aligned_data.pkl', 'wb') as f:
            filtered_data = {
                'validated_alignments': validated_alignments,
                'invalid_alignments': invalid_alignments
            }
            p.dump(filtered_data, f)
        print(f"\nSaved {len(validated_alignments)} validated and {len(invalid_alignments)} invalid structures to 'filtered_aligned_data.pkl'")
        
        return validated_alignments, invalid_alignments

    def copy_validated_structures(self, validated_alignments, source_dir=None, target_dir="Results/activation_segments/nonReconstructed_mustang_filtered"):
        """
        Copy structure files for validated alignments to a new directory.
        
        Args:
            validated_alignments (list): List of validated analyse_alignment objects
            source_dir (str): Source directory containing original structure files
            target_dir (str): Target directory for filtered structures
        """
        if source_dir is None:
            source_dir = self.alignment_dir
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        copied_count = 0
        failed_count = 0
        
        for alignment in tqdm(validated_alignments, desc="Copying validated structures"):
            # Look for structure files associated with this alignment
            structure_name = alignment.name
            
            # Search for various file types that might be associated with this structure
            possible_extensions = ['.pdb', '.afasta', '.html', '.msf']
            
            for ext in possible_extensions:
                # Check in subdirectories
                for subdir in os.listdir(source_dir):
                    subdir_path = os.path.join(source_dir, subdir)
                    if os.path.isdir(subdir_path):
                        source_file = os.path.join(subdir_path, structure_name + ext)
                        if os.path.exists(source_file):
                            # Create corresponding subdirectory in target
                            target_subdir = os.path.join(target_dir, subdir)
                            os.makedirs(target_subdir, exist_ok=True)
                            
                            target_file = os.path.join(target_subdir, structure_name + ext)
                            try:
                                shutil.copy2(source_file, target_file)
                                copied_count += 1
                            except Exception as e:
                                print(f"Failed to copy {source_file}: {e}")
                                failed_count += 1
        
        print(f"Successfully copied {copied_count} files")
        if failed_count > 0:
            print(f"Failed to copy {failed_count} files")

    def filter_and_save_structures_with_motifs(self, target_dir, source_dir=None):
        """
        Filter structures that have both DFG and APE motifs in their sequence and save them to target directory.
        
        Args:
            target_dir (str): Target directory to save filtered structures
            source_dir (str): Source directory containing original structure files (defaults to self.alignment_dir)
            
        Returns:
            tuple: (valid_alignments, invalid_alignments) where valid_alignments contains structures 
                   with both DFG and APE motifs, and invalid_alignments contains those without
        """
        if source_dir is None:
            source_dir = self.alignment_dir
        
        # Load alignments
        print("Loading alignments...")
        alignments = self.load_alignments()
        
        # Filter alignments based on DFG and APE motifs
        print("Filtering alignments based on DFG and APE motifs...")
        valid_alignments = []
        invalid_alignments = []
        
        for alignment in alignments:
            # Check if both DFG and APE motifs are present in seq1
            has_dfg = alignment.seq1.find("DFG") != -1
            has_ape = alignment.seq1.find("APE") != -1
            
            if has_dfg and has_ape:
                valid_alignments.append(alignment)
            else:
                invalid_alignments.append(alignment)
        
        print(f"Found {len(valid_alignments)} structures with both DFG and APE motifs")
        print(f"Excluded {len(invalid_alignments)} structures without required motifs")
        
        if len(valid_alignments) == 0:
            print("No valid structures found with both DFG and APE motifs.")
            return valid_alignments, invalid_alignments
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy valid structures to target directory
        print(f"Copying valid structures to {target_dir}...")
        copied_count = 0
        failed_count = 0
        
        for alignment in tqdm(valid_alignments, desc="Copying structures with DFG and APE motifs"):
            structure_name = alignment.name
            
            # Search for various file types that might be associated with this structure
            possible_extensions = ['.pdb', '.afasta', '.html', '.msf']
            
            for ext in possible_extensions:
                # Check in subdirectories
                for subdir in os.listdir(source_dir):
                    subdir_path = os.path.join(source_dir, subdir)
                    if os.path.isdir(subdir_path):
                        source_file = os.path.join(subdir_path, structure_name + ext)
                        if os.path.exists(source_file):
                            # Create corresponding subdirectory in target
                            target_subdir = os.path.join(target_dir, subdir)
                            os.makedirs(target_subdir, exist_ok=True)
                            
                            target_file = os.path.join(target_subdir, structure_name + ext)
                            try:
                                shutil.copy2(source_file, target_file)
                                copied_count += 1
                            except Exception as e:
                                print(f"Failed to copy {source_file}: {e}")
                                failed_count += 1
        
        print(f"Successfully copied {copied_count} files")
        if failed_count > 0:
            print(f"Failed to copy {failed_count} files")
        
        # Save filtered alignments for future use
        filtered_pickle_path = os.path.join(target_dir, "filtered_alignments_with_motifs.pkl")
        with open(filtered_pickle_path, 'wb') as f:
            p.dump(valid_alignments, f)
        print(f"Saved {len(valid_alignments)} filtered alignments to {filtered_pickle_path}")
        
        return valid_alignments, invalid_alignments

    def __getitem__(self, idx):
        """Allow indexing of alignment object."""
        return (self.seq1[idx], self.seq2[idx])

    def __repr__(self):
        """String representation of alignment object."""
        return self.name if hasattr(self, 'name') else "analyse_alignment"


class visualise_sequence_alignment:
    """
    Subclass for visualizing sequence alignments in HTML format.
    """
    
    def __init__(self, mustang_dir="Results/activation_segments/nonReconstructed_mustang"):
        self.mustang_dir = mustang_dir
    
    def clean_sequence(self, seq_str):
        """Remove newlines and convert to single string"""
        return ''.join(seq_str.split('\n')).strip()
    
    def read_afasta_file(self, afasta_file):
        """Read an afasta file and return the reference and structure sequences"""
        sequences = []
        with open(afasta_file, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences.append((current_id, self.clean_sequence(''.join(current_seq))))
                    current_id = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None:
                sequences.append((current_id, self.clean_sequence(''.join(current_seq))))
        return sequences
    
    def generate_multi_alignment_html(self, alignment_file, output_file, reference_name="6UAN_chainD"):
        """
        Create an HTML visualization of multi-structure 3Di alignment with reference at the top.
        
        Args:
            alignment_file (str): Path to the multi-structure 3Di alignment file (msa_3di.fa)
            output_file (str): Output HTML filename
            reference_name (str): Name/substring to identify the reference structure
        
        Returns:
            dict: Dictionary with statistics about the alignment
        """
        alignment_type = "3Di"
        # Read all sequences from the alignment file
        sequences = self.read_afasta_file(alignment_file)
        
        if not sequences:
            print(f"Error: No sequences found in {alignment_file}")
            return None
        
        # Find the reference sequence
        ref_seq = None
        ref_id = None
        other_sequences = []
        
        for seq_id, seq in sequences:
            if reference_name in seq_id:
                ref_seq = seq
                ref_id = seq_id
            else:
                other_sequences.append((seq_id, seq))
        
        # If reference not found, use first sequence
        if ref_seq is None:
            print(f"Warning: Reference '{reference_name}' not found. Using first sequence as reference.")
            ref_id, ref_seq = sequences[0]
            other_sequences = sequences[1:]
        
        seq_length = len(ref_seq)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Structure {alignment_type.title()} Alignment</title>
            <style>
                body {{ 
                    font-family: monospace; 
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .stats {{
                    background-color: white;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }}
                .alignment-container {{ 
                    overflow-x: auto; 
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .sequence-row {{ 
                    margin: 1px 0;
                    white-space: nowrap;
                }}
                .sequence-id {{ 
                    display: inline-block; 
                    width: 250px; 
                    font-weight: bold;
                    padding: 2px 5px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .sequence {{ 
                    letter-spacing: 1px;
                    font-size: 12px;
                }}
                .residue {{ 
                    display: inline-block; 
                    width: 12px; 
                    text-align: center;
                }}
                .position-marker {{ 
                    color: #7f8c8d;
                    font-size: 10px;
                    border-bottom: 2px solid #bdc3c7;
                    margin-bottom: 5px;
                    background-color: #ecf0f1;
                }}
                .reference-row {{ 
                    background-color: #e8f4f8;
                    color: #2980b9;
                    font-weight: bold;
                    border-top: 3px solid #3498db;
                    border-bottom: 3px solid #3498db;
                    padding: 3px 0;
                    margin-bottom: 10px;
                }}
                .structure-row {{ 
                    color: #555;
                }}
                .structure-row:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .structure-row:hover {{
                    background-color: #fff9e6;
                }}
                h1, h2 {{ 
                    font-family: Arial, sans-serif;
                    margin: 0 0 10px 0;
                }}
                .gap {{
                    color: #bdc3c7;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Structure {alignment_type.title()} Alignment</h1>
                <p>Alignment file: {os.path.basename(alignment_file)}</p>
            </div>
            
            <div class="stats">
                <p><strong>Reference structure:</strong> {html.escape(ref_id)}</p>
                <p><strong>Total structures:</strong> {len(other_sequences) + 1}</p>
                <p><strong>Alignment length:</strong> {seq_length} positions</p>
                <p><strong>Alignment type:</strong> {alignment_type}</p>
            </div>
            
            <div class="alignment-container">
        """
        
        # Add position markers
        html_content += """
                <div class="sequence-row position-marker">
                    <div class="sequence-id">Position</div>
                    <div class="sequence">
        """
        
        for j in range(0, seq_length, 10):
            pos = str(j+1)
            html_content += f'<span class="residue">{pos[0] if pos else " "}</span>'
            for k in range(1, min(10, seq_length - j)):
                if k < len(pos):
                    html_content += f'<span class="residue">{pos[k]}</span>'
                else:
                    html_content += '<span class="residue"> </span>'
        
        html_content += """
                    </div>
                </div>
        """
        
        # Add reference sequence
        html_content += f"""
                <div class="sequence-row reference-row">
                    <div class="sequence-id" title="{html.escape(ref_id)}">{html.escape(ref_id[:40] + '...' if len(ref_id) > 40 else ref_id)}</div>
                    <div class="sequence">
        """
        
        for residue in ref_seq:
            css_class = 'residue gap' if residue == '-' else 'residue'
            html_content += f'<span class="{css_class}">{residue}</span>'
        
        html_content += """
                    </div>
                </div>
        """
        
        # Add all other sequences
        for i, (struct_id, struct_seq) in enumerate(other_sequences):
            html_content += f"""
                <div class="sequence-row structure-row">
                    <div class="sequence-id" title="{html.escape(struct_id)}">{html.escape(struct_id[:40] + '...' if len(struct_id) > 40 else struct_id)}</div>
                    <div class="sequence">
            """
            
            for residue in struct_seq:
                css_class = 'residue gap' if residue == '-' else 'residue'
                html_content += f'<span class="{css_class}">{residue}</span>'
            
            html_content += """
                    </div>
                </div>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Multi-structure alignment visualization created: {output_file}")
        print(f"Total structures visualized: {len(other_sequences) + 1}")
        
        return {
            'reference_id': ref_id,
            'n_structures': len(other_sequences) + 1,
            'alignment_length': seq_length,
            'output_file': output_file
        }
    
    def generate_pairwise_alignment_html(self, output_file="pairwise_alignments.html"):
        """
        Create a visualization where each structure sequence is shown with its 
        corresponding reference sequence above it (as in each .afasta file).
        
        Args:
            output_file (str): Output HTML filename
        """
        # Find all .afasta files
        afasta_files = sorted(glob.glob(os.path.join(self.mustang_dir, "*/*.afasta")))
        
        # Process all alignment files
        alignments = []
        
        for afasta_file in afasta_files:
            sequences = self.read_afasta_file(afasta_file)
            
            # Skip if we don't have at least 2 sequences
            if len(sequences) < 2:
                print(f"Warning: Skipping {afasta_file} - not enough sequences")
                continue
            
            # Get reference and structure sequences
            ref_id, ref_seq = sequences[0]
            struct_id, struct_seq = sequences[1]
            
            # Add to our alignments list
            alignments.append({
                'file': os.path.basename(afasta_file),
                'reference': {
                    'id': ref_id,
                    'seq': ref_seq,
                    'ungapped': len(ref_seq.replace('-', ''))
                },
                'structure': {
                    'id': struct_id,
                    'seq': struct_seq,
                    'ungapped': len(struct_seq.replace('-', ''))
                }
            })
        
        # Generate HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pairwise MUSTANG Alignments</title>
            <style>
                body { font-family: monospace; }
                .container { margin-bottom: 40px; }
                .alignment-container { overflow-x: auto; white-space: nowrap; margin-top: 10px; }
                .sequence-row { margin: 2px 0; }
                .sequence-id { display: inline-block; width: 200px; font-weight: bold; }
                .sequence { letter-spacing: 0.5px; }
                .residue { display: inline-block; width: 10px; text-align: center; }
                .position-marker { color: #999; font-size: 0.8em; border-bottom: 1px solid #ccc; }
                .alignment-header { background-color: #f0f0f0; padding: 10px; margin-top: 20px; }
                .reference-row { color: #0066cc; }
                .structure-row { color: #cc6600; }
                h1, h2 { font-family: Arial, sans-serif; }
            </style>
        </head>
        <body>
            <h1>Pairwise MUSTANG Alignments</h1>
            <p>Each alignment is shown with the reference sequence above the structure sequence, exactly as in the original .afasta files.</p>
        """
        
        # Add each pairwise alignment
        for i, alignment in enumerate(alignments):
            html_content += f"""
            <div class="container">
                <div class="alignment-header">
                    <h2>Alignment #{i+1}: {alignment['file']}</h2>
                    <p><strong>Reference:</strong> {alignment['reference']['id']} ({alignment['reference']['ungapped']} residues)</p>
                    <p><strong>Structure:</strong> {alignment['structure']['id']} ({alignment['structure']['ungapped']} residues)</p>
                </div>
                <div class="alignment-container">
                    <!-- Position markers -->
                    <div class="sequence-row position-marker">
                        <div class="sequence-id">Position</div>
                        <div class="sequence">
            """
            
            # Add position markers
            seq_length = len(alignment['reference']['seq'])
            for j in range(0, seq_length, 10):
                pos = str(j+1)
                html_content += f'<span class="residue">{pos[0]}</span>'
                for k in range(1, len(pos)):
                    html_content += f'<span class="residue">{pos[k]}</span>'
                for k in range(len(pos), 10):
                    html_content += f'<span class="residue">&nbsp;</span>'
            
            html_content += """
                        </div>
                    </div>
            """
            
            # Add reference sequence
            html_content += f"""
                    <div class="sequence-row reference-row">
                        <div class="sequence-id">{html.escape(alignment['reference']['id'])}</div>
                        <div class="sequence">
            """
            
            for residue in alignment['reference']['seq']:
                html_content += f'<span class="residue">{residue}</span>'
            
            html_content += """
                        </div>
                    </div>
            """
            
            # Add structure sequence
            html_content += f"""
                    <div class="sequence-row structure-row">
                        <div class="sequence-id">{html.escape(alignment['structure']['id'])}</div>
                        <div class="sequence">
            """
            
            for residue in alignment['structure']['seq']:
                html_content += f'<span class="residue">{residue}</span>'
            
            html_content += """
                        </div>
                    </div>
                </div>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Pairwise alignment visualization created: {output_file}")
        print(f"Total alignments visualized: {len(alignments)}")
        
        return alignments
#!/usr/bin/env python3
"""
Protein Sequence Reconstruction Pipeline

This program extracts SEQRES and ATOM sequences from PDB files, aligns them,
identifies gaps, and uses MODELLER to reconstruct missing residues when needed.
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
from modeller import *
from modeller.automodel import *


class ProteinReconstructor:
    """
    A class to handle protein sequence reconstruction from PDB files.
    
    This class combines sequence extraction, alignment, gap detection,
    and MODELLER-based reconstruction in a single pipeline.
    """
    
    def __init__(self, input_dir, full_pdb_dir, output_dir, max_gap_length=4):
        """
        Initialize the protein reconstructor.
        
        Args:
            input_dir: Directory containing PDB segments to process
            full_pdb_dir: Directory containing full PDB files for SEQRES extraction
            output_dir: Directory to save reconstructed structures
            max_gap_length: Maximum gap length allowed for reconstruction
        """
        self.input_dir = input_dir
        self.full_pdb_dir = full_pdb_dir
        self.output_dir = output_dir
        self.max_gap_length = max_gap_length
        
        # Non-natural amino acid substitutions
        self.substitutions = {
            'X': 'G',  # Glycine
            'B': 'N',  # Asparagine
            'Z': 'Q',  # Glutamine
            'J': 'L'   # Leucine
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_seqres_sequence(self, pdb_file):
        """Extract SEQRES sequences for each chain from a PDB file."""
        seq_dict = {}
        with open(pdb_file, "r") as file:
            lines = file.readlines()

        current_chain = None
        current_seq = []

        for line in lines:
            if line.startswith("SEQRES"):
                parts = line.split()
                chain_id = parts[2]
                if chain_id != current_chain:
                    if current_chain is not None:
                        seq_dict[current_chain] = ''.join(seq1(residue) for residue in current_seq)
                    current_chain = chain_id
                    current_seq = []
                current_seq.extend(parts[4:])

        if current_chain is not None:
            seq_dict[current_chain] = ''.join(seq1(residue) for residue in current_seq)

        return seq_dict
    
    def extract_atom_sequence(self, pdb_file, chain_id):
        """Extract sequence from atomic coordinates for a specific chain."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB', pdb_file)
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                ppb = PPBuilder()
                sequence = ''
                for pp in ppb.build_peptides(chain):
                    sequence += pp.get_sequence()
                return str(sequence)
        return None
    
    def find_motif_indices(self, sequence, motif):
        """Find the start index of a motif in a sequence."""
        index = sequence.find(motif)
        return index if index != -1 else None
    
    def align_sequences(self, seqres_segment, atom_segment):
        """Align SEQRES and ATOM segments and calculate gap information."""
        aligned_seqres = ''
        aligned_atom = ''
        atom_index = 0
        max_gap_length = 0
        current_gap_length = 0

        for res_seqres in seqres_segment:
            if atom_index < len(atom_segment) and res_seqres == atom_segment[atom_index]:
                aligned_seqres += res_seqres
                aligned_atom += atom_segment[atom_index]
                atom_index += 1
                current_gap_length = 0
            else:
                aligned_seqres += res_seqres
                aligned_atom += '-'
                current_gap_length += 1
                max_gap_length = max(max_gap_length, current_gap_length)

        return aligned_seqres, aligned_atom, max_gap_length
    
    def substitute_non_natural_amino_acids(self, aligned_seqres, aligned_atom):
        """Substitute non-natural amino acids with their natural counterparts."""
        corrected_seqres = ''
        
        for index, (res_seqres, res_atom) in enumerate(zip(aligned_seqres, aligned_atom)):
            if res_seqres in self.substitutions:
                # Don't substitute 'X' if it's surrounded by gaps
                if res_seqres == 'X':
                    if ((index > 0 and aligned_atom[index - 1] == '-') or
                        (index < len(aligned_atom) - 1 and aligned_atom[index + 1] == '-')):
                        corrected_seqres += res_seqres
                        continue
                
                corrected_residue = self.substitutions[res_seqres]
                corrected_seqres += corrected_residue
                print(f"Substituting non-natural amino acid '{res_seqres}' with '{corrected_residue}'")
            else:
                corrected_seqres += res_seqres
        
        return corrected_seqres
    
    def remove_remark_lines(self, pdb_file):
        """Remove lines starting with 'REMARK' from the PDB file."""
        with open(pdb_file, 'r') as file:
            lines = file.readlines()
        
        with open(pdb_file, 'w') as file:
            for line in lines:
                if not line.startswith("REMARK"):
                    file.write(line)
    
    def reconstruct_with_modeller(self, pdb_chain_id, pdb_path, full_sequence, atom_sequence):
        """Use MODELLER to reconstruct missing residues."""
        print(f"Reconstructing sequence for {pdb_chain_id} using MODELLER")
        
        # Set up MODELLER environment
        env = environ()
        aln = alignment(env)
        
        # Read the template structure
        mdl = model(env, file=pdb_path)
        aln.append_model(mdl, align_codes='template', atom_files=pdb_path)
        
        # Append the target sequence
        aln.append_sequence(full_sequence)
        aln[-1].code = 'target'
        
        # Perform alignment and build model
        aln.align2d(max_gap_length=50)
        
        a = automodel(env, alnfile=aln, knowns='template', sequence='target')
        a.starting_model = 1
        a.ending_model = 1
        a.make()
        
        # Save the reconstructed model
        output_path = os.path.join(self.output_dir, f"{pdb_chain_id}_reconstructed.pdb")
        os.rename(a.outputs[0]['name'], output_path)
        
        # Remove REMARK lines
        self.remove_remark_lines(output_path)
        
        print(f"Reconstruction completed for {pdb_chain_id}")
        return output_path
    
    def process_structure(self, pdb_file):
        """Process a single PDB structure."""
        pdb_name = os.path.basename(pdb_file)
        pdb_id, chain_id = os.path.splitext(pdb_name)[0].split('_')
        
        # Get full PDB file path
        full_pdb_path = os.path.join(self.full_pdb_dir, f"{pdb_id}.pdb")
        if not os.path.isfile(full_pdb_path):
            print(f"Full PDB file not found for {pdb_id}")
            return False
        
        # Extract sequences
        seqres_seqs = self.extract_seqres_sequence(full_pdb_path)
        atom_seq = self.extract_atom_sequence(full_pdb_path, chain_id)
        
        if chain_id not in seqres_seqs or not atom_seq:
            print(f"Could not extract sequences for {pdb_id}_{chain_id}")
            return False
        
        # Get activation segment sequences
        seqres_sequence = seqres_seqs[chain_id]
        seqres_dfg_index = self.find_motif_indices(seqres_sequence, 'DFG')
        seqres_ape_index = self.find_motif_indices(seqres_sequence, 'APE')
        atom_dfg_index = self.find_motif_indices(atom_seq, 'DFG')
        atom_ape_index = self.find_motif_indices(atom_seq, 'APE')
        
        # Check if motifs are found
        if None in [seqres_dfg_index, seqres_ape_index, atom_dfg_index, atom_ape_index]:
            print(f"Required motifs not found in {pdb_id}_{chain_id}")
            return False
        
        # Extract activation segments
        seqres_start = min(seqres_dfg_index, seqres_ape_index)
        seqres_end = max(seqres_dfg_index + 3, seqres_ape_index + 3)
        atom_start = min(atom_dfg_index, atom_ape_index)
        atom_end = max(atom_dfg_index + 3, atom_ape_index + 3)
        
        seqres_segment = seqres_sequence[seqres_start:seqres_end]
        atom_segment = atom_seq[atom_start:atom_end]
        
        # Align sequences and check gaps
        aligned_seqres, aligned_atom, max_gap_length = self.align_sequences(seqres_segment, atom_segment)
        
        # Check if gap length is acceptable
        if max_gap_length > self.max_gap_length:
            print(f"Gap length ({max_gap_length}) exceeds maximum ({self.max_gap_length}) for {pdb_id}_{chain_id}")
            return False
        
        # Substitute non-natural amino acids
        corrected_seqres = self.substitute_non_natural_amino_acids(aligned_seqres, aligned_atom)
        
        # Check if reconstruction is needed
        if corrected_seqres != aligned_atom.replace('-', ''):
            # Reconstruct with MODELLER
            self.reconstruct_with_modeller(f"{pdb_id}_{chain_id}", pdb_file, seqres_sequence, atom_seq)
        else:
            # Copy original file
            output_path = os.path.join(self.output_dir, f"{pdb_id}_{chain_id}.pdb")
            shutil.copy(pdb_file, output_path)
            print(f"No reconstruction needed for {pdb_id}_{chain_id}, copied original file")
        
        print(f"Processed {pdb_id}_{chain_id} - Max gap length: {max_gap_length}")
        return True
    
    def run_modeller_pipeline(self):
        """Run the complete reconstruction pipeline."""
        pdb_files = glob(os.path.join(self.input_dir, "*.pdb"))
        
        if not pdb_files:
            print(f"No PDB files found in {self.input_dir}")
            return
        
        print(f"Processing {len(pdb_files)} PDB files...")
        
        processed_count = 0
        for pdb_file in tqdm(pdb_files, desc="Processing structures"):
            if self.process_structure(pdb_file):
                processed_count += 1
        
        print(f"\nPipeline completed! Successfully processed {processed_count}/{len(pdb_files)} structures")
        print(f"Results saved to: {self.output_dir}")
        
        # Create nonreconstructed folder after processing
        self.create_nonreconstructed_folder()
    
    def create_nonreconstructed_folder(self):
        """Create a folder with unaligned structures that have matching entries in the reconstructed folder."""
        target_dir = "Results/activation_segments/nonReconstructed4Mustang"
        
        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created directory: {target_dir}")
        
        # Get list of files in both directories
        unaligned_files = [f for f in os.listdir(self.input_dir) if f.endswith('.pdb')]
        reconstructed_files = [f for f in os.listdir(self.output_dir) if f.endswith('.pdb')]
        
        # Extract prefixes (first 6 characters) from reconstructed files
        reconstructed_prefixes = {f[:6] for f in reconstructed_files}
        
        # Counter for copied files
        copied_count = 0
        skipped_count = 0
        
        # For each unaligned file, check if its prefix exists in reconstructed files
        for unaligned_file in unaligned_files:
            file_prefix = unaligned_file[:6]
            
            if file_prefix in reconstructed_prefixes:
                source_path = os.path.join(self.input_dir, unaligned_file)
                target_path = os.path.join(target_dir, unaligned_file)
                
                # Copy the file
                shutil.copy(source_path, target_path)
                copied_count += 1
            else:
                skipped_count += 1
        
        print(f"\nCreated nonreconstructed folder:")
        print(f"Copied {copied_count} files from '{self.input_dir}' to '{target_dir}'")
        print(f"Skipped {skipped_count} files that don't have matching entries in '{self.output_dir}'")
        print(f"These files have matching entries in '{self.output_dir}' based on the first 6 characters")
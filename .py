import os
from glob import glob
from pymol import cmd
from Bio.PDB import PDBParser, PPBuilder

# Paths
pdb_dir = "Results/activation_segments/mustangs/"
reference_pdb = "6UAN_chainD.pdb"
output_dir = "Results/activation_segments/mustangs_realigned/"
image_output_path = "Results/aligned_loops.png"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the reference structure and set up its selections
cmd.load(reference_pdb, "6UAN_chainD")
cmd.select("6UAN_chainD_dfg_selection", "6UAN_chainD and resi 594-596 and name CA")
cmd.select("6UAN_chainD_ape_selection", "6UAN_chainD and resi 621-623 and name CA")
cmd.select("6UAN_chainD_ends_selection", "6UAN_chainD and (resi 594-596 or resi 621-623) and name CA")


def extract_atom_sequence(pdb_file):
    """
    Extracts the amino acid sequences of all chains from a PDB file and returns the sequence 
    and corresponding residue numbers of the first chain.
    This implementation parses all models and chains in one go to avoid repeatedly creating PPBuilder objects for each chain.
    Returns:
        (sequence, sequence_ids)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_file)
    ppb = PPBuilder()  # Create a global PPBuilder object
    sequence = ""
    sequence_ids = []
    # Iterate over all chains in the first model (only the first chain is used)
    for model in structure:
        for chain in model:
            for pp in ppb.build_peptides(chain):
                sequence_fragment = str(pp.get_sequence())
                sequence += sequence_fragment
                # For each residue in the peptide fragment, record its actual PDB residue number
                for residue in pp:
                    sequence_ids.append(residue.get_id()[1])
            return sequence, sequence_ids
    return None, None


def find_motif_indices(seq, seq_ids, motif):
    """
    Finds the specified motif in the given sequence and returns the corresponding PDB residue number range 
    using the residue number list.
    Note: It is assumed that the order in seq_ids corresponds to the character order in seq.
    """
    index = seq.find(motif)
    if index == -1:
        return None
    # Directly use the matched position in the residue number list to get the corresponding PDB residue number
    start_pdb_num = seq_ids[index]
    end_pdb_num = start_pdb_num + len(motif) - 1
    return start_pdb_num, end_pdb_num


def print_residues_in_selection(selection_name):
    model = cmd.get_model(selection_name)
    residues = sorted(set((atom.resi, atom.resn) for atom in model.atom))
    print(f"Residues in {selection_name}: {residues}")


def process_structure(pdb_file, ref_name="6UAN_chainD"):
    pdb_code = os.path.basename(pdb_file).split('.')[0]
    cmd.load(pdb_file, pdb_code)

    # Parse and extract the sequence and residue numbers (to avoid redundant parsing)
    seq, seq_ids = extract_atom_sequence(pdb_file)
    if not seq:
        print(f"Skipping {pdb_code} due to inability to extract sequence.")
        return
    print(f"Sequence for {pdb_code}: {seq}")

    # Find the PDB residue number ranges corresponding to the DFG and APE motifs based on the sequence
    dfg_range = find_motif_indices(seq, seq_ids, "DFG")
    ape_range = find_motif_indices(seq, seq_ids, "APE")
    print("DFG CHECK", dfg_range)
    print("APE CHECK", ape_range)
    if not dfg_range or not ape_range:
        print(f"Skipping {pdb_code} due to missing motifs.")
        return

    # Construct selection strings; note that here we use the range directly without adding 1
    cmd.select(f"{pdb_code}_dfg_selection", f"{pdb_code} and resi {dfg_range[0]}-{dfg_range[1]} and name CA")
    cmd.select(f"{pdb_code}_ape_selection", f"{pdb_code} and resi {ape_range[0]}-{ape_range[1]} and name CA")
    cmd.select(f"{pdb_code}_ends_selection", f"{pdb_code} and (resi {dfg_range[0]}-{dfg_range[1]} or resi {ape_range[0]}-{ape_range[1]}) and name CA")

    # Print the information of the selected residues
    print_residues_in_selection(f"{pdb_code}_dfg_selection")
    print_residues_in_selection(f"{pdb_code}_ape_selection")
    print_residues_in_selection(f"{pdb_code}_ends_selection")

    # Calculate the RMSD before alignment
    rms_dfg_before = cmd.rms_cur(f"{pdb_code}_dfg_selection", f"{ref_name}_dfg_selection", matchmaker=-1)
    rms_ape_before = cmd.rms_cur(f"{pdb_code}_ape_selection", f"{ref_name}_ape_selection", matchmaker=-1)
    rms_ends_before = cmd.rms_cur(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", matchmaker=-1)
    print(f"Before alignment RMSD for {pdb_code}: DFG={rms_dfg_before}, APE={rms_ape_before}, ENDS={rms_ends_before}")

    # Align the structure
    cmd.align(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", cycles=0, transform=1)

    # Save the aligned structure
    aligned_pdb_path = os.path.join(output_dir, f"{pdb_code}_aligned.pdb")
    cmd.save(aligned_pdb_path, pdb_code)
    print(f"Saved aligned structure to {aligned_pdb_path}")

    # Calculate the RMSD after alignment
    rms_dfg_after = cmd.rms_cur(f"{pdb_code}_dfg_selection", f"{ref_name}_dfg_selection", matchmaker=-1)
    rms_ape_after = cmd.rms_cur(f"{pdb_code}_ape_selection", f"{ref_name}_ape_selection", matchmaker=-1)
    rms_ends_after = cmd.rms_cur(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", matchmaker=-1)
    print(f"After alignment RMSD for {pdb_code}: DFG={rms_dfg_after}, APE={rms_ape_after}, ENDS={rms_ends_after}\n")


# Iterate over all PDB files and process them
# (Optionally, multithreading/multiprocessing can be used for large datasets,
# but be aware of PyMOL command thread safety.)
fps = glob(os.path.join(pdb_dir, "*", "*.pdb"))
for fp in fps:
    print(fp)
    process_structure(fp)

# Visualization and coloring
cmd.select("all_loops", "byres 6UAN_chainD_ends_selection")
cmd.color("red", "6UAN_chainD and 6UAN_chainD_ends_selection")
cmd.color("white", "not 6UAN_chainD and all_loops")
cmd.show("cartoon")
cmd.hide("lines")

# Save image
cmd.png(image_output_path, width=1200, height=800, dpi=300, ray=1)
print(f"Image saved to {image_output_path}")

cmd.quit()
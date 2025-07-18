import os
import subprocess
from glob import glob
from time import time
import pymol
from pymol import cmd
from Bio.PDB import PDBParser, PPBuilder


class Alignment:
    """
    A class to handle protein structure alignment using both MUSTANG and PyMOL.
    
    This class combines functionality for:
    1. MUSTANG-based structural alignment
    2. PyMOL-based motif alignment using DFG and APE motifs
    """
    
    def __init__(self, mustang_path="/home/marmatt/Downloads/MUSTANG_v3.2.4/bin/mustang-3.2.4"):
        """
        Initialize the Alignment class.
        
        Args:
            mustang_path: Path to the MUSTANG executable
        """
        self.mustang_path = mustang_path
        self.pymol_initialized = False
        
    def initialize_pymol(self):
        """Initialize PyMOL in headless mode for notebook use."""
        if not self.pymol_initialized:
            try:
                # Initialize PyMOL in headless mode
                pymol.pymol_argv = ['pymol', '-c']  # -c flag for command line mode
                pymol.finish_launching()
                self.pymol_initialized = True
                print("PyMOL initialized in headless mode")
            except Exception as e:
                print(f"Error initializing PyMOL: {e}")
                return False
        return True
    
    def cleanup_pymol(self):
        """Clean up PyMOL objects without quitting."""
        try:
            cmd.delete("all")
            cmd.reset()
            print("PyMOL cleanup completed")
        except Exception as e:
            print(f"Error during PyMOL cleanup: {e}")
    
    def find_pdbs(self, directory):
        """Find all PDB files in the given directory and subdirectories."""
        pdb_files = []
        
        # First, try to find PDB files directly in the directory
        direct_pdbs = glob(os.path.join(directory, "*.pdb"))
        pdb_files.extend(direct_pdbs)
        
        # If no direct PDB files found, search recursively in subdirectories
        if not direct_pdbs:
            print(f"No PDB files found directly in {directory}, searching subdirectories...")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.pdb'):
                        pdb_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdb_files)} PDB files in {directory}")
        return pdb_files
    
    def fname(self, path):
        """Extract filename without extension."""
        return os.path.splitext(os.path.basename(path))[0]
    
    def run_mustang(self, template_pdb, input_pdb, target_dir, name=None):
        """
        Run MUSTANG alignment on a PDB file against a template.
        
        Args:
            template_pdb: Path to the template PDB file
            input_pdb: Path to the input PDB file to align
            target_dir: Directory to save the aligned output
            name: Optional name for the output file
        
        Returns:
            Path to the output file or None if failed
        """
        if name is None:
            name = self.fname(input_pdb)
        
        try:
            # Create a dedicated directory for this structure
            struct_dir = os.path.join(target_dir, name)
            os.makedirs(struct_dir, exist_ok=True)
            
            # Define output path
            new_fp = os.path.join(struct_dir, name)
            
            # Define structure list for the command
            structs = f"{template_pdb} {input_pdb}"
            
            # Prepare and run the MUSTANG command
            command = f"{self.mustang_path} -i {structs} -o {new_fp} -F fasta -s ON"
            print(f"Running MUSTANG: {command}")
            
            # Run the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"MUSTANG failed: {result.stderr}")
                return None
            
            # Look for the aligned PDB file
            expected_output = os.path.join(struct_dir, f"{name}.pdb")
            
            # Check for alternative output names 
            if not os.path.exists(expected_output):
                alt_files = glob(os.path.join(struct_dir, "*.pdb"))
                if alt_files:
                    print(f"Expected output {expected_output} not found, but found alternatives: {alt_files}")
                    expected_output = alt_files[0]
                else:
                    print(f"No PDB files found in {struct_dir}")
                    return None
                    
            print(f"Successfully aligned {name}, saved to {expected_output}")
            return expected_output
            
        except Exception as e:
            print(f"Error running MUSTANG for {name}: {e}")
            return None
    
    def process_mustang_alignment(self, pdb_path, target_dir, template_pdb):
        """
        Process multiple PDB files through MUSTANG alignment.
        
        Args:
            pdb_path: Directory containing PDB files to align
            target_dir: Directory to save aligned outputs
            template_pdb: Template PDB file for alignment
        """
        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Input directory: {pdb_path}")
        print(f"Output directory: {target_dir}")
        print(f"Template PDB: {template_pdb}")
        
        # Get list of PDB files
        pdbs = self.find_pdbs(pdb_path)
        print(f"Found {len(pdbs)} PDB files to process")
        
        # Process each file
        t1 = time()
        
        for pdb in pdbs:
            name = self.fname(pdb)
            new_fp = self.run_mustang(template_pdb, pdb, target_dir, name=name)
        
        t2 = time()
        print(f"Processing complete in {round(t2-t1, 3)} seconds")
    
    def extract_sequence_and_residue_numbers(self, pdb_file):
        """Extract sequence and corresponding PDB residue numbers from atomic coordinates for the first chain found in the PDB file."""
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
                print(residue_numbers)
                return str(sequence), residue_numbers
        return None, None
    
    def find_motif_indices(self, seq, motif):
        """Find the indices of a motif in a sequence."""
        index = seq.find(motif)
        if index == -1:
            return None
        return index, index + len(motif)
    
    def print_residues_in_selection(self, selection_name):
        """Print residues in a PyMOL selection."""
        try:
            model = cmd.get_model(selection_name)
            print(model)
            residues = set((atom.resi, atom.resn) for atom in model.atom)
            print(residues)
            print(f"Residues in {selection_name}: {sorted(residues)}")
        except Exception as e:
            print(f"Error printing residues in {selection_name}: {e}")
    
    def process_structure(self, pdb_file, output_dir, ref_name="6UAN_chainD"):
        """
        Process a single structure for PyMOL-based alignment.
        
        Args:
            pdb_file: Path to the PDB file to process
            output_dir: Directory to save aligned output
            ref_name: Name of the reference structure in PyMOL
        """
        try:
            pdb_code = os.path.basename(pdb_file).split('.')[0]
            cmd.load(pdb_file, pdb_code)

            # Extract sequence and corresponding residue numbers from atomic coordinates
            seq, residue_numbers = self.extract_sequence_and_residue_numbers(pdb_file)
            if seq is None or residue_numbers is None:
                print(f"Skipping {pdb_code} due to inability to extract sequence and residue numbers.")
                return
            print(f"Sequence for {pdb_code}: {seq}")
            print(f"Residue numbers: {residue_numbers}")

            # Find indices for DFG and APE in the sequence
            dfg_indices = self.find_motif_indices(seq, "DFG")
            ape_indices = self.find_motif_indices(seq, "APE")
            print(f"DFG sequence indices: {dfg_indices}")
            print(f"APE sequence indices: {ape_indices}")

            if not dfg_indices or not ape_indices:
                print(f"Skipping {pdb_code} due to missing motifs.")
                return

            # Map sequence indices to actual PDB residue numbers
            dfg_pdb_residues = [residue_numbers[i] for i in range(dfg_indices[0], dfg_indices[1])]
            ape_pdb_residues = [residue_numbers[i] for i in range(ape_indices[0], ape_indices[1])]
            
            print(f"DFG PDB residue numbers: {dfg_pdb_residues}")
            print(f"APE PDB residue numbers: {ape_pdb_residues}")
            print(pdb_code)
            
            # Create selections using actual PDB residue numbers - handle alternate conformations
            cmd.select(f"{pdb_code}_dfg_selection", f"{pdb_code} and resi {dfg_pdb_residues[0]}-{dfg_pdb_residues[-1]} and name CA and not alt B+C+D")
            cmd.select(f"{pdb_code}_ape_selection", f"{pdb_code} and resi {ape_pdb_residues[0]}-{ape_pdb_residues[-1]} and name CA and not alt B+C+D")
            cmd.select(f"{pdb_code}_ends_selection", f"{pdb_code} and (resi {dfg_pdb_residues[0]}-{dfg_pdb_residues[-1]} or resi {ape_pdb_residues[0]}-{ape_pdb_residues[-1]}) and name CA and not alt B+C+D")
            
            # Print residues in selections
            self.print_residues_in_selection(f"{pdb_code}_dfg_selection")
            self.print_residues_in_selection(f"{pdb_code}_ape_selection")
            self.print_residues_in_selection(f"{pdb_code}_ends_selection")

            # RMSD before alignment
            rms_dfg_before = cmd.rms_cur(f"{pdb_code}_dfg_selection", f"{ref_name}_dfg_selection", matchmaker=-1)
            rms_ape_before = cmd.rms_cur(f"{pdb_code}_ape_selection", f"{ref_name}_ape_selection", matchmaker=-1)
            rms_ends_before = cmd.rms_cur(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", matchmaker=-1)

            print(f"Before alignment RMSD for {pdb_code}: DFG={rms_dfg_before}, APE={rms_ape_before}, ENDS={rms_ends_before}")

            # Aligning
            cmd.align(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", cycles=0, transform=1)

            # Save aligned structure
            aligned_pdb_path = os.path.join(output_dir, f"{pdb_code}_aligned.pdb")
            cmd.save(aligned_pdb_path, pdb_code)
            print(f"Saved aligned structure to {aligned_pdb_path}")

            # RMSD after alignment
            rms_dfg_after = cmd.rms_cur(f"{pdb_code}_dfg_selection", f"{ref_name}_dfg_selection", matchmaker=-1)
            rms_ape_after = cmd.rms_cur(f"{pdb_code}_ape_selection", f"{ref_name}_ape_selection", matchmaker=-1)
            rms_ends_after = cmd.rms_cur(f"{pdb_code}_ends_selection", f"{ref_name}_ends_selection", matchmaker=-1)

            print(f"After alignment RMSD for {pdb_code}: DFG={rms_dfg_after}, APE={rms_ape_after}, ENDS={rms_ends_after}")
            
            # Clean up selections
            cmd.delete(f"{pdb_code}_dfg_selection")
            cmd.delete(f"{pdb_code}_ape_selection")
            cmd.delete(f"{pdb_code}_ends_selection")
            cmd.delete(pdb_code)
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
    
    def setup_reference_structure(self, reference_pdb, ref_name="6UAN_chainD"):
        """
        Set up the reference structure for PyMOL-based alignment.
        
        Args:
            reference_pdb: Path to the reference PDB file
            ref_name: Name to assign to the reference structure
        """
        try:
            # Load the reference structure
            cmd.load(reference_pdb, ref_name)

            # Create dynamic selections for the reference structure
            ref_seq, ref_residue_numbers = self.extract_sequence_and_residue_numbers(reference_pdb)
            if ref_seq and ref_residue_numbers:
                ref_dfg_indices = self.find_motif_indices(ref_seq, "DFG")
                ref_ape_indices = self.find_motif_indices(ref_seq, "APE")
                
                if ref_dfg_indices and ref_ape_indices:
                    ref_dfg_pdb_residues = [ref_residue_numbers[i] for i in range(ref_dfg_indices[0], ref_dfg_indices[1])]
                    ref_ape_pdb_residues = [ref_residue_numbers[i] for i in range(ref_ape_indices[0], ref_ape_indices[1])]
                    
                    print(f"Reference DFG PDB residue numbers: {ref_dfg_pdb_residues}")
                    print(f"Reference APE PDB residue numbers: {ref_ape_pdb_residues}")
                    
                    cmd.select(f"{ref_name}_dfg_selection", f"{ref_name} and resi {ref_dfg_pdb_residues[0]}-{ref_dfg_pdb_residues[-1]} and name CA")
                    cmd.select(f"{ref_name}_ape_selection", f"{ref_name} and resi {ref_ape_pdb_residues[0]}-{ref_ape_pdb_residues[-1]} and name CA")
                    cmd.select(f"{ref_name}_ends_selection", f"{ref_name} and (resi {ref_dfg_pdb_residues[0]}-{ref_dfg_pdb_residues[-1]} or resi {ref_ape_pdb_residues[0]}-{ref_ape_pdb_residues[-1]}) and name CA")
                    
                    # Verify reference selections
                    print("Reference structure selections:")
                    self.print_residues_in_selection(f"{ref_name}_dfg_selection")
                    self.print_residues_in_selection(f"{ref_name}_ape_selection")
                    self.print_residues_in_selection(f"{ref_name}_ends_selection")
                    
                    return True
                else:
                    print("Error: Could not find DFG or APE motifs in reference structure")
                    return False
            else:
                print("Error: Could not extract sequence from reference structure")
                return False
        except Exception as e:
            print(f"Error setting up reference structure: {e}")
            return False
    
    def process_pymol_alignment(self, pdb_dir, reference_pdb, output_dir, ref_name="6UAN_chainD"):
        """
        Process multiple PDB files through PyMOL-based alignment.
        
        Args:
            pdb_dir: Directory containing PDB files to align
            reference_pdb: Reference PDB file for alignment
            output_dir: Directory to save aligned outputs
            ref_name: Name to assign to the reference structure
        """
        # Initialize PyMOL if not already done
        if not self.initialize_pymol():
            print("Failed to initialize PyMOL")
            return
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup reference structure
            if not self.setup_reference_structure(reference_pdb, ref_name):
                print("Failed to setup reference structure")
                return
            
            # Load all PDB files and process them
            fps = self.find_pdbs(pdb_dir)
            if not fps:
                print("ERROR: No PDB files found to process!")
                return
            
            print(f"Found {len(fps)} PDB files to process")
            for fp in fps:
                print(f"Processing: {fp}")
                self.process_structure(fp, output_dir, ref_name)

            # Clean up instead of quitting
            self.cleanup_pymol()
            print("PyMOL alignment processing completed successfully")
            
        except Exception as e:
            print(f"Error during PyMOL alignment processing: {e}")
            self.cleanup_pymol()




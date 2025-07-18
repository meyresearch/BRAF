import os
import subprocess
from glob import glob
from utilities import fname

def run_mustang(template_pdb, input_pdb, target_dir, name=None):
    """
    Run MUSTANG alignment on a PDB file against a template.
    
    Args:
        template_pdb: Path to the template PDB file
        input_pdb: Path to the input PDB file to align
        target_dir: Directory where output should be saved
        name: Optional name for the output file
    
    Returns:
        Path to the output file or None if failed
    """
    if name is None:
        name = fname(input_pdb)
    
    try:
        # Create a dedicated directory for this structure
        struct_dir = os.path.join(target_dir, name)
        os.makedirs(struct_dir, exist_ok=True)
        
        # Define output path
        new_fp = os.path.join(struct_dir, name)
        
        # Define structure list for the command
        structs = f"{template_pdb} {input_pdb}"
        
        # Prepare and run the MUSTANG command
        command = f"/home/marmatt/Downloads/MUSTANG_v3.2.4/bin/mustang-3.2.4 -i {structs} -o {new_fp} -F fasta -s ON"
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

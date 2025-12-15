import os
import multiprocessing
import MDAnalysis as mda
import pandas as pd


class PDBChainExtractor:
    """
    A class to extract specific chains from PDB files and write them to new PDB files.
    
    This class implements parallel processing for efficient chain extraction,
    post-processing cleanup, and error handling.
    """
    
    def __init__(self, pdb_source_dir="Results/InterProPDBs", default_output_dir="Results/activation_segments/unaligned"):
        """
        Initialize the PDBChainExtractor.
        
        Args:
            pdb_source_dir (str): Directory containing source PDB files
            default_output_dir (str): Default directory for output PDB files
        """
        self.pdb_source_dir = pdb_source_dir
        self.default_output_dir = default_output_dir
    
    def get_pdb_path(self, pdb_id):
        """
        Generate a cross-platform PDB file path.
        
        Args:
            pdb_id (str): PDB ID
            
        Returns:
            str: Full path to the PDB file
        """
        return os.path.join(self.pdb_source_dir, f"{pdb_id}.pdb")
    
    def post_process_pdb(self, fname):
        """
        Remove unnecessary 'TER' lines from the PDB file to clean the output.
        
        Args:
            fname (str): Path to the PDB file to post-process
        """
        try:
            with open(fname, "r") as f:
                lines = f.readlines()

            # Keep only necessary TER lines (last two lines)
            final_lines = lines[-2:]
            no_ter = [line for line in lines if not line.startswith("TER") or line in final_lines]

            if len(no_ter) != len(lines):
                with open(fname, "w") as f:
                    f.writelines(no_ter)
        except Exception as e:
            print(f"Error post-processing {fname}: {e}")
    
    def extract_chain_from_pdb(self, pdb_file, chain_id):
        """
        Extract a specific chain from a PDB file.
        
        Args:
            pdb_file (str): Path to the PDB file
            chain_id (str): Chain ID to extract
            
        Returns:
            MDAnalysis.AtomGroup or None: Selected chain atoms or None if error
        """
        try:
            u = mda.Universe(pdb_file)
            chain = u.select_atoms(f"protein and chainID {chain_id}")
            if len(chain) == 0:
                return None
            return chain
        except Exception as e:
            print(f"Error loading {pdb_file}: {e}")
            return None
    
    def process_single_pdb_entry(self, entry):
        """
        Process a single PDB entry, extracting chains and writing output.
        
        Args:
            entry (tuple): (accession, chain_list, target_dir)
            
        Returns:
            bool: True if successful, False otherwise
        """
        accession, chain_list, target_dir = entry
        file_path = self.get_pdb_path(accession)

        if not os.path.exists(file_path):
            print(f"Skipping {file_path}: File not found")
            return False

        try:
            # Load PDB file once
            u = mda.Universe(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False

        success = True
        for chain_id in chain_list:
            try:
                chain = u.select_atoms(f"protein and chainID {chain_id}")
                if chain.n_atoms > 0:
                    output_pdb = os.path.join(target_dir, f"{accession}_{chain_id}.pdb")
                    with mda.Writer(output_pdb) as w:
                        w.write(chain)
                    self.post_process_pdb(output_pdb)
                    print(f"Processed {output_pdb}")
                else:
                    print(f"Chain {chain_id} not found in {file_path}")
                    success = False
            except Exception as e:
                print(f"Error processing {file_path} (Chain {chain_id}): {e}")
                success = False
        
        # Release memory
        del u
        return success
    
    def extract_chains_parallel(self, pdb_data, target_dir=None, max_workers=None):
        """
        Process multiple PDB files in parallel to extract chains.
        
        Args:
            pdb_data (pd.DataFrame): DataFrame with PDB information
            target_dir (str): Output directory for processed files
            max_workers (int): Maximum number of worker processes
        """
        if target_dir is None:
            target_dir = self.default_output_dir
        
        os.makedirs(target_dir, exist_ok=True)

        # Filter downloaded PDB entries
        pdb_entries = pdb_data[pdb_data['Downloaded'] == True].copy()

        # Convert chain list from semicolon-separated string to a list
        pdb_entries['Chain_list'] = pdb_entries['Chains'].apply(lambda x: x.split(';'))

        # Prepare input list for multiprocessing
        task_list = [(row['Accession'], row['Chain_list'], target_dir) for _, row in pdb_entries.iterrows()]

        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 10)

        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=10) as pool:
            pool.map(self.process_single_pdb_entry, task_list)

        print("All PDB processing completed.")
    
    def extract_single_chain(self, accession, chain_id, target_dir=None):
        """
        Extract a single chain from a PDB file.
        
        Args:
            accession (str): PDB accession code
            chain_id (str): Chain ID to extract
            target_dir (str): Output directory
            
        Returns:
            str or None: Path to output file if successful, None otherwise
        """
        if target_dir is None:
            target_dir = self.default_output_dir
        
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = self.get_pdb_path(accession)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        try:
            u = mda.Universe(file_path)
            chain = u.select_atoms(f"protein and chainID {chain_id}")
            
            if chain.n_atoms > 0:
                output_pdb = os.path.join(target_dir, f"{accession}_{chain_id}.pdb")
                with mda.Writer(output_pdb) as w:
                    w.write(chain)
                self.post_process_pdb(output_pdb)
                print(f"Processed {output_pdb}")
                return output_pdb
            else:
                print(f"Chain {chain_id} not found in {file_path}")
                return None
        except Exception as e:
            print(f"Error processing {file_path} (Chain {chain_id}): {e}")
            return None
    
    def load_and_mark_downloaded_pdbs(self, tsv_path):
        """
        Load PDB data from TSV file and mark which ones were downloaded.
        
        Args:
            tsv_path (str): Path to the TSV file with PDB information
            
        Returns:
            pd.DataFrame: DataFrame with PDB data and download status
        """
        try:
            # Load PDB data from TSV file
            pdb_data = pd.read_csv(tsv_path, sep="\t", header=0, engine='python')
            pdb_data['Accession'] = pdb_data['Accession'].str.upper()

            # Get downloaded PDBs
            if os.path.exists(self.pdb_source_dir):
                file_names = [os.path.splitext(f)[0] for f in os.listdir(self.pdb_source_dir) 
                             if os.path.isfile(os.path.join(self.pdb_source_dir, f))]
                
                # Convert to uppercase for correct matching
                pdb_raw = pd.DataFrame({"PDBs": file_names})
                pdb_raw['PDBs'] = pdb_raw['PDBs'].str.upper()

                # Mark downloaded PDBs
                pdb_data['Downloaded'] = pdb_data['Accession'].isin(pdb_raw['PDBs'])
            else:
                print(f"Source directory {self.pdb_source_dir} does not exist")
                pdb_data['Downloaded'] = False
            
            return pdb_data
        except Exception as e:
            print(f"Error loading pdb_data: {e}")
            return None 
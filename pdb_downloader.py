import os
import time
import requests
import multiprocessing
import concurrent.futures


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

    #def parallel_download(self, pdb_list, pdir=None, max_workers=None):
    def parallel_download(self, pdb_list, pdir=None):
        """
        Download PDB files in parallel using multiple threads.
        
        Args:
            pdb_list (list): List of PDB codes to download
            pdir (str): Directory to save the files
            max_workers (int): Maximum number of worker threads
        """
        # if max_workers is None:
        #     max_workers = min(20, multiprocessing.cpu_count() * 2)
        max_workers = min(20, multiprocessing.cpu_count() * 2)
        chunk_size = max(10, len(pdb_list) // max_workers)
        splited_pdb_lists = [pdb_list[i:i+chunk_size] for i in range(0, len(pdb_list), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_multiple, chunk, pdir) for chunk in splited_pdb_lists]
            # Wait for all downloads to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
    #Below should be done by logging step
    '''
    def get_download_stats(self, pdb_list, pdir=None):
        """
        Get statistics about downloaded vs failed PDB files.
        
        Args:
            pdb_list (list): List of PDB codes that were attempted
            pdir (str): Directory where files were saved
            
        Returns:
            dict: Dictionary with 'downloaded' and 'failed' counts
        """
        directory = os.path.abspath(pdir if pdir else self.default_dir)
        if not os.path.exists(directory):
            return {'downloaded': 0, 'failed': len(pdb_list)}
        
        existing_files = {os.path.splitext(f)[0] for f in os.listdir(directory) 
                         if os.path.isfile(os.path.join(directory, f))}
        
        downloaded = len([code for code in pdb_list if code in existing_files])
        failed = len(pdb_list) - downloaded
        
        return {'downloaded': downloaded, 'failed': failed} 
    '''
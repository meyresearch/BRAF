import os
import subprocess
from time import time
import logging
import traceback
from datetime import datetime
try:
    from .utilities import find_pdbs, find_pdbs_recursive, fname
except ImportError:  # pragma: no cover
    from utilities import find_pdbs, find_pdbs_recursive, fname


class AlignmentFoldMason:
    """
    A class to handle protein structure alignment using FoldMason.
    
    This class provides functionality for FoldMason-based multiple structural alignment,
    mirroring the interface of the MUSTANG-based Alignment class.
    """
    
    def __init__(self, foldmason_path="foldmason", log_file=None):
        """
        Initialize the AlignmentFoldMason class.
        
        Args:
            foldmason_path: Path to the FoldMason executable (default: "foldmason" assumes it's in PATH)
            log_file: Optional path to log file. If None, creates alignment_foldmason_TIMESTAMP.log
        """
        self.foldmason_path = foldmason_path
        
        # Set up logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"alignment_foldmason_{timestamp}.log"
        
        self.log_file = log_file
        self.logger = logging.getLogger(f"AlignmentFoldMason_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create file handler with detailed logging
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with less verbose output
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("="*80)
        self.logger.info(f"AlignmentFoldMason class initialized with log file: {log_file}")
        self.logger.info(f"FoldMason path: {foldmason_path}")
        self.logger.info("="*80)
    
    def run_foldmason_multiple(self, pdb_files, target_dir, out_name="msa", report_mode=1):
        """
        Run FoldMason to align multiple structures in a single run.
        
        Args:
            pdb_files: List of absolute PDB file paths (length >= 2)
            target_dir: Directory to write outputs
            out_name: Output prefix for generated files
            report_mode: FoldMason report mode (0: no report, 1: HTML report)
        
        Returns:
            Output prefix path (target_dir/out_name) or None if failed
        """
        self.logger.info("="*80)
        self.logger.info("Starting run_foldmason_multiple")
        self.logger.info("="*80)
        
        try:
            # Validate inputs
            self.logger.debug(f"Number of PDB files provided: {len(pdb_files) if pdb_files else 0}")
            if not pdb_files or len(pdb_files) < 2:
                self.logger.error("FoldMason multi requires at least two PDB files")
                self.logger.error(f"Received: {pdb_files}")
                return None
            
            self.logger.info(f"Processing {len(pdb_files)} PDB files for multi-structure alignment")
            
            # Log each PDB file
            for idx, pdb_file in enumerate(pdb_files, 1):
                self.logger.debug(f"  [{idx}/{len(pdb_files)}] {pdb_file}")
                if not os.path.exists(pdb_file):
                    self.logger.warning(f"  WARNING: File does not exist: {pdb_file}")
            
            # Create target directory
            self.logger.debug(f"Creating target directory: {target_dir}")
            os.makedirs(target_dir, exist_ok=True)
            self.logger.debug(f"Target directory created/verified successfully")
            
            # Create temporary directory for FoldMason intermediate files
            tmp_dir = os.path.join(target_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self.logger.debug(f"Temporary directory: {tmp_dir}")
            
            out_prefix = os.path.join(target_dir, out_name)
            self.logger.info(f"Output prefix: {out_prefix}")
            
            # Build argument list for FoldMason easy-msa
            # FoldMason easy-msa expects: foldmason easy-msa <pdb1> <pdb2> ... <output> <tmp> [options]
            args = [self.foldmason_path, 'easy-msa'] + pdb_files + [out_prefix, tmp_dir]
            
            # Add report mode if specified
            if report_mode is not None:
                args.extend(['--report-mode', str(report_mode)])
            
            self.logger.info(f"FoldMason executable: {self.foldmason_path}")
            self.logger.debug(f"Full command arguments ({len(args)} args):")
            self.logger.debug(f"  Executable: {args[0]}")
            self.logger.debug(f"  Command: {args[1]}")
            self.logger.debug(f"  Input files: {len(pdb_files)} files")
            for idx, pdb in enumerate(pdb_files, 1):
                self.logger.debug(f"    [{idx}] {pdb}")
            self.logger.debug(f"  Output prefix: {out_prefix}")
            self.logger.debug(f"  Temporary directory: {tmp_dir}")
            self.logger.debug(f"  Report mode: {report_mode}")
            
            # Log the full command for debugging
            full_cmd = ' '.join(args)
            self.logger.debug(f"Full command string (length={len(full_cmd)} chars):")
            self.logger.debug(f"{full_cmd}")
            
            self.logger.info("Executing FoldMason multi-structure alignment...")
            start_time = time()
            
            result = subprocess.run(args, shell=False, capture_output=True, text=True)
            
            end_time = time()
            elapsed = round(end_time - start_time, 3)
            
            self.logger.info(f"FoldMason execution completed in {elapsed} seconds")
            self.logger.debug(f"Return code: {result.returncode}")
            
            # Log stdout (FoldMason output)
            if result.stdout:
                self.logger.debug("FoldMason stdout:")
                self.logger.debug("-" * 40)
                for line in result.stdout.splitlines():
                    self.logger.debug(f"  {line}")
                self.logger.debug("-" * 40)
            else:
                self.logger.debug("FoldMason stdout: (empty)")
            
            # Log stderr (FoldMason errors/warnings)
            if result.stderr:
                self.logger.debug("FoldMason stderr:")
                self.logger.debug("-" * 40)
                for line in result.stderr.splitlines():
                    self.logger.debug(f"  {line}")
                self.logger.debug("-" * 40)
            else:
                self.logger.debug("FoldMason stderr: (empty)")
            
            # Check return code
            if result.returncode != 0:
                self.logger.error(f"FoldMason multi failed with return code: {result.returncode}")
                self.logger.error("Error details:")
                self.logger.error(f"  stderr: {result.stderr}")
                self.logger.error(f"  stdout: {result.stdout}")
                return None
            
            # Verify output files were created
            self.logger.debug("Checking for output files...")
            
            # List all files in target directory for verification
            try:
                all_files = os.listdir(target_dir)
                self.logger.debug(f"All files in target directory ({len(all_files)} files):")
                for f in all_files:
                    full_path = os.path.join(target_dir, f)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        self.logger.info(f"Output file: {f} ({size} bytes)")
                    else:
                        self.logger.debug(f"  Directory: {f}")
            except Exception as e:
                self.logger.warning(f"Could not list target directory: {e}")
            
            self.logger.info("="*80)
            self.logger.info(f"Successfully ran multi-structure FoldMason; outputs under prefix {out_prefix}")
            self.logger.info("="*80)
            return out_prefix
            
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error(f"CRITICAL ERROR in run_foldmason_multiple: {e}")
            self.logger.error("="*80)
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error("-" * 40)
            for line in traceback.format_exc().splitlines():
                self.logger.error(f"  {line}")
            self.logger.error("-" * 40)
            return None
    
    def process_foldmason_alignment_multi(self, pdb_path, target_dir, template_pdb=None, out_name="msa", report_mode=1):
        """
        Run a single multi-structure FoldMason alignment over all PDBs in a directory,
        optionally including a template structure.
        
        Args:
            pdb_path: Directory containing PDB files to include
            target_dir: Directory to save outputs
            template_pdb: Optional path to a template/reference PDB to include first
            out_name: Output prefix to use for FoldMason outputs
            report_mode: FoldMason report mode (0: no report, 1: HTML report)
        
        Returns:
            Output prefix path or None if failed
        """
        self.logger.info("="*80)
        self.logger.info("Starting process_foldmason_alignment_multi")
        self.logger.info("="*80)
        
        # Ensure the target directory exists
        self.logger.debug(f"Creating target directory: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        self.logger.info(f"Input directory: {pdb_path}")
        self.logger.info(f"Output directory: {target_dir}")
        if template_pdb:
            self.logger.info(f"Template PDB (included): {template_pdb}")
            self.logger.debug(f"Template exists: {os.path.exists(template_pdb)}")
        else:
            self.logger.info("No template PDB specified")
        
        # Gather PDB files
        self.logger.debug(f"Searching for PDB files in: {pdb_path}")
        pdbs = find_pdbs(pdb_path)
        
        # If no direct PDB files found, search recursively
        if not pdbs:
            self.logger.info(f"No PDB files found directly in {pdb_path}, searching subdirectories...")
            pdbs = find_pdbs_recursive(pdb_path)
        
        self.logger.info(f"Found {len(pdbs)} PDB files in {pdb_path}")
        
        if not pdbs:
            self.logger.error("No PDB files found to include in multi-structure alignment")
            return None
        
        # Optionally place template first
        if template_pdb:
            pdb_list = [template_pdb] + pdbs
            self.logger.debug(f"Template will be included as first structure")
        else:
            pdb_list = pdbs
        
        self.logger.info(f"Running FoldMason multi on {len(pdb_list)} structures")
        t1 = time()
        out_prefix = self.run_foldmason_multiple(pdb_list, target_dir, out_name=out_name, report_mode=report_mode)
        t2 = time()
        elapsed = round(t2 - t1, 3)
        
        self.logger.info("="*80)
        if out_prefix:
            self.logger.info(f"Multi-structure alignment complete in {elapsed} seconds")
            self.logger.info(f"Output prefix: {out_prefix}")
        else:
            self.logger.error(f"Multi-structure alignment FAILED after {elapsed} seconds")
        self.logger.info("="*80)
        
        return out_prefix


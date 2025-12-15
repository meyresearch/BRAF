"""
Dunbrack Kinase Conformation Assignment Module

This module provides classes for assigning kinase conformational states
to protein structures using the KinCore tool from the Dunbrack Lab.

The KinCore tool classifies kinase structures based on:
- DFG (Asp-Phe-Gly) motif conformation (in/out)
- Chelix (Regulatory spine) position
- Activation loop conformation

Note: This module assumes KinCore is already installed.
"""

import os
import sys
import subprocess
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class ConformationAssigner:
    """
    Class for assigning kinase conformations using KinCore.
    """
    
    # Conformation definitions
    CONFORMATIONS = {
        'DFGin_Chelix_in': 'Active conformation (Type I)',
        'DFGin_Chelix_out': 'Inactive conformation (Type I/Type II-like)',
        'DFGout_Chelix_in': 'DFG-out conformation (Type II)',
        'DFGout_Chelix_out': 'DFG-out inactive conformation',
        'unknown': 'Unable to classify'
    }
    
    def __init__(self, kincore_dir: str = "kincore_tool", log_file: Optional[str] = None,
                 kincore_python: Optional[str] = None):
        """
        Initialize the conformation assigner.
        
        Args:
            kincore_dir: Directory where KinCore is installed
            log_file: Optional log file path
            kincore_python: Path to Python interpreter in kincore environment
                           (default: /home/marmatt/miniforge3/envs/kincore-standalone/bin/python)
            
        Note:
            The script looks for 'kinase_state.py' in the kincore_dir.
            KinCore requires its conda environment to be available.
            When KinCore fails to classify a structure, it will be marked as 'failed'.
        """
        self.kincore_dir = Path(kincore_dir)
        self.kincore_script = self.kincore_dir / "kinase_state.py"
        
        # Use kincore environment's Python by default
        if kincore_python is None:
            self.kincore_python = "/home/marmatt/miniforge3/envs/kincore-standalone/bin/python"
        else:
            self.kincore_python = kincore_python
            
        self.logger = self._setup_logger(log_file)
        self.results = None
        
    def _setup_logger(self, log_file: Optional[str] = None) -> logging.Logger:
        """Set up logging for the assigner."""
        logger = logging.getLogger("ConformationAssigner")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if specified
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        return logger
    
    def check_kincore_installed(self) -> bool:
        """
        Check if KinCore is installed and ready to use.
        
        Returns:
            True if KinCore is available, False otherwise
        """
        if not self.kincore_script.exists():
            self.logger.error(f"KinCore script not found at {self.kincore_script}")
            self.logger.error("Please ensure KinCore is installed and kincore_dir points to the correct location")
            return False
        return True
    
    def find_pdb_files(self, directory: str) -> List[Path]:
        """
        Find all PDB files in a directory.
        
        Args:
            directory: Directory to search for PDB files
            
        Returns:
            List of PDB file paths
        """
        pdb_dir = Path(directory)
        if not pdb_dir.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []
            
        pdb_files = list(pdb_dir.glob("*.pdb"))
        self.logger.info(f"Found {len(pdb_files)} PDB files in {directory}")
        return pdb_files
    
    def assign_single_structure(self, pdb_file: str) -> Optional[Dict]:
        """
        Assign conformation to a single PDB structure.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Dictionary with conformation assignment results, or None if failed
        """
        if not self.check_kincore_installed():
            return None
            
        try:
            # Convert to absolute path and escape for shell
            abs_pdb_file = str(Path(pdb_file).absolute())
            
            # Run KinCore using conda environment activation to get proper PATH
            # The 'True' argument tells KinCore to align to HMMs and auto-identify conserved residues
            # Use explicit quoting to ensure arguments are passed correctly
            conda_cmd = (
                f"source /home/marmatt/miniforge3/etc/profile.d/conda.sh && "
                f"conda activate kincore-standalone && "
                f"python '{self.kincore_script}' '{abs_pdb_file}' True"
            )
            
            result = subprocess.run(
                conda_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout since alignment is slower
                cwd=str(self.kincore_dir)  # Run from kincore directory
            )
            
            if result.returncode != 0:
                self.logger.warning(f"KinCore failed for {pdb_file}: {result.stderr}")
                return None
                
            # Parse the output
            output = result.stdout
            conformation_data = self._parse_kincore_output(output, pdb_file)
            
            return conformation_data
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"KinCore timed out for {pdb_file}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {pdb_file}: {e}")
            return None
    
    def _parse_kincore_output(self, output: str, pdb_file: str) -> Dict:
        """
        Parse KinCore output to extract conformation and ligand information.
        
        Args:
            output: KinCore output string
            pdb_file: Original PDB file path
            
        Returns:
            Dictionary with parsed conformation data and ligand information
        """
        # Initialize result dictionary
        result = {
            'pdb_file': Path(pdb_file).name,
            'pdb_code': Path(pdb_file).stem,
            'dfg_conformation': 'unknown',
            'chelix_conformation': 'unknown',
            'overall_conformation': 'unknown',
            'conformation_description': 'unknown',
            'ligand': 'unknown',
            'ligand_label': 'unknown',
            'raw_output': output
        }
        
        # Parse output lines
        lines = output.strip().split('\n')
        
        # Find the header line and data line in tabular output
        for i, line in enumerate(lines):
            # Look for the header line with column names
            if 'Ligand' in line and 'Ligand_label' in line:
                # Next line(s) should contain the data
                if i + 1 < len(lines):
                    data_line = lines[i + 1]
                    # Split by whitespace and parse
                    parts = data_line.split()
                    
                    # Try to find Ligand and Ligand_label in the data
                    # The format is: ... Ligand Ligand_label ...
                    try:
                        ligand_idx = line.split().index('Ligand')
                        ligand_label_idx = line.split().index('Ligand_label')
                        
                        if len(parts) > ligand_idx:
                            result['ligand'] = parts[ligand_idx]
                        if len(parts) > ligand_label_idx:
                            result['ligand_label'] = parts[ligand_label_idx]
                    except (ValueError, IndexError):
                        pass  # Keep as 'unknown' if parsing fails
        
        # Parse conformation information from text
        for line in lines:
            line_lower = line.lower()
            
            # Check for DFG conformation
            if 'dfg' in line_lower:
                if 'in' in line_lower and 'out' not in line_lower:
                    result['dfg_conformation'] = 'in'
                elif 'out' in line_lower:
                    result['dfg_conformation'] = 'out'
                    
            # Check for Chelix conformation
            if 'chelix' in line_lower or 'c-helix' in line_lower or 'regulatory spine' in line_lower:
                if 'in' in line_lower and 'out' not in line_lower:
                    result['chelix_conformation'] = 'in'
                elif 'out' in line_lower:
                    result['chelix_conformation'] = 'out'
        
        # Determine overall conformation
        dfg = result['dfg_conformation']
        chelix = result['chelix_conformation']
        
        if dfg != 'unknown' and chelix != 'unknown':
            conf_key = f"DFG{dfg}_Chelix_{chelix}"
            result['overall_conformation'] = conf_key
            result['conformation_description'] = self.CONFORMATIONS.get(conf_key, 'unknown')
        
        return result
    
    def assign_directory(self, input_dir: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Assign conformations to all PDB structures in a directory.
        
        Args:
            input_dir: Directory containing PDB files
            output_csv: Optional path to save results as CSV
            
        Returns:
            DataFrame with conformation assignments for all structures
        """
        self.logger.info(f"Starting conformation assignment for structures in {input_dir}")
        
        # Find all PDB files
        pdb_files = self.find_pdb_files(input_dir)
        
        if not pdb_files:
            self.logger.error("No PDB files found")
            return pd.DataFrame()
        
        # Process each structure
        results_list = []
        failed_count = 0
        
        self.logger.info(f"Processing {len(pdb_files)} structures...")
        
        for pdb_file in tqdm(pdb_files, desc="Assigning conformations"):
            result = self.assign_single_structure(str(pdb_file))
            
            if result:
                results_list.append(result)
            else:
                failed_count += 1
                # Add entry for failed structures
                results_list.append({
                    'pdb_file': pdb_file.name,
                    'pdb_code': pdb_file.stem,
                    'dfg_conformation': 'failed',
                    'chelix_conformation': 'failed',
                    'overall_conformation': 'failed',
                    'conformation_description': 'Analysis failed',
                    'raw_output': ''
                })
        
        # Create DataFrame
        self.results = pd.DataFrame(results_list)
        
        # Save to CSV if requested
        if output_csv:
            self.results.to_csv(output_csv, index=False)
            self.logger.info(f"Results saved to {output_csv}")
        
        # Print summary
        self.logger.info("\n" + "="*80)
        self.logger.info("CONFORMATION ASSIGNMENT SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total structures: {len(pdb_files)}")
        self.logger.info(f"Successfully analyzed: {len(pdb_files) - failed_count}")
        self.logger.info(f"Failed: {failed_count}")
        
        if not self.results.empty:
            self.logger.info("\nConformation distribution:")
            conf_counts = self.results['overall_conformation'].value_counts()
            for conf, count in conf_counts.items():
                self.logger.info(f"  {conf}: {count}")
        
        return self.results
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of conformation assignments.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.results is None or self.results.empty:
            self.logger.warning("No results available. Run assign_directory first.")
            return {}
        
        stats = {
            'total_structures': len(self.results),
            'dfg_in': len(self.results[self.results['dfg_conformation'] == 'in']),
            'dfg_out': len(self.results[self.results['dfg_conformation'] == 'out']),
            'chelix_in': len(self.results[self.results['chelix_conformation'] == 'in']),
            'chelix_out': len(self.results[self.results['chelix_conformation'] == 'out']),
            'active_type_I': len(self.results[self.results['overall_conformation'] == 'DFGin_Chelix_in']),
            'inactive': len(self.results[self.results['overall_conformation'] == 'DFGin_Chelix_out']),
            'dfg_out_type_II': len(self.results[self.results['overall_conformation'] == 'DFGout_Chelix_in']),
            'unknown': len(self.results[self.results['overall_conformation'] == 'unknown']),
            'failed': len(self.results[self.results['overall_conformation'] == 'failed'])
        }
        
        return stats


class DunbrackWorkflow:
    """
    Complete workflow for Dunbrack kinase conformation assignment.
    """
    
    def __init__(self, input_dir: str, output_dir: str = "Results/dunbrack_assignments",
                 kincore_dir: str = "/home/marmatt/Documents/Kincore-standalone",
                 kincore_python: Optional[str] = None):
        """
        Initialize the workflow.
        
        Args:
            input_dir: Directory containing PDB structures to analyze
            output_dir: Directory to save results
            kincore_dir: Directory where KinCore is installed
            kincore_python: Path to Python in kincore environment (auto-detected if None)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.kincore_dir = Path(kincore_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize assigner with kincore environment Python
        self.assigner = ConformationAssigner(
            str(self.kincore_dir),
            log_file=str(self.output_dir / "dunbrack_assignment.log"),
            kincore_python=kincore_python
        )
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the workflow."""
        logger = logging.getLogger("DunbrackWorkflow")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def run(self, output_csv: str = "conformation_assignments.csv") -> pd.DataFrame:
        """
        Run the complete Dunbrack conformation assignment workflow.
        
        Args:
            output_csv: Name of output CSV file
            
        Returns:
            DataFrame with conformation assignments
        """
        self.logger.info("="*80)
        self.logger.info("DUNBRACK KINASE CONFORMATION ASSIGNMENT WORKFLOW")
        self.logger.info("="*80)
        
        # Assign conformations
        self.logger.info(f"\nStep 1: Assigning conformations to structures in {self.input_dir}...")
        output_path = self.output_dir / output_csv
        results = self.assigner.assign_directory(str(self.input_dir), str(output_path))
        
        # Generate summary
        self.logger.info("\nStep 2: Generating summary statistics...")
        stats = self.assigner.get_summary_statistics()
        
        if stats:
            self.logger.info("\n" + "="*80)
            self.logger.info("SUMMARY STATISTICS")
            self.logger.info("="*80)
            for key, value in stats.items():
                percentage = (value / stats['total_structures'] * 100) if stats['total_structures'] > 0 else 0
                self.logger.info(f"{key:.<40} {value:>5} ({percentage:.1f}%)")
        
        self.logger.info("\n" + "="*80)
        self.logger.info("WORKFLOW COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Results saved to: {output_path}")
        
        return results


# Convenience function for quick usage
def assign_conformations(input_dir: str, output_dir: str = "Results/dunbrack_assignments") -> pd.DataFrame:
    """
    Convenience function to run the complete Dunbrack conformation assignment workflow.
    
    Args:
        input_dir: Directory containing PDB structures to analyze
        output_dir: Directory to save results
        
    Returns:
        DataFrame with conformation assignments
    """
    workflow = DunbrackWorkflow(input_dir, output_dir)
    return workflow.run()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Assign kinase conformations using KinCore')
    parser.add_argument('input_dir', help='Directory containing PDB files')
    parser.add_argument('--output-dir', default='Results/dunbrack_assignments',
                       help='Output directory for results')
    parser.add_argument('--kincore-dir', default='/home/marmatt/Documents/Kincore-standalone',
                       help='Directory where KinCore is installed')
    
    args = parser.parse_args()
    
    workflow = DunbrackWorkflow(args.input_dir, args.output_dir, args.kincore_dir)
    results = workflow.run()
    
    print(f"\nProcessed {len(results)} structures")


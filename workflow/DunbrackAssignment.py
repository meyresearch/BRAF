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
import re
import glob
import shutil
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    # ------------------------------------------------------------------
    # Plotting / post-processing helpers (to keep notebooks concise)
    # ------------------------------------------------------------------
    @staticmethod
    def extract_dunbrack_state(raw_output: str) -> str:
        """
        Extract the Dihedral_label (Dunbrack state) from KinCore raw output.
        """
        if pd.isna(raw_output) or 'not a protein kinase' in str(raw_output):
            return 'unknown'

        # Look for patterns like BLBminus, BLAplus, BLBplus, BBAminus, etc.
        match = re.search(
            r'\b(BL[AB][mp][il][nu][us]s?|BB[AB][mp][il][nu][us]s?|AB[AB][mp][il][nu][us]s?)\b',
            str(raw_output),
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

        # Fallback: "Dihedral_label <STATE>"
        match = re.search(r'Dihedral_label\s+(\S+)', str(raw_output))
        if match:
            return match.group(1)

        return 'unknown'

    @staticmethod
    def classify_activation(conformation_description: str) -> str:
        """
        Map KinCore conformation_description to a simple activation label.
        """
        if pd.isna(conformation_description) or conformation_description == 'unknown':
            return 'Unknown'
        desc = str(conformation_description)
        if 'Active' in desc:
            return 'Active'
        if 'inactive' in desc.lower():
            return 'Inactive'
        return 'Unknown'

    @classmethod
    def load_and_annotate_assignments(cls, assignments_csv: str) -> pd.DataFrame:
        """
        Load KinCore assignment CSV and add:
        - dunbrack_state (parsed from raw_output)
        - activation_state (parsed from conformation_description)
        """
        df = pd.read_csv(assignments_csv)
        if 'raw_output' in df.columns:
            df['dunbrack_state'] = df['raw_output'].apply(cls.extract_dunbrack_state)
        else:
            df['dunbrack_state'] = 'unknown'

        if 'conformation_description' in df.columns:
            df['activation_state'] = df['conformation_description'].apply(cls.classify_activation)
        else:
            df['activation_state'] = 'Unknown'

        return df

    @classmethod
    def plot_conformation_distribution(
        cls,
        assignments_csv: str = "Results/dunbrack_assignments/kinase_conformation_assignments.csv",
        output_png: str = "Results/dunbrack_assignments/conformation_distribution.png",
        show: bool = True,
        print_dunbrack_summary: bool = True,
    ) -> Dict[str, int]:
        """
        Reproduce the multi-panel distribution plot used in the notebook, and save it.

        Returns:
            dict: dunbrack_state -> count
        """
        results = cls.load_and_annotate_assignments(assignments_csv)

        # Create figure with subplots (2x3 layout)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Plot 1: Overall conformation distribution
        conf_counts = results['overall_conformation'].value_counts() if 'overall_conformation' in results.columns else pd.Series(dtype=int)
        axes[0].bar(range(len(conf_counts)), conf_counts.values, color='steelblue')
        axes[0].set_xticks(range(len(conf_counts)))
        axes[0].set_xticklabels(conf_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Overall Conformation Distribution')
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: DFG motif distribution
        dfg_counts = results['dfg_conformation'].value_counts() if 'dfg_conformation' in results.columns else pd.Series(dtype=int)
        colors = ['#2ecc71' if x == 'in' else '#e74c3c' if x == 'out' else '#95a5a6' for x in dfg_counts.index]
        axes[1].bar(range(len(dfg_counts)), dfg_counts.values, color=colors)
        axes[1].set_xticks(range(len(dfg_counts)))
        axes[1].set_xticklabels(dfg_counts.index, rotation=45, ha='right')
        axes[1].set_ylabel('Count')
        axes[1].set_title('DFG Motif Conformation')
        axes[1].grid(axis='y', alpha=0.3)

        # Plot 3: C-helix distribution
        chelix_counts = results['chelix_conformation'].value_counts() if 'chelix_conformation' in results.columns else pd.Series(dtype=int)
        colors = ['#3498db' if x == 'in' else '#f39c12' if x == 'out' else '#95a5a6' for x in chelix_counts.index]
        axes[2].bar(range(len(chelix_counts)), chelix_counts.values, color=colors)
        axes[2].set_xticks(range(len(chelix_counts)))
        axes[2].set_xticklabels(chelix_counts.index, rotation=45, ha='right')
        axes[2].set_ylabel('Count')
        axes[2].set_title('C-helix Conformation')
        axes[2].grid(axis='y', alpha=0.3)

        # Plot 4: Dunbrack States Distribution (BLBplus, BLAminus, etc.)
        dunbrack_counts = results['dunbrack_state'].value_counts()
        dunbrack_colors = {
            'BLAminus': '#2ecc71',   # Green - active-like
            'BLAplus': '#27ae60',    # Dark green
            'BLBminus': '#3498db',   # Blue
            'BLBplus': '#2980b9',    # Dark blue
            'BBAminus': '#e74c3c',   # Red - inactive-like
            'BBAplus': '#c0392b',    # Dark red
            'ABAminus': '#9b59b6',   # Purple
            'unknown': '#95a5a6'     # Gray
        }
        colors = [dunbrack_colors.get(x, '#7f8c8d') for x in dunbrack_counts.index]
        axes[3].bar(range(len(dunbrack_counts)), dunbrack_counts.values, color=colors, edgecolor='black', linewidth=0.5)
        axes[3].set_xticks(range(len(dunbrack_counts)))
        axes[3].set_xticklabels(dunbrack_counts.index, rotation=45, ha='right', fontsize=10)
        axes[3].set_ylabel('Count')
        axes[3].set_title('Dunbrack States Distribution\n(Dihedral Label)')
        axes[3].grid(axis='y', alpha=0.3)
        for i, (count, label) in enumerate(zip(dunbrack_counts.values, dunbrack_counts.index)):
            axes[3].text(i, count + 1, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Plot 5: Active/Inactive/Unknown Distribution
        activation_counts = results['activation_state'].value_counts()
        order = ['Active', 'Inactive', 'Unknown']
        activation_counts = activation_counts.reindex([x for x in order if x in activation_counts.index])
        activation_colors = {'Active': '#2ecc71', 'Inactive': '#e74c3c', 'Unknown': '#95a5a6'}
        colors = [activation_colors.get(x, '#7f8c8d') for x in activation_counts.index]
        bars = axes[4].bar(range(len(activation_counts)), activation_counts.values, color=colors, edgecolor='black', linewidth=1)
        axes[4].set_xticks(range(len(activation_counts)))
        axes[4].set_xticklabels(activation_counts.index, rotation=0, fontsize=11)
        axes[4].set_ylabel('Count')
        axes[4].set_title('Activation State Distribution')
        axes[4].grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, activation_counts.values):
            pct = count / len(results) * 100 if len(results) else 0.0
            axes[4].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f'{count}\n({pct:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

        # Hide the 6th subplot (empty)
        axes[5].axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png), exist_ok=True)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        if print_dunbrack_summary:
            print("\n=== Dunbrack States Summary ===")
            for state, count in dunbrack_counts.items():
                pct = count / len(results) * 100 if len(results) else 0.0
                print(f"  {state}: {count} ({pct:.1f}%)")
            print(f"\nVisualization saved to: {output_png}")

        return dunbrack_counts.to_dict()

    # ------------------------------------------------------------------
    # Notebook-facing helpers (non-KLIFS logic only)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_kincore_results_for_report(df: pd.DataFrame) -> pd.DataFrame:
        """Make a KinCore-like DataFrame from in-memory results or saved CSV."""
        out = df.copy()

        rename_map = {
            "pdb": "pdb_code",
            "pdb_id": "pdb_code",
            "chain": "chain_id",
            "chainID": "chain_id",
        }
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out = out.rename(columns={k: v})

        if "pdb_file" not in out.columns:
            if {"pdb_code", "chain_id"}.issubset(out.columns):
                out["pdb_file"] = (
                    out["pdb_code"].astype(str).str.upper() + "_" + out["chain_id"].astype(str) + ".pdb"
                )
            elif "pdb_code" in out.columns:
                out["pdb_file"] = out["pdb_code"].astype(str).str.upper() + ".pdb"

        if "chain_id" not in out.columns and "pdb_file" in out.columns:
            out["chain_id"] = (
                out["pdb_file"].astype(str).str.extract(r"^[^_]+_([A-Za-z0-9])\.", expand=False).fillna("")
            )

        return out

    @classmethod
    def print_conformation_and_ligand_report(
        cls,
        results_df: pd.DataFrame,
        *,
        output_true_ligand_csv: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Print conformation + ligand report for a KinCore-like results table.
        """
        results_df = cls.normalize_kincore_results_for_report(results_df)

        print(f"\n{'='*80}")
        print("CONFORMATION ASSIGNMENT RESULTS")
        print(f"{'='*80}\n")
        try:
            from IPython.display import display
            display(results_df.head(10))
        except Exception:
            print(results_df.head(10).to_string(index=False))

        if "overall_conformation" in results_df.columns:
            print("\nConformation Distribution:")
            print(results_df["overall_conformation"].value_counts())
        if "dfg_conformation" in results_df.columns:
            print("\nDFG Motif Distribution:")
            print(results_df["dfg_conformation"].value_counts())
        if "chelix_conformation" in results_df.columns:
            print("\nC-helix Distribution:")
            print(results_df["chelix_conformation"].value_counts())

        if "ligand" not in results_df.columns:
            print("\n(no 'ligand' column found; skipping ligand report)")
            return None

        print("\n" + "=" * 80)
        print("LIGAND INFORMATION")
        print("=" * 80)

        print("\nRaw ligand distribution (KinCore output):")
        raw_ligand_counts = results_df["ligand"].value_counts()
        print(raw_ligand_counts)
        print(f"\nTotal unique ligand strings (raw): {len(raw_ligand_counts)}")

        ligand_str = results_df["ligand"].astype(str)
        raw_has_ligand = ~ligand_str.isin(["No_ligand", "unknown", "nan", "None", ""])
        print(f"Structures with raw ligand (excluding 'unknown'): {int(raw_has_ligand.sum())}")
        print(f"Structures without raw ligand: {int((~raw_has_ligand).sum())}")

        ligand_label = results_df.get("ligand_label", None)
        if ligand_label is not None:
            aa_artifact = ligand_label.astype(str).str.contains("amino acid", case=False, na=False)
        else:
            aa_artifact = False

        aa_like_ligand = ligand_str.str.match(r"^[A-Z]{3}:\d+$", na=False)
        true_has_ligand = raw_has_ligand & (~aa_artifact) & (~aa_like_ligand)
        true_ligand_counts = results_df.loc[true_has_ligand, "ligand"].value_counts()

        print("\nTrue ligand distribution (excluding amino-acid artifacts):")
        print(true_ligand_counts)
        print(f"\nStructures with true ligand: {int(true_has_ligand.sum())}")
        print(f"Structures without true ligand: {int((~true_has_ligand).sum())}")

        if ligand_label is not None and "pdb_file" in results_df.columns:
            artifact_rows = results_df[raw_has_ligand & aa_artifact][["pdb_file", "ligand", "ligand_label"]]
            if len(artifact_rows):
                print("\nAmino-acid ligand artifacts (treat as apo):")
                print(artifact_rows.to_string(index=False))

        ligand_report = results_df.copy()
        if "pdb_file" in ligand_report.columns:
            ligand_report["pdb_file"] = ligand_report["pdb_file"].astype(str)
            if "chain_id" not in ligand_report.columns:
                ligand_report["chain_id"] = (
                    ligand_report["pdb_file"].str.extract(r"^[^_]+_([A-Za-z0-9])\.", expand=False).fillna("")
                )

        ligand_report["raw_has_ligand"] = raw_has_ligand.values
        ligand_report["aa_artifact"] = (aa_artifact.values if hasattr(aa_artifact, "values") else aa_artifact)
        ligand_report["true_has_ligand"] = true_has_ligand.values

        cols = ["pdb_file", "chain_id", "ligand", "true_has_ligand"]
        if "pdb_code" in ligand_report.columns:
            cols.insert(1, "pdb_code")
        if "ligand_label" in ligand_report.columns:
            cols.insert(cols.index("ligand") + 1, "ligand_label")
        cols = [c for c in cols if c in ligand_report.columns]

        print("\nChains with TRUE ligand bound (excluding amino-acid artifacts):")
        true_rows = ligand_report[ligand_report["true_has_ligand"]]
        try:
            from IPython.display import display
            if len(true_rows) == 0:
                print("  (none)")
            else:
                display(true_rows[cols].sort_values(["pdb_file"]))
            print("\nAll chains (raw ligand field; first 50 rows):")
            display(ligand_report[cols].sort_values(["pdb_file"]).head(50))
        except Exception:
            if len(true_rows) == 0:
                print("  (none)")
            else:
                print(true_rows[cols].sort_values(["pdb_file"]).to_string(index=False))
            print("\nAll chains (raw ligand field; first 50 rows):")
            print(ligand_report[cols].sort_values(["pdb_file"]).head(50).to_string(index=False))
        print("(showing first 50 rows)")

        print("\nTop 10 true ligands (excluding apo + amino-acid artifacts):")
        for lig, count in true_ligand_counts.head(10).items():
            print(f"  {lig:.<20} {count:>4}")

        if output_true_ligand_csv:
            out_dir = os.path.dirname(output_true_ligand_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_cols = [
                c
                for c in [
                    "pdb_code",
                    "chain_id",
                    "pdb_file",
                    "ligand",
                    "ligand_label",
                    "raw_has_ligand",
                    "aa_artifact",
                    "true_has_ligand",
                ]
                if c in ligand_report.columns
            ]
            ligand_report[out_cols].to_csv(output_true_ligand_csv, index=False)
            print(f"\nSaved true-ligand classification to: {output_true_ligand_csv}")

        return ligand_report

    @classmethod
    def report_from_assignments_csv(
        cls,
        assignments_csv: str = "Results/dunbrack_assignments/kinase_conformation_assignments.csv",
        true_ligand_csv: str = "Results/dunbrack_assignments/true_ligand_classification.csv",
    ) -> pd.DataFrame:
        """Load assignment CSV and print/write the conformation + ligand report."""
        if not os.path.exists(assignments_csv):
            raise FileNotFoundError(
                f"KinCore assignments CSV not found: {assignments_csv}. "
                "Run the Dunbrack/KinCore assignment first."
            )
        results_csv = pd.read_csv(assignments_csv)
        out = cls.print_conformation_and_ligand_report(
            results_csv,
            output_true_ligand_csv=true_ligand_csv,
        )
        return out if out is not None else results_csv

    @staticmethod
    def _label_has_type1(ligand_label: str) -> bool:
        tokens = [t.strip() for t in str(ligand_label).split(",") if t.strip()]
        return "Type1" in tokens

    @classmethod
    def get_type1_chains_in_dir(
        cls,
        pdb_dir: str = "Results/activation_segments/motif_filtered+ligands",
        true_ligand_csv: str = "Results/dunbrack_assignments/true_ligand_classification.csv",
    ) -> pd.DataFrame:
        """
        Return Type1 true-ligand chain records whose PDB files exist in `pdb_dir`.
        """
        pdb_paths = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
        if not pdb_paths:
            raise FileNotFoundError(
                f"No .pdb files found in: {pdb_dir}. "
                "Double-check the path (it is relative to the notebook working directory)."
            )
        pdb_files_in_dir = {os.path.basename(p) for p in pdb_paths}

        if not os.path.exists(true_ligand_csv):
            raise FileNotFoundError(
                f"Required KinCore type table not found: {true_ligand_csv}. "
                "Generate it from report_from_assignments_csv first."
            )

        lig_df = pd.read_csv(true_ligand_csv)
        required_cols = {"pdb_code", "chain_id", "pdb_file", "ligand_label", "true_has_ligand"}
        missing = required_cols - set(lig_df.columns)
        if missing:
            raise ValueError(f"{true_ligand_csv} is missing required columns: {sorted(missing)}")

        sel = lig_df[
            lig_df["pdb_file"].astype(str).isin(pdb_files_in_dir)
            & lig_df["true_has_ligand"].astype(bool)
            & lig_df["ligand_label"].map(cls._label_has_type1)
        ].copy()
        return sel

    @classmethod
    def summarize_type1_chains(
        cls,
        pdb_dir: str = "Results/activation_segments/motif_filtered+ligands",
        true_ligand_csv: str = "Results/dunbrack_assignments/true_ligand_classification.csv",
    ) -> pd.DataFrame:
        """Print and return Type1 chain selection in the motif+ligand directory."""
        pdb_paths = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
        pdb_files_in_dir = {os.path.basename(p) for p in pdb_paths}
        sel = cls.get_type1_chains_in_dir(pdb_dir=pdb_dir, true_ligand_csv=true_ligand_csv)
        pdb_list = sorted(sel["pdb_file"].astype(str).str[:4].str.upper().unique().tolist())
        print(f"PDB chains in {pdb_dir}: {len(pdb_files_in_dir)}")
        print(f"Type1 chains found in {true_ligand_csv}: {len(sel)}")
        print(f"Unique Type1 PDB IDs: {len(pdb_list)}")
        return sel

    @staticmethod
    def copy_atp_analogue_structures(
        pdb_dir: str = "Results/activation_segments/motif_filtered+ligands",
        true_ligand_csv: str = "Results/dunbrack_assignments/true_ligand_classification.csv",
        cluster_label_files: Optional[List[str]] = None,
        combined_out_dir: str = "Results/activation_segments/ATP+analogues",
        atp_only_codes: Optional[set] = None,
        atp_analogue_codes: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Copy all ATP/analogue chain PDBs into one folder and write a manifest.
        """
        if atp_only_codes is None:
            atp_only_codes = {"ATP"}
        if atp_analogue_codes is None:
            atp_analogue_codes = {"ANP", "ACP", "AGS", "ADP", "AMP", "ADN"}

        if cluster_label_files is None:
            cluster_label_files = [
                p for p in ["labels.csv", "filtered_labels.csv", "corr_filtered_labels.csv"] if os.path.exists(p)
            ]

        if not os.path.exists(true_ligand_csv):
            raise FileNotFoundError(f"Required file not found: {true_ligand_csv}")

        lig_df = pd.read_csv(true_ligand_csv)
        lig_df["lig_code"] = lig_df["ligand"].astype(str).str.extract(r"^([A-Z0-9]+):", expand=False)
        pdb_files_in_dir = {os.path.basename(p) for p in glob.glob(os.path.join(pdb_dir, "*.pdb"))}

        all_atp_codes = set(atp_only_codes) | set(atp_analogue_codes)
        atp_all_sel = lig_df[
            lig_df["pdb_file"].astype(str).isin(pdb_files_in_dir) & lig_df["lig_code"].isin(all_atp_codes)
        ].copy()

        atp_all_chain_files = sorted(atp_all_sel["pdb_file"].astype(str).unique().tolist())
        print(f"Total chains with ATP or analogues: {len(atp_all_chain_files)}")
        print(f"Ligand codes included: {sorted(all_atp_codes)}")

        os.makedirs(combined_out_dir, exist_ok=True)
        copied = 0
        missing_src = 0
        for f in atp_all_chain_files:
            src = os.path.join(pdb_dir, f)
            dst = os.path.join(combined_out_dir, f)
            if not os.path.exists(src):
                missing_src += 1
                continue
            shutil.copy2(src, dst)
            copied += 1
        print(f"Copied: {copied} -> {combined_out_dir}")
        if missing_src:
            print(f"WARNING: {missing_src} files were missing in source dir (not copied)")

        manifest_df = atp_all_sel.copy()
        if len(manifest_df):
            manifest_df["structure"] = (
                manifest_df["pdb_file"].astype(str).str.replace(".pdb", "", regex=False).str.upper()
            )
            for clf in cluster_label_files:
                cdf = pd.read_csv(clf)
                if not {"structure", "label"}.issubset(cdf.columns):
                    continue
                col_name = f"cluster_label__{os.path.splitext(os.path.basename(clf))[0]}"
                cdf = cdf[["structure", "label"]].rename(columns={"label": col_name}).copy()
                cdf["structure"] = cdf["structure"].astype(str).str.upper()
                manifest_df = manifest_df.merge(cdf, on="structure", how="left")
            manifest_df.drop(columns=["structure"], inplace=True, errors="ignore")

        manifest_csv = os.path.join(combined_out_dir, "atp_analogues_manifest.csv")
        manifest_df.to_csv(manifest_csv, index=False)
        print(f"Saved manifest: {manifest_csv}")
        return manifest_df

    @staticmethod
    def build_output_paths(
        output_dir: str = "Results/dunbrack_assignments",
        assignments_filename: str = "kinase_conformation_assignments.csv",
        true_ligand_filename: str = "true_ligand_classification.csv",
        conformation_plot_filename: str = "conformation_distribution.png",
    ) -> Dict[str, str]:
        """
        Build canonical Dunbrack artifact paths for notebook use.
        """
        return {
            "output_dir": output_dir,
            "assignments_csv": os.path.join(output_dir, assignments_filename),
            "true_ligand_csv": os.path.join(output_dir, true_ligand_filename),
            "conformation_plot_png": os.path.join(output_dir, conformation_plot_filename),
        }

    @classmethod
    def ensure_assignments_cached(
        cls,
        *,
        input_dir: str = "Results/activation_segments/motif_filtered+ligands/",
        output_dir: str = "Results/dunbrack_assignments",
        kincore_dir: str = "/home/marmatt/Documents/Kincore-standalone",
        assignments_filename: str = "kinase_conformation_assignments.csv",
        force: bool = False,
    ) -> str:
        """
        Ensure KinCore assignments CSV exists; run workflow only when needed.
        Returns the assignments CSV path.
        """
        paths = cls.build_output_paths(output_dir=output_dir, assignments_filename=assignments_filename)
        assignments_csv = paths["assignments_csv"]

        if (not os.path.exists(assignments_csv)) or bool(force):
            workflow = cls(
                input_dir=input_dir,
                output_dir=output_dir,
                kincore_dir=kincore_dir,
            )
            _ = workflow.run(output_csv=assignments_filename)
        else:
            print(f"Found existing KinCore assignments; skipping run: {assignments_csv}")

        return assignments_csv

    @classmethod
    def show_or_plot_conformation_distribution(
        cls,
        *,
        assignments_csv: str = "Results/dunbrack_assignments/kinase_conformation_assignments.csv",
        output_png: str = "Results/dunbrack_assignments/conformation_distribution.png",
        show: bool = True,
        print_dunbrack_summary: bool = True,
    ) -> Dict[str, int]:
        """
        Display cached conformation-distribution PNG if present; otherwise generate it.
        """
        try:
            from IPython.display import Image, display
        except Exception:
            Image = None
            display = None

        if os.path.exists(output_png) and Image is not None and display is not None:
            display(Image(filename=output_png))
            return {}

        return cls.plot_conformation_distribution(
            assignments_csv=assignments_csv,
            output_png=output_png,
            show=show,
            print_dunbrack_summary=print_dunbrack_summary,
        )


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


"""Pairwise MUSTANG structural alignment (template + one PDB).

Ported from writingWorkflowNotebook20_10_25/align.py for use in 08c.
Kept separate from the MDAnalysis-based ``workflow.align.Alignment`` class.
"""

from __future__ import annotations

import logging
import os
import subprocess
import traceback
from datetime import datetime
from glob import glob
from time import time
from typing import List, Optional

DEFAULT_MUSTANG_BIN = "/home/marmatt/Downloads/MUSTANG_v3.2.4/bin/mustang-3.2.4"


def _fname(pdb_path: str) -> str:
    base = os.path.basename(pdb_path)
    if base.lower().endswith(".pdb"):
        return base[:-4]
    return os.path.splitext(base)[0]


def _find_pdbs(directory: str) -> List[str]:
    return sorted(glob(os.path.join(directory, "*.pdb")))


class AlignmentMustang:
    """Pairwise MUSTANG aligner: template + one query PDB per call."""

    def __init__(
        self,
        mustang_path: str = DEFAULT_MUSTANG_BIN,
        log_file: Optional[str] = None,
    ):
        self.mustang_path = mustang_path

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"alignment_mustang_{timestamp}.log"

        self.log_file = log_file
        self.logger = logging.getLogger(f"AlignmentMustang_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []

        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("=" * 80)
        self.logger.info(f"AlignmentMustang initialized with log file: {log_file}")
        self.logger.info(f"MUSTANG path: {mustang_path}")
        self.logger.info("=" * 80)

    def run_mustang(
        self,
        template_pdb: str,
        input_pdb: str,
        target_dir: str,
        name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Run MUSTANG on template + one query PDB.

        Returns path to the produced ``.afasta`` file, or None on failure.
        """
        if name is None:
            name = _fname(input_pdb)

        self.logger.info(f"Starting MUSTANG alignment for: {name}")
        if not os.path.exists(self.mustang_path):
            self.logger.error(f"MUSTANG binary not found: {self.mustang_path}")
            return None
        if not os.path.exists(template_pdb):
            self.logger.error(f"Template PDB not found: {template_pdb}")
            return None
        if not os.path.exists(input_pdb):
            self.logger.error(f"Input PDB not found: {input_pdb}")
            return None

        try:
            struct_dir = os.path.join(target_dir, name)
            os.makedirs(struct_dir, exist_ok=True)
            out_prefix = os.path.join(struct_dir, name)

            args = [
                self.mustang_path,
                "-i",
                os.path.abspath(template_pdb),
                os.path.abspath(input_pdb),
                "-o",
                out_prefix,
                "-F",
                "fasta",
                "-s",
                "ON",
            ]
            self.logger.info(f"Running MUSTANG command: {' '.join(args)}")
            result = subprocess.run(args, shell=False, capture_output=True, text=True)

            if result.stdout:
                self.logger.debug(f"MUSTANG stdout:\n{result.stdout}")
            if result.stderr:
                self.logger.debug(f"MUSTANG stderr:\n{result.stderr}")

            if result.returncode != 0:
                self.logger.error(f"MUSTANG failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return None

            afasta = out_prefix + ".afasta"
            if not os.path.exists(afasta):
                # Some builds write .fasta instead
                alt = out_prefix + ".fasta"
                if os.path.exists(alt):
                    afasta = alt
                else:
                    found = glob(os.path.join(struct_dir, "*.afasta")) + glob(
                        os.path.join(struct_dir, "*.fasta")
                    )
                    if found:
                        afasta = found[0]
                    else:
                        self.logger.error(f"No FASTA/AFASTA found in {struct_dir}")
                        return None

            self.logger.info(f"Successfully aligned {name}, afasta: {afasta}")
            return afasta

        except Exception as e:
            self.logger.error(f"Error running MUSTANG for {name}: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def process_mustang_alignment(
        self,
        pdb_path: str,
        target_dir: str,
        template_pdb: str,
    ) -> dict:
        """
        Align every PDB in *pdb_path* to *template_pdb* (pairwise loop).

        Skips files whose basename matches the template. Returns a summary dict
        with keys: successful, failed, elapsed_s, n_input, afasta_paths.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting process_mustang_alignment")
        self.logger.info("=" * 80)

        os.makedirs(target_dir, exist_ok=True)
        self.logger.info(f"Input directory: {pdb_path}")
        self.logger.info(f"Output directory: {target_dir}")
        self.logger.info(f"Template PDB: {template_pdb}")

        if not os.path.isdir(pdb_path):
            raise FileNotFoundError(
                f"Sample pool directory not found: {pdb_path}. "
                "Regenerate Results/activation_segments/misaligned_filter/ upstream."
            )

        pdbs = _find_pdbs(pdb_path)

        template_base = os.path.basename(template_pdb)
        pdbs = [p for p in pdbs if os.path.basename(p) != template_base]
        self.logger.info(f"Found {len(pdbs)} PDB files to process")
        if not pdbs:
            raise FileNotFoundError(
                f"No .pdb files found in: {pdb_path}. "
                "Regenerate Results/activation_segments/misaligned_filter/ upstream."
            )

        t1 = time()
        successful = 0
        failed = 0
        afasta_paths: List[str] = []

        for idx, pdb in enumerate(pdbs, 1):
            self.logger.info(f"Processing structure {idx}/{len(pdbs)}")
            name = _fname(pdb)
            out = self.run_mustang(template_pdb, pdb, target_dir, name=name)
            if out:
                successful += 1
                afasta_paths.append(out)
            else:
                failed += 1
                self.logger.warning(f"Failed to process: {pdb}")

        elapsed = round(time() - t1, 3)
        self.logger.info("=" * 80)
        self.logger.info(f"Processing complete in {elapsed} seconds")
        self.logger.info(f"Successful alignments: {successful}/{len(pdbs)}")
        if failed:
            self.logger.warning(f"Failed alignments: {failed}/{len(pdbs)}")
        self.logger.info("=" * 80)

        return {
            "successful": successful,
            "failed": failed,
            "elapsed_s": float(elapsed),
            "n_input": len(pdbs),
            "afasta_paths": afasta_paths,
        }

"""MUSCLE-based MSA for DFG/APE motif identification (benchmark / experiment).

Outputs default to ``Results/muscle_msa_experiment``. Used by the August
``03c`` loop-filter benchmark and the March MUSCLE experiment notebook.
"""

import os
import sys
import shutil
import subprocess
from collections import OrderedDict
from typing import Dict, List

# Ensure the August workflow package root is importable.
_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import pandas as pd
from Bio import AlignIO
from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm

DEFAULT_MSA_RESULTS_DIR = "Results/muscle_msa_experiment"
MSA_PASSING_BASENAMES_FILE = "msa_passing_basenames.txt"
ACTIVATION_LOOP_TSV_FILE = "activation_loop_sequences.tsv"


def path_msa_passing_basenames(results_dir: str = DEFAULT_MSA_RESULTS_DIR) -> str:
    """Path to the newline-separated list of MSA-passing chain basenames."""
    return os.path.join(results_dir, MSA_PASSING_BASENAMES_FILE)


def path_activation_loop_tsv(results_dir: str = DEFAULT_MSA_RESULTS_DIR) -> str:
    """Path to the activation-loop subsequence TSV produced by ``AlignmentMUSCLE.run()``."""
    return os.path.join(results_dir, ACTIVATION_LOOP_TSV_FILE)


def load_msa_passing_basenames(results_dir: str = DEFAULT_MSA_RESULTS_DIR) -> List[str]:
    """Load basenames that passed the MSA DFG/APE filter."""
    path = path_msa_passing_basenames(results_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run AlignmentMUSCLE.run() first."
        )
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


class AlignmentMUSCLE:
    """Whole-dataset amino-acid MSA using MUSCLE v5 (`-super5` for N > 1000).

    Identifies DFG / APE motif columns in the reference sequence, classifies
    each input chain by whether those columns are non-gapped, and extracts
    activation-loop sub-sequences for passing chains.
    """

    def __init__(
        self,
        muscle_bin: str,
        results_dir: str = DEFAULT_MSA_RESULTS_DIR,
        dfg_motif: str = "DFG",
        ape_motif: str = "APE",
    ):
        self.muscle_bin = muscle_bin
        self.results_dir = results_dir
        self.dfg_motif = dfg_motif
        self.ape_motif = ape_motif
        os.makedirs(results_dir, exist_ok=True)

    @property
    def fasta_path(self) -> str:
        return os.path.join(self.results_dir, "interpro_sequences.fasta")

    @property
    def msa_path(self) -> str:
        return os.path.join(self.results_dir, "muscle_msa.fasta")

    @property
    def loop_tsv_path(self) -> str:
        return os.path.join(self.results_dir, ACTIVATION_LOOP_TSV_FILE)

    @property
    def html_path(self) -> str:
        return os.path.join(self.results_dir, "muscle_msa.html")

    def extract_sequences(
        self,
        input_dir: str,
        reference_pdb: str,
        reference_name: str,
    ) -> OrderedDict:
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        sequences: OrderedDict = OrderedDict()

        try:
            ref_struct = parser.get_structure(reference_name, reference_pdb)
            ref_seq = "".join(
                str(pp.get_sequence()) for pp in ppb.build_peptides(ref_struct)
            )
            if not ref_seq:
                raise ValueError("Empty sequence extracted from reference PDB.")
            sequences[reference_name] = ref_seq
        except Exception as exc:
            raise RuntimeError(
                f"Could not extract reference sequence from '{reference_pdb}': {exc}"
            ) from exc

        pdb_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".pdb")
        )
        n_failed = 0
        for fname in tqdm(pdb_files, desc="Extracting sequences"):
            basename = os.path.splitext(fname)[0]
            try:
                structure = parser.get_structure(
                    basename, os.path.join(input_dir, fname)
                )
                seq = "".join(
                    str(pp.get_sequence()) for pp in ppb.build_peptides(structure)
                )
                if seq:
                    sequences[basename] = seq
                else:
                    n_failed += 1
            except Exception:
                n_failed += 1

        with open(self.fasta_path, "w") as fh:
            for name, seq in sequences.items():
                fh.write(f">{name}\n{seq}\n")

        print(
            f"Extracted {len(sequences)} sequences "
            f"(including reference, {n_failed} skipped) → {self.fasta_path}"
        )
        return sequences

    def run_muscle(self, super5_threshold: int = 1000) -> None:
        n_seqs = sum(1 for line in open(self.fasta_path) if line.startswith(">"))
        use_super5 = n_seqs > super5_threshold

        mode_flag = "-super5" if use_super5 else "-align"
        mode_name = "super5" if use_super5 else "align"
        print(
            f"Running MUSCLE MSA in '{mode_name}' mode "
            f"({n_seqs} sequences) — this may take a few minutes..."
        )

        result = subprocess.run(
            [self.muscle_bin, mode_flag, self.fasta_path, "-output", self.msa_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MUSCLE failed:\n{result.stderr}")
        print(f"MUSCLE MSA written → {self.msa_path}")

    def parse_msa(self, reference_name: str) -> dict:
        alignment = AlignIO.read(self.msa_path, "fasta")
        msa_dict = {rec.id: str(rec.seq) for rec in alignment}
        if reference_name not in msa_dict:
            raise ValueError(
                f"Reference '{reference_name}' not found in MSA. "
                f"First 5 IDs: {list(msa_dict.keys())[:5]}"
            )
        return msa_dict

    @staticmethod
    def find_motif_columns(aligned_seq: str, motif: str):
        ungapped, col_map = "", []
        for col, aa in enumerate(aligned_seq):
            if aa != "-":
                col_map.append(col)
                ungapped += aa
        idx = ungapped.find(motif)
        return None if idx == -1 else [col_map[idx + i] for i in range(len(motif))]

    def classify_structures(
        self, msa_dict: dict, reference_name: str, dfg_cols: list, ape_cols: list
    ) -> dict:
        msa_passing, failing_both, failing_dfg_only, failing_ape_only = [], [], [], []
        for name, aligned_seq in msa_dict.items():
            if name == reference_name:
                continue
            dfg_ok = all(aligned_seq[c] != "-" for c in dfg_cols)
            ape_ok = all(aligned_seq[c] != "-" for c in ape_cols)
            if dfg_ok and ape_ok:
                msa_passing.append(name)
            elif not dfg_ok and not ape_ok:
                failing_both.append(name)
            elif not dfg_ok:
                failing_dfg_only.append(name)
            else:
                failing_ape_only.append(name)

        print(f"\nMSA alignment results:")
        print(f"  Passing (non-gapped DFG + APE) : {len(msa_passing)}")
        print(f"  Failing both DFG and APE       : {len(failing_both)}")
        print(f"  Failing DFG only               : {len(failing_dfg_only)}")
        print(f"  Failing APE only               : {len(failing_ape_only)}")

        for label, lst in [
            ("Failing both", failing_both),
            ("Failing DFG only", failing_dfg_only),
            ("Failing APE only", failing_ape_only),
        ]:
            if lst:
                preview = ", ".join(lst[:20])
                suffix = f" ... and {len(lst) - 20} more" if len(lst) > 20 else ""
                print(f"\n{label} ({len(lst)}): {preview}{suffix}")

        return {
            "msa_passing": msa_passing,
            "failing_both": failing_both,
            "failing_dfg_only": failing_dfg_only,
            "failing_ape_only": failing_ape_only,
        }

    def extract_loop_sequences(
        self, msa_dict: dict, msa_passing: list, dfg_cols: list, ape_cols: list
    ) -> pd.DataFrame:
        loop_start_col = dfg_cols[-1] + 1
        loop_end_col = ape_cols[0]
        if loop_start_col >= loop_end_col:
            raise ValueError(
                f"DFG ends at column {dfg_cols[-1]} but APE starts at "
                f"{ape_cols[0]}. Unexpected DFG–APE order in alignment."
            )
        records = []
        for name in msa_passing:
            loop_seq = msa_dict[name][loop_start_col:loop_end_col].replace("-", "")
            records.append({"pdb_basename": name, "loop_sequence": loop_seq})
        loop_df = pd.DataFrame(records)
        loop_df.to_csv(self.loop_tsv_path, sep="\t", index=False)
        ll = loop_df["loop_sequence"].str.len()
        print(f"\nSaved {len(loop_df)} activation-loop sequences → {self.loop_tsv_path}")
        print(
            f"Loop region: MSA columns {loop_start_col}–{loop_end_col - 1} "
            f"({loop_end_col - loop_start_col} alignment columns)"
        )
        print(
            f"Sequence lengths: min={ll.min()}, max={ll.max()}, "
            f"median={ll.median():.0f}"
        )
        return loop_df

    def generate_html(self, reference_name: str) -> dict:
        from workflow.analyse_alignment_foldmason import visualise_sequence_alignment

        visualizer = visualise_sequence_alignment()
        stats = visualizer.generate_multi_alignment_html(
            alignment_file=self.msa_path,
            output_file=self.html_path,
            reference_name=reference_name,
            alignment_type="amino acid",
        )
        return stats

    def run(
        self,
        input_dir: str,
        reference_pdb: str,
        reference_name: str,
        generate_html: bool = True,
    ) -> dict:
        self.extract_sequences(input_dir, reference_pdb, reference_name)
        self.run_muscle()
        msa_dict = self.parse_msa(reference_name)
        ref_aligned = msa_dict[reference_name]

        dfg_cols = self.find_motif_columns(ref_aligned, self.dfg_motif)
        ape_cols = self.find_motif_columns(ref_aligned, self.ape_motif)
        if dfg_cols is None:
            raise ValueError(
                f"'{self.dfg_motif}' not found in reference sequence '{reference_name}'"
            )
        if ape_cols is None:
            raise ValueError(
                f"'{self.ape_motif}' not found in reference sequence '{reference_name}'"
            )

        print(f"DFG columns in MSA : {dfg_cols}")
        print(f"APE columns in MSA : {ape_cols}")

        classification = self.classify_structures(
            msa_dict, reference_name, dfg_cols, ape_cols
        )

        pb_path = path_msa_passing_basenames(self.results_dir)
        with open(pb_path, "w", encoding="utf-8") as fh:
            fh.writelines(n + "\n" for n in classification["msa_passing"])
        print(
            f"Saved MSA-passing basenames ({len(classification['msa_passing'])}) → {pb_path}"
        )

        loop_df = self.extract_loop_sequences(
            msa_dict, classification["msa_passing"], dfg_cols, ape_cols
        )

        html_stats = None
        if generate_html:
            html_stats = self.generate_html(reference_name)

        return {
            **classification,
            "dfg_cols": dfg_cols,
            "ape_cols": ape_cols,
            "msa_dict": msa_dict,
            "loop_df": loop_df,
            "paths": {
                "fasta": self.fasta_path,
                "msa": self.msa_path,
                "loop_tsv": self.loop_tsv_path,
                "html": self.html_path,
                "passing_basenames": pb_path,
            },
            "html_stats": html_stats,
        }

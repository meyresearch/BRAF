import os
import subprocess
from time import time
import logging
import traceback
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
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
    
    def run_foldmason_multiple(self, pdb_files, target_dir, out_name="msa", report_mode=2):
        """
        Run FoldMason to align multiple structures in a single run.
        
        Args:
            pdb_files: List of absolute PDB file paths (length >= 2)
            target_dir: Directory to write outputs
            out_name: Output prefix for generated files
            report_mode: FoldMason report mode (0: no report, 1: HTML report, 2: JSON report)
        
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

    @staticmethod
    def export_tm_scores_from_report_json(json_path: str, output_csv: str) -> bool:
        """
        Best-effort export of TM-score estimates from a FoldMason/MMseqs JSON report.

        FoldMason `--report-mode 2` triggers an mmseqs JSON report (msa2lddtjson).
        Depending on version, TM-score fields can appear in different shapes.
        This function searches common patterns and writes a CSV:

        - Pairwise pattern: query,target,tm_score
        - Per-entry pattern: name,tm_score
        """
        import csv
        import json

        if not os.path.exists(json_path):
            return False

        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        def get_tm(d: dict):
            # direct keys
            for k in ("tmScore", "tm_score", "tmscore", "TM-score", "TMscore"):
                if k in d:
                    try:
                        return float(d[k])
                    except Exception:
                        return None
            # case-insensitive scan
            for k, v in d.items():
                if str(k).lower() in ("tmscore", "tm_score", "tm-score", "tmscoreestimate", "tmscore_estimate"):
                    try:
                        return float(v)
                    except Exception:
                        return None
            return None

        # Recursively collect dicts that include a TM-score-like field
        records = []

        def walk(obj):
            if isinstance(obj, dict):
                tm = get_tm(obj)
                if tm is not None:
                    records.append(obj)
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

        # Pattern A: pairwise-like records
        rows_pairwise = []
        for r in records:
            tm = get_tm(r)
            if tm is None:
                continue
            q = r.get("query") or r.get("q") or r.get("name1") or r.get("source") or r.get("from")
            t = r.get("target") or r.get("t") or r.get("name2") or r.get("dest") or r.get("to")
            if q is not None and t is not None:
                rows_pairwise.append((str(q), str(t), float(tm)))

        if rows_pairwise:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["query", "target", "tm_score"])
                w.writerows(rows_pairwise)
            return True

        # Pattern B: per-entry tmScore (relative to reference/consensus)
        rows_entry = []
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            for e in data["entries"]:
                if not isinstance(e, dict):
                    continue
                tm = get_tm(e)
                if tm is None:
                    continue
                name = e.get("name") or e.get("id") or e.get("entry") or e.get("structure")
                if name is None:
                    continue
                rows_entry.append((str(name), float(tm)))

        if rows_entry:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["name", "tm_score"])
                w.writerows(rows_entry)
            return True

        return False
    
    def process_foldmason_alignment_multi(self, pdb_path, target_dir, template_pdb=None, out_name="msa", report_mode=2):
        """
        Run a single multi-structure FoldMason alignment over all PDBs in a directory,
        optionally including a template structure.
        
        Args:
            pdb_path: Directory containing PDB files to include
            target_dir: Directory to save outputs
            template_pdb: Optional path to a template/reference PDB to include first
            out_name: Output prefix to use for FoldMason outputs
            report_mode: FoldMason report mode (0: no report, 1: HTML report, 2: JSON report)
        
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

            # If we requested a JSON report, attempt to export TM-scores to a simple CSV
            try:
                json_path = f"{out_prefix}.json"
                if os.path.exists(json_path):
                    out_csv = f"{out_prefix}_tmscore.csv"
                    ok = self.export_tm_scores_from_report_json(json_path, out_csv)
                    if ok:
                        self.logger.info(f"✅ Exported TM-score estimates to: {out_csv}")
                    else:
                        self.logger.warning(
                            f"JSON report found at {json_path}, but TM-score export did not find recognizable TM fields."
                        )
            except Exception as e:
                self.logger.warning(f"TM-score export step failed (non-fatal): {e}")
        else:
            self.logger.error(f"Multi-structure alignment FAILED after {elapsed} seconds")
        self.logger.info("="*80)
        
        return out_prefix

    @staticmethod
    def _parse_newick_to_tree(newick_str: str):
        """Parse a minimal Newick string into a nested node dict."""
        i = 0
        n = len(newick_str)

        def skip_ws():
            nonlocal i
            while i < n and newick_str[i].isspace():
                i += 1

        def parse_name_and_len():
            nonlocal i
            skip_ws()
            name = []
            while i < n and newick_str[i] not in ":,();":
                name.append(newick_str[i])
                i += 1
            nm = "".join(name).strip() or None

            ln = None
            skip_ws()
            if i < n and newick_str[i] == ":":
                i += 1
                skip_ws()
                num = []
                while i < n and newick_str[i] not in ",();":
                    num.append(newick_str[i])
                    i += 1
                try:
                    ln = float("".join(num).strip())
                except Exception:
                    ln = None
            return nm, ln

        def parse_subtree():
            nonlocal i
            skip_ws()
            if i < n and newick_str[i] == "(":
                i += 1
                children = []
                while True:
                    children.append(parse_subtree())
                    skip_ws()
                    if i < n and newick_str[i] == ",":
                        i += 1
                        continue
                    if i < n and newick_str[i] == ")":
                        i += 1
                        break
                    raise ValueError(f"Unexpected character in Newick at pos {i}: {newick_str[i:i+20]!r}")
                nm, ln = parse_name_and_len()
                return {"name": nm, "length": ln, "children": children}
            nm, ln = parse_name_and_len()
            return {"name": nm, "length": ln, "children": []}

        root = parse_subtree()
        skip_ws()
        if i < n and newick_str[i] == ";":
            i += 1
        return root

    @classmethod
    def plot_guide_tree_dendrogram(
        cls,
        nw_path: str = "Results/activation_segments/multi_aligned_foldmason/msa.nw",
        label_height: float = 10.0,
        figsize: tuple = (22, 9),
        show: bool = True,
    ) -> dict:
        """
        Render FoldMason guide tree (Newick) as a dendrogram with truncated labels.
        """
        try:
            from scipy.cluster.hierarchy import dendrogram, fcluster
        except ImportError as e:
            raise ImportError(
                "plot_guide_tree_dendrogram requires SciPy (scipy.cluster.hierarchy)."
            ) from e

        if not os.path.exists(nw_path):
            raise FileNotFoundError(f"Newick file not found: {nw_path}")

        with open(nw_path, "r", encoding="utf-8", errors="ignore") as f:
            newick_str = f.read().strip()

        has_branch_lengths = ":" in newick_str
        default_edge_len = 0.0 if has_branch_lengths else 1.0
        root = cls._parse_newick_to_tree(newick_str)

        leaves = []

        def collect_leaves(node):
            if not node["children"]:
                leaves.append(node)
                return
            for child in node["children"]:
                collect_leaves(child)

        collect_leaves(root)
        leaf_names = [leaf["name"] or f"leaf_{k}" for k, leaf in enumerate(leaves)]
        leaf_id = {id(leaf): k for k, leaf in enumerate(leaves)}

        Z = []
        next_id = [len(leaves)]

        def edge_len(child):
            return float(child["length"]) if child["length"] is not None else float(default_edge_len)

        def build_linkage(node):
            if not node["children"]:
                return leaf_id[id(node)], 0.0, 1

            child_infos = []
            for child in node["children"]:
                cid, ch, cn = build_linkage(child)
                child_infos.append((cid, ch + edge_len(child), cn))

            node_h = max(h for _, h, _ in child_infos) if child_infos else 0.0
            cid0, _, n0 = child_infos[0]
            cur_id, cur_n = cid0, n0
            for cid, _, cn in child_infos[1:]:
                Z.append([cur_id, cid, float(node_h), int(cur_n + cn)])
                cur_id = next_id[0]
                next_id[0] += 1
                cur_n += cn

            return cur_id, node_h, cur_n

        build_linkage(root)
        Z = np.asarray(Z, dtype=float)

        clusters_at_cut = fcluster(Z, t=label_height, criterion="distance")
        num_clusters = int(len(set(clusters_at_cut)))
        n = len(leaves)

        def cluster_count(cid):
            return 1 if cid < n else int(Z[cid - n, 3])

        def rep_leaf(cid):
            while cid >= n:
                cid = int(Z[cid - n, 0])
            return cid

        def leaf_label_func(cid):
            cnt = cluster_count(cid)
            if cnt == 1:
                return leaf_names[cid]
            rep = leaf_names[rep_leaf(cid)]
            return f"{rep} (n={cnt})"

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=num_clusters,
            leaf_label_func=leaf_label_func,
            leaf_rotation=90,
            leaf_font_size=7,
            color_threshold=label_height,
        )

        ax.axhline(label_height, color="black", lw=1, ls="--")
        ax.set_title(f"FoldMason guide tree dendrogram (labels shown up to height={label_height})")
        ax.set_xlabel("clusters at cut height (representative label + cluster size)")
        ax.set_ylabel(
            "tree height (cumulative branch length)"
            if has_branch_lengths
            else "tree height (topological depth; unit edges)"
        )
        plt.tight_layout()
        if show:
            plt.show()

        print(f"Leaves: {n}")
        print(f"Branch lengths present in Newick: {has_branch_lengths}")
        print(f"Cut height: {label_height} -> showing ~{num_clusters} clusters")
        if not has_branch_lengths:
            print("Note: height is topological depth (not TM-score/lDDT) because `msa.nw` has no branch lengths.")

        return {
            "n_leaves": n,
            "has_branch_lengths": has_branch_lengths,
            "label_height": float(label_height),
            "num_clusters": num_clusters,
            "figure": fig,
            "axis": ax,
        }


import os
import re
import warnings
import multiprocessing
from contextlib import nullcontext

import MDAnalysis as mda
import pandas as pd
from typing import Optional, List, Dict, Set, Tuple, Iterable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


def _pool_init_suppress_warnings() -> None:
    """Pool worker initializer: hide MDAnalysis/NumPy/etc. warnings during extraction."""
    import logging

    warnings.filterwarnings("ignore")
    for name in ("MDAnalysis", "MDAnalysis.core", "MDAnalysis.coordinates", "MDAnalysis.topology"):
        logging.getLogger(name).setLevel(logging.ERROR)


class PDBChainExtractor:
    """
    A class to extract specific chains from PDB files and write them to new PDB files.
    
    This class implements parallel processing for efficient chain extraction,
    post-processing cleanup, and error handling.
    """
    
    def __init__(self, pdb_source_dir="Results/InterPro_PDBs", default_output_dir="Results/activation_segments/unaligned"):
        """
        Initialize the PDBChainExtractor.
        
        Args:
            pdb_source_dir (str): Directory containing source PDB files
            default_output_dir (str): Default directory for output PDB files
        """
        self.pdb_source_dir = pdb_source_dir
        self.default_output_dir = default_output_dir

        # MDAnalysis selection language varies slightly across versions; some do not support
        # the 'water' selection token. We therefore use explicit water residue names.
        self._water_resnames = ("HOH", "WAT", "H2O", "TIP3", "TIP3P", "SOL")

    def _mda_not_water_clause(self) -> str:
        """Return an MDAnalysis selection clause that excludes common water residue names."""
        # Example: 'and not (resname HOH WAT H2O TIP3 TIP3P SOL)'
        return "and not (resname " + " ".join(self._water_resnames) + ")"
    
    def get_pdb_path(self, pdb_id):
        """
        Path to coordinates for this PDB ID in ``pdb_source_dir``.

        Prefers ``{id}.pdb`` when present; otherwise ``{id}.cif`` (mmCIF from the same
        RCSB download layout). If neither exists, returns the ``.pdb`` path for checks
        and error messages.

        Args:
            pdb_id (str): PDB ID

        Returns:
            str: Full path to an existing ``.pdb`` or ``.cif``, else the ``.pdb`` path
        """
        pid = str(pdb_id).strip().upper()
        pdb_path = os.path.join(self.pdb_source_dir, f"{pid}.pdb")
        cif_path = os.path.join(self.pdb_source_dir, f"{pid}.cif")
        if os.path.isfile(pdb_path):
            return pdb_path
        if os.path.isfile(cif_path):
            return cif_path
        return pdb_path

    def _write_small_molecule_inventory_tsv(self, out_path: str, atomgroup) -> None:
        """
        Write a small TSV listing non-protein residues present in `atomgroup`.

        This is meant as bookkeeping for what hetero residues were kept alongside the chain.
        Water is excluded.
        """
        try:
            # Avoid 'water' token for compatibility across MDAnalysis versions.
            het = atomgroup.select_atoms(f"not protein {self._mda_not_water_clause()}")
        except Exception:
            return

        if het.n_atoms == 0:
            return

        rows = []
        for r in het.residues:
            rows.append({
                "resname": getattr(r, "resname", ""),
                "resid": getattr(r, "resid", ""),
                "chainID": getattr(r, "chainID", ""),
                "segid": getattr(r, "segid", ""),
            })

        if not rows:
            return

        inv_path = out_path + ".small_molecules.tsv"
        pd.DataFrame(rows).to_csv(inv_path, sep="\t", index=False)

    @staticmethod
    def _parse_pdb_small_molecules_from_lines(
        lines: List[str],
        exclude_waters: bool = True,
        exclude_amino_acids: bool = True,
        ignore_resnames: Optional[Set[str]] = None,
        min_heavy_atoms: int = 6,
    ) -> List[Dict]:
        """
        Parse small-molecule candidates from PDB text lines by scanning HETATM records.

        Returns a list of dicts with residue identifiers:
        - resname (3-letter code)
        - chain_id (single-char chain)
        - resid (integer residue number)
        - heavy_atoms (integer heavy atom count for that residue)

        Notes:
        - This is a *PDB-format* parser (fixed columns), intentionally simple and fast.
        - If exclude_amino_acids=True, standard amino acids written as HETATM
          (common in chain-break/disconnected fragment cases) are excluded.
        - If ignore_resnames is provided (or defaults), those residue names are excluded.
        - If min_heavy_atoms is set, residues with < min_heavy_atoms heavy atoms are excluded
          (heavy atoms are non-H/non-D).
        """
        molecules: List[Dict] = []
        seen: Set[Tuple[str, str, int]] = set()

        water_names = {"HOH", "WAT", "H2O"}
        aa_names = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }

        default_ignore: Set[str] = {
            "HOH", "WAT", "DOD", "SOL",
            "NA", "K", "CL", "BR", "I",
            "CA", "MG", "MN", "FE", "ZN", "CU", "CO", "NI", "CD", "HG",
            "SO4", "PO4", "NO3",
            "GOL", "EDO", "MPD", "DMS", "ACT", "ACE",
            "PEG", "PGE", "PG4", "PG5", "PEO",
            "TRS", "MES", "HEP", "BIS",
            "CIT", "FMT", "TAR",
            "BOG", "NAG", "NDG", "BMA", "FUC", "MAN", "GAL",
            "SPM", "PUT", "LDAO", "CHAPS", "SDS",
        }
        ignore = default_ignore if ignore_resnames is None else set(ignore_resnames)

        def _guess_element(pdb_line: str) -> str:
            """
            Best-effort element extraction from a PDB ATOM/HETATM line.

            Prefers the element column (77-78, 1-based). Falls back to atom name (13-16).
            """
            if len(pdb_line) >= 78:
                el = pdb_line[76:78].strip().upper()
                if el:
                    return el

            atom_name = pdb_line[12:16].strip().upper() if len(pdb_line) >= 16 else ""
            if not atom_name:
                return ""

            # Strip leading digits (e.g., 1HG1) to get element token.
            i = 0
            while i < len(atom_name) and atom_name[i].isdigit():
                i += 1
            atom_name = atom_name[i:]
            if not atom_name:
                return ""

            two = atom_name[:2]
            common_two = {
                "CL", "BR", "NA", "CA", "ZN", "MG", "FE", "MN", "CU", "CO", "NI", "CD", "HG"
            }
            if two in common_two:
                return two
            return atom_name[0]

        # First pass: count heavy atoms per residue instance (resname, chain, resid).
        heavy_counts: Dict[Tuple[str, str, int], int] = {}
        for ln in lines:
            if not ln.startswith("HETATM"):
                continue
            if len(ln) < 26:
                continue

            resname = ln[17:20].strip().upper()
            chain_id = ln[21].strip()
            resid_str = ln[22:26].strip()
            try:
                resid = int(resid_str)
            except ValueError:
                continue

            if exclude_waters and resname in water_names:
                continue
            if exclude_amino_acids and resname in aa_names:
                continue
            if resname in ignore:
                continue

            el = _guess_element(ln)
            if not el or el in {"H", "D"}:
                continue

            key = (resname, chain_id, resid)
            heavy_counts[key] = heavy_counts.get(key, 0) + 1

        for ln in lines:
            if not ln.startswith("HETATM"):
                continue
            if len(ln) < 26:
                continue

            resname = ln[17:20].strip().upper()
            chain_id = ln[21].strip()
            resid_str = ln[22:26].strip()
            try:
                resid = int(resid_str)
            except ValueError:
                continue

            if exclude_waters and resname in water_names:
                continue
            if exclude_amino_acids and resname in aa_names:
                continue
            if resname in ignore:
                continue

            heavy_atoms = heavy_counts.get((resname, chain_id, resid), 0)
            if heavy_atoms < int(min_heavy_atoms):
                continue

            key = (resname, chain_id, resid)
            if key in seen:
                continue
            seen.add(key)

            molecules.append(
                {"resname": resname, "chain_id": chain_id, "resid": resid, "heavy_atoms": heavy_atoms}
            )

        return molecules

    def report_small_molecules_in_directory(
        self,
        pdb_dir: str,
        output_csv: Optional[str] = None,
        exclude_waters: bool = True,
        exclude_amino_acids: bool = True,
        ignore_resnames: Optional[Set[str]] = None,
        min_heavy_atoms: int = 6,
        print_examples: int = 20,
    ) -> pd.DataFrame:
        """
        Scan a directory of PDB files and report which PDB+chain contain small molecules (HETATM).

        This is meant to be run independently of `extract_chains_parallel()` to generate a
        human-readable summary and an optional CSV for downstream analysis.

        Args:
            pdb_dir: Directory containing extracted PDB files (e.g. ``Results/InterPro_protein_small_molecules/``)
            output_csv: If provided, write the per-chain small-molecule table to this path.
                        (If relative, it is interpreted relative to the current working directory.)
            exclude_waters: Ignore HOH/WAT/H2O.
            exclude_amino_acids: Ignore standard amino acids written as HETATM (artifact cases).
            ignore_resnames: If provided, residue names to ignore (e.g., ions/solvents/buffers).
                            If None, a default ignore list is applied.
            min_heavy_atoms: Minimum heavy-atom count required for a residue to count as a small molecule.
                             Residues with fewer heavy atoms are excluded.
            print_examples: Print up to N example chains that include a small molecule.

        Returns:
            DataFrame with one row per (pdb_file, small-molecule residue) mapping.
        """
        if not os.path.isdir(pdb_dir):
            raise FileNotFoundError(f"pdb_dir not found: {pdb_dir}")

        pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb")])
        rows: List[Dict] = []

        for fn in pdb_files:
            path = os.path.join(pdb_dir, fn)
            try:
                with open(path, "r") as f:
                    lines = f.readlines()
            except Exception:
                continue

            hits = self._parse_pdb_small_molecules_from_lines(
                lines,
                exclude_waters=exclude_waters,
                exclude_amino_acids=exclude_amino_acids,
                ignore_resnames=ignore_resnames,
                min_heavy_atoms=min_heavy_atoms,
            )

            # try to split "1ABC_A.pdb" -> pdb_id=1ABC, chain_from_name=A
            stem = os.path.splitext(fn)[0]
            m = re.match(r"^([^_]+)_([A-Za-z0-9])$", stem)
            pdb_id = m.group(1) if m else stem
            chain_from_name = m.group(2) if m else ""

            if not hits:
                rows.append({
                    "pdb_file": fn,
                    "pdb_id": pdb_id,
                    "chain_from_name": chain_from_name,
                    "has_small_molecule": False,
                    "small_molecule_id": "",
                    "small_molecule_resname": "",
                    "small_molecule_chain": "",
                    "small_molecule_resid": "",
                })
                continue

            for sm in hits:
                sm_id = f"{sm['resname']}:{sm['resid']}"
                rows.append({
                    "pdb_file": fn,
                    "pdb_id": pdb_id,
                    "chain_from_name": chain_from_name,
                    "has_small_molecule": True,
                    "small_molecule_id": sm_id,
                    "small_molecule_resname": sm["resname"],
                    "small_molecule_chain": sm["chain_id"],
                    "small_molecule_resid": sm["resid"],
                    "small_molecule_heavy_atoms": sm.get("heavy_atoms", ""),
                })

        df = pd.DataFrame(rows)

        # Summary prints
        n_files = df["pdb_file"].nunique() if not df.empty else 0
        sm_files = df[df["has_small_molecule"]]["pdb_file"].nunique() if not df.empty else 0
        print("\n" + "=" * 60)
        print("SMALL MOLECULE REPORT (from PDB HETATM records)")
        print("=" * 60)
        print(f"Directory: {pdb_dir}")
        print(f"Files scanned: {n_files}")
        print(f"Files with small molecule(s): {sm_files}")

        if sm_files:
            unique_sm = df[df["has_small_molecule"]]["small_molecule_resname"].value_counts()
            print("\nMost common small-molecule residue names:")
            print(unique_sm.head(15).to_string())

            print("\nExample chains containing a small molecule:")
            examples = df[df["has_small_molecule"]][["pdb_file", "small_molecule_id"]].drop_duplicates().head(int(print_examples))
            for _, r in examples.iterrows():
                print(f"  {r['pdb_file']}: {r['small_molecule_id']}")

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nSaved report to: {output_csv}")

        return df

    def plot_small_molecule_chain_counts(
        self,
        df: pd.DataFrame,
        *,
        figsize: Tuple[float, float] = (5.0, 4.0),
        title: str = "Chains by small-molecule content (first-pass HETATM screen)",
        ylabel: str = "Number of chains (PDB files)",
        show: bool = True,
    ) -> Tuple[int, int]:
        """
        Bar chart of chain PDB files with at least one retained small molecule vs none.

        Expects ``pdb_file`` and ``has_small_molecule`` as returned by
        :meth:`report_small_molecules_in_directory`.

        Returns:
            ``(n_with_small_molecule, n_without)`` counts over distinct ``pdb_file`` values.
        """
        import matplotlib.pyplot as plt

        if df.empty:
            n_with_sm, n_without_sm = 0, 0
        else:
            required = {"pdb_file", "has_small_molecule"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"df is missing required columns: {sorted(missing)}")
            has_sm = df["has_small_molecule"]
            if has_sm.dtype == object:
                has_sm = has_sm.astype(str).str.lower().isin(("true", "1"))
            n_total = df["pdb_file"].nunique()
            n_with_sm = df.loc[has_sm, "pdb_file"].nunique()
            n_without_sm = n_total - n_with_sm

        fig, ax = plt.subplots(figsize=figsize)
        labels = ["With small molecule", "Without"]
        counts = [n_with_sm, n_without_sm]
        ax.bar(labels, counts, color=["#2a6f97", "#c4c4c4"], edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ymax = max(counts) if counts else 1
        for i, c in enumerate(counts):
            ax.text(i, c + ymax * 0.02, str(c), ha="center", va="bottom", fontsize=11)
        fig.tight_layout()
        if show:
            plt.show()
        return n_with_sm, n_without_sm

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
            if not getattr(self, "_extract_quiet", False):
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
    
    def process_single_pdb_entry(self, entry) -> Tuple[bool, str]:
        """
        Process a single PDB entry, extracting chains and writing output.

        Args:
            entry (tuple): (accession, chain_list, target_dir, include_small_molecules,
                small_molecule_distance, keep_waters, write_small_molecule_inventory)

        Returns:
            (success, message): ``message`` is non-empty when ``success`` is False and
            ``_extract_quiet`` is set (caller prints via ``tqdm.write``). In verbose mode
            ``message`` is always empty (details are printed in this process).
        """
        (
            accession,
            chain_list,
            target_dir,
            include_small_molecules,
            small_molecule_distance,
            keep_waters,
            write_small_molecule_inventory,
        ) = entry
        quiet = getattr(self, "_extract_quiet", False)
        file_path = self.get_pdb_path(accession)

        def fail(msg_verbose: str, msg_quiet: str) -> Tuple[bool, str]:
            if quiet:
                return False, msg_quiet
            print(msg_verbose)
            return False, ""

        if not os.path.exists(file_path):
            return fail(
                f"Skipping {file_path}: File not found",
                f"Extract failed: {accession}: source coordinates not found",
            )

        u = None
        try:
            try:
                u = mda.Universe(file_path)
            except Exception as e:
                return fail(
                    f"Error loading {file_path}: {e}",
                    f"Extract failed: {accession}: load error ({e})",
                )

            success = True
            quiet_notes: List[str] = []

            for chain_id in chain_list:
                try:
                    protein_sel = f"protein and chainID {chain_id}"
                    chain = u.select_atoms(protein_sel)

                    if chain.n_atoms > 0:
                        output_pdb = os.path.join(target_dir, f"{accession}_{chain_id}.pdb")

                        out_atoms = chain
                        if include_small_molecules:
                            water_clause = "" if keep_waters else self._mda_not_water_clause()
                            het_near = u.select_atoms(
                                f"(not protein {water_clause}) and around {float(small_molecule_distance)} ({protein_sel})"
                            )
                            out_atoms = chain + het_near

                        with mda.Writer(output_pdb) as w:
                            w.write(out_atoms)
                        self.post_process_pdb(output_pdb)
                        if not quiet:
                            print(f"Processed {output_pdb}")

                        if include_small_molecules and write_small_molecule_inventory:
                            try:
                                self._write_small_molecule_inventory_tsv(output_pdb, out_atoms)
                            except Exception as e:
                                if quiet:
                                    quiet_notes.append(f"small molecule inventory: {e}")
                                else:
                                    print(
                                        f"Warning: failed to write small-molecule inventory for {output_pdb}: {e}"
                                    )
                    else:
                        success = False
                        if quiet:
                            quiet_notes.append(f"chain {chain_id} not found")
                        else:
                            print(f"Chain {chain_id} not found in {file_path}")
                except Exception as e:
                    success = False
                    if quiet:
                        quiet_notes.append(f"chain {chain_id}: {e}")
                    else:
                        print(f"Error processing {file_path} (Chain {chain_id}): {e}")

            if quiet:
                if not success:
                    detail = "; ".join(quiet_notes) if quiet_notes else "unknown error"
                    return False, f"Extract failed: {accession}: {detail}"
                return True, ""
            return success, ""
        finally:
            if u is not None:
                del u
    
    def extract_chains_parallel(
        self,
        pdb_data,
        target_dir=None,
        max_workers=None,
        include_small_molecules: bool = False,
        small_molecule_distance: float = 5.0,
        keep_waters: bool = False,
        write_small_molecule_inventory: bool = True,
        show_progress: bool = True,
        show_errors: bool = False,
        **kwargs,
    ):
        """
        Process multiple PDB files in parallel to extract chains.

        Args:
            pdb_data (pd.DataFrame): DataFrame with PDB information
            target_dir (str): Output directory for processed files
            max_workers (int): Maximum number of worker processes
            include_small_molecules (bool): If True, include non-protein atoms within
                ``small_molecule_distance`` Å of the selected protein chain in the output PDB.
                This allows KinCore to report small molecules after extraction.
            small_molecule_distance (float): Distance cutoff (Å) for including nearby hetero atoms.
            keep_waters (bool): If True, include waters as well (default False).
            write_small_molecule_inventory (bool): If True, write ``<output>.small_molecules.tsv`` files listing
                non-protein residues that were included.
            show_progress (bool): If True, show a tqdm bar and suppress per-entry success prints.
            show_errors (bool): If True (and ``show_progress``), print failure lines with
                ``tqdm.write``. If False, only the progress bar is shown (failures are silent).

        Keyword Args (deprecated names, accepted for compatibility):
            include_ligands: alias for ``include_small_molecules``
            ligand_distance: alias for ``small_molecule_distance``
            write_ligand_inventory: alias for ``write_small_molecule_inventory``
        """
        if "include_ligands" in kwargs:
            include_small_molecules = bool(kwargs.pop("include_ligands"))
        if "ligand_distance" in kwargs:
            small_molecule_distance = float(kwargs.pop("ligand_distance"))
        if "write_ligand_inventory" in kwargs:
            write_small_molecule_inventory = bool(kwargs.pop("write_ligand_inventory"))
        if kwargs:
            raise TypeError(
                "extract_chains_parallel() got unexpected keyword arguments: "
                f"{sorted(kwargs.keys())}"
            )
        if target_dir is None:
            target_dir = self.default_output_dir
        
        os.makedirs(target_dir, exist_ok=True)

        # Validate required columns early with a helpful error
        required_cols = {"Accession", "Chains"}
        missing_cols = required_cols - set(getattr(pdb_data, "columns", []))
        if missing_cols:
            raise KeyError(
                f"pdb_data is missing required column(s): {sorted(missing_cols)}. "
                f"Available columns: {list(getattr(pdb_data, 'columns', []))}"
            )

        # Ensure we have a Downloaded column; many notebooks build pdb_data without it.
        if "Downloaded" not in pdb_data.columns:
            pdb_data = pdb_data.copy()
            if os.path.exists(self.pdb_source_dir):
                file_names = [
                    os.path.splitext(f)[0].upper()
                    for f in os.listdir(self.pdb_source_dir)
                    if os.path.isfile(os.path.join(self.pdb_source_dir, f))
                    and f.lower().endswith((".pdb", ".cif"))
                ]
                downloaded_set = set(file_names)
                pdb_data["Downloaded"] = (
                    pdb_data["Accession"].astype(str).str.upper().isin(downloaded_set)
                )
            else:
                # Fall back to processing all rows (extract step will skip missing files).
                if not show_progress:
                    print(
                        f"Warning: 'Downloaded' column not found and source directory "
                        f"{self.pdb_source_dir} does not exist. Proceeding as if all entries were downloaded."
                    )
                pdb_data["Downloaded"] = True

        # Filter downloaded PDB entries
        pdb_entries = pdb_data[pdb_data["Downloaded"].astype(bool)].copy()

        # Convert chain list from semicolon-separated string to a list
        pdb_entries['Chain_list'] = pdb_entries['Chains'].apply(lambda x: x.split(';'))

        # Prepare input list for multiprocessing
        task_list = [
            (
                row["Accession"],
                row["Chain_list"],
                target_dir,
                bool(include_small_molecules),
                float(small_molecule_distance),
                bool(keep_waters),
                bool(write_small_molecule_inventory),
            )
            for _, row in pdb_entries.iterrows()
        ]

        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 10)

        self._extract_quiet = bool(show_progress)
        try:
            pool_kw = {}
            if show_progress:
                pool_kw["initializer"] = _pool_init_suppress_warnings

            warn_ctx = warnings.catch_warnings() if show_progress else nullcontext()
            with warn_ctx:
                if show_progress:
                    warnings.simplefilter("ignore")
                with multiprocessing.Pool(
                    processes=max_workers, maxtasksperchild=10, **pool_kw
                ) as pool:
                    if not task_list:
                        pass
                    elif show_progress and tqdm is not None:
                        chunksize = max(1, len(task_list) // (max_workers * 8))
                        with tqdm(
                            total=len(task_list),
                            desc="Extract chains",
                            unit="entry",
                            smoothing=0.05,
                        ) as pbar:
                            for _ok, msg in pool.imap_unordered(
                                self.process_single_pdb_entry,
                                task_list,
                                chunksize=chunksize,
                            ):
                                pbar.update(1)
                                if msg and show_errors:
                                    tqdm.write(msg)
                    elif show_progress and tqdm is None:
                        for _ok, msg in pool.imap_unordered(
                            self.process_single_pdb_entry,
                            task_list,
                            chunksize=max(1, len(task_list) // (max_workers * 8)),
                        ):
                            if msg and show_errors:
                                print(msg)
                    else:
                        pool.map(self.process_single_pdb_entry, task_list)
        finally:
            if hasattr(self, "_extract_quiet"):
                delattr(self, "_extract_quiet")

        if not show_progress:
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
                file_names = [
                    os.path.splitext(f)[0]
                    for f in os.listdir(self.pdb_source_dir)
                    if os.path.isfile(os.path.join(self.pdb_source_dir, f))
                    and f.lower().endswith((".pdb", ".cif"))
                ]
                
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


# Backward-compatible alias for older notebooks
PDBChainExtractor.report_ligands_in_directory = PDBChainExtractor.report_small_molecules_in_directory
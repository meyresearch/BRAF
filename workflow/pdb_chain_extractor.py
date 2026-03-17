import os
import multiprocessing
import MDAnalysis as mda
import pandas as pd
import re
from typing import Optional, List, Dict, Set, Tuple, Iterable


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

        # MDAnalysis selection language varies slightly across versions; some do not support
        # the 'water' selection token. We therefore use explicit water residue names.
        self._water_resnames = ("HOH", "WAT", "H2O", "TIP3", "TIP3P", "SOL")

    def _mda_not_water_clause(self) -> str:
        """Return an MDAnalysis selection clause that excludes common water residue names."""
        # Example: 'and not (resname HOH WAT H2O TIP3 TIP3P SOL)'
        return "and not (resname " + " ".join(self._water_resnames) + ")"
    
    def get_pdb_path(self, pdb_id):
        """
        Generate a cross-platform PDB file path.
        
        Args:
            pdb_id (str): PDB ID
            
        Returns:
            str: Full path to the PDB file
        """
        return os.path.join(self.pdb_source_dir, f"{pdb_id}.pdb")

    def _write_ligand_inventory_tsv(self, out_path: str, atomgroup) -> None:
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

        inv_path = out_path + ".ligands.tsv"
        pd.DataFrame(rows).to_csv(inv_path, sep="\t", index=False)

    @staticmethod
    def _parse_pdb_ligands_from_lines(
        lines: List[str],
        exclude_waters: bool = True,
        exclude_amino_acids: bool = True,
        ignore_resnames: Optional[Set[str]] = None,
        min_heavy_atoms: int = 6,
    ) -> List[Dict]:
        """
        Parse ligands from PDB text lines by scanning HETATM records.

        Returns a list of dicts with ligand identifiers:
        - resname (3-letter code)
        - chain_id (single-char chain)
        - resid (integer residue number)
        - heavy_atoms (integer heavy atom count for that residue)

        Notes:
        - This is a *PDB-format* parser (fixed columns), intentionally simple and fast.
        - If exclude_amino_acids=True, standard amino acids written as HETATM
          (common in chain-break/disconnected fragment cases) are excluded.
        - If ignore_resnames is provided (or defaults), those residue names are excluded.
        - If min_heavy_atoms is set, ligands with < min_heavy_atoms heavy atoms are excluded
          (heavy atoms are non-H/non-D).
        """
        ligands: List[Dict] = []
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

            ligands.append(
                {"resname": resname, "chain_id": chain_id, "resid": resid, "heavy_atoms": heavy_atoms}
            )

        return ligands

    def report_ligands_in_directory(
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
        Scan a directory of PDB files and report which PDB+chain contain ligands (HETATM).

        This is meant to be run independently of `extract_chains_parallel()` to generate a
        human-readable summary and an optional CSV for downstream analysis.

        Args:
            pdb_dir: Directory containing extracted PDB files (e.g., Results/.../unaligned+ligands/)
            output_csv: If provided, write the per-chain ligand table to this path.
                        (If relative, it is interpreted relative to the current working directory.)
            exclude_waters: Ignore HOH/WAT/H2O.
            exclude_amino_acids: Ignore standard amino acids written as HETATM (artifact cases).
            ignore_resnames: If provided, residue names to ignore (e.g., ions/solvents/buffers).
                            If None, a default ignore list is applied.
            min_heavy_atoms: Minimum heavy-atom count required for a residue to be considered a ligand.
                             Residues with fewer heavy atoms are excluded.
            print_examples: Print up to N example ligand-bearing chains.

        Returns:
            DataFrame with one row per (pdb_file, ligand residue) mapping.
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

            ligs = self._parse_pdb_ligands_from_lines(
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

            if not ligs:
                rows.append({
                    "pdb_file": fn,
                    "pdb_id": pdb_id,
                    "chain_from_name": chain_from_name,
                    "has_ligand": False,
                    "ligand_id": "",
                    "ligand_resname": "",
                    "ligand_chain": "",
                    "ligand_resid": "",
                })
                continue

            for lig in ligs:
                ligand_id = f"{lig['resname']}:{lig['resid']}"
                rows.append({
                    "pdb_file": fn,
                    "pdb_id": pdb_id,
                    "chain_from_name": chain_from_name,
                    "has_ligand": True,
                    "ligand_id": ligand_id,
                    "ligand_resname": lig["resname"],
                    "ligand_chain": lig["chain_id"],
                    "ligand_resid": lig["resid"],
                    "ligand_heavy_atoms": lig.get("heavy_atoms", ""),
                })

        df = pd.DataFrame(rows)

        # Summary prints
        n_files = df["pdb_file"].nunique() if not df.empty else 0
        ligand_files = df[df["has_ligand"]]["pdb_file"].nunique() if not df.empty else 0
        print("\n" + "=" * 60)
        print("LIGAND REPORT (from PDB HETATM records)")
        print("=" * 60)
        print(f"Directory: {pdb_dir}")
        print(f"Files scanned: {n_files}")
        print(f"Files with ligand(s): {ligand_files}")

        if ligand_files:
            unique_ligs = df[df["has_ligand"]]["ligand_resname"].value_counts()
            print("\nMost common ligand residue names:")
            print(unique_ligs.head(15).to_string())

            print("\nExample ligand-bearing chains:")
            examples = df[df["has_ligand"]][["pdb_file", "ligand_id"]].drop_duplicates().head(int(print_examples))
            for _, r in examples.iterrows():
                print(f"  {r['pdb_file']}: {r['ligand_id']}")

        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nSaved report to: {output_csv}")

        return df
    
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
            entry (tuple): (accession, chain_list, target_dir, include_ligands, ligand_distance, keep_waters, write_ligand_inventory)
            
        Returns:
            bool: True if successful, False otherwise
        """
        accession, chain_list, target_dir, include_ligands, ligand_distance, keep_waters, write_ligand_inventory = entry
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
                protein_sel = f"protein and chainID {chain_id}"
                chain = u.select_atoms(protein_sel)

                if chain.n_atoms > 0:
                    output_pdb = os.path.join(target_dir, f"{accession}_{chain_id}.pdb")

                    out_atoms = chain
                    if include_ligands:
                        # Include hetero atoms near the protein chain so KinCore can detect ligands.
                        # This keeps the extracted structure mostly protein-only, plus bound ligands/cofactors/ions.
                        water_clause = "" if keep_waters else self._mda_not_water_clause()
                        het_near = u.select_atoms(
                            f"(not protein {water_clause}) and around {float(ligand_distance)} ({protein_sel})"
                        )
                        out_atoms = chain + het_near

                    with mda.Writer(output_pdb) as w:
                        w.write(out_atoms)
                    self.post_process_pdb(output_pdb)
                    print(f"Processed {output_pdb}")

                    if include_ligands and write_ligand_inventory:
                        try:
                            self._write_ligand_inventory_tsv(output_pdb, out_atoms)
                        except Exception as e:
                            print(f"Warning: failed to write ligand inventory for {output_pdb}: {e}")
                else:
                    print(f"Chain {chain_id} not found in {file_path}")
                    success = False
            except Exception as e:
                print(f"Error processing {file_path} (Chain {chain_id}): {e}")
                success = False
        
        # Release memory
        del u
        return success
    
    def extract_chains_parallel(
        self,
        pdb_data,
        target_dir=None,
        max_workers=None,
        include_ligands: bool = False,
        ligand_distance: float = 5.0,
        keep_waters: bool = False,
        write_ligand_inventory: bool = True,
    ):
        """
        Process multiple PDB files in parallel to extract chains.
        
        Args:
            pdb_data (pd.DataFrame): DataFrame with PDB information
            target_dir (str): Output directory for processed files
            max_workers (int): Maximum number of worker processes
            include_ligands (bool): If True, include non-protein atoms within `ligand_distance`
                                   Å of the selected protein chain in the output PDB.
                                   This allows KinCore to report ligands after extraction.
            ligand_distance (float): Distance cutoff (Å) for including nearby hetero atoms.
            keep_waters (bool): If True, include waters as well (default False).
            write_ligand_inventory (bool): If True, write `<output>.ligands.tsv` files listing
                                           non-protein residues that were included.
        """
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
                ]
                downloaded_set = set(file_names)
                pdb_data["Downloaded"] = (
                    pdb_data["Accession"].astype(str).str.upper().isin(downloaded_set)
                )
            else:
                # Fall back to processing all rows (extract step will skip missing files).
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
                bool(include_ligands),
                float(ligand_distance),
                bool(keep_waters),
                bool(write_ligand_inventory),
            )
            for _, row in pdb_entries.iterrows()
        ]

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
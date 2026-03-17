import os
from glob import glob
from time import time
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.align as mda_align
import MDAnalysis.analysis.rms as mda_rms
from Bio.PDB import PDBParser, PPBuilder

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class Alignment:
    """
    A class to handle motif-based protein structure alignment using MDAnalysis.
    
    This class aligns structures to a reference using CA atoms in the DFG and APE motif
    regions (mirroring the previous PyMOL workflow), and writes aligned PDBs.
    """
    
    def __init__(self, altloc_exclude=("B", "C", "D")):
        """
        Initialize the Alignment class.
        
        Args:
            altloc_exclude: Alternate location identifiers to exclude when selecting atoms.
        """
        self.altloc_exclude = set(altloc_exclude or [])
        self._ref_universe = None
        self._ref_name = None
        self._ref_resids = {}
        # Cached for convenience plotting / downstream steps
        self._last_conservation = None
        self._last_anchor_info = None
        
    def initialize_pymol(self):
        """
        Backwards-compatible no-op.

        The alignment implementation no longer relies on PyMOL.
        """
        return True
    
    def cleanup_pymol(self):
        """Backwards-compatible no-op (PyMOL is no longer used)."""
        return
    
    def find_pdbs(self, directory):
        """Find all PDB files in the given directory and subdirectories."""
        pdb_files = []
        
        # First, try to find PDB files directly in the directory
        direct_pdbs = glob(os.path.join(directory, "*.pdb"))
        pdb_files.extend(direct_pdbs)
        
        # If no direct PDB files found, search recursively in subdirectories
        if not direct_pdbs:
            print(f"No PDB files found directly in {directory}, searching subdirectories...")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.pdb'):
                        pdb_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdb_files)} PDB files in {directory}")
        return pdb_files
    
    def fname(self, path):
        """Extract filename without extension."""
        return os.path.splitext(os.path.basename(path))[0]
    
    def extract_sequence_and_residue_numbers(self, pdb_file):
        """Extract sequence and corresponding PDB residue numbers from atomic coordinates for the first chain found in the PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB', pdb_file)
        
        for model in structure:
            for chain in model:
                ppb = PPBuilder()
                sequence = ''
                residue_numbers = []
                
                for pp in ppb.build_peptides(chain):
                    sequence += pp.get_sequence()
                    # Get the actual residue numbers from the PDB
                    for residue in pp:
                        residue_numbers.append(residue.get_id()[1])  # residue.get_id()[1] gives the residue number
                return str(sequence), residue_numbers
        return None, None
    
    def find_motif_indices(self, seq, motif):
        """Find the indices of a motif in a sequence."""
        index = seq.find(motif)
        if index == -1:
            return None
        return index, index + len(motif)
    
    def print_residues_in_selection(self, atoms, label="selection"):
        """Print residues in an MDAnalysis AtomGroup (for debugging)."""
        try:
            if atoms is None:
                print(f"{label}: <None>")
                return
            residues = sorted({(int(a.resid), str(a.resname)) for a in atoms})
            print(f"Residues in {label}: {residues}")
        except Exception as e:
            print(f"Error printing residues in {label}: {e}")

    def _select_ca_resids(self, u, resids, exclude_altloc=True):
        """
        Select CA atoms for a list of PDB residue numbers (resids), optionally excluding altlocs.
        """
        if not resids:
            return u.atoms[0:0]

        # MDAnalysis selection supports whitespace-separated resid lists.
        resid_str = " ".join(str(int(r)) for r in sorted(set(resids)))
        atoms = u.select_atoms(f"name CA and resid {resid_str}")

        if not exclude_altloc or not self.altloc_exclude or len(atoms) == 0:
            return atoms

        # Robust altloc filtering: prefer AtomGroup.altLocs if available, otherwise keep as-is.
        altlocs = None
        for attr in ("altLocs", "altLoc", "altlocs", "altloc"):
            if hasattr(atoms, attr):
                try:
                    altlocs = getattr(atoms, attr)
                except Exception:
                    altlocs = None
                break

        if altlocs is None:
            return atoms

        altlocs = np.asarray(altlocs)
        # Keep blanks / 'A' etc; drop B/C/D etc.
        mask = np.array([str(a) not in self.altloc_exclude for a in altlocs], dtype=bool)
        return atoms[mask]

    def _motif_resids_for_file(self, pdb_file):
        """
        Return (dfg_resids, ape_resids, ends_resids) as PDB residue numbers for the first chain.
        """
        seq, residue_numbers = self.extract_sequence_and_residue_numbers(pdb_file)
        if seq is None or residue_numbers is None:
            return None, None, None, None

        dfg_indices = self.find_motif_indices(seq, "DFG")
        ape_indices = self.find_motif_indices(seq, "APE")
        if not dfg_indices or not ape_indices:
            return None, None, None, seq

        dfg_resids = [residue_numbers[i] for i in range(dfg_indices[0], dfg_indices[1])]
        ape_resids = [residue_numbers[i] for i in range(ape_indices[0], ape_indices[1])]
        ends_resids = sorted(set(dfg_resids + ape_resids))
        return dfg_resids, ape_resids, ends_resids, seq
    
    def process_structure(self, pdb_file, output_dir, ref_name="6UAN_chainD"):
        """
        Process a single structure for motif-based alignment using MDAnalysis.
        
        Args:
            pdb_file: Path to the PDB file to process
            output_dir: Directory to save aligned output
            ref_name: Reference label (kept for backwards compatibility; not used for object naming)
        """
        try:
            if self._ref_universe is None:
                raise ValueError("Reference not initialized. Call setup_reference_structure(reference_pdb) first.")

            pdb_code = os.path.basename(pdb_file).split('.')[0]

            dfg_resids, ape_resids, ends_resids, seq = self._motif_resids_for_file(pdb_file)
            if dfg_resids is None or ape_resids is None or ends_resids is None:
                print(f"Skipping {pdb_code} due to missing motifs.")
                return

            # Load target PDB
            u = mda.Universe(pdb_file)

            # Build selections (CA-only, optionally excluding altlocs)
            mob_dfg = self._select_ca_resids(u, dfg_resids, exclude_altloc=True)
            mob_ape = self._select_ca_resids(u, ape_resids, exclude_altloc=True)
            mob_ends = self._select_ca_resids(u, ends_resids, exclude_altloc=True)

            ref_dfg = self._select_ca_resids(self._ref_universe, self._ref_resids["DFG"], exclude_altloc=False)
            ref_ape = self._select_ca_resids(self._ref_universe, self._ref_resids["APE"], exclude_altloc=False)
            ref_ends = self._select_ca_resids(self._ref_universe, self._ref_resids["ENDS"], exclude_altloc=False)

            # Validate atom counts (must match to compute RMSD/alignment)
            if len(mob_ends) == 0 or len(ref_ends) == 0 or len(mob_ends) != len(ref_ends):
                print(
                    f"Skipping {pdb_code}: selection mismatch for ENDS "
                    f"(mobile {len(mob_ends)} atoms vs reference {len(ref_ends)} atoms)."
                )
                return

            # RMSD before alignment (no fitting; mirrors PyMOL rms_cur behavior)
            def _safe_rms(a, b):
                if len(a) == 0 or len(b) == 0 or len(a) != len(b):
                    return np.nan
                return float(mda_rms.rmsd(a.positions, b.positions, center=False, superposition=False))

            rms_dfg_before = _safe_rms(mob_dfg, ref_dfg)
            rms_ape_before = _safe_rms(mob_ape, ref_ape)
            rms_ends_before = _safe_rms(mob_ends, ref_ends)

            print(
                f"Before alignment RMSD for {pdb_code}: "
                f"DFG={rms_dfg_before}, APE={rms_ape_before}, ENDS={rms_ends_before}"
            )

            # Compute Kabsch rotation from ENDS CA atoms and apply to ALL atoms of the mobile structure.
            R, _rms_after_fit = mda_align.rotation_matrix(mob_ends.positions, ref_ends.positions)
            mob_com = mob_ends.center_of_mass()
            ref_com = ref_ends.center_of_mass()

            u.atoms.translate(-mob_com)
            u.atoms.rotate(R)
            u.atoms.translate(ref_com)

            # Save aligned structure
            os.makedirs(output_dir, exist_ok=True)
            aligned_pdb_path = os.path.join(output_dir, f"{pdb_code}_aligned.pdb")
            u.atoms.write(aligned_pdb_path)
            print(f"Saved aligned structure to {aligned_pdb_path}")

            # RMSD after alignment (no fitting)
            rms_dfg_after = _safe_rms(self._select_ca_resids(u, dfg_resids, exclude_altloc=True), ref_dfg)
            rms_ape_after = _safe_rms(self._select_ca_resids(u, ape_resids, exclude_altloc=True), ref_ape)
            rms_ends_after = _safe_rms(self._select_ca_resids(u, ends_resids, exclude_altloc=True), ref_ends)

            print(
                f"After alignment RMSD for {pdb_code}: "
                f"DFG={rms_dfg_after}, APE={rms_ape_after}, ENDS={rms_ends_after}"
            )
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    def compute_anchor_positions_from_conservation(
        self,
        *,
        reference_pdb: str,
        conservation,
        threshold: float = 0.997,
    ):
        """
        Identify anchored regions adjacent to DFG and APE using a conservation/coverage array.

        Interpretation of `conservation` here matches the notebook's current implementation:
        for each reference position, it is the fraction of structures that are NOT gaps at that position.

        We define two anchor regions:
        - left anchor: consecutive positions immediately before the DFG motif with conservation >= threshold
        - right anchor: consecutive positions immediately after the APE motif with conservation >= threshold

        Returns:
            dict with keys:
              - dfg_indices, ape_indices (tuple start,end; end exclusive)
              - left_anchor_positions (list of ints, ascending)
              - right_anchor_positions (list of ints, ascending)
              - anchor_positions (combined list of ints, ascending)
        """
        conservation = np.asarray(conservation, dtype=float)

        seq, _resnums = self.extract_sequence_and_residue_numbers(reference_pdb)
        if seq is None:
            raise ValueError(f"Could not extract sequence from reference_pdb={reference_pdb}")

        dfg_indices = self.find_motif_indices(seq, "DFG")
        ape_indices = self.find_motif_indices(seq, "APE")
        if not dfg_indices or not ape_indices:
            raise ValueError("Could not find DFG and/or APE motifs in reference structure sequence.")

        dfg_start = dfg_indices[0]
        ape_end = ape_indices[1]  # exclusive

        if dfg_start <= 0 or ape_end >= len(conservation):
            raise ValueError(
                f"Motif indices out of bounds for conservation array: "
                f"dfg_start={dfg_start}, ape_end={ape_end}, len(conservation)={len(conservation)}"
            )

        # LEFT side: scan leftwards from DFG until we *find* a conserved position,
        # then collect the full consecutive conserved block and stop.
        left = []
        i = dfg_start - 1
        # skip non-conserved positions
        while i >= 0 and conservation[i] < threshold:
            i -= 1
        # collect the consecutive conserved block
        while i >= 0 and conservation[i] >= threshold:
            left.append(i)
            i -= 1
        left = sorted(left)

        # RIGHT side: scan rightwards from APE until we *find* a conserved position,
        # then collect the full consecutive conserved block and stop.
        right = []
        i = ape_end
        # skip non-conserved positions
        while i < len(conservation) and conservation[i] < threshold:
            i += 1
        # collect the consecutive conserved block
        while i < len(conservation) and conservation[i] >= threshold:
            right.append(i)
            i += 1

        anchor_positions = sorted(set(left + right))

        anchor_info = {
            "dfg_indices": dfg_indices,
            "ape_indices": ape_indices,
            "left_anchor_positions": left,
            "right_anchor_positions": right,
            "anchor_positions": anchor_positions,
            "threshold": float(threshold),
        }
        # Cache for later (e.g., plotting without recomputing anything)
        self._last_conservation = conservation
        self._last_anchor_info = anchor_info
        return anchor_info

    def save_conservation_plot_with_anchor_rectangles(
        self,
        *,
        conservation=None,
        reference_residues=None,
        anchor_info=None,
        output_file: str,
        total_structures: int = None,
        figsize=(15, 6),
        tick_interval: int = 10,
        bordeaux: str = "#800020",
        alpha: float = 0.22,
        show_plot: bool = False,
    ):
        """
        Save a conservation plot (same style as analyse_alignment.visualize_residue_conservation),
        but with two full-height bordeaux rectangles spanning the left/right anchor ranges.

        This is designed so you don't need to recompute anything:
        - pass the in-memory `conservation` array from the notebook, OR
        - omit it and this will use the last cached conservation from
          `compute_anchor_positions_from_conservation`.

        Args:
            conservation: 1D array of per-position conservation in [0,1]
            reference_residues: list like ["ALA-449", ...] used for x tick labels
            anchor_info: dict returned by compute_anchor_positions_from_conservation
            output_file: where to write the PNG
            total_structures: optional integer for the title ("across N Structures")
            tick_interval: show every Nth x tick label
            bordeaux: rectangle color (default #800020)
            alpha: rectangle alpha
            show_plot: if True, display interactively (otherwise just saves)
        """
        # Local import so non-plotting workflows don't require matplotlib at import time.
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.ticker import FuncFormatter

        if conservation is None:
            conservation = self._last_conservation
        if anchor_info is None:
            anchor_info = self._last_anchor_info

        if conservation is None:
            raise ValueError(
                "No conservation array provided and none cached. "
                "Pass `conservation=...` from the notebook."
            )
        if anchor_info is None:
            raise ValueError(
                "No anchor_info provided and none cached. "
                "Pass `anchor_info=...` returned by compute_anchor_positions_from_conservation."
            )

        conservation = np.asarray(conservation, dtype=float)
        x = list(range(len(conservation)))

        # Helper function to format residue labels (matches analyse_alignment implementation)
        def _format_residue_label(idx: int) -> str:
            if reference_residues and idx < len(reference_residues):
                res = reference_residues[idx]  # e.g., "ALA-123"
                parts = str(res).split("-")
                if len(parts) == 2:
                    return f"{parts[0]}{parts[1]} ({idx})"
                return f"{res} ({idx})"
            return f"({idx})"

        fig, ax = plt.subplots(1, figsize=figsize)

        # Bars
        ax.bar(
            x,
            conservation,
            linewidth=0.05,
            width=1,
            color="royalblue",
            alpha=0.5,
            edgecolor="steelblue",
            zorder=10,
        )

        # Structural regions (same defaults as analyse_alignment.visualize_residue_conservation)
        color_regions = [
            (14, 21, "blue", "pLoop"),
            (43, 58, "green", "alphaC"),
            (145, 168, "hotpink", "DFG-APE"),
        ]
        for start, end, color, _label in color_regions:
            for i in range(start, end + 1):
                if i < len(conservation):
                    bar_height = float(conservation[i])
                    rect_height = 1.0 - bar_height
                    if rect_height > 0:
                        rect = patches.Rectangle(
                            (i - 0.5, bar_height),
                            1,
                            rect_height,
                            facecolor=color,
                            alpha=0.7,
                            zorder=5,
                        )
                        ax.add_patch(rect)

        # Bordeaux rectangles spanning the full plot height for the left/right anchor ranges
        left_positions = list(anchor_info.get("left_anchor_positions") or [])
        right_positions = list(anchor_info.get("right_anchor_positions") or [])

        def _axvspan_positions(pos_list):
            if not pos_list:
                return
            lo = int(min(pos_list))
            hi = int(max(pos_list))
            # span full height (ymin=0..ymax=1 in axes coords); expand by 0.5 to cover full bars
            ax.axvspan(
                lo - 0.5,
                hi + 0.5,
                ymin=0,
                ymax=1,
                color=bordeaux,
                alpha=float(alpha),
                zorder=7,
            )

        _axvspan_positions(left_positions)
        _axvspan_positions(right_positions)

        # X ticks
        tick_positions = x[:: max(int(tick_interval), 1)]
        tick_labels = [_format_residue_label(i) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, fontsize=8, ha="right")

        # Formatting
        ax.set_xlim(-0.5, len(conservation) - 0.5)
        ax.set_ylim(0, 1.0001)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=0, labelsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

        plt.xlabel("ResiduePDBindex (0-based index)", fontsize=12)
        plt.ylabel("Conservation Percentage", fontsize=12)
        if total_structures is not None:
            plt.title(f"Residue Conservation across {int(total_structures)} Structures", fontsize=14)
        else:
            plt.title("Residue Conservation", fontsize=14)

        plt.tight_layout()

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Conservation plot with anchors saved to {output_file}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def _sequence_indices_to_pdb_resids(self, pdb_file: str, indices):
        """
        Map sequence indices (0-based) -> PDB residue numbers using Bio.PDB peptide mapping.
        Returns a list of PDB residue numbers (same order as indices).
        """
        _seq, residue_numbers = self.extract_sequence_and_residue_numbers(pdb_file)
        if residue_numbers is None:
            return None

        resids = []
        for idx in indices:
            if idx < 0 or idx >= len(residue_numbers):
                return None
            resids.append(residue_numbers[idx])
        return resids

    def process_anchored_alignment(
        self,
        *,
        pdb_dir: str,
        reference_pdb: str,
        output_dir: str,
        anchor_positions,
        reference_residues=None,
        matrices_csv: str = None,
        threshold_label: str = "0.997",
        ref_name: str = "6UAN_chainD",
    ):
        """
        Anchored alignment:
        - filters out structures missing any anchor positions (sequence indices)
        - aligns each structure to the reference using CA atoms at those anchor positions
        - saves per-structure rotation/translation matrices
        - writes aligned PDBs to output_dir
        """
        import pandas as pd

        anchor_positions = list(anchor_positions) if anchor_positions is not None else []
        if len(anchor_positions) == 0:
            raise ValueError(
                "No anchor positions were provided (empty anchor_positions). "
                "This usually means your ≥99.7% conservation criterion found no consecutive residues "
                "immediately adjacent to DFG/APE. Try lowering the threshold or inspect the conservation array."
            )

        os.makedirs(output_dir, exist_ok=True)
        fps = self.find_pdbs(pdb_dir)
        if not fps:
            raise ValueError(f"No PDB files found in: {pdb_dir}")

        # Prepare reference selections
        self._ref_universe = mda.Universe(reference_pdb)
        ref_anchor_resids = self._sequence_indices_to_pdb_resids(reference_pdb, anchor_positions)
        if ref_anchor_resids is None:
            raise ValueError("Could not map anchor_positions to reference PDB residue numbers.")
        ref_atoms = self._select_ca_resids(self._ref_universe, ref_anchor_resids, exclude_altloc=False)
        if len(ref_atoms) == 0:
            raise ValueError("Reference anchor CA selection is empty.")

        # Print anchor positions in BRAF numbering if provided
        if reference_residues is not None:
            print(f"\n=== Anchoring regions (threshold ≥ {threshold_label}) ===")
            for idx in anchor_positions:
                if idx < len(reference_residues):
                    print(f"  Position {idx:3d}: {reference_residues[idx]}")
                else:
                    print(f"  Position {idx:3d}: (no reference residue label available)")

        excluded = []
        kept = []
        results = []

        iterator = tqdm(fps, desc="Anchored aligning PDBs") if tqdm is not None else fps
        for fp in iterator:
            name = os.path.basename(fp)

            mob_anchor_resids = self._sequence_indices_to_pdb_resids(fp, anchor_positions)
            if mob_anchor_resids is None:
                excluded.append(name)
                continue

            u = mda.Universe(fp)
            mob_atoms = self._select_ca_resids(u, mob_anchor_resids, exclude_altloc=True)

            if len(mob_atoms) != len(ref_atoms) or len(mob_atoms) == 0:
                excluded.append(name)
                continue

            # Compute transform from anchor CA atoms
            R, rms_fit = mda_align.rotation_matrix(mob_atoms.positions, ref_atoms.positions)
            mob_com = mob_atoms.center_of_mass()
            ref_com = ref_atoms.center_of_mass()
            t = ref_com - mob_com @ R  # for completeness; we apply using translate/rotate/translate below

            # Apply to all atoms
            u.atoms.translate(-mob_com)
            u.atoms.rotate(R)
            u.atoms.translate(ref_com)

            out_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_aligned.pdb")
            u.atoms.write(out_path)

            kept.append(name)

            # Store matrices (R and translation to map x -> (x - mob_com)R + ref_com)
            results.append(
                {
                    "pdb_file": name,
                    "output_pdb": os.path.basename(out_path),
                    "n_anchor_atoms": int(len(mob_atoms)),
                    "rms_fit": float(rms_fit),
                    "R00": float(R[0, 0]),
                    "R01": float(R[0, 1]),
                    "R02": float(R[0, 2]),
                    "R10": float(R[1, 0]),
                    "R11": float(R[1, 1]),
                    "R12": float(R[1, 2]),
                    "R20": float(R[2, 0]),
                    "R21": float(R[2, 1]),
                    "R22": float(R[2, 2]),
                    "mob_com_x": float(mob_com[0]),
                    "mob_com_y": float(mob_com[1]),
                    "mob_com_z": float(mob_com[2]),
                    "ref_com_x": float(ref_com[0]),
                    "ref_com_y": float(ref_com[1]),
                    "ref_com_z": float(ref_com[2]),
                    "t_x": float(t[0]),
                    "t_y": float(t[1]),
                    "t_z": float(t[2]),
                }
            )

        print(f"\n=== Anchored alignment filtering ===")
        print(f"Total structures: {len(fps)}")
        print(f"Kept (have all anchor positions): {len(kept)}")
        print(f"Excluded (missing anchor positions / mismatch): {len(excluded)}")
        if excluded:
            print("\nExcluded structures:")
            for s in excluded:
                print(f"  - {s}")

        df = pd.DataFrame(results)
        if matrices_csv is None:
            matrices_csv = os.path.join(output_dir, "anchored_alignment_matrices.csv")
        df.to_csv(matrices_csv, index=False)
        print(f"\nSaved matrices to: {matrices_csv}")

        return {
            "kept": kept,
            "excluded": excluded,
            "matrices": df,
            "output_dir": output_dir,
            "ref_name": ref_name,
        }
    
    def setup_reference_structure(self, reference_pdb, ref_name="6UAN_chainD"):
        """
        Set up the reference structure for motif-based alignment using MDAnalysis.
        
        Args:
            reference_pdb: Path to the reference PDB file
            ref_name: Name to assign to the reference structure
        """
        try:
            self._ref_universe = mda.Universe(reference_pdb)
            self._ref_name = ref_name

            dfg_resids, ape_resids, ends_resids, _seq = self._motif_resids_for_file(reference_pdb)
            if dfg_resids is None or ape_resids is None or ends_resids is None:
                print("Error: Could not find DFG or APE motifs in reference structure")
                self._ref_universe = None
                return False

            self._ref_resids = {"DFG": dfg_resids, "APE": ape_resids, "ENDS": ends_resids}

            print(f"Reference DFG PDB residue numbers: {dfg_resids}")
            print(f"Reference APE PDB residue numbers: {ape_resids}")
            print("Reference structure selections:")
            self.print_residues_in_selection(
                self._select_ca_resids(self._ref_universe, dfg_resids, exclude_altloc=False),
                label=f"{ref_name}_dfg_selection",
            )
            self.print_residues_in_selection(
                self._select_ca_resids(self._ref_universe, ape_resids, exclude_altloc=False),
                label=f"{ref_name}_ape_selection",
            )
            self.print_residues_in_selection(
                self._select_ca_resids(self._ref_universe, ends_resids, exclude_altloc=False),
                label=f"{ref_name}_ends_selection",
            )
            return True
        except Exception as e:
            print(f"Error setting up reference structure: {e}")
            return False
    
    def process_pymol_alignment(self, pdb_dir, reference_pdb, output_dir, ref_name="6UAN_chainD"):
        """
        Process multiple PDB files through motif-based alignment using MDAnalysis.
        
        Args:
            pdb_dir: Directory containing PDB files to align
            reference_pdb: Reference PDB file for alignment
            output_dir: Directory to save aligned outputs
            ref_name: Name to assign to the reference structure
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup reference structure
            if not self.setup_reference_structure(reference_pdb, ref_name):
                print("Failed to setup reference structure")
                return
            
            # Load all PDB files and process them
            fps = self.find_pdbs(pdb_dir)
            if not fps:
                print("ERROR: No PDB files found to process!")
                return
            
            print(f"Found {len(fps)} PDB files to process")
            iterator = tqdm(fps, desc="Aligning PDBs") if tqdm is not None else fps
            for fp in iterator:
                print(f"Processing: {fp}")
                self.process_structure(fp, output_dir, ref_name)

            print("MDAnalysis alignment processing completed successfully")
            
        except Exception as e:
            print(f"Error during alignment processing: {e}")




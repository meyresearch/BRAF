import os
from glob import glob
from time import time
import numpy as np

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
        try:
            from Bio.PDB import PDBParser, PPBuilder
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Biopython is required for extract_sequence_and_residue_numbers(). "
                "Install it with `pip install biopython` (or your environment equivalent)."
            ) from e

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
            try:
                import MDAnalysis as mda
                import MDAnalysis.analysis.align as mda_align
                import MDAnalysis.analysis.rms as mda_rms
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "MDAnalysis is required for structure alignment. "
                    "Install it with `pip install MDAnalysis` (or your environment equivalent)."
                ) from e

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
        threshold: float = 0.90,
        n_values=range(6, 41, 2),
    ):
        """
        Identify anchor positions around DFG and APE using a conservation/coverage array,
        allowing non-consecutive residues.

        Interpretation of `conservation` here matches the notebook:
        for each reference position, it is the fraction of structures that are NOT gaps at that position.

        Anchor definition:
        - consider all positions with conservation >= threshold (not necessarily consecutive)
        - INCLUDE the DFG and APE motif residue positions in every anchor set (regardless of conservation)
        - build multiple anchor sets for N in n_values (e.g., 6, 8, 10, ..., 44):
          * left side: start from the first residue of DFG and walk left; select conserved positions
            strictly BEFORE DFG (i < dfg_start)
          * right side: start from the last residue of APE and walk right; select conserved positions
            strictly AFTER APE (i >= ape_end)
          * for each N: take the first N positions from the left walk AND the first N positions
            from the right walk (so total anchors contributed by conservation = 2N),
            then add the motif positions (DFG + APE) and combine them (ascending)

        Returns dict with keys:
          - dfg_indices, ape_indices (tuple start,end; end exclusive)
          - dfg_start, ape_end (0-based sequence indices defining the walk boundaries)
          - conserved_positions (sorted list of indices where conservation >= threshold)
          - left_walk, right_walk (full ordered walks)
          - anchor_sets: dict mapping N -> list[int] anchor_positions (ascending)
          - left_anchor_positions, right_anchor_positions, anchor_positions: aliases for the max-N set
            (kept for backwards compatibility / printing)
        """
        conservation = np.asarray(conservation, dtype=float)
        if conservation.ndim != 1:
            raise ValueError("conservation must be a 1D array")

        seq, _resnums = self.extract_sequence_and_residue_numbers(reference_pdb)
        if seq is None:
            raise ValueError(f"Could not extract sequence from reference_pdb={reference_pdb}")

        dfg_indices = self.find_motif_indices(seq, "DFG")
        ape_indices = self.find_motif_indices(seq, "APE")
        if not dfg_indices or not ape_indices:
            raise ValueError("Could not find DFG and/or APE motifs in reference structure sequence.")

        dfg_start, dfg_end = dfg_indices  # end exclusive
        ape_start, ape_end = ape_indices  # end exclusive

        # Sanity check bounds for walk boundaries
        if dfg_start < 0 or dfg_start >= len(conservation) or ape_end < 0 or ape_end > len(conservation):
            raise ValueError(
                f"Motif indices out of bounds for conservation array: "
                f"dfg_start={dfg_start}, ape_end={ape_end}, len(conservation)={len(conservation)}"
            )

        conserved = set(np.where(conservation >= float(threshold))[0].tolist())
        conserved_positions = sorted(conserved)

        # Motif positions (always included in the anchors, even if below threshold).
        motif_positions = sorted(
            {
                *range(int(dfg_start), int(dfg_end)),
                *range(int(ape_start), int(ape_end)),
            }
        )

        # Build non-consecutive "walks" over conserved positions OUTSIDE the motifs.
        # Left: positions strictly before DFG start, nearest-to-DFG first.
        left_walk = sorted([i for i in conserved if i < int(dfg_start)], reverse=True)
        # Right: positions strictly after APE end, nearest-to-APE first.
        right_walk = sorted([i for i in conserved if i >= int(ape_end)])

        anchor_sets = {}
        for n in list(n_values):
            n = int(n)
            if n <= 0:
                raise ValueError(f"n_values must contain positive integers; got N={n}")
            if len(left_walk) < n or len(right_walk) < n:
                raise ValueError(
                    f"Not enough positions to build N={n} anchors on both sides: "
                    f"left has {len(left_walk)}, right has {len(right_walk)}. "
                    "Try lowering threshold or reducing max N."
                )
            left_sel = left_walk[:n]
            right_sel = right_walk[:n]
            anchors = sorted(set(left_sel + right_sel + motif_positions))
            expected = (2 * n) + len(motif_positions)
            if len(anchors) != expected:
                raise ValueError(
                    f"Unexpected anchor count for N={n}: got {len(anchors)} unique positions (expected {expected}). "
                    "This may indicate overlapping motif regions or a logic error."
                )
            anchor_sets[n] = anchors

        # Backwards-compatible aliases: use the max-N selection
        max_n = max(int(x) for x in list(n_values)) if n_values is not None else None
        anchor_positions = anchor_sets[max_n] if max_n is not None else []
        left_anchor_positions = sorted(set(left_walk[:max_n])) if max_n is not None else []
        right_anchor_positions = sorted(set(right_walk[:max_n])) if max_n is not None else []

        anchor_info = {
            "dfg_indices": dfg_indices,
            "ape_indices": ape_indices,
            "dfg_start": int(dfg_start),
            "ape_end": int(ape_end),
            "motif_positions": motif_positions,
            "conserved_positions": conserved_positions,
            "left_walk": left_walk,
            "right_walk": right_walk,
            "anchor_sets": anchor_sets,
            "left_anchor_positions": left_anchor_positions,
            "right_anchor_positions": right_anchor_positions,
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
        highlight_individual_positions: bool = True,
        show_plot: bool = False,
    ):
        """
        Save a conservation plot (same style as analyse_alignment.visualize_residue_conservation),
        with anchor positions highlighted.

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
            highlight_individual_positions: if True (default), highlight each anchor residue
                index individually; if False, highlight the whole left/right anchor ranges.
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

        # Anchor highlighting
        left_positions = list(anchor_info.get("left_anchor_positions") or [])
        right_positions = list(anchor_info.get("right_anchor_positions") or [])
        motif_positions = list(anchor_info.get("motif_positions") or [])

        if highlight_individual_positions:
            # Highlight *individual* anchor residue positions used in the alignment.
            # We draw a full-height 1-residue-wide rectangle centered on each index.
            anchor_positions = sorted({int(i) for i in (left_positions + right_positions + motif_positions)})
            for i in anchor_positions:
                if 0 <= i < len(conservation):
                    rect = patches.Rectangle(
                        (i - 0.5, 0.0),
                        1.0,
                        1.0,
                        facecolor=bordeaux,
                        alpha=float(alpha),
                        linewidth=0.0,
                        zorder=2,  # behind bars (zorder=10) so the bars remain readable
                    )
                    ax.add_patch(rect)
        else:
            # Backwards-compatible behavior: highlight the *ranges* spanned by the
            # left and right anchor selections.
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
        try:
            import MDAnalysis as mda
            import MDAnalysis.analysis.align as mda_align
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MDAnalysis is required for anchored alignment. "
                "Install it with `pip install MDAnalysis` (or your environment equivalent)."
            ) from e

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

    # ---------------------------------------------------------------------
    # FoldMason-based truncation helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _ordered_residue_keys_from_pdb_atoms(pdb_path: str):
        """
        Return ordered residue keys encountered in ATOM records.

        Key format: (chain_id, resseq_int, icode_str)
        """
        residue_keys = []
        seen = set()
        with open(pdb_path, "r") as fin:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21].strip()
                try:
                    resseq = int(line[22:26])
                except Exception:
                    continue
                icode = line[26].strip()
                key = (chain, resseq, icode)
                if key not in seen:
                    seen.add(key)
                    residue_keys.append(key)
        return residue_keys

    @classmethod
    def _truncate_pdb_by_target_indices(cls, src_pdb: str, out_pdb: str, target_indices):
        """
        Truncate a PDB by residue indices (0-based) and write to out_pdb.

        Priority order:
        - MDAnalysis (if installed): select protein residues by mapped indices and write
        - Pure-PDB fallback: use ATOM-record residue order as residue-index order and write ATOM lines only
        """
        if not target_indices:
            return {"wrote": False, "reason": "empty target_indices"}

        # --- Preferred path: MDAnalysis (keeps only protein atoms cleanly)
        try:
            import MDAnalysis as mda

            u = mda.Universe(src_pdb)
            protein = u.select_atoms("protein")
            if len(protein) == 0:
                return {"wrote": False, "reason": "no protein atoms"}

            residue_numbers = None
            try:
                align_util = cls()
                _seq, residue_numbers = align_util.extract_sequence_and_residue_numbers(src_pdb)
            except Exception:
                residue_numbers = None

            if residue_numbers is None:
                residue_numbers = [int(r.resid) for r in protein.residues]

            target_indices = [i for i in target_indices if 0 <= i < len(residue_numbers)]
            if not target_indices:
                return {"wrote": False, "reason": "target_indices out of bounds"}

            tar_resids = sorted({int(residue_numbers[i]) for i in target_indices})
            resid_str = " ".join(str(r) for r in tar_resids)

            sel = protein.select_atoms(f"resid {resid_str}")
            if len(sel) == 0:
                return {"wrote": False, "reason": "MDAnalysis selection empty"}

            os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
            sel.write(out_pdb)
            return {"wrote": True, "n_atoms": int(len(sel))}
        except Exception:
            pass

        # --- Pure-PDB fallback (no external deps)
        residue_keys = cls._ordered_residue_keys_from_pdb_atoms(src_pdb)
        if not residue_keys:
            return {"wrote": False, "reason": "no ATOM residues found"}

        target_indices = [i for i in target_indices if 0 <= i < len(residue_keys)]
        if not target_indices:
            return {"wrote": False, "reason": "target_indices out of bounds"}

        keep_keys = {residue_keys[i] for i in target_indices}

        os.makedirs(os.path.dirname(out_pdb) or ".", exist_ok=True)
        wrote_atoms = 0
        with open(src_pdb, "r") as fin, open(out_pdb, "w") as fout:
            for line in fin:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21].strip()
                try:
                    resseq = int(line[22:26])
                except Exception:
                    continue
                icode = line[26].strip()
                if (chain, resseq, icode) in keep_keys:
                    fout.write(line)
                    wrote_atoms += 1
            fout.write("END\n")

        if wrote_atoms == 0:
            return {"wrote": False, "reason": "no atoms written"}

        return {"wrote": True, "n_atoms": int(wrote_atoms)}

    def truncate_aligned_anchor_datasets_foldmason(
        self,
        *,
        anchor_info: dict = None,
        n_values=None,
        reference_residues: list,
        alignment_file: str = "Results/activation_segments/multi_aligned_foldmason/msa_3di.fa",
        reference_id: str = "6UAN_chainD",
        ref_start_resnum: int = 528,
        ref_end_resnum: int = 697,
        in_root: str = "Results/activation_segments",
        in_prefix: str = "aligned_anchor_",
        out_prefix: str = "aligned_anchor_trunc_",
        overwrite: bool = True,
    ):
        """
        Copy+truncate structures from aligned anchor datasets into aligned_anchor_trunc_{N}.

        Truncation criterion:
        - keep only residues that are structurally aligned (FoldMason) to the reference region
          whose residue numbers are in [ref_start_resnum, ref_end_resnum] in `reference_residues`.

        Inputs:
        - `anchor_info["anchor_sets"]` provides the N values to process (optional).
        - Alternatively pass `n_values=[6,7,8,...]`.
        - If neither is provided, this auto-discovers `aligned_anchor_<N>/` folders under `in_root`.
        - `alignment_file` must contain both `reference_id` and the target structure IDs used
          in the FoldMason multi-alignment (e.g., `1A9U_A`, `6UAN_chainD`).
        """
        from workflow.analyse_alignment_foldmason import analyse_alignment

        wanted_ref_indices = analyse_alignment.reference_indices_for_resnum_range(
            reference_residues, start_resnum=ref_start_resnum, end_resnum=ref_end_resnum
        )
        if not wanted_ref_indices:
            raise ValueError(
                f"No reference indices found for residue numbers {ref_start_resnum}–{ref_end_resnum}. "
                "Check `reference_residues` numbering."
            )
        wanted_ref_set = set(wanted_ref_indices)

        seq_by_id = analyse_alignment.load_alignment_dict(alignment_file)
        if reference_id not in seq_by_id:
            raise ValueError(f"Reference '{reference_id}' not found in FoldMason alignment file: {alignment_file}")
        ref_aln = seq_by_id[reference_id]

        cache_target_indices = {}
        summary = {}

        # Determine which N values to run
        n_list = None
        if n_values is not None:
            n_list = [int(x) for x in list(n_values)]
        else:
            anchor_sets = (anchor_info or {}).get("anchor_sets") or {}
            if anchor_sets:
                n_list = [int(x) for x in anchor_sets.keys()]
            else:
                # Auto-discover folders like aligned_anchor_6, aligned_anchor_10, ...
                n_list = []
                for d in glob(os.path.join(in_root, f"{in_prefix}*")):
                    if not os.path.isdir(d):
                        continue
                    suffix = os.path.basename(d)[len(in_prefix) :]
                    try:
                        n_list.append(int(suffix))
                    except Exception:
                        continue

        n_list = sorted(set(n_list))
        if not n_list:
            raise ValueError(
                "No N values provided/found. Provide `anchor_info`, or `n_values=[...]`, or ensure "
                f"folders like '{in_prefix}<N>/' exist under in_root='{in_root}'."
            )

        for N in n_list:
            in_dir = os.path.join(in_root, f"{in_prefix}{N}")
            out_dir = os.path.join(in_root, f"{out_prefix}{N}")
            os.makedirs(out_dir, exist_ok=True)

            pdb_files = sorted(glob(os.path.join(in_dir, "*.pdb")))
            wrote = 0
            skipped = 0

            for src_pdb in pdb_files:
                out_pdb = os.path.join(out_dir, os.path.basename(src_pdb))
                if (not overwrite) and os.path.exists(out_pdb):
                    wrote += 1
                    continue

                msa_id = analyse_alignment.pdb_to_msa_id(src_pdb)
                target_aln = seq_by_id.get(msa_id)
                if target_aln is None:
                    skipped += 1
                    continue

                if msa_id not in cache_target_indices:
                    cache_target_indices[msa_id] = analyse_alignment.target_indices_aligned_to_reference_set(
                        ref_aln=ref_aln,
                        target_aln=target_aln,
                        wanted_ref_set=wanted_ref_set,
                    )

                res = self._truncate_pdb_by_target_indices(src_pdb, out_pdb, cache_target_indices[msa_id])
                if res.get("wrote"):
                    wrote += 1
                else:
                    skipped += 1

            summary[int(N)] = {
                "total": int(len(pdb_files)),
                "wrote": int(wrote),
                "skipped": int(skipped),
                "out_dir": out_dir,
            }
            print(f"N={N}: wrote {wrote}/{len(pdb_files)} truncated PDBs -> {out_dir} (skipped={skipped})")

        return summary
    
    def setup_reference_structure(self, reference_pdb, ref_name="6UAN_chainD"):
        """
        Set up the reference structure for motif-based alignment using MDAnalysis.
        
        Args:
            reference_pdb: Path to the reference PDB file
            ref_name: Name to assign to the reference structure
        """
        try:
            try:
                import MDAnalysis as mda
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "MDAnalysis is required for structure alignment. "
                    "Install it with `pip install MDAnalysis` (or your environment equivalent)."
                ) from e

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




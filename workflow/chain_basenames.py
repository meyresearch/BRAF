"""Shared chain-basename helpers (lightweight — no heavy pipeline imports)."""

from __future__ import annotations

import os


def normalize_chain_basename(name: str) -> str:
    """Return ``PDB_CHAIN`` form without a ``.pdb`` suffix.

    Accepts ``3KMW_A``, ``3KMW_A.pdb``, or a full path ending in either form.
    """
    base = os.path.basename(str(name).strip())
    if base.lower().endswith(".pdb"):
        return os.path.splitext(base)[0]
    return base


def load_excluded_pseudokinase_basenames(path: str, *, required: bool = False) -> set[str]:
    """Load pseudokinase exclusion basenames (one ``PDB_CHAIN`` per line).

    Parameters
    ----------
    path : str
        Path to ``excluded_pseudokinase_basenames.txt``.
    required : bool
        When ``True``, raise ``FileNotFoundError`` if *path* does not exist.

    Returns
    -------
    set[str]
        Normalized basenames (no ``.pdb`` suffix). Empty set when the file
        exists but is empty and *required* is ``False``.
    """
    if not path:
        if required:
            raise FileNotFoundError("Pseudokinase exclusion path is empty.")
        return set()
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(
                f"Pseudokinase exclusion file not found: {path}\n"
                "Run KinaseGroupLabeller.annotate_dataset_chains_with_kinome(...) "
                "first (Workflow 1 §2.1)."
            )
        return set()
    with open(path, encoding="utf-8") as fh:
        return {normalize_chain_basename(line) for line in fh if line.strip()}

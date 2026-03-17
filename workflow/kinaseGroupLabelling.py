"""
Minimal utilities to annotate PDB IDs with UniProt accession, HGNC gene symbol,
and Kinome group/family (via KLIFS).

Refactored into a class with an auto-discovery method to collect PDB IDs
from a downloads directory. Safe to import in notebooks.
"""

import os
import re
import glob
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from pathlib import Path
import zipfile
import time

class KinaseGroupLabeller:
    """
    Annotate PDB entries with UniProt accession, gene symbol, and kinome group/family.

    - Collect PDB IDs from a directory tree (e.g., Results/InterProPDBs)
    - Map PDB -> UniProt (PDBe SIFTS)
    - Map UniProt -> gene symbol (UniProt REST)
    - Map UniProt -> group/family/class/species (UniProt REST)
    """

    def __init__(self, downloads_dir: str = "Results/InterProPDBs", filter_species: str | None = None):
        self.downloads_dir = downloads_dir
        self.filter_species = filter_species
        self._kinase_info_cache = None
        self._uniprot_json_cache: dict[str, dict | None] = {}
        self._http = requests.Session()

    def collect_pdb_ids_from_dir(self, root: str | None = None) -> list:
        """
        Discover PDB IDs by scanning file names in a directory tree.
        Matches 4-character PDB codes at the start of filenames.
        """
        search_root = root or self.downloads_dir
        ids = set()
        for path in glob.glob(os.path.join(search_root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            base = os.path.basename(path)
            m = re.match(r"([0-9][A-Za-z0-9]{3})", base)
            if m:
                ids.add(m.group(1).upper())
        return sorted(ids)

    def collect_pdb_chains_from_dataset(self, dataset_dir: str) -> pd.DataFrame:
        """
        Collect (pdb_id, chain_id) pairs from an extracted-chain dataset directory.

        Expected filenames: <PDB>_<CHAIN>.pdb (e.g. 1A06_A.pdb)

        Returns DataFrame columns: pdb_file, pdb_id, chain_id
        """
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

        rows = []
        for path in glob.glob(os.path.join(dataset_dir, "*.pdb")):
            base = os.path.basename(path)
            # NOTE: use '\.pdb' (not '\\.pdb') so we match a literal dot before the extension.
            m = re.match(r"^([0-9][A-Za-z0-9]{3})_([A-Za-z0-9])\.pdb$", base, flags=re.IGNORECASE)
            if not m:
                continue
            rows.append({
                "pdb_file": base,
                "pdb_id": m.group(1).upper(),
                "chain_id": m.group(2),
            })
        return pd.DataFrame(rows, columns=["pdb_file", "pdb_id", "chain_id"]).drop_duplicates()

    def annotate_dataset_chains_with_kinome(
        self,
        dataset_dir: str,
        output_csv: str | None = None,
        caution_output_csv: str | None = None,
    ) -> pd.DataFrame:
        """
        Annotate each extracted PDB chain in `dataset_dir` with UniProt/gene/group/family/etc.

        This differs from `run()` (which only discovers 4-letter PDB IDs) by retaining chain_id
        and joining annotations back onto each chain file.

        Returns DataFrame columns:
          pdb_file, pdb_id, chain_id, uniprot_acc, gene, group, family, kinase_class, species,
          uniprot_caution, pseudokinase
        """
        chains = self.collect_pdb_chains_from_dataset(dataset_dir)
        if chains.empty:
            return pd.DataFrame(columns=[
                "pdb_file", "pdb_id", "chain_id", "uniprot_acc", "gene", "group", "family", "kinase_class", "species",
                "uniprot_caution", "pseudokinase"
            ])

        pdb_ids = sorted(chains["pdb_id"].unique().tolist())
        annot = self.annotate_pdbs_with_kinome(pdb_ids)
        out = chains.merge(annot, on=["pdb_id", "chain_id"], how="left")

        # Add UniProt CAUTION section + pseudokinase label
        if "uniprot_acc" in out.columns and out["uniprot_acc"].notna().any():
            caution_df = self.fetch_uniprot_caution_info(out["uniprot_acc"].dropna().unique().tolist())
            if not caution_df.empty:
                out = out.assign(uniprot_root=out["uniprot_acc"].map(self._normalize_uniprot))
                out = out.merge(caution_df, on="uniprot_root", how="left")
                # Ensure consistent boolean output even when UniProt has no CAUTION section
                if "pseudokinase" in out.columns:
                    out["pseudokinase"] = out["pseudokinase"].fillna(False).astype(bool)
                # Drop helper join key from the returned table
                if "uniprot_root" in out.columns:
                    out = out.drop(columns=["uniprot_root"])
        else:
            out["uniprot_caution"] = None
            out["pseudokinase"] = False

        # Save a per-chain document of CAUTION text (as requested)
        # Default: if output_csv is provided, write alongside it as *_uniprot_caution.csv
        if caution_output_csv is None and output_csv:
            base, ext = os.path.splitext(output_csv)
            caution_output_csv = f"{base}_uniprot_caution.csv"

        if caution_output_csv:
            try:
                os.makedirs(os.path.dirname(caution_output_csv) or ".", exist_ok=True)
            except Exception:
                pass
            cols = [c for c in [
                "pdb_file", "pdb_id", "chain_id", "uniprot_acc", "gene",
                "uniprot_caution", "pseudokinase"
            ] if c in out.columns]
            out[cols].to_csv(caution_output_csv, index=False)

        if output_csv:
            try:
                os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            except Exception:
                pass
            out.to_csv(output_csv, index=False)

        return out

    def pdb_to_uniprot(self, pdb_ids: list) -> pd.DataFrame:
        """
        Map PDB IDs -> UniProt accessions per chain using PDBe SIFTS.
        Returns DataFrame columns: pdb_id, chain_id, uniprot_acc
        """
        rows = []
        for pdb_id in pdb_ids:
            try:
                url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
                r = self._http.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                data = r.json().get(pdb_id.lower(), {}).get("UniProt", {})
                for acc, entry in data.items():
                    for m in entry.get("mappings", []):
                        rows.append({
                            "pdb_id": pdb_id.upper(),
                            "chain_id": m.get("chain_id"),
                            "uniprot_acc": acc
                        })
            except Exception:
                # Keep minimal and robust; skip on errors
                continue
        return pd.DataFrame(rows).drop_duplicates()

    @staticmethod
    def _normalize_uniprot(acc: str) -> str:
        """
        Normalize UniProt accession by removing isoform suffix and uppercasing.
        Example: P00533-2 -> P00533
        """
        if not isinstance(acc, str):
            return acc
        return acc.split("-")[0].upper()

    @staticmethod
    def _extract_gene_name(uniprot_json: dict) -> str | None:
        """
        Extract primary HGNC-like gene symbol from UniProt JSON if present.
        """
        genes = uniprot_json.get("genes", [])
        # Prefer primary geneName.value
        for g in genes:
            val = g.get("geneName", {}).get("value")
            if val:
                return val
        # Fallback to a synonym if available
        for g in genes:
            for syn in g.get("synonyms", []) or []:
                if isinstance(syn, dict):
                    val = syn.get("value")
                else:
                    val = syn
                if val:
                    return val
        return None

    def uniprot_to_gene(self, uniprot_accs: list) -> pd.DataFrame:
        """
        Fetch gene symbol for each UniProt accession using UniProt REST.
        Returns DataFrame columns: uniprot_acc, gene
        """
        rows = []
        unique_accs = sorted(set(uniprot_accs))
        for acc in unique_accs:
            try:
                data = self._fetch_uniprot_json(acc)
                gene = self._extract_gene_name(data or {}) if data else None
                rows.append({"uniprot_acc": acc, "gene": gene})
            except Exception:
                rows.append({"uniprot_acc": acc, "gene": None})
        return pd.DataFrame(rows)

    def fetch_uniprot_kinase_info(self, uniprot_accs: list) -> pd.DataFrame:
        """
        Fetch kinase metadata from UniProt to add group/family/class/species.
        Returns DataFrame columns: uniprot_acc, group, family, kinase_class, species
        """
        if self._kinase_info_cache is not None:
            return self._kinase_info_cache
        
        rows = []
        unique_accs = sorted(set(uniprot_accs))
        
        for acc in unique_accs:
            try:
                # Normalize accession (remove isoform suffix)
                root_acc = self._normalize_uniprot(acc)
                data = self._fetch_uniprot_json(root_acc)
                if not data:
                    rows.append({
                        "uniprot_acc": root_acc,
                        "group": None,
                        "family": None,
                        "kinase_class": None,
                        "species": None
                    })
                    continue
                
                # Extract species from organism
                species = None
                if "organism" in data:
                    species = data["organism"].get("scientificName")
                
                # Extract protein family information
                family = None
                group = None
                kinase_class = None
                
                # Try to get family from protein description families
                if "proteinDescription" in data:
                    protein_desc = data["proteinDescription"]
                    # Check for domain information
                    if "domain" in protein_desc:
                        for domain in protein_desc.get("domain", []):
                            domain_name = domain.get("name", "")
                            if "kinase" in domain_name.lower():
                                kinase_class = domain_name
                                break
                
                # Extract from comments (particularly SIMILARITY or DOMAIN comments)
                if "comments" in data:
                    for comment in data["comments"]:
                        if comment.get("commentType") == "SIMILARITY":
                            text = comment.get("texts", [{}])[0].get("value", "")
                            if "kinase" in text.lower():
                                # Try to extract family from text like "Belongs to the protein kinase superfamily"
                                if not family:
                                    family = self._extract_family_from_text(text)
                        elif comment.get("commentType") == "DOMAIN":
                            text = comment.get("texts", [{}])[0].get("value", "")
                            if not kinase_class and "kinase" in text.lower():
                                kinase_class = "Protein kinase"
                
                # Extract from keywords
                if "keywords" in data:
                    for kw in data["keywords"]:
                        kw_val = kw.get("name", "")
                        if "kinase" in kw_val.lower():
                            if not kinase_class:
                                kinase_class = kw_val
                            # Try to identify kinase group from keywords
                            if "serine/threonine" in kw_val.lower():
                                group = "STE" if not group else group
                            elif "tyrosine" in kw_val.lower():
                                group = "TK" if not group else group
                
                # Extract from protein families (proteinDescription -> includedName or family annotation)
                if "uniProtKBCrossReferences" in data:
                    for xref in data["uniProtKBCrossReferences"]:
                        if xref.get("database") == "InterPro":
                            # InterPro contains family/domain information
                            props = xref.get("properties", [])
                            for prop in props:
                                if prop.get("key") == "EntryName":
                                    entry_name = prop.get("value", "")
                                    if "kinase" in entry_name.lower() and not family:
                                        family = entry_name
                
                rows.append({
                    "uniprot_acc": root_acc,
                    "group": group,
                    "family": family,
                    "kinase_class": kinase_class,
                    "species": species
                })
                
            except Exception as e:
                # On error, append empty row
                rows.append({
                    "uniprot_acc": self._normalize_uniprot(acc) if isinstance(acc, str) else acc,
                    "group": None,
                    "family": None,
                    "kinase_class": None,
                    "species": None
                })
        
        self._kinase_info_cache = pd.DataFrame(rows).drop_duplicates()
        
        # Filter by species if specified
        if self.filter_species and str(self.filter_species).strip() and "species" in self._kinase_info_cache.columns:
            self._kinase_info_cache = self._kinase_info_cache[
                self._kinase_info_cache["species"].str.contains(self.filter_species, case=False, na=False)
            ]
        
        return self._kinase_info_cache

    def _fetch_uniprot_json(self, uniprot_acc: str) -> dict | None:
        """
        Fetch UniProt entry JSON for an accession, with in-memory caching.
        """
        root = self._normalize_uniprot(uniprot_acc)
        if root in self._uniprot_json_cache:
            return self._uniprot_json_cache[root]

        try:
            url = f"https://rest.uniprot.org/uniprotkb/{root}.json"
            r = self._http.get(url, timeout=30)
            if r.status_code != 200:
                self._uniprot_json_cache[root] = None
                return None
            data = r.json()
            self._uniprot_json_cache[root] = data
            return data
        except Exception:
            self._uniprot_json_cache[root] = None
            return None

    @staticmethod
    def _extract_uniprot_caution(uniprot_json: dict) -> str | None:
        """
        Extract UniProt 'CAUTION' comment text(s) if present.
        Returns a single string (joined by ' | ' if multiple).
        """
        if not isinstance(uniprot_json, dict):
            return None
        comments = uniprot_json.get("comments") or []
        cautions: list[str] = []
        for c in comments:
            if not isinstance(c, dict):
                continue
            if c.get("commentType") != "CAUTION":
                continue
            texts = c.get("texts") or []
            for t in texts:
                if isinstance(t, dict):
                    val = t.get("value")
                else:
                    val = None
                if val:
                    cautions.append(str(val).strip())
        if not cautions:
            return None
        # De-duplicate while preserving order
        seen = set()
        uniq = []
        for x in cautions:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return " | ".join(uniq)

    def fetch_uniprot_caution_info(self, uniprot_accs: list) -> pd.DataFrame:
        """
        For each UniProt accession, fetch the UniProt 'CAUTION' comment and label pseudokinase
        based on whether the caution text contains any of these keywords (case-insensitive):
        inactivity, inactive, activity, catalysis, pseudokinase.

        Returns DataFrame columns: uniprot_root, uniprot_caution, pseudokinase
        """
        rows = []
        unique_accs = sorted(set([a for a in uniprot_accs if isinstance(a, str) and a.strip()]))
        for acc in unique_accs:
            root = self._normalize_uniprot(acc)
            data = self._fetch_uniprot_json(root)
            caution = self._extract_uniprot_caution(data or {}) if data else None
            rows.append({"uniprot_root": root, "uniprot_caution": caution})

        df = pd.DataFrame(rows).drop_duplicates(subset=["uniprot_root"])
        if df.empty:
            return pd.DataFrame(columns=["uniprot_root", "uniprot_caution", "pseudokinase"])

        # User-defined heuristic: any occurrence of these keywords flags a pseudokinase.
        # Note: we intentionally do substring matching (not word-boundary matching) to catch
        # variants like "inactivated" etc.
        pseudo_keywords = ["inactivity", "inactive", "activity", "catalysis", "pseudokinase"]
        pseudo_regex = "|".join(re.escape(k) for k in pseudo_keywords)
        df["pseudokinase"] = (
            df["uniprot_caution"]
            .fillna("")
            .astype(str)
            .str.contains(pseudo_regex, case=False, regex=True)
        )
        return df
    
    @staticmethod
    def _extract_family_from_text(text: str) -> str | None:
        """
        Try to extract protein family name from similarity text.
        """
        # Look for patterns like "Belongs to the X family" or "member of the X family"
        import re
        patterns = [
            r"Belongs to the ([^.]+?) family",
            r"member of the ([^.]+?) family",
            r"([A-Z][A-Za-z0-9]+) family"
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def annotate_pdbs_with_kinome(self, pdb_ids: list) -> pd.DataFrame:
        """
        End-to-end: PDB -> UniProt (PDBe) -> gene (UniProt) + group/family/class/species (UniProt).
        Returns DataFrame columns: pdb_id, chain_id, uniprot_acc, gene, group, family, kinase_class, species
        """
        pdb_u = self.pdb_to_uniprot(pdb_ids)
        if pdb_u.empty:
            return pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_acc", "gene", "group", "family", "kinase_class", "species"])
        # Normalize UniProt accession for better joining across sources
        pdb_u = pdb_u.assign(uniprot_root=pdb_u["uniprot_acc"].map(self._normalize_uniprot))
        u_gene = self.uniprot_to_gene(pdb_u["uniprot_acc"])  # gene fetch handles isoforms internally
        u_gene = u_gene.assign(uniprot_root=u_gene["uniprot_acc"].map(self._normalize_uniprot))
        
        # Fetch kinase info from UniProt (instead of KLIFS)
        uniprot_kinase_info = self.fetch_uniprot_kinase_info(pdb_u["uniprot_acc"].tolist())
        if not uniprot_kinase_info.empty and "uniprot_acc" in uniprot_kinase_info.columns:
            uniprot_kinase_info = uniprot_kinase_info.assign(
                uniprot_root=uniprot_kinase_info["uniprot_acc"].map(self._normalize_uniprot)
            )

        # Merge on normalized UniProt accession
        kinase_cols = [c for c in ["uniprot_root", "group", "family", "kinase_class", "species"] 
                       if c in uniprot_kinase_info.columns]
        annot = (pdb_u
                 .merge(u_gene[["uniprot_root", "gene"]], on="uniprot_root", how="left")
                 .merge(uniprot_kinase_info[kinase_cols], on="uniprot_root", how="left"))

        # Optionally, uppercase gene for consistent sub-family labels
        if "gene" in annot.columns:
            annot["gene"] = annot["gene"].astype(str).str.upper().replace({"NONE": None})
        # Arrange columns and drop helper
        cols = ["pdb_id", "chain_id", "uniprot_acc", "gene"]
        for col in ["group", "family", "kinase_class", "species"]:
            if col in annot.columns:
                cols.append(col)
        return annot[cols]

    def plot_distribution(
        self, 
        df: pd.DataFrame, 
        column: str, 
        title: str | None = None,
        top_n: int | None = 15,
        figsize: Tuple[int, int] = (10, 8),
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Create a pie chart for the distribution of values in a specified column.
        
        Args:
            df: Annotation DataFrame from annotate_pdbs_with_kinome()
            column: Column name to visualize ('family', 'kinase_class', 'species', 'group')
            title: Plot title (auto-generated if None)
            top_n: Show only top N categories, group rest as "Other" (None = show all)
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available: {df.columns.tolist()}")
        
        # Count values, excluding None/NaN
        value_counts = df[column].dropna().value_counts()
        
        if value_counts.empty:
            print(f"Warning: No data available for column '{column}'")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No data available for {column}', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Apply top_n filter if specified
        if top_n is not None and len(value_counts) > top_n:
            top_values = value_counts.head(top_n)
            other_count = value_counts[top_n:].sum()
            if other_count > 0:
                value_counts = pd.concat([top_values, pd.Series({'Other': other_count})])
            else:
                value_counts = top_values
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate colors
        colors = plt.cm.Set3(range(len(value_counts)))
        
        # Create pie chart with better formatting
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*value_counts.sum())})',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 9}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        
        # Set title
        if title is None:
            title = f'Distribution of {column.replace("_", " ").title()}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add total count
        total = len(df[column].dropna())
        fig.text(0.5, 0.02, f'Total entries: {total}', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {save_path}")
            except Exception as e:
                print(f"Warning: Could not save figure to {save_path}: {e}")
        
        return fig
    
    def plot_distribution_bars(
        self, 
        df: pd.DataFrame, 
        column: str, 
        title: str | None = None,
        top_n: int | None = 15,
        figsize: Tuple[int, int] = (10, 8),
        save_path: str | None = None,
        color: str = '#3498db'
    ) -> plt.Figure:
        """
        Create a horizontal bar chart for the distribution of values in a specified column.
        Labels are easily readable on the y-axis.
        
        Args:
            df: Annotation DataFrame from annotate_pdbs_with_kinome()
            column: Column name to visualize ('family', 'kinase_class', 'species', 'group')
            title: Plot title (auto-generated if None)
            top_n: Show only top N categories (None = show all)
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            color: Bar color
            
        Returns:
            matplotlib Figure object
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available: {df.columns.tolist()}")
        
        # Count values, excluding None/NaN
        value_counts = df[column].dropna().value_counts()
        
        if value_counts.empty:
            print(f"Warning: No data available for column '{column}'")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No data available for {column}', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Apply top_n filter if specified
        if top_n is not None and len(value_counts) > top_n:
            value_counts = value_counts.head(top_n)
        
        # Reverse order so highest is at top
        value_counts = value_counts[::-1]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars
        bars = ax.barh(value_counts.index, value_counts.values, color=color, edgecolor='white')
        
        # Add count labels on bars
        for bar, count in zip(bars, value_counts.values):
            width = bar.get_width()
            percentage = count / value_counts.sum() * 100
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{count} ({percentage:.1f}%)', 
                   va='center', ha='left', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel(column.replace("_", " ").title(), fontsize=12)
        
        if title is None:
            title = f'Distribution of {column.replace("_", " ").title()}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Adjust x-axis to make room for labels
        ax.set_xlim(0, value_counts.max() * 1.25)
        
        # Add grid for readability
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add total count
        total = len(df[column].dropna())
        fig.text(0.5, 0.02, f'Total entries: {total}', 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {save_path}")
            except Exception as e:
                print(f"Warning: Could not save figure to {save_path}: {e}")
        
        return fig
    
    def plot_all_distributions(
        self,
        df: pd.DataFrame,
        top_n: int | None = 15,
        save_dir: str | None = None
    ) -> dict:
        """
        Create pie charts for family, kinase_class, and species distributions.
        
        Args:
            df: Annotation DataFrame from annotate_pdbs_with_kinome()
            top_n: Show only top N categories per plot, group rest as "Other"
            save_dir: Optional directory to save all figures
            
        Returns:
            Dictionary mapping column names to Figure objects
        """
        columns = ['family', 'kinase_class', 'species']
        figures = {}
        
        for col in columns:
            if col not in df.columns:
                print(f"Skipping '{col}' - column not found in DataFrame")
                continue
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{col}_distribution.png")
            
            try:
                fig = self.plot_distribution(
                    df=df,
                    column=col,
                    title=f'{col.replace("_", " ").title()} Distribution',
                    top_n=top_n,
                    save_path=save_path
                )
                figures[col] = fig
            except Exception as e:
                print(f"Error creating plot for '{col}': {e}")
        
        return figures
    
    def plot_group_distribution(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 8),
        save_path: str | None = None
    ) -> plt.Figure:
        """
        Create a pie chart specifically for kinome group distribution.
        Groups are typically: TK (Tyrosine Kinase), STE (Serine/Threonine), etc.
        
        Args:
            df: Annotation DataFrame from annotate_pdbs_with_kinome()
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        return self.plot_distribution(
            df=df,
            column='group',
            title='Kinome Group Distribution',
            top_n=None,  # Usually few groups, show all
            figsize=figsize,
            save_path=save_path
        )

    def plot_chain_annotation_distributions(
        self,
        annot_path: str = "Results/kinase_annotation_chains.csv",
    ) -> dict:
        """
        Load cached per-chain annotation CSV, print pseudokinase summary,
        and generate pseudokinase/family/species/class distribution plots.

        Returns:
            dict with keys:
              - annot: loaded annotation DataFrame
              - fig_pseudokinase: pseudokinase pie chart Figure
              - fig_family: family bar chart Figure
              - fig_species: species bar chart Figure
              - fig_class: kinase_class bar chart Figure
        """
        if not os.path.exists(annot_path):
            raise FileNotFoundError(
                f"Expected annotation table not found: {annot_path}\n"
                "Run the previous KinaseGroupLabeller annotation step first."
            )

        annot = pd.read_csv(annot_path)

        if "pseudokinase" not in annot.columns:
            raise KeyError(
                "Expected a 'pseudokinase' column in the annotation table. "
                "Re-run KinaseGroupLabeller.annotate_dataset_chains_with_kinome(...)."
            )

        n_total = int(len(annot))
        n_pseudo = int(annot["pseudokinase"].fillna(False).astype(bool).sum())
        n_non = n_total - n_pseudo
        pct_pseudo = (n_pseudo / n_total * 100.0) if n_total else 0.0
        print(f"Pseudokinase chains: {n_pseudo}/{n_total} ({pct_pseudo:.1f}%)")

        fig_pseudokinase, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [n_pseudo, n_non],
            labels=["Pseudokinase", "Non-pseudokinase"],
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            colors=["#DD8452", "#4C72B0"],
        )
        ax.set_title("Pseudokinase fraction (all chains)")
        ax.axis("equal")
        plt.show()

        fig_family = self.plot_distribution_bars(annot, "family", top_n=10)
        fig_species = self.plot_distribution_bars(annot, "species", top_n=15)
        fig_class = self.plot_distribution_bars(annot, "kinase_class")

        return {
            "annot": annot,
            "fig_pseudokinase": fig_pseudokinase,
            "fig_family": fig_family,
            "fig_species": fig_species,
            "fig_class": fig_class,
        }

    def run(self, pdb_ids: list | None = None, output_csv: str = "Results/kinase_annotation.csv") -> pd.DataFrame:
        """
        Convenience wrapper: discover PDB IDs if not provided, annotate, and write CSV.
        Returns the annotation DataFrame.
        """
        ids = pdb_ids or self.collect_pdb_ids_from_dir()
        annot = self.annotate_pdbs_with_kinome(ids)
        try:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        except Exception:
            pass
        annot.to_csv(output_csv, index=False)
        return annot


if __name__ == "__main__":
    # Example: run with autodiscovered PDB IDs and write default CSV
    labeller = KinaseGroupLabeller()
    df = labeller.run()
    try:
        from IPython.display import display  # type: ignore
        display(df.head())
    except Exception:
        print(df.head())
    print(f"Saved annotation for {len(df)} rows to Results/kinase_annotation.csv")
    
    # Example: Create visualization plots
    print("\nGenerating distribution plots...")
    figures = labeller.plot_all_distributions(df, save_dir="Results/plots")
    
    # Also create group distribution if data is available
    if 'group' in df.columns and df['group'].notna().any():
        labeller.plot_group_distribution(df, save_path="Results/plots/group_distribution.png")
    
    # Show plots if running interactively
    try:
        plt.show()
    except Exception:
        print("Plots saved to Results/plots/")
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
from typing import Tuple


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
                # Try the original accession; if it fails, try normalized root accession
                url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
                r = self._http.get(url, timeout=20)
                if r.status_code != 200:
                    root = self._normalize_uniprot(acc)
                    url2 = f"https://rest.uniprot.org/uniprotkb/{root}.json"
                    r = self._http.get(url2, timeout=20)
                if r.status_code != 200:
                    rows.append({"uniprot_acc": acc, "gene": None})
                    continue
                gene = self._extract_gene_name(r.json())
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
                url = f"https://rest.uniprot.org/uniprotkb/{root_acc}.json"
                r = self._http.get(url, timeout=20)
                
                if r.status_code != 200:
                    rows.append({
                        "uniprot_acc": root_acc,
                        "group": None,
                        "family": None,
                        "kinase_class": None,
                        "species": None
                    })
                    continue
                
                data = r.json()
                
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
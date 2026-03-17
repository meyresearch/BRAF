"""
Guide-tree clustering utilities (FoldMason Newick output).

This module is used to prevent data leakage when evaluating ML models on highly
similar structures: we cluster structures using FoldMason's `msa.nw` guide tree
and then split train/test by cluster (group-aware splitting).

Notes on "height":
- If the Newick contains explicit branch lengths (tokens like `:0.123`), heights
  are in those cumulative branch-length units.
- FoldMason's `msa.nw` is often topology-only (no branch lengths). In that case
  we treat each edge as length 1.0 and "height" becomes *topological depth*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Node:
    name: Optional[str] = None
    length: Optional[float] = None  # branch length to parent (None if not present)
    children: Optional[List["Node"]] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


def parse_newick(newick_str: str) -> Node:
    """
    Minimal Newick parser that supports leaf/internal names and optional branch lengths.
    """
    s = newick_str.strip()
    i = 0
    n = len(s)

    def skip_ws() -> None:
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def parse_name_and_len() -> Tuple[Optional[str], Optional[float]]:
        nonlocal i
        skip_ws()
        name_chars: List[str] = []
        while i < n and s[i] not in ":,();":
            name_chars.append(s[i])
            i += 1
        nm = "".join(name_chars).strip() or None

        ln: Optional[float] = None
        skip_ws()
        if i < n and s[i] == ":":
            i += 1
            skip_ws()
            num_chars: List[str] = []
            while i < n and s[i] not in ",();":
                num_chars.append(s[i])
                i += 1
            try:
                ln = float("".join(num_chars).strip())
            except Exception:
                ln = None
        return nm, ln

    def parse_subtree() -> Node:
        nonlocal i
        skip_ws()
        if i < n and s[i] == "(":
            i += 1
            children: List[Node] = []
            while True:
                children.append(parse_subtree())
                skip_ws()
                if i < n and s[i] == ",":
                    i += 1
                    continue
                if i < n and s[i] == ")":
                    i += 1
                    break
                raise ValueError(f"Unexpected character in Newick at pos {i}: {s[i:i+20]!r}")
            nm, ln = parse_name_and_len()
            return Node(name=nm, length=ln, children=children)

        nm, ln = parse_name_and_len()
        return Node(name=nm, length=ln, children=[])

    root = parse_subtree()
    skip_ws()
    if i < n and s[i] == ";":
        i += 1
    return root


def has_branch_lengths(newick_str: str) -> bool:
    # Newick branch lengths are signaled by ':' tokens.
    return ":" in newick_str


def iter_leaf_names(root: Node) -> Iterable[str]:
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.children:
            if node.name is not None:
                yield node.name
            continue
        stack.extend(reversed(node.children))


def compute_node_heights(root: Node, default_edge_len: float) -> Dict[int, float]:
    """
    Compute height of each node = max distance from node to any leaf in its subtree.
    Returns mapping id(node) -> height.
    """
    heights: Dict[int, float] = {}

    def edge_len(child: Node) -> float:
        return float(child.length) if child.length is not None else float(default_edge_len)

    def postorder(node: Node) -> float:
        if not node.children:
            h = 0.0
        else:
            h = max(postorder(c) + edge_len(c) for c in node.children)
        heights[id(node)] = h
        return h

    postorder(root)
    return heights


def cluster_leaves_by_height_cut(
    root: Node,
    *,
    cut_height: float,
    default_edge_len_if_missing: float = 1.0,
) -> Tuple[Dict[str, int], float]:
    """
    Cluster leaves by cutting the dendrogram at `cut_height`.

    Returns:
    - leaf_to_cluster: mapping leaf_name -> cluster_id (1..K)
    - max_height: height of the root (useful for choosing scan range)

    Behavior:
    - If the Newick has no branch lengths, each edge is treated as length 1.0.
    - A node becomes a cluster if its subtree height <= cut_height and its parent's height > cut_height.
    """
    # Determine edge length behavior.
    # If the Newick has any explicit branch lengths anywhere in the tree, then
    # nodes without explicit lengths default to 0.0. Otherwise treat the tree
    # as topology-only and use unit edges (default 1.0).
    stack = [root]
    any_lengths = False
    while stack:
        cur = stack.pop()
        if cur.length is not None:
            any_lengths = True
            break
        if cur.children:
            stack.extend(cur.children)

    default_edge_len = 0.0 if any_lengths else float(default_edge_len_if_missing)

    heights = compute_node_heights(root, default_edge_len=default_edge_len)
    root_h = heights[id(root)]

    leaf_to_cluster: Dict[str, int] = {}
    next_cluster = 1

    def assign_all_leaves(node: Node, cluster_id: int) -> None:
        stack = [node]
        while stack:
            cur = stack.pop()
            if not cur.children:
                if cur.name is not None:
                    leaf_to_cluster[cur.name] = cluster_id
                continue
            stack.extend(cur.children)

    def recurse(node: Node) -> None:
        nonlocal next_cluster
        if not node.children:
            # singleton leaf
            if node.name is not None:
                leaf_to_cluster[node.name] = next_cluster
                next_cluster += 1
            return

        node_h = heights[id(node)]
        if node_h <= cut_height:
            cid = next_cluster
            next_cluster += 1
            assign_all_leaves(node, cid)
            return

        for c in node.children:
            recurse(c)

    recurse(root)
    return leaf_to_cluster, root_h


"""
03_community_detection.py
Community Detection using the Louvain Algorithm

Applies the Louvain method for community detection on the co-comment
interaction networks. Analyzes both the full network and the cross-posting
subgraph to identify hidden community structures.

Author: Gowri Balasubramaniam
Course: IS 527 Network Analysis, UIUC
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import community as community_louvain  # python-louvain
import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_graph(filename: str) -> nx.Graph:
    """Load a graph from GraphML format."""
    path = DATA_DIR / filename
    G = nx.read_graphml(path)
    logger.info(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """
    Apply Louvain community detection algorithm.

    The Louvain method optimizes modularity through a hierarchical
    agglomeration process, iteratively merging communities to maximize
    the density of intra-community edges relative to a random graph.

    Parameters
    ----------
    G : nx.Graph
        Input network
    resolution : float
        Resolution parameter for Louvain algorithm.
        Higher values produce more, smaller communities.

    Returns
    -------
    dict
        Mapping of node -> community_id
    """
    partition = community_louvain.best_partition(G, resolution=resolution)
    modularity = community_louvain.modularity(partition, G)

    num_communities = len(set(partition.values()))
    logger.info(f"Detected {num_communities} communities (modularity: {modularity:.3f})")

    return partition


def analyze_community_composition(
    G: nx.Graph, partition: dict
) -> pd.DataFrame:
    """
    Analyze the category composition of each detected community.

    For each community, calculates the proportion of users from each
    subreddit category (youth, parenting, technology) to identify
    communities with meaningful cross-category mixing.

    Parameters
    ----------
    G : nx.Graph
        Network with 'category' node attributes
    partition : dict
        Community assignments from Louvain detection

    Returns
    -------
    pd.DataFrame
        Community composition statistics
    """
    community_data = defaultdict(lambda: {
        "size": 0,
        "categories": Counter(),
        "subreddits": Counter(),
    })

    for node, comm_id in partition.items():
        community_data[comm_id]["size"] += 1
        category = G.nodes[node].get("category", "unknown")
        community_data[comm_id]["categories"][category] += 1

        subreddits_str = G.nodes[node].get("subreddits", "")
        if subreddits_str:
            for sub in subreddits_str.split(","):
                community_data[comm_id]["subreddits"][sub] += 1

    # Build summary DataFrame
    rows = []
    for comm_id, data in sorted(community_data.items(), key=lambda x: -x[1]["size"]):
        total = data["size"]
        cats = data["categories"]

        row = {
            "community_id": comm_id,
            "size": total,
            "youth_count": cats.get("youth_focused", 0),
            "parent_count": cats.get("parent_focused", 0),
            "tech_count": cats.get("technology_focused", 0),
            "youth_pct": cats.get("youth_focused", 0) / total * 100,
            "parent_pct": cats.get("parent_focused", 0) / total * 100,
            "tech_pct": cats.get("technology_focused", 0) / total * 100,
            "dominant_category": max(cats, key=cats.get),
            "is_mixed": len([c for c in cats.values() if c / total > 0.1]) > 1,
            "top_subreddits": ", ".join(
                s for s, _ in data["subreddits"].most_common(5)
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def identify_bridge_nodes(G: nx.Graph, partition: dict, top_n: int = 20) -> list:
    """
    Identify users who bridge multiple communities.

    Bridge nodes have high betweenness centrality AND participate in
    multiple subreddit categories. These users are key to understanding
    how information flows between youth, parenting, and tech communities.

    Parameters
    ----------
    G : nx.Graph
        Network graph
    partition : dict
        Community assignments
    top_n : int
        Number of top bridge nodes to return

    Returns
    -------
    list[dict]
        Top bridge nodes with their metrics
    """
    # Calculate betweenness centrality (approximate for large graphs)
    if G.number_of_nodes() > 10000:
        betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    else:
        betweenness = nx.betweenness_centrality(G)

    bridge_scores = []
    for node in G.nodes():
        num_cats = G.nodes[node].get("num_categories", 1)
        bc = betweenness.get(node, 0)

        # Bridge score: combines centrality with cross-category participation
        bridge_score = bc * num_cats

        if bridge_score > 0:
            bridge_scores.append({
                "user": node,
                "betweenness_centrality": round(bc, 6),
                "num_categories": num_cats,
                "bridge_score": round(bridge_score, 6),
                "community": partition.get(node, -1),
                "category": G.nodes[node].get("category", "unknown"),
                "subreddits": G.nodes[node].get("subreddits", ""),
            })

    bridge_scores.sort(key=lambda x: -x["bridge_score"])
    return bridge_scores[:top_n]


def save_results(
    partition: dict,
    composition_df: pd.DataFrame,
    bridge_nodes: list,
    prefix: str,
):
    """Save community detection results."""
    # Community assignments
    partition_df = pd.DataFrame([
        {"user": k, "community": v} for k, v in partition.items()
    ])
    partition_df.to_csv(DATA_DIR / f"{prefix}_communities.csv", index=False)

    # Composition analysis
    composition_df.to_csv(DATA_DIR / f"{prefix}_composition.csv", index=False)

    # Bridge nodes
    bridge_df = pd.DataFrame(bridge_nodes)
    bridge_df.to_csv(DATA_DIR / f"{prefix}_bridge_nodes.csv", index=False)

    logger.info(f"Saved results with prefix '{prefix}'")


if __name__ == "__main__":
    # ── Full Network Analysis ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FULL NETWORK COMMUNITY DETECTION")
    logger.info("=" * 60)

    G_full = load_graph("full_network.graphml")
    partition_full = detect_communities(G_full)
    composition_full = analyze_community_composition(G_full, partition_full)

    logger.info(f"\nTop 10 Communities by Size:")
    logger.info(composition_full.head(10).to_string(index=False))

    mixed = composition_full[composition_full["is_mixed"]]
    logger.info(f"\nMixed communities (>10% from 2+ categories): {len(mixed)}")

    bridge_full = identify_bridge_nodes(G_full, partition_full)
    save_results(partition_full, composition_full, bridge_full, "full")

    # ── Cross-Posting Subgraph Analysis ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-POSTING SUBGRAPH COMMUNITY DETECTION")
    logger.info("=" * 60)

    G_cross = load_graph("crossposting_network.graphml")
    partition_cross = detect_communities(G_cross)
    composition_cross = analyze_community_composition(G_cross, partition_cross)

    logger.info(f"\nTop 10 Communities by Size:")
    logger.info(composition_cross.head(10).to_string(index=False))

    bridge_cross = identify_bridge_nodes(G_cross, partition_cross)
    logger.info(f"\nTop Bridge Nodes:")
    for node in bridge_cross[:5]:
        logger.info(f"  {node['user']}: score={node['bridge_score']}, "
                    f"categories={node['num_categories']}, "
                    f"subreddits={node['subreddits'][:50]}...")

    save_results(partition_cross, composition_cross, bridge_cross, "cross")

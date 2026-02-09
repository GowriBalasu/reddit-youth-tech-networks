"""
02_network_construction.py
Build Co-Comment Interaction Networks from Reddit Data

Constructs a user interaction graph where nodes are Reddit users and edges
represent co-comment interactions (users commenting in the same thread).
Adapted from the Hamilton et al. (2017) model from Stanford SNAP.

Output: NetworkX graph saved as GraphML for Gephi visualization.

Author: Gowri Balasubramaniam
Course: IS 527 Network Analysis, UIUC
"""

import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_comments() -> pd.DataFrame:
    """Load cleaned comment data."""
    path = DATA_DIR / "comments_raw.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} comments")
    return df


def build_thread_map(comments_df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Group comments by their parent submission thread.

    Each comment's parent_id links it to a submission (t3_*) or another
    comment (t1_*). We trace up to the root submission to group all
    commenters in the same thread.

    Parameters
    ----------
    comments_df : pd.DataFrame
        Comment records with 'parent_id', 'author', 'subreddit', 'category'

    Returns
    -------
    dict
        Mapping of submission_id -> list of {author, subreddit, category} dicts
    """
    thread_map = defaultdict(list)

    for _, row in comments_df.iterrows():
        parent = row.get("parent_id", "")
        # Extract root submission ID (strip t3_ or t1_ prefix)
        thread_id = parent.split("_")[-1] if isinstance(parent, str) else "unknown"

        thread_map[thread_id].append({
            "author": row["author"],
            "subreddit": row["subreddit"],
            "category": row["category"],
        })

    logger.info(f"Identified {len(thread_map):,} unique threads")
    return dict(thread_map)


def build_cocomment_graph(thread_map: dict) -> nx.Graph:
    """
    Construct a co-comment interaction network.

    Two users are connected by an edge if they commented in the same thread.
    Edge weight counts the number of shared threads. Node attributes include
    the user's primary subreddit category and list of active subreddits.

    Parameters
    ----------
    thread_map : dict
        Mapping of thread_id -> list of commenter info dicts

    Returns
    -------
    nx.Graph
        Co-comment interaction network
    """
    G = nx.Graph()

    # Track user metadata across all threads
    user_categories = defaultdict(lambda: defaultdict(int))
    user_subreddits = defaultdict(set)

    for thread_id, commenters in thread_map.items():
        # Get unique authors in this thread
        unique_authors = {}
        for c in commenters:
            author = c["author"]
            unique_authors[author] = c
            user_categories[author][c["category"]] += 1
            user_subreddits[author].add(c["subreddit"])

        # Create edges between all pairs of co-commenters
        author_list = list(unique_authors.keys())
        for a1, a2 in combinations(author_list, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)

    # Add node attributes
    for user in G.nodes():
        # Primary category = most frequent category
        cats = user_categories.get(user, {})
        primary_cat = max(cats, key=cats.get) if cats else "unknown"
        G.nodes[user]["category"] = primary_cat
        G.nodes[user]["subreddits"] = ",".join(sorted(user_subreddits.get(user, set())))
        G.nodes[user]["num_categories"] = len(
            set(c for c in cats.keys())
        )

    logger.info(f"Built graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_graph_statistics(G: nx.Graph) -> dict:
    """
    Compute basic network statistics.

    Returns
    -------
    dict
        Network metrics including density, avg degree, component info
    """
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "connected_components": nx.number_connected_components(G),
    }

    # Category distribution
    categories = nx.get_node_attributes(G, "category")
    cat_counts = defaultdict(int)
    for cat in categories.values():
        cat_counts[cat] += 1
    stats["category_distribution"] = dict(cat_counts)

    # Cross-category users
    num_cats = nx.get_node_attributes(G, "num_categories")
    stats["users_in_1_category"] = sum(1 for v in num_cats.values() if v == 1)
    stats["users_in_2_categories"] = sum(1 for v in num_cats.values() if v == 2)
    stats["users_in_3_categories"] = sum(1 for v in num_cats.values() if v == 3)

    return stats


def extract_crossposting_subgraph(G: nx.Graph) -> nx.Graph:
    """
    Extract subgraph of users active in 2+ subreddit categories.

    This subgraph reveals the 'bridge' users who connect teen,
    parenting, and technology communities.

    Parameters
    ----------
    G : nx.Graph
        Full co-comment network

    Returns
    -------
    nx.Graph
        Subgraph of cross-posting users
    """
    cross_users = [
        node for node, data in G.nodes(data=True)
        if data.get("num_categories", 1) >= 2
    ]

    subgraph = G.subgraph(cross_users).copy()
    logger.info(
        f"Cross-posting subgraph: {subgraph.number_of_nodes():,} nodes, "
        f"{subgraph.number_of_edges():,} edges"
    )
    return subgraph


def save_graphs(full_graph: nx.Graph, cross_graph: nx.Graph):
    """Save graphs in GraphML format for Gephi visualization."""
    full_path = DATA_DIR / "full_network.graphml"
    cross_path = DATA_DIR / "crossposting_network.graphml"

    nx.write_graphml(full_graph, full_path)
    nx.write_graphml(cross_graph, cross_path)

    logger.info(f"Saved full network to {full_path}")
    logger.info(f"Saved cross-posting network to {cross_path}")


if __name__ == "__main__":
    logger.info("Building co-comment interaction networks\n")

    comments_df = load_comments()
    thread_map = build_thread_map(comments_df)

    # Build full network
    G = build_cocomment_graph(thread_map)
    stats = compute_graph_statistics(G)
    logger.info(f"\nFull Network Statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Extract cross-posting subgraph
    cross_G = extract_crossposting_subgraph(G)
    cross_stats = compute_graph_statistics(cross_G)
    logger.info(f"\nCross-Posting Subgraph Statistics:")
    for k, v in cross_stats.items():
        logger.info(f"  {k}: {v}")

    save_graphs(G, cross_G)

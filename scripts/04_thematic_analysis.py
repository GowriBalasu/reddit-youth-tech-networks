"""
04_thematic_analysis.py
Thematic Content Analysis of Cross-Community Discussions

Performs automated and manual-assisted thematic analysis on submission
and comment text from the top identified communities. Identifies key
themes in technology discourse across youth, parenting, and tech communities.

Author: Gowri Balasubramaniam
Course: IS 527 Network Analysis, UIUC
"""

import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Theme keywords identified through open coding of community content
# These were developed iteratively through qualitative analysis
THEME_KEYWORDS = {
    "social_connection": [
        "friend", "friends", "friendship", "lonely", "loneliness",
        "meet", "connect", "relationship", "social", "community",
        "talk", "chat", "hang out", "together",
    ],
    "gaming": [
        "game", "games", "gaming", "video game", "minecraft", "fortnite",
        "console", "pc gaming", "steam", "xbox", "playstation", "nintendo",
        "esports", "multiplayer", "online game",
    ],
    "music": [
        "music", "song", "songs", "album", "artist", "playlist",
        "spotify", "listen", "concert", "band", "genre",
    ],
    "privacy_safety": [
        "privacy", "safe", "safety", "data", "tracking", "surveillance",
        "personal information", "location", "predator", "creepy",
        "protect", "danger", "risk", "parental controls", "monitoring",
    ],
    "mental_health": [
        "mental health", "anxiety", "depression", "stress", "therapy",
        "wellbeing", "self-care", "burnout", "screen time", "addiction",
        "dopamine", "attention span",
    ],
    "education_learning": [
        "school", "homework", "study", "learn", "education", "teacher",
        "college", "grade", "class", "course", "university",
    ],
    "ai_technology": [
        "ai", "artificial intelligence", "chatgpt", "machine learning",
        "algorithm", "automation", "robot", "deepfake", "chatbot",
        "neural network", "openai", "gpt",
    ],
    "pets_animals": [
        "pet", "pets", "dog", "cat", "kitten", "puppy", "animal",
    ],
}


def preprocess_text(text: str) -> str:
    """Clean and normalize text for analysis."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_themes(text: str) -> list[str]:
    """
    Classify a text into one or more themes based on keyword matching.

    Parameters
    ----------
    text : str
        Preprocessed text content

    Returns
    -------
    list[str]
        List of detected theme labels
    """
    detected = []
    for theme, keywords in THEME_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                detected.append(theme)
                break  # Only count each theme once per text
    return detected


def analyze_community_themes(
    texts_df: pd.DataFrame,
    communities_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze theme distribution across detected communities.

    Joins text content with community assignments and computes theme
    frequencies for each community, enabling comparison of discourse
    patterns between youth-dominant, parent-dominant, and mixed communities.

    Parameters
    ----------
    texts_df : pd.DataFrame
        Submissions/comments with 'author', 'body', 'category' columns
    communities_df : pd.DataFrame
        Community assignments with 'user', 'community' columns

    Returns
    -------
    pd.DataFrame
        Theme counts per community
    """
    # Merge texts with community assignments
    merged = texts_df.merge(
        communities_df, left_on="author", right_on="user", how="inner"
    )
    logger.info(f"Matched {len(merged):,} texts to community assignments")

    # Classify themes for each text
    merged["clean_text"] = merged["body"].apply(preprocess_text)
    merged["themes"] = merged["clean_text"].apply(classify_themes)

    # Aggregate by community
    community_themes = []
    for comm_id in merged["community"].unique():
        comm_texts = merged[merged["community"] == comm_id]
        all_themes = [t for themes in comm_texts["themes"] for t in themes]
        theme_counts = Counter(all_themes)

        row = {
            "community": comm_id,
            "total_texts": len(comm_texts),
            **{f"theme_{theme}": theme_counts.get(theme, 0) for theme in THEME_KEYWORDS},
        }
        community_themes.append(row)

    return pd.DataFrame(community_themes)


def analyze_category_themes(texts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare theme distribution across subreddit categories.

    This is the key analysis: comparing what youth, parents, and tech
    community users discuss about technology.
    """
    texts_df["clean_text"] = texts_df["body"].apply(preprocess_text)
    texts_df["themes"] = texts_df["clean_text"].apply(classify_themes)

    category_themes = []
    for category in texts_df["category"].unique():
        cat_texts = texts_df[texts_df["category"] == category]
        all_themes = [t for themes in cat_texts["themes"] for t in themes]
        theme_counts = Counter(all_themes)
        total_themed = sum(theme_counts.values())

        row = {
            "category": category,
            "total_texts": len(cat_texts),
            "texts_with_themes": sum(1 for t in cat_texts["themes"] if t),
        }
        for theme in THEME_KEYWORDS:
            count = theme_counts.get(theme, 0)
            row[f"{theme}_count"] = count
            row[f"{theme}_pct"] = round(count / total_themed * 100, 1) if total_themed else 0

        category_themes.append(row)

    return pd.DataFrame(category_themes)


def generate_sample_for_coding(
    texts_df: pd.DataFrame,
    n_per_category: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a stratified sample for manual qualitative coding.

    Samples top-scoring posts from each category to ensure representation
    of high-engagement content across communities.

    Parameters
    ----------
    texts_df : pd.DataFrame
        Full text dataset
    n_per_category : int
        Number of posts to sample per category
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Sampled texts ready for manual coding
    """
    samples = []
    for category in texts_df["category"].unique():
        cat_df = texts_df[texts_df["category"] == category]
        # Prioritize high-engagement posts (by score)
        if "score" in cat_df.columns:
            cat_df = cat_df.sort_values("score", ascending=False)
        sample = cat_df.head(n_per_category * 2).sample(
            n=min(n_per_category, len(cat_df)), random_state=seed
        )
        samples.append(sample)

    sample_df = pd.concat(samples, ignore_index=True)
    # Add columns for manual coding
    sample_df["manual_themes"] = ""
    sample_df["notes"] = ""

    logger.info(f"Generated coding sample: {len(sample_df)} texts")
    return sample_df


if __name__ == "__main__":
    logger.info("Running thematic analysis\n")

    # Load data
    comments_df = pd.read_csv(DATA_DIR / "comments_raw.csv")
    communities_df = pd.read_csv(DATA_DIR / "cross_communities.csv")

    # Category-level theme analysis
    logger.info("=" * 60)
    logger.info("THEME DISTRIBUTION BY CATEGORY")
    logger.info("=" * 60)
    category_themes = analyze_category_themes(comments_df)
    logger.info(f"\n{category_themes.to_string(index=False)}")
    category_themes.to_csv(DATA_DIR / "category_themes.csv", index=False)

    # Community-level theme analysis
    logger.info("\n" + "=" * 60)
    logger.info("THEME DISTRIBUTION BY COMMUNITY")
    logger.info("=" * 60)
    community_themes = analyze_community_themes(comments_df, communities_df)
    community_themes.to_csv(DATA_DIR / "community_themes.csv", index=False)

    # Generate sample for manual coding
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING MANUAL CODING SAMPLE")
    logger.info("=" * 60)
    sample = generate_sample_for_coding(comments_df)
    sample.to_csv(DATA_DIR / "coding_sample.csv", index=False)

    logger.info("\nThematic analysis complete.")

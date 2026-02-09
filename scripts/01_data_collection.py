"""
01_data_collection.py
Reddit Data Collection for Youth-Tech Community Analysis

Collects submissions and comments from target subreddits using the Pushshift API.
Adapted from the Hamilton et al. (2017) web-RedditNetworks model (Stanford SNAP).

Note: Pushshift API was deprecated in 2023. This script documents the original
methodology. For reproducibility, see the processed dataset or contact the author.

Author: Gowri Balasubramaniam
Course: IS 527 Network Analysis, UIUC
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.pushshift.io/reddit"
RATE_LIMIT_SECONDS = 1.5  # polite rate limiting
DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = DATA_DIR / "subreddit_config.json"


def load_subreddit_config() -> dict:
    """Load subreddit categories and metadata from config file."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def fetch_submissions(
    subreddit: str,
    after: int,
    before: int,
    size: int = 500,
    fields: Optional[list] = None,
) -> list[dict]:
    """
    Fetch submissions from a subreddit within a time range.

    Parameters
    ----------
    subreddit : str
        Name of the subreddit (without r/ prefix)
    after : int
        Unix timestamp for start of time range
    before : int
        Unix timestamp for end of time range
    size : int
        Number of results per request (max 500)
    fields : list, optional
        Specific fields to retrieve

    Returns
    -------
    list[dict]
        List of submission records
    """
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "size": size,
    }
    if fields:
        params["fields"] = ",".join(fields)

    try:
        response = requests.get(f"{BASE_URL}/search/submission/", params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info(f"  Fetched {len(data)} submissions from r/{subreddit}")
        return data
    except requests.RequestException as e:
        logger.error(f"  Error fetching submissions from r/{subreddit}: {e}")
        return []


def fetch_comments(
    subreddit: str,
    after: int,
    before: int,
    size: int = 500,
    fields: Optional[list] = None,
) -> list[dict]:
    """
    Fetch comments from a subreddit within a time range.

    Parameters
    ----------
    subreddit : str
        Name of the subreddit (without r/ prefix)
    after : int
        Unix timestamp for start of time range
    before : int
        Unix timestamp for end of time range
    size : int
        Number of results per request (max 500)
    fields : list, optional
        Specific fields to retrieve

    Returns
    -------
    list[dict]
        List of comment records
    """
    params = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "size": size,
    }
    if fields:
        params["fields"] = ",".join(fields)

    try:
        response = requests.get(f"{BASE_URL}/search/comment/", params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info(f"  Fetched {len(data)} comments from r/{subreddit}")
        return data
    except requests.RequestException as e:
        logger.error(f"  Error fetching comments from r/{subreddit}: {e}")
        return []


def paginate_collection(
    fetch_fn,
    subreddit: str,
    after: int,
    before: int,
    fields: list,
    max_results: int = 50000,
) -> list[dict]:
    """
    Paginate through API results to collect all data within a time range.

    Uses the created_utc field of the last result to advance the pagination
    cursor, collecting data in chronological batches until no more results
    are returned or the max_results limit is reached.

    Parameters
    ----------
    fetch_fn : callable
        Either fetch_submissions or fetch_comments
    subreddit : str
        Target subreddit name
    after : int
        Start of collection window (unix timestamp)
    before : int
        End of collection window (unix timestamp)
    fields : list
        Fields to collect per record
    max_results : int
        Safety limit on total records collected

    Returns
    -------
    list[dict]
        All collected records
    """
    all_data = []
    current_after = after

    while len(all_data) < max_results:
        batch = fetch_fn(subreddit, current_after, before, size=500, fields=fields)
        if not batch:
            break

        all_data.extend(batch)
        # Advance cursor to last record's timestamp
        current_after = batch[-1].get("created_utc", before)
        time.sleep(RATE_LIMIT_SECONDS)

    return all_data


def add_category_label(records: list[dict], category: str) -> list[dict]:
    """Add category label to each record for downstream analysis."""
    for record in records:
        record["category"] = category
    return records


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate and invalid records.

    Filters out:
    - Deleted/removed users ([deleted], AutoModerator)
    - Duplicate records by ID
    - Records with missing critical fields
    """
    initial_count = len(df)

    # Remove deleted/bot users
    df = df[~df["author"].isin(["[deleted]", "AutoModerator", "[removed]"])]

    # Remove duplicates
    id_col = "id" if "id" in df.columns else df.columns[0]
    df = df.drop_duplicates(subset=[id_col])

    # Remove records with missing author or body
    if "body" in df.columns:
        df = df.dropna(subset=["author", "body"])
    else:
        df = df.dropna(subset=["author"])

    logger.info(
        f"  Cleaned: {initial_count} → {len(df)} records "
        f"({initial_count - len(df)} removed)"
    )
    return df


def collect_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main collection pipeline. Iterates through all configured subreddits,
    collects submissions and comments, labels by category, and cleans.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (submissions_df, comments_df)
    """
    config = load_subreddit_config()
    time_range = config["data_collection"]["time_range"]
    fields = config["data_collection"]["fields"]

    # Convert date strings to unix timestamps
    after_ts = int(datetime.strptime(time_range["start"], "%Y-%m-%d").timestamp())
    before_ts = int(datetime.strptime(time_range["end"], "%Y-%m-%d").timestamp())

    all_submissions = []
    all_comments = []

    for category_key, category_data in config["categories"].items():
        category_name = category_key  # e.g., "youth_focused"
        subreddits = category_data["subreddits"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Collecting from category: {category_name}")
        logger.info(f"Subreddits: {len(subreddits)}")
        logger.info(f"{'='*60}")

        for sub_info in subreddits:
            sub_name = sub_info["name"]
            logger.info(f"\nProcessing r/{sub_name} ({sub_info['members']:,} members)")

            # Collect submissions
            submissions = paginate_collection(
                fetch_submissions, sub_name, after_ts, before_ts, fields
            )
            submissions = add_category_label(submissions, category_name)
            all_submissions.extend(submissions)

            # Collect comments
            comments = paginate_collection(
                fetch_comments, sub_name, after_ts, before_ts, fields
            )
            comments = add_category_label(comments, category_name)
            all_comments.extend(comments)

    # Convert to DataFrames and clean
    submissions_df = clean_records(pd.DataFrame(all_submissions))
    comments_df = clean_records(pd.DataFrame(all_comments))

    logger.info(f"\n{'='*60}")
    logger.info(f"COLLECTION COMPLETE")
    logger.info(f"Total submissions: {len(submissions_df):,}")
    logger.info(f"Total comments: {len(comments_df):,}")
    logger.info(f"{'='*60}")

    return submissions_df, comments_df


def save_data(submissions_df: pd.DataFrame, comments_df: pd.DataFrame):
    """Save collected data to CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    submissions_path = DATA_DIR / "submissions_raw.csv"
    comments_path = DATA_DIR / "comments_raw.csv"

    submissions_df.to_csv(submissions_path, index=False)
    comments_df.to_csv(comments_path, index=False)

    logger.info(f"Saved submissions to {submissions_path}")
    logger.info(f"Saved comments to {comments_path}")


if __name__ == "__main__":
    logger.info("Starting Reddit data collection pipeline")
    logger.info("NOTE: Pushshift API deprecated as of 2023.")
    logger.info("This script documents the original collection methodology.\n")

    submissions_df, comments_df = collect_all_data()
    save_data(submissions_df, comments_df)

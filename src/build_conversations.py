import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RAW_DATA = Path("data/raw/twcs.csv")
BRAND_DATA = Path("data/processed/apple_support.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "apple_support_conversations.csv"

CHUNK_SIZE = 100_000


# ============================================================
# Helper function
# ============================================================

def clean_id(series):
    """
    Convert tweet IDs to a consistent string format.

    Handles values such as:
        706704
        706,704
        706704.0
        NaN
    """

    return (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


# ============================================================
# Step 1 — Load AppleSupport tweets
# ============================================================

print("=" * 70)
print("STEP 1: Loading AppleSupport tweets")
print("=" * 70)

apple = pd.read_csv(BRAND_DATA)

print(f"AppleSupport tweets: {len(apple):,}")


# ============================================================
# Step 2 — Clean AppleSupport IDs
# ============================================================

apple["tweet_id_clean"] = clean_id(
    apple["tweet_id"]
)

apple["parent_id_clean"] = clean_id(
    apple["in_response_to_tweet_id"]
)

apple["response_id_clean"] = clean_id(
    apple["response_tweet_id"]
)


# ============================================================
# Step 3 — Collect related tweet IDs
# ============================================================

print()
print("=" * 70)
print("STEP 2: Collecting related tweet IDs")
print("=" * 70)


parent_ids = set(
    apple.loc[
        apple["in_response_to_tweet_id"].notna(),
        "parent_id_clean"
    ]
)

response_ids = set(
    apple.loc[
        apple["response_tweet_id"].notna(),
        "response_id_clean"
    ]
)

apple_ids = set(
    apple["tweet_id_clean"]
)

related_ids = (
    apple_ids
    | parent_ids
    | response_ids
)


print(f"AppleSupport tweet IDs: {len(apple_ids):,}")
print(f"Parent tweet IDs:       {len(parent_ids):,}")
print(f"Response tweet IDs:     {len(response_ids):,}")
print(f"Total related IDs:      {len(related_ids):,}")


# ============================================================
# Step 4 — Scan original dataset
# ============================================================

print()
print("=" * 70)
print("STEP 3: Extracting related tweets")
print("=" * 70)


related_chunks = []

total_rows = 0
matched_rows = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(RAW_DATA, chunksize=CHUNK_SIZE),
    start=1
):

    total_rows += len(chunk)

    # Clean tweet IDs in current chunk
    chunk["tweet_id_clean"] = clean_id(
        chunk["tweet_id"]
    )

    # Find tweets connected to AppleSupport conversations
    mask = chunk["tweet_id_clean"].isin(
        related_ids
    )

    matched = chunk[mask].copy()

    if not matched.empty:

        related_chunks.append(matched)

        matched_rows += len(matched)


    print(
        f"Processed {total_rows:,} rows | "
        f"Matched {matched_rows:,} related tweets"
    )


# ============================================================
# Step 5 — Combine results
# ============================================================

print()
print("=" * 70)
print("STEP 4: Combining data")
print("=" * 70)


if not related_chunks:

    raise RuntimeError(
        "No related tweets were found."
    )


conversations = pd.concat(
    related_chunks,
    ignore_index=True
)


# ============================================================
# Step 6 — Remove duplicates
# ============================================================

conversations = conversations.drop_duplicates(
    subset=["tweet_id_clean"]
)


# ============================================================
# Step 7 — Sort chronologically
# ============================================================

conversations["created_at_dt"] = pd.to_datetime(
    conversations["created_at"],
    errors="coerce"
)

conversations = conversations.sort_values(
    "created_at_dt"
)


# ============================================================
# Step 8 — Add direction
# ============================================================

conversations["direction"] = conversations["inbound"].map(
    {
        True: "customer",
        False: "brand"
    }
)


# ============================================================
# Step 9 — Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

conversations.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("CONVERSATION EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"Original dataset rows:    {total_rows:,}"
)

print(
    f"Related tweets extracted: {len(conversations):,}"
)

print()
print("Direction:")

print(
    conversations["direction"]
    .value_counts()
    .to_string()
)

print()
print("Saved to:")

print(OUTPUT_FILE)
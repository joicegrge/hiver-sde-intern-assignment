import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/apple_support_conversations.csv"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "apple_support_pairs.csv"


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("LOADING APPLESUPPORT CONVERSATIONS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# Normalize tweet IDs
# ============================================================

def clean_id(series):
    return (
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


df["tweet_id_clean"] = clean_id(df["tweet_id"])

df["parent_id_clean"] = clean_id(
    df["in_response_to_tweet_id"]
)


# ============================================================
# Build lookup table
# ============================================================

tweet_lookup = (
    df.set_index("tweet_id_clean")
)


# ============================================================
# Find AppleSupport → customer pairs
# ============================================================

print()
print("=" * 70)
print("CREATING CUSTOMER → APPLESUPPORT PAIRS")
print("=" * 70)


brand_tweets = df[
    df["author_id"].astype(str).str.strip()
    == "AppleSupport"
].copy()


pairs = []


for _, brand_row in brand_tweets.iterrows():

    parent_id = brand_row["parent_id_clean"]

    # Ignore missing parent IDs
    if (
        not parent_id
        or parent_id == "nan"
    ):
        continue


    # Find parent tweet
    if parent_id not in tweet_lookup.index:
        continue


    customer_row = tweet_lookup.loc[parent_id]


    # Make sure the parent is actually a customer tweet
    if customer_row["inbound"] is not True:

        # Pandas may represent booleans differently,
        # so also handle the string form.
        if str(customer_row["inbound"]).lower() != "true":
            continue


    pairs.append(
        {
            "customer_tweet_id":
                customer_row["tweet_id"],

            "brand_tweet_id":
                brand_row["tweet_id"],

            "customer_message":
                customer_row["text"],

            "brand_response":
                brand_row["text"],

            "customer_created_at":
                customer_row["created_at"],

            "brand_created_at":
                brand_row["created_at"],
        }
    )


# ============================================================
# Create DataFrame
# ============================================================

pairs_df = pd.DataFrame(pairs)


if pairs_df.empty:

    raise RuntimeError(
        "No customer → AppleSupport pairs were found."
    )


# ============================================================
# Remove duplicates
# ============================================================

pairs_df = pairs_df.drop_duplicates(
    subset=[
        "customer_tweet_id",
        "brand_tweet_id"
    ]
)


# ============================================================
# Basic cleaning
# ============================================================

pairs_df["customer_message"] = (
    pairs_df["customer_message"]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)


pairs_df["brand_response"] = (
    pairs_df["brand_response"]
    .astype(str)
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()
)


# Remove empty messages
pairs_df = pairs_df[
    (pairs_df["customer_message"].str.len() > 5)
    &
    (pairs_df["brand_response"].str.len() > 5)
]


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


pairs_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("PAIR EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"AppleSupport replies examined: "
    f"{len(brand_tweets):,}"
)

print(
    f"Customer → AppleSupport pairs: "
    f"{len(pairs_df):,}"
)

print()
print("Saved to:")

print(OUTPUT_FILE)


# ============================================================
# Show examples
# ============================================================

print()
print("=" * 70)
print("SAMPLE CUSTOMER → BRAND PAIRS")
print("=" * 70)


for _, row in pairs_df.head(10).iterrows():

    print()
    print("-" * 70)

    print("CUSTOMER:")
    print(row["customer_message"])

    print()
    print("APPLE SUPPORT:")
    print(row["brand_response"])
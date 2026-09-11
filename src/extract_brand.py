import pandas as pd
from pathlib import Path


# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_PATH = Path("data/raw/twcs.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "apple_support.csv"

BRAND = "AppleSupport"

# Read the huge CSV in chunks instead of loading
# the entire 3M-row dataset into memory.
CHUNK_SIZE = 100_000


# ---------------------------------------
# Prepare output directory
# ---------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_brand_chunks = []

total_rows = 0
brand_rows = 0


# ---------------------------------------
# Process dataset
# ---------------------------------------

print("Starting extraction...")
print(f"Looking for brand: {BRAND}")
print()


for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    total_rows += len(chunk)

    # Find tweets sent by AppleSupport
    brand_chunk = chunk[
        chunk["author_id"].astype(str).str.strip() == BRAND
    ]

    if not brand_chunk.empty:
        all_brand_chunks.append(brand_chunk)
        brand_rows += len(brand_chunk)

    print(
        f"Processed {total_rows:,} rows | "
        f"Found {brand_rows:,} {BRAND} tweets"
    )


# ---------------------------------------
# Combine results
# ---------------------------------------

if not all_brand_chunks:
    raise ValueError(
        f"No tweets found for brand: {BRAND}"
    )


brand_df = pd.concat(
    all_brand_chunks,
    ignore_index=True
)


# ---------------------------------------
# Save
# ---------------------------------------

brand_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------
# Summary
# ---------------------------------------

print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total rows scanned: {total_rows:,}")
print(f"{BRAND} tweets found: {len(brand_df):,}")

print()
print("Inbound / outbound:")

print(
    brand_df["inbound"].value_counts().to_string()
)

print()
print(f"Saved to:")
print(OUTPUT_PATH)
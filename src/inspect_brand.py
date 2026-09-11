import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/twcs.csv")

df = pd.read_csv(DATA_PATH)

BRAND = "AppleSupport"

# Get all AppleSupport tweets
brand_tweets = df[df["author_id"] == BRAND]

print("=" * 80)
print(f"{BRAND} - {len(brand_tweets)} tweets")
print("=" * 80)

for _, row in brand_tweets.iterrows():

    print("\n" + "-" * 80)

    print("Tweet ID:", row["tweet_id"])
    print("Date:", row["created_at"])

    print("Response to:", row["in_response_to_tweet_id"])
    print("Responds with:", row["response_tweet_id"])

    print("BRAND:")
    print(row["text"])

    # Find the customer tweet that this brand tweet responds to
    parent_id = row["in_response_to_tweet_id"]

    if pd.notna(parent_id):

        parent_id = int(parent_id)

        parent = df[df["tweet_id"] == parent_id]

        if not parent.empty:

            print("\nCUSTOMER:")
            print(parent.iloc[0]["text"])

print("\n" + "=" * 80)
print("END")
print("=" * 80)
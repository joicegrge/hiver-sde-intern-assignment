import pandas as pd
from pathlib import Path
import re


INPUT_FILE = Path(
    "data/processed/apple_support_pairs.csv"
)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("LOADING APPLESUPPORT PAIRS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Total pairs: {len(df):,}")


# ============================================================
# Basic cleaning
# ============================================================

df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
)

df["brand_response"] = (
    df["brand_response"]
    .fillna("")
    .astype(str)
)


# ============================================================
# Remove Twitter usernames and URLs
# ============================================================

def clean_text(text):

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_customer_message"] = (
    df["customer_message"]
    .apply(clean_text)
)


# ============================================================
# Message length
# ============================================================

df["message_length"] = (
    df["clean_customer_message"]
    .str.len()
)


print()
print("=" * 70)
print("MESSAGE LENGTH")
print("=" * 70)

print(
    df["message_length"].describe()
)


# ============================================================
# Very short messages
# ============================================================

print()
print("=" * 70)
print("SHORT CUSTOMER MESSAGES")
print("=" * 70)

short_messages = df[
    df["message_length"] < 50
]

print(
    f"Messages shorter than 50 chars: "
    f"{len(short_messages):,}"
)


# ============================================================
# Random sample
# ============================================================

print()
print("=" * 70)
print("RANDOM SAMPLE OF CUSTOMER MESSAGES")
print("=" * 70)

sample = df.sample(
    n=min(100, len(df)),
    random_state=42
)


for i, (_, row) in enumerate(
    sample.iterrows(),
    start=1
):

    print()
    print(f"{i}. {row['clean_customer_message']}")


# ============================================================
# Keyword exploration
# ============================================================

keywords = {
    "icloud": [
        "icloud",
        "apple id"
    ],

    "iphone": [
        "iphone"
    ],

    "ipad": [
        "ipad"
    ],

    "mac": [
        "mac",
        "macbook"
    ],

    "app": [
        "app",
        "application"
    ],

    "battery": [
        "battery",
        "charge",
        "charging"
    ],

    "payment": [
        "payment",
        "paid",
        "charge",
        "charged",
        "purchase"
    ],

    "password": [
        "password",
        "passcode",
        "locked",
        "security"
    ],

    "sim_network": [
        "sim",
        "network",
        "signal",
        "carrier",
        "cellular"
    ],

    "homekit": [
        "homekit",
        "home kit"
    ],

    "notification": [
        "notification",
        "notifications"
    ],

    "update": [
        "update",
        "ios"
    ],

    "refund": [
        "refund",
        "money back"
    ],

    "subscription": [
        "subscription",
        "subscribe"
    ],

    "bluetooth": [
        "bluetooth"
    ],

    "wifi": [
        "wifi",
        "wi-fi"
    ],

    "facetime": [
        "facetime"
    ],

    "imessage": [
        "imessage",
        "i message"
    ],
}


print()
print("=" * 70)
print("KEYWORD COUNTS")
print("=" * 70)


for category, words in keywords.items():

    pattern = "|".join(
        re.escape(word)
        for word in words
    )

    mask = (
        df["clean_customer_message"]
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
    )

    count = mask.sum()

    print(
        f"{category:20s}: {count:,}"
    )


# ============================================================
# Save a discovery sample
# ============================================================

discovery_sample = df.sample(
    n=min(1000, len(df)),
    random_state=42
)

OUTPUT_FILE = Path(
    "data/processed/intent_discovery_sample.csv"
)

discovery_sample[
    [
        "customer_tweet_id",
        "customer_message",
        "brand_response",
        "clean_customer_message"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("DISCOVERY SAMPLE SAVED")
print("=" * 70)

print(OUTPUT_FILE)
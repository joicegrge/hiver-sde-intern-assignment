import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/processed/apple_support_pairs.csv")
OUTPUT = Path("evaluation/golden_set_candidates.csv")

# Target number of examples we eventually want for each intent.
TARGETS = {
    "ios_update_issue": 20,
    "battery_issue": 20,
    "device_hardware_issue": 20,
    "connectivity_issue": 20,
    "apple_account_icloud": 20,
    "apps_and_app_store": 20,
    "apple_services": 20,
    "payments_and_billing": 15,
    "device_features_settings": 15,
    "general_support": 30,
}

# Keywords are ONLY used to find candidate examples.
# We will manually assign the final labels later.
PATTERNS = {
    "ios_update_issue": r"\bios\s*\d|\bios\d|update|upgrad",
    "battery_issue": r"battery|charging|charge|drain",
    "device_hardware_issue": (
        r"screen|display|speaker|microphone|mic|broken|"
        r"crack|hardware|fan|keyboard"
    ),
    "connectivity_issue": (
        r"wifi|wi-fi|bluetooth|sim|network|cellular|"
        r"signal|internet"
    ),
    "apple_account_icloud": (
        r"icloud|apple id|apple account|locked|password|"
        r"login|sign in"
    ),
    "apps_and_app_store": (
        r"\bapp\b|apps|app store|download|install|delete app"
    ),
    "apple_services": (
        r"apple music|itunes|podcast|facetime|imessage|"
        r"siri|homekit"
    ),
    "payments_and_billing": (
        r"payment|paid|charge|refund|receipt|billing|"
        r"subscription|purchase"
    ),
    "device_features_settings": (
        r"notification|screenshot|airdrop|keyboard|"
        r"settings|mute|feature"
    ),
}


print("=" * 70)
print("LOADING APPLESUPPORT PAIRS")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"Total pairs: {len(df):,}")

# Clean customer messages
df["customer_message"] = (
    df["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty messages
df = df[df["customer_message"].str.len() > 0].copy()

print(f"Non-empty messages: {len(df):,}")


# ------------------------------------------------------------
# CATEGORY CANDIDATES
# ------------------------------------------------------------

selected = []

for intent, target in TARGETS.items():

    # General support will be selected separately.
    if intent == "general_support":
        continue

    pattern = PATTERNS[intent]

    candidates = df[
        df["customer_message"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ].copy()

    if len(candidates) == 0:
        print(f"WARNING: No candidates found for {intent}")
        continue

    # We deliberately collect more candidates than needed.
    candidates = candidates.sample(
        frac=1,
        random_state=42
    )

    candidates = candidates.head(target * 2)

    candidates["candidate_type"] = "category_candidate"

    selected.append(candidates)

    print(
        f"{intent:30s}: "
        f"{len(candidates):4d} candidates"
    )


# ------------------------------------------------------------
# AMBIGUOUS / SHORT MESSAGES
# ------------------------------------------------------------

short_df = df[
    df["customer_message"].str.len() < 50
].copy()

print(
    f"\nShort messages (<50 chars): "
    f"{len(short_df):,}"
)

short_count = min(150, len(short_df))

if short_count > 0:
    short_messages = short_df.sample(
        n=short_count,
        random_state=123
    ).copy()

    short_messages["candidate_type"] = "ambiguous"

    selected.append(short_messages)


# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

golden = pd.concat(
    selected,
    ignore_index=True
)

# Remove duplicate customer messages
golden = golden.drop_duplicates(
    subset=["customer_message"]
)

# Shuffle
golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Keep maximum 500 candidates
golden = golden.head(500)


# ------------------------------------------------------------
# CREATE ANNOTATION FILE
# ------------------------------------------------------------

output = pd.DataFrame({
    "pair_id": golden.index,
    "customer_message": golden["customer_message"],
    "historical_reply": golden["brand_response"],
    "candidate_type": golden["candidate_type"],

    # YOU will fill these manually later.
    "gold_intent": "",

    # We'll use this later for escalation evaluation.
    "escalation_label": "",

    "notes": ""
})

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT,
    index=False
)


print("\n" + "=" * 70)
print("GOLDEN SET CANDIDATES CREATED")
print("=" * 70)

print(f"Total candidates: {len(output):,}")
print(f"Saved to: {OUTPUT}")

print("\nColumns:")
print("  pair_id")
print("  customer_message")
print("  historical_reply")
print("  candidate_type")
print("  gold_intent")
print("  escalation_label")
print("  notes")

print("\nIMPORTANT:")
print("Do NOT label the examples yet.")
print("We will inspect the candidates first.")
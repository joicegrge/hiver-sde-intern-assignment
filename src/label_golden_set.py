import pandas as pd
import re
from pathlib import Path

FILE = Path("evaluation/golden_set.csv")

INTENTS = {
    "1": "ios_update_issue",
    "2": "battery_issue",
    "3": "device_hardware_issue",
    "4": "connectivity_issue",
    "5": "apple_account_icloud",
    "6": "apps_and_app_store",
    "7": "apple_services",
    "8": "payments_and_billing",
    "9": "device_features_settings",
    "10": "general_support",
}

# Strong patterns for automatic suggestions.
PATTERNS = {
    "ios_update_issue": [
        r"\bios\s*\d",
        r"\bios\d",
        r"\bios\b.*update",
        r"update.*ios",
        r"updated.*iphone",
        r"upgrade.*ios",
        r"upgrading.*ios",
    ],

    "battery_issue": [
        r"battery",
        r"charging",
        r"charge",
        r"drain",
        r"not holding charge",
    ],

    "device_hardware_issue": [
        r"screen",
        r"display",
        r"speaker",
        r"microphone",
        r"\bmic\b",
        r"volume button",
        r"power button",
        r"keyboard.*not work",
        r"trackpad",
        r"broken",
        r"crack",
        r"shattered",
        r"hardware",
        r"shuts down",
        r"turned off",
    ],

    "connectivity_issue": [
        r"\bwifi\b",
        r"wi-fi",
        r"bluetooth",
        r"\bsim\b",
        r"network",
        r"cellular",
        r"signal",
        r"internet",
        r"pairing",
        r"connect.*car",
    ],

    "apple_account_icloud": [
        r"icloud",
        r"apple id",
        r"apple account",
        r"password",
        r"login",
        r"log in",
        r"sign in",
        r"locked",
        r"credentials",
        r"keychain",
    ],

    "apps_and_app_store": [
        r"\bapp\b",
        r"\bapps\b",
        r"app store",
        r"download.*app",
        r"install.*app",
        r"delete.*app",
    ],

    "apple_services": [
        r"apple music",
        r"itunes",
        r"podcast",
        r"facetime",
        r"imessage",
        r"siri",
        r"homekit",
    ],

    "payments_and_billing": [
        r"payment",
        r"paid",
        r"refund",
        r"receipt",
        r"billing",
        r"subscription",
        r"purchase",
        r"charged",
        r"charging my card",
        r"credit card",
        r"payment plan",
    ],

    "device_features_settings": [
        r"notification",
        r"screenshot",
        r"airdrop",
        r"keyboard",
        r"settings",
        r"mute",
        r"control center",
        r"screen recording",
    ],
}


def suggest_intent(text):
    """
    Returns a suggested intent based on keyword/pattern matching.
    We deliberately return general_support when there is
    insufficient evidence.
    """

    text = str(text).lower()

    scores = {}

    for intent, patterns in PATTERNS.items():
        score = 0

        for pattern in patterns:
            if re.search(pattern, text):
                score += 1

        if score > 0:
            scores[intent] = score

    if not scores:
        return "general_support"

    # Highest number of matching patterns wins.
    best_intent = max(scores, key=scores.get)

    # If there is a weak/ambiguous match, prefer clarification.
    if scores[best_intent] == 1 and len(text) < 35:
        return "general_support"

    return best_intent


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(FILE)

df["gold_intent"] = df["gold_intent"].fillna("")
df["notes"] = df["notes"].fillna("")

print("=" * 70)
print("GOLDEN SET LABELING")
print("=" * 70)
print(f"Total examples: {len(df)}")

already = (
    df["gold_intent"]
    .astype(str)
    .str.strip()
    != ""
).sum()

print(f"Already labeled: {already}")
print(f"Remaining: {len(df) - already}")


# ------------------------------------------------------------
# LABEL
# ------------------------------------------------------------

for index, row in df.iterrows():

    # Resume support
    if str(row["gold_intent"]).strip():
        continue

    message = str(row["customer_message"])

    suggestion = suggest_intent(message)

    print("\n" + "#" * 70)
    print(
        f"EXAMPLE {index + 1} / {len(df)}"
    )
    print("#" * 70)

    print("\nCUSTOMER:")
    print(message)

    print("\nHISTORICAL APPLESUPPORT RESPONSE:")
    print(row["historical_reply"])

    print("\n" + "-" * 70)
    print("SUGGESTED INTENT:")
    print(f"  {suggestion}")
    print("-" * 70)

    print("\n1  = ios_update_issue")
    print("2  = battery_issue")
    print("3  = device_hardware_issue")
    print("4  = connectivity_issue")
    print("5  = apple_account_icloud")
    print("6  = apps_and_app_store")
    print("7  = apple_services")
    print("8  = payments_and_billing")
    print("9  = device_features_settings")
    print("10 = general_support")

    print("\nENTER = accept suggestion")
    print("s     = skip")
    print("q     = quit and save")

    while True:

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            df.to_csv(FILE, index=False)
            print("\nProgress saved.")
            raise SystemExit

        if choice == "s":
            print("Skipped.")
            break

        if choice == "":
            # Accept automatic suggestion
            df.at[index, "gold_intent"] = suggestion

            df.to_csv(FILE, index=False)

            print(f"Accepted: {suggestion}")
            break

        if choice in INTENTS:

            selected = INTENTS[choice]

            df.at[index, "gold_intent"] = selected

            # Record that the label was manually corrected/selected.
            df.at[index, "notes"] = "human_reviewed"

            df.to_csv(FILE, index=False)

            print(f"Saved: {selected}")
            break

        print("Invalid input. Press ENTER, 1-10, s, or q.")


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

df.to_csv(FILE, index=False)

print("\n" + "=" * 70)
print("LABELING COMPLETE")
print("=" * 70)

print(
    "Labeled:",
    (
        df["gold_intent"]
        .astype(str)
        .str.strip()
        != ""
    ).sum()
)

print("\nIntent distribution:")
print(df["gold_intent"].value_counts())
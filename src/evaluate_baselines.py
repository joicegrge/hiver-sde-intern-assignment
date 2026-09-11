import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PAIRS_PATH = "data/processed/apple_support_pairs.csv"
GOLDEN_PATH = "evaluation/golden_set.csv"

MODEL_DIR = "evaluation/models"
OUTPUT_DIR = "evaluation"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading data...")

pairs = pd.read_csv(PAIRS_PATH)
golden = pd.read_csv(GOLDEN_PATH)

pairs["customer_message"] = pairs["customer_message"].fillna("").astype(str)
golden["customer_message"] = golden["customer_message"].fillna("").astype(str)

golden_ids = set(golden["customer_message"])

# ---------------------------------------------------------
# IMPORTANT:
# Keep Golden Set completely separate from training.
# Remove any exact Golden Set messages from training.
# ---------------------------------------------------------

train_df = pairs[~pairs["customer_message"].isin(golden_ids)].copy()

print(f"Total historical pairs: {len(pairs):,}")
print(f"Golden Set: {len(golden):,}")
print(f"Training pool: {len(train_df):,}")

X_train = train_df["customer_message"]
X_gold = golden["customer_message"]
y_gold = golden["gold_intent"]

# ---------------------------------------------------------
# BASELINE 1: Majority Class
# ---------------------------------------------------------

majority_class = y_gold.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_gold)

majority_accuracy = accuracy_score(y_gold, majority_predictions)
majority_macro_f1 = f1_score(
    y_gold,
    majority_predictions,
    average="macro",
    zero_division=0,
)
majority_weighted_f1 = f1_score(
    y_gold,
    majority_predictions,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 70)
print("BASELINE 1 — MAJORITY CLASS")
print("=" * 70)

print(f"Majority intent: {majority_class}")
print(f"Accuracy:       {majority_accuracy:.4f}")
print(f"Macro F1:       {majority_macro_f1:.4f}")
print(f"Weighted F1:    {majority_weighted_f1:.4f}")

# ---------------------------------------------------------
# BASELINE 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)

print("Training TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)

print(f"TF-IDF matrix: {X_train_tfidf.shape}")

# ---------------------------------------------------------
# We need labels for training.
#
# The original Twitter dataset does not contain intent labels.
# We therefore create weak labels using the same deterministic
# taxonomy rules used during Golden Set suggestion.
# ---------------------------------------------------------

INTENTS = [
    "ios_update_issue",
    "battery_issue",
    "device_hardware_issue",
    "connectivity_issue",
    "apple_account_icloud",
    "apps_and_app_store",
    "apple_services",
    "payments_and_billing",
    "device_features_settings",
    "general_support",
]

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
        r"\bbattery\b",
        r"\bcharging\b",
        r"\bcharge\b",
        r"\bdrain\b",
        r"not holding charge",
    ],
    "device_hardware_issue": [
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\bspeaker\b",
        r"\bmicrophone\b",
        r"\bmic\b",
        r"volume button",
        r"power button",
        r"\bkeyboard\b.*not work",
        r"\btrackpad\b",
        r"\bbroken\b",
        r"\bcrack\b",
        r"\bshattered\b",
        r"\bhardware\b",
        r"\bshuts down\b",
        r"\bturned off\b",
    ],
    "connectivity_issue": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\bbluetooth\b",
        r"\bsim\b",
        r"\bnetwork\b",
        r"\bcellular\b",
        r"\bsignal\b",
        r"\binternet\b",
        r"\bpairing\b",
        r"\bconnect\b.*\bcar\b",
    ],
    "apple_account_icloud": [
        r"\bicloud\b",
        r"\bapple id\b",
        r"\bapple account\b",
        r"\bpassword\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\bsign in\b",
        r"\blocked\b",
        r"\bcredentials\b",
        r"\bkeychain\b",
    ],
    "apps_and_app_store": [
        r"\bapp\b",
        r"\bapps\b",
        r"\bapp store\b",
        r"download.*app",
        r"install.*app",
        r"delete.*app",
    ],
    "apple_services": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bpodcast\b",
        r"\bfacetime\b",
        r"\bimessage\b",
        r"\bsiri\b",
        r"\bhomekit\b",
    ],
    "payments_and_billing": [
        r"\bpayment\b",
        r"\bpaid\b",
        r"\brefund\b",
        r"\breceipt\b",
        r"\bbilling\b",
        r"\bsubscription\b",
        r"\bpurchase\b",
        r"\bcharged\b",
        r"charging my card",
        r"\bcredit card\b",
        r"\bpayment plan\b",
    ],
    "device_features_settings": [
        r"\bnotification\b",
        r"\bscreenshot\b",
        r"\bairdrop\b",
        r"\bkeyboard\b",
        r"\bsettings\b",
        r"\bmute\b",
        r"\bcontrol center\b",
        r"\bscreen recording\b",
    ],
}

import re


def weak_label(text):
    """
    Assign a weak intent label using deterministic rules.

    Order is deliberate: more specific intents are checked
    before broad app/general patterns.
    """

    text = text.lower().strip()

    # Very short/contextless messages
    if len(text) < 8:
        return "general_support"

    # Specific priority ordering
    priority = [
        "ios_update_issue",
        "battery_issue",
        "device_hardware_issue",
        "connectivity_issue",
        "apple_account_icloud",
        "payments_and_billing",
        "apple_services",
        "device_features_settings",
        "apps_and_app_store",
    ]

    for intent in priority:
        for pattern in PATTERNS[intent]:
            if re.search(pattern, text):
                return intent

    return "general_support"


print("Creating weak training labels...")

train_df["weak_intent"] = train_df["customer_message"].apply(weak_label)

print("\nWeak-label distribution:")
print(train_df["weak_intent"].value_counts())

y_train = train_df["weak_intent"]

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

classifier.fit(X_train_tfidf, y_train)

X_gold_tfidf = vectorizer.transform(X_gold)

predictions = classifier.predict(X_gold_tfidf)

accuracy = accuracy_score(y_gold, predictions)
macro_f1 = f1_score(
    y_gold,
    predictions,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_gold,
    predictions,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 70)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(f"Accuracy:       {accuracy:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")

print("\nPer-intent results:")
print(
    classification_report(
        y_gold,
        predictions,
        labels=INTENTS,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_gold,
    predictions,
    labels=INTENTS,
)

cm_df = pd.DataFrame(
    cm,
    index=INTENTS,
    columns=INTENTS,
)

print(cm_df)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"),
)

joblib.dump(
    classifier,
    os.path.join(MODEL_DIR, "intent_classifier.joblib"),
)

print("\nSaved:")
print("  evaluation/models/tfidf_vectorizer.joblib")
print("  evaluation/models/intent_classifier.joblib")
print("  evaluation/confusion_matrix.csv")

# ---------------------------------------------------------
# Save predictions for failure analysis
# ---------------------------------------------------------

results = golden.copy()

results["predicted_intent"] = predictions
results["correct"] = (
    results["gold_intent"] == results["predicted_intent"]
)

results.to_csv(
    os.path.join(OUTPUT_DIR, "baseline_predictions.csv"),
    index=False,
)

print("  evaluation/baseline_predictions.csv")

print("\nDONE.")
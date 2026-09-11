import pandas as pd
from pathlib import Path

INPUT = Path("evaluation/golden_set_candidates.csv")
OUTPUT = Path("evaluation/golden_set.csv")

df = pd.read_csv(INPUT)

# Remove duplicate messages
df = df.drop_duplicates(
    subset=["customer_message"]
).reset_index(drop=True)

# Separate candidates
category = df[df["candidate_type"] == "category_candidate"].copy()
ambiguous = df[df["candidate_type"] == "ambiguous"].copy()

# We want 30 difficult examples.
ambiguous_n = min(30, len(ambiguous))

ambiguous = ambiguous.sample(
    n=ambiguous_n,
    random_state=100
)

# Approximate number of category examples.
# We'll manually balance the final set if necessary.
category_n = 200 - len(ambiguous)

category = category.sample(
    n=min(category_n, len(category)),
    random_state=200
)

golden = pd.concat(
    [category, ambiguous],
    ignore_index=True
)

golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Keep exactly 200 if enough candidates exist.
golden = golden.head(200)

# Create clean annotation columns.
result = pd.DataFrame({
    "example_id": range(1, len(golden) + 1),
    "customer_message": golden["customer_message"],
    "historical_reply": golden["historical_reply"],
    "candidate_type": golden["candidate_type"],
    "gold_intent": "",
    "escalation_label": "",
    "notes": ""
})

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("FINAL GOLDEN SET CREATED")
print("=" * 70)
print(f"Examples: {len(result)}")
print(f"Output: {OUTPUT}")

print("\nCandidate type:")
print(result["candidate_type"].value_counts())

print("\nOpen this file in Excel:")
print("evaluation/golden_set.csv")
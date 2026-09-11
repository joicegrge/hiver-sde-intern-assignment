import pandas as pd
from pathlib import Path


# -----------------------------
# 1. Load dataset
# -----------------------------

DATA_PATH = Path("data/raw/sample.csv")

df = pd.read_csv(DATA_PATH)


# -----------------------------
# 2. Basic information
# -----------------------------

print("\n========== DATASET INFO ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)


# -----------------------------
# 3. First few rows
# -----------------------------

print("\n========== FIRST 5 ROWS ==========")

print(df.head().to_string())


# -----------------------------
# 4. Missing values
# -----------------------------

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# -----------------------------
# 5. Inbound / outbound
# -----------------------------

print("\n========== INBOUND / OUTBOUND ==========")

print(df["inbound"].value_counts())


# -----------------------------
# 6. Number of tweets per account
# -----------------------------

print("\n========== AUTHORS ==========")

author_counts = df["author_id"].value_counts()

print(author_counts.to_string())


# -----------------------------
# 7. Unique authors
# -----------------------------

print("\n========== UNIQUE AUTHORS ==========")

print("Number of unique authors:", df["author_id"].nunique())


# -----------------------------
# 8. Tweet length
# -----------------------------

df["text_length"] = df["text"].astype(str).str.len()

print("\n========== TWEET LENGTH ==========")

print(df["text_length"].describe())


# -----------------------------
# 9. Sample tweets
# -----------------------------

print("\n========== SAMPLE TWEETS ==========")

for i, row in df.head(10).iterrows():

    direction = "CUSTOMER" if row["inbound"] else "BRAND"

    print("\n--------------------------------")
    print("Direction:", direction)
    print("Author:", row["author_id"])
    print("Tweet:", row["text"])
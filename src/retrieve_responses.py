import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


PAIRS_PATH = "data/processed/apple_support_pairs.csv"
MODEL_DIR = "evaluation/models"
OUTPUT_DIR = "evaluation"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading historical conversations...")

    df = pd.read_csv(PAIRS_PATH)

    df["customer_message"] = df["customer_message"].fillna("").map(clean_text)
    df["brand_response"] = df["brand_response"].fillna("").astype(str)

    # Remove empty conversations
    df = df[df["customer_message"].str.len() > 0].copy()

    print(f"Historical pairs: {len(df):,}")

    # Use a manageable retrieval index.
    # 30K examples gives good coverage while keeping the demo fast.
    if len(df) > 30000:
        df = df.sample(
            n=30000,
            random_state=42
        ).reset_index(drop=True)

    print(f"Retrieval index: {len(df):,}")

    print("Building TF-IDF retrieval index...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(df["customer_message"])

    print(f"TF-IDF matrix: {matrix.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        vectorizer,
        os.path.join(MODEL_DIR, "retrieval_vectorizer.joblib")
    )

    # Save retrieval data separately
    df.to_pickle(
        os.path.join(MODEL_DIR, "retrieval_pairs.pkl")
    )

    joblib.dump(
        matrix,
        os.path.join(MODEL_DIR, "retrieval_matrix.joblib")
    )

    print("\nSaved:")
    print(" ", os.path.join(MODEL_DIR, "retrieval_vectorizer.joblib"))
    print(" ", os.path.join(MODEL_DIR, "retrieval_pairs.pkl"))
    print(" ", os.path.join(MODEL_DIR, "retrieval_matrix.joblib"))

    # ---------------------------------------------------------
    # Interactive test
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    while True:
        query = input("\nEnter a customer message (or 'q' to quit): ").strip()

        if query.lower() == "q":
            break

        if not query:
            continue

        query_vector = vectorizer.transform([clean_text(query)])

        scores = cosine_similarity(
            query_vector,
            matrix
        ).flatten()

        top_indices = scores.argsort()[-3:][::-1]

        print("\nTop historical matches:")

        for rank, idx in enumerate(top_indices, start=1):
            print("\n" + "-" * 70)
            print(f"#{rank}  Similarity: {scores[idx]:.3f}")
            print(f"Customer: {df.iloc[idx]['customer_message']}")
            print(f"AppleSupport: {df.iloc[idx]['brand_response']}")


if __name__ == "__main__":
    main()
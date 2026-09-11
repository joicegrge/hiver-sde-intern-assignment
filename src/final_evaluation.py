import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run_agent


GOLDEN_PATH = "evaluation/golden_set.csv"
PREDICTIONS_PATH = "evaluation/agent_predictions.csv"
SUMMARY_PATH = "evaluation/final_metrics.txt"


def main():

    print("=" * 70)
    print("FINAL AGENT EVALUATION")
    print("=" * 70)

    golden = pd.read_csv(GOLDEN_PATH)

    # ---------------------------------------------------------
    # Run agent on Golden Set
    # ---------------------------------------------------------

    results = []

    for i, row in golden.iterrows():

        message = str(row["customer_message"])

        result = run_agent(message)

        results.append({
            "customer_message": message,
            "gold_intent": row["gold_intent"],
            "predicted_intent": result["intent"],
            "confidence": result["intent_confidence"],
            "draft_response": result["draft_response"],
            "escalate": result["escalate"],
            "escalation_reason": result["escalation_reason"],
            "top_similarity":
                result["retrieved_examples"][0]["similarity"],
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(golden)}")

    df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Intent metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        df["gold_intent"],
        df["predicted_intent"]
    )

    macro_f1 = f1_score(
        df["gold_intent"],
        df["predicted_intent"],
        average="macro"
    )

    weighted_f1 = f1_score(
        df["gold_intent"],
        df["predicted_intent"],
        average="weighted"
    )

    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION")
    print("=" * 70)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-intent results:")

    report = classification_report(
        df["gold_intent"],
        df["predicted_intent"],
        digits=3
    )

    print(report)

    # ---------------------------------------------------------
    # Escalation statistics
    # ---------------------------------------------------------

    escalation_rate = df["escalate"].mean()

    print("=" * 70)
    print("ESCALATION")
    print("=" * 70)

    print(
        f"Escalation rate: "
        f"{escalation_rate:.3f} "
        f"({escalation_rate * 100:.1f}%)"
    )

    print(
        "\nNote: Golden Set contains intent labels but "
        "does not contain independent human escalation labels."
    )

    print(
        "Therefore escalation is reported as a system-policy "
        "metric, not as measured escalation accuracy."
    )

    # ---------------------------------------------------------
    # Retrieval statistics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("HISTORICAL RETRIEVAL")
    print("=" * 70)

    print(
        f"Mean top-1 similarity: "
        f"{df['top_similarity'].mean():.3f}"
    )

    print(
        f"Median top-1 similarity: "
        f"{df['top_similarity'].median():.3f}"
    )

    print(
        f"Similarity < 0.20: "
        f"{(df['top_similarity'] < 0.20).sum()}"
    )

    # ---------------------------------------------------------
    # Failure examples
    # ---------------------------------------------------------

    failures = df[
        df["gold_intent"] != df["predicted_intent"]
    ]

    print("\n" + "=" * 70)
    print("TOP FAILURE EXAMPLES")
    print("=" * 70)

    for _, row in failures.head(10).iterrows():

        print("\nCustomer:")
        print(row["customer_message"])

        print(
            "Gold:",
            row["gold_intent"]
        )

        print(
            "Predicted:",
            row["predicted_intent"]
        )

        print(
            "Confidence:",
            round(row["confidence"], 3)
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("HIVER SUPPORT AGENT — FINAL METRICS\n\n")

        f.write(
            f"Golden Set size: {len(df)}\n"
        )

        f.write(
            f"Intent accuracy: {accuracy:.4f}\n"
        )

        f.write(
            f"Intent macro F1: {macro_f1:.4f}\n"
        )

        f.write(
            f"Intent weighted F1: {weighted_f1:.4f}\n"
        )

        f.write(
            f"Escalation rate: {escalation_rate:.4f}\n"
        )

        f.write(
            f"Mean retrieval similarity: "
            f"{df['top_similarity'].mean():.4f}\n"
        )

        f.write(
            f"Median retrieval similarity: "
            f"{df['top_similarity'].median():.4f}\n"
        )

        f.write("\nClassification report:\n")
        f.write(report)

    print("\nSaved:")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
import os
import sys
import pandas as pd

# Allow importing agent.py without starting the interactive CLI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run_agent


GOLDEN_PATH = "evaluation/golden_set.csv"
OUTPUT_PATH = "evaluation/agent_predictions.csv"


def main():

    print("=" * 70)
    print("EVALUATING COMPLETE SUPPORT AGENT")
    print("=" * 70)

    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Golden Set: {len(golden)} examples")

    results = []

    for i, row in golden.iterrows():

        message = str(row["customer_message"])
        gold_intent = str(row["gold_intent"])

        result = run_agent(message)

        results.append({
            "customer_message": message,
            "gold_intent": gold_intent,
            "predicted_intent": result["intent"],
            "confidence": result["intent_confidence"],
            "draft_response": result["draft_response"],
            "escalate": result["escalate"],
            "escalation_reason": result["escalation_reason"],
            "top_similarity": (
                result["retrieved_examples"][0]["similarity"]
            ),
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(golden)}")

    predictions = pd.DataFrame(results)

    predictions["correct"] = (
        predictions["gold_intent"]
        == predictions["predicted_intent"]
    )

    accuracy = predictions["correct"].mean()

    print("\n" + "=" * 70)
    print("AGENT INTENT RESULTS")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.4f}"
        f" ({accuracy * 100:.1f}%)"
    )

    print("\nPer-intent accuracy:")

    per_intent = (
        predictions
        .groupby("gold_intent")["correct"]
        .agg(["count", "mean"])
        .sort_values("mean")
    )

    print(per_intent)

    print("\n" + "=" * 70)
    print("TOP INTENT FAILURES")
    print("=" * 70)

    failures = predictions[
        ~predictions["correct"]
    ].copy()

    print(
        f"Incorrect: {len(failures)} / "
        f"{len(predictions)}"
    )

    for _, row in failures.head(15).iterrows():

        print("\n" + "-" * 70)

        print("Customer:")
        print(row["customer_message"])

        print(
            f"Gold:      {row['gold_intent']}"
        )

        print(
            f"Predicted: {row['predicted_intent']}"
        )

        print(
            f"Confidence: {row['confidence']:.3f}"
        )

    predictions.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
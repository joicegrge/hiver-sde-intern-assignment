# Hiver SDE Intern Assignment — AI Customer Support Agent

An AI-assisted customer support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system takes an incoming customer message and:

1. Classifies it into one of 10 support intents.
2. Retrieves historically similar AppleSupport interactions.
3. Drafts a historically grounded response.
4. Decides whether the request should be auto-handled or escalated, with a reason.

## 🚀 How to Run the Project

### 1. Install dependencies

`pip install -r requirements.txt`

### 2. Start the interactive demo

`streamlit run src/app.py`

### 3. Open the app

After running the command, open:

`http://localhost:8501`

### 4. Try the agent

You can either select a quick test case or type your own customer support message.

The app will show:

- Intent Classification
- Confidence
- Suggested Response
- Auto-handle / Escalate decision
- Escalation reason
- Historical supporting examples

---

---



## 1. Problem Framing

Customer support teams receive large volumes of repetitive requests. A useful support agent should be able to identify the customer's problem, use previous support interactions as evidence, provide a helpful first response, and avoid automatically handling sensitive or uncertain cases.

For this project, I selected **AppleSupport** from the Customer Support on Twitter dataset.

The goal is not to replace human support agents. Instead, the system is designed as an **assistive first-response agent** with conservative escalation for risky or uncertain requests.

---

## 2. Dataset

The project uses the Customer Support on Twitter (TWCS) dataset.

The full dataset contains approximately 2.8 million tweets.

For this project:

| Stage | Count |
|---|---:|
| Total tweets | 2,811,774 |
| AppleSupport tweets | 106,860 |
| Customer → AppleSupport pairs | 106,646 |

Customer-support pairs were constructed by linking AppleSupport replies to their corresponding customer tweets using the response relationships in the dataset.

The original `twcs.csv` dataset is intentionally not committed to this repository because of its large size. The processed AppleSupport data used by the project is included.

---

## 3. Intent Taxonomy

I defined a compact taxonomy of 10 support intents based on recurring problems in the AppleSupport data.

| Intent | Description |
|---|---|
| `ios_update_issue` | iOS updates, update failures, update-related problems, downgrade/rollback requests |
| `battery_issue` | Battery drain, charging, battery health or battery not holding charge |
| `device_hardware_issue` | Physical/device hardware problems |
| `connectivity_issue` | Wi-Fi, Bluetooth, cellular, SIM or network connectivity failures |
| `apple_account_icloud` | Apple ID, iCloud, passwords, sign-in and account access |
| `apps_and_app_store` | Apps, App Store, downloading, installing or updating apps |
| `apple_services` | Apple Music, iTunes, Podcasts, FaceTime, iMessage, Siri and HomeKit |
| `payments_and_billing` | Purchases, refunds, subscriptions, billing and payment issues |
| `device_features_settings` | Notifications, settings, screenshots, keyboard, AirDrop and device features |
| `general_support` | Vague or insufficiently specified support requests |

When a message contains multiple symptoms, the classifier attempts to identify the primary customer problem rather than simply selecting the first keyword that appears.

---

## 4. System Architecture

```text
                    Customer Message
                           |
                           v
                 +-------------------+
                 |  Intent Classifier |
                 +---------+---------+
                           |
                           v
                    Predicted Intent
                           |
                 +---------+---------+
                 |                   |
                 v                   v
       Historical Retrieval    Risk Detection
                 |                   |
                 v                   v
        Similar Interactions   Escalate / Auto
                 |                 Handle
                 v
        Response Generation
                 |
                 v
          Suggested Response

Intent classification

The classifier uses a hybrid approach:

deterministic high-signal rules for domain-specific phrases;
TF-IDF + Logistic Regression as a statistical fallback;
confidence-based handling for uncertain predictions.

This combination works well for support messages containing distinctive terms such as battery, Apple ID, Wi-Fi, App Store, iOS updates, etc.

Historical retrieval

A TF-IDF retrieval index is built from 30,000 sampled historical customer-support interactions.

For an incoming message, similar historical customer messages are retrieved together with their associated AppleSupport responses.

The historical responses are used as evidence rather than blindly copying a retrieved response.

Escalation

The agent escalates when:

the message contains high-risk signals such as fraud, hacking, unauthorized charges, legal threats or data-breach concerns;
classifier confidence is below 0.45;
historical retrieval similarity is below 0.20.

Otherwise, the system recommends auto-handling.

5. Evaluation
Golden Set

A 200-example human-reviewed Golden Set was created.

Candidates were selected programmatically to achieve coverage across the taxonomy and were then manually reviewed and corrected.

| Intent                     | Examples |
| -------------------------- | -------: |
| `device_features_settings` |       35 |
| `battery_issue`            |       30 |
| `general_support`          |       30 |
| `ios_update_issue`         |       26 |
| `apple_services`           |       21 |
| `payments_and_billing`     |       14 |
| `apps_and_app_store`       |       13 |
| `connectivity_issue`       |       13 |
| `apple_account_icloud`     |       10 |
| `device_hardware_issue`    |        8 |
| **Total**                  |  **200** |
Exact duplicate customer messages from the Golden Set were excluded from the baseline training pool.

6. Baselines
Baseline 1 — Majority Class

The majority baseline predicts the most frequent intent for every message.
| Metric      | Result |
| ----------- | -----: |
| Accuracy    | 17.50% |
| Macro F1    |  2.98% |
| Weighted F1 |  5.21% |


Baseline 2 — TF-IDF + Logistic Regression

A traditional text-classification baseline was implemented using:

TF-IDF features
unigram + bigram features
Logistic Regression

The historical training examples were assigned weak labels using deterministic taxonomy rules.
| Metric      | Result |
| ----------- | -----: |
| Accuracy    | 52.00% |
| Macro F1    | 52.62% |
| Weighted F1 | 51.28% |

| Intent                     |    F1 |
| -------------------------- | ----: |
| `apple_account_icloud`     | 0.640 |
| `apple_services`           | 0.698 |
| `apps_and_app_store`       | 0.615 |
| `battery_issue`            | 0.784 |
| `connectivity_issue`       | 0.435 |
| `device_features_settings` | 0.621 |
| `device_hardware_issue`    | 0.615 |
| `general_support`          | 0.676 |
| `ios_update_issue`         | 0.509 |
| `payments_and_billing`     | 0.571 |


Retrieval

On the Golden Set:

Mean top-1 retrieval similarity: 0.609
Median top-1 retrieval similarity: 0.459
Examples below 0.20 similarity: 0 / 200
Escalation

The escalation policy triggered on approximately 2.5% of Golden Set examples.

This is a policy metric rather than an escalation accuracy metric, because the Golden Set does not contain independent human escalation labels.

8. Top Failure Cases
1. Ambiguous Wi-Fi / proxy issue

Message:

When I try to input manually a wifi/proxy and I click save, it always says "deactivated".

Gold: ios_update_issue
Prediction: connectivity_issue

Hypothesis: The message contains a strong Wi-Fi/connectivity signal, while the manually assigned label does not align cleanly with the message-only taxonomy. This illustrates ambiguity between connectivity, settings and update-related categories.

2. Update with multiple symptoms

Message:

The IOS update is causing my apps to freeze. Also, the touch screen is having a hard time with the small icons.

Gold: connectivity_issue
Prediction: ios_update_issue

Hypothesis: Multiple symptoms appear in the same message. The classifier follows the strong update signal, while the human label is difficult to infer from the message alone.

3. App download / account / white screen

Message:

Im having trouble downloading apps it ask me to review my id then its just a white screen.

Gold: general_support
Prediction: apps_and_app_store

Hypothesis: "Downloading apps" strongly indicates App Store, while the actual failure could also be related to account authentication or a UI issue.

4. Extremely vague request

Message:

@668313 @115858 FIX THIS

Gold: ios_update_issue
Prediction: general_support

Hypothesis: A message-only model cannot reliably infer the intended issue. Conversation history would be required to determine what the customer is referring to.

5. Update mentioned but battery is the main problem

Message:

Upgrade to iOS 11 and battery drain...

Gold: battery_issue
Prediction: ios_update_issue

Hypothesis: Keyword-based signals can over-weight "iOS/update" even when the customer's actual problem is battery behavior. A stronger model should explicitly reason about the primary requested problem.

9. What Is Misleading About My Headline Number?

The 63% accuracy should not be interpreted as "the agent correctly solves 63% of all Apple customer-support requests."

There are several limitations:

The evaluation set contains only 200 examples.
Candidates were selected programmatically for coverage and then human-reviewed.
Several support messages contain inherently ambiguous or overlapping problems.
Some labels are difficult to reproduce from the customer message alone.
The traditional classifier uses weak historical labels.
The rules were iteratively refined after examining failures, so this is a development evaluation rather than a pristine untouched test set.
There are no independent human escalation labels.
Retrieval similarity is an evidence signal, not a direct response-quality metric.

Therefore, the most accurate interpretation is:

The final agent reaches 63% accuracy and 61.65% macro F1 on a 200-example human-reviewed development Golden Set, substantially above the 17.5% majority baseline and 52.0% TF-IDF baseline.

A larger independently labeled and untouched test set would be required to estimate production performance.

10. LLM-as-a-Judge

The assignment calls for LLM-as-a-judge evaluation of generated responses.

I designed the response generation and retrieval pipeline so that responses can be evaluated on:

relevance
correctness
helpfulness
historical grounding
unsupported claims
escalation appropriateness

However, I did not run the LLM-as-a-judge evaluation because the available OpenAI API account did not have API credits.

I have therefore intentionally not fabricated judge scores.

The current evaluation reports automated classification metrics and human-reviewed Golden Set labels.

A production evaluation should add an independent judge set and measure agreement between LLM ratings and human ratings before using the LLM judge as the primary quality metric.

11. Decision Log
Brand selection: Selected AppleSupport because it provides a large number of support interactions.
Conversation construction: Linked customer tweets to AppleSupport replies using response relationships.
Compact taxonomy: Chose 10 broad intents instead of hundreds of narrow labels.
Primary-problem labeling: Used the primary customer problem when multiple symptoms were present.
Message-only classification: Avoided using the historical response to leak the target intent.
Golden Set: Created a 200-example human-reviewed evaluation set.
Leakage prevention: Removed exact Golden Set customer-message duplicates from the baseline training pool.
Weak-label baseline: Used deterministic rules to create labels for the traditional classifier.
Retrieval index: Used 30,000 historical interactions to keep local inference practical.
Hybrid classifier: Combined deterministic rules with statistical classification.
Historical grounding: Used retrieved interactions as evidence for response drafting.
Response generation: Used deterministic templates so the project runs without an external API.
Escalation: Added explicit handling for security, fraud, legal and uncertain cases.
Confidence thresholds: Used classifier confidence and retrieval similarity as conservative signals.
Honest evaluation: Did not report fabricated LLM-as-a-judge results when API execution was unavailable.
12. One-Week Next Steps

If I had another week, I would prioritize:

1. Better labeled data

Independently label 500–1,000+ examples and create a truly held-out test set.

2. Taxonomy refinement

Review confusion pairs such as:

update vs battery
connectivity vs settings
account vs App Store
services vs payments

and refine the taxonomy where necessary.

3. Better retrieval

Experiment with semantic embeddings and reranking instead of relying only on TF-IDF similarity.

4. Response evaluation

Have humans independently rate responses for:

relevance
correctness
helpfulness
grounding
escalation appropriateness

Then validate an LLM judge against those human ratings.

5. Escalation model

Create explicit human labels for:

safe to auto-handle
needs human review
security-sensitive
billing-sensitive
insufficient information
6. Production readiness

Add:

PII redaction
confidence calibration
monitoring
feedback loops
model/version tracking
retrieval-quality monitoring
audit logs


13. Running the Project
Requirements

Python 3.10+ recommended.

Install dependencies:

pip install -r requirements.txt
streamlit run src/app.py

The application provides:

customer message input
intent classification
confidence
suggested response
auto-handle/escalate decision
escalation reason
historical supporting examples

python src/extract_brand.py
python src/build_conversations.py
python src/create_pairs.py
python src/analyze_intents.py
python src/evaluate_baselines.py
python src/retrieve_responses.py
python src/final_evaluation.py
The core system runs locally without requiring an LLM API.

hiver-sde-intern-assignment/
│
├── data/
│   ├── raw/
│   │   └── sample.csv
│   │
│   └── processed/
│       ├── apple_support.csv
│       ├── apple_support_conversations.csv
│       ├── apple_support_pairs.csv
│       └── intent_discovery_sample.csv
│
├── evaluation/
│   ├── agent_predictions.csv
│   ├── baseline_predictions.csv
│   ├── confusion_matrix.csv
│   ├── final_metrics.txt
│   ├── golden_set.csv
│   └── golden_set_candidates.csv
│
├── src/
│   ├── agent.py
│   ├── app.py
│   ├── analyze_intents.py
│   ├── build_conversations.py
│   ├── create_golden_set.py
│   ├── create_pairs.py
│   ├── evaluate_agent.py
│   ├── evaluate_baselines.py
│   ├── explore_data.py
│   ├── extract_brand.py
│   ├── final_evaluation.py
│   ├── inspect_brand.py
│   ├── label_golden_set.py
│   ├── retrieve_responses.py
│   └── select_golden_set.py
│
├── .gitignore
├── requirements.txt
└── README.md

15. Summary

The project demonstrates a lightweight customer-support agent built around:

Classify → Retrieve → Decide

The final system achieves:

63.00% Accuracy
61.65% Macro F1
63.20% Weighted F1

on a 200-example human-reviewed development Golden Set.

This substantially improves over:

17.50% majority baseline accuracy
52.00% TF-IDF + Logistic Regression accuracy

The major limitations are the small evaluation set, ambiguous support labels, weak historical training labels and the absence of the requested LLM-as-a-judge run due to unavailable API credits.

The next major improvements would be a larger independently labeled dataset, a true held-out test set, semantic retrieval, human-validated response evaluation and a calibrated escalation model.
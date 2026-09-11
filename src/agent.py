import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = "evaluation/models"


# ============================================================
# INTENTS
# ============================================================

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


# ============================================================
# TEXT PATTERNS
# ============================================================

PATTERNS = {
    "ios_update_issue": [
        r"\bios\s*\d",
        r"\bios\s*11\b",
        r"\bios\s*12\b",
        r"\bios\s*13\b",
        r"\bios\s*14\b",
        r"\bios\s*15\b",
        r"\bios\s*16\b",
        r"\bios\s*17\b",
        r"\bios\s*18\b",
        r"\bios\s*19\b",
        r"\bupdate\b.*\bios\b",
        r"\bios\b.*\bupdate\b",
        r"\bupgrade\b",
        r"\bdowngrade\b",
    ],

    "battery_issue": [
        r"\bbattery\b",
        r"\bcharging\b",
        r"\bcharge\b",
        r"\bdrain(?:ing)?\b",
        r"\bdying\b",
        r"\bholding charge\b",
    ],

    "device_hardware_issue": [
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\bspeaker\b",
        r"\bmicrophone\b",
        r"\bmic\b",
        r"\bbutton\b",
        r"\bcamera\b",
        r"\bcrack(?:ed)?\b",
        r"\bbroken\b",
        r"\bflicker(?:ing)?\b",
        r"\bnoise\b",
        r"\btrackpad\b",
        r"\bhardware\b",
    ],

    "connectivity_issue": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\bbluetooth\b",
        r"\bsim\b",
        r"\bcellular\b",
        r"\bnetwork\b",
        r"\bdisconnect(?:ed|ing)?\b",
        r"\bpair(?:ing)?\b",
    ],

    "apple_account_icloud": [
        r"\bicloud\b",
        r"\bapple id\b",
        r"\blog ?in\b",
        r"\bsign ?in\b",
        r"\bpassword\b",
        r"\blocked\b",
        r"\baccount\b",
        r"\bkeychain\b",
    ],

    "apps_and_app_store": [
        r"\bapp\b",
        r"\bapps\b",
        r"\bapp store\b",
        r"\bdownload\b",
        r"\binstall\b",
        r"\buninstall\b",
        r"\bdelete apps?\b",
        r"\bapplication\b",
    ],

    "apple_services": [
        r"\bitunes\b",
        r"\bapple music\b",
        r"\bpodcast\b",
        r"\bfacetime\b",
        r"\bimessage\b",
        r"\bsiri\b",
        r"\bhomekit\b",
    ],

    "payments_and_billing": [
        r"\bpayment\b",
        r"\bpay\b",
        r"\bpaid\b",
        r"\bcharged\b",
        r"\brefund\b",
        r"\bbilling\b",
        r"\bsubscription\b",
        r"\breceipt\b",
        r"\bwarranty\b",
        r"\bapplecare\b",
    ],

    "device_features_settings": [
        r"\bnotification\b",
        r"\bnotifications\b",
        r"\bsetting\b",
        r"\bsettings\b",
        r"\bkeyboard\b",
        r"\bscreenshot\b",
        r"\bairdrop\b",
        r"\bcontrol center\b",
        r"\bbrightness\b",
        r"\bgesture\b",
        r"\bmultitask\b",
        r"\bvoicemail\b",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def matches(text, patterns):
    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

intent_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.joblib"
    )
)

intent_classifier = joblib.load(
    os.path.join(
        MODEL_DIR,
        "intent_classifier.joblib"
    )
)

retrieval_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "retrieval_vectorizer.joblib"
    )
)

retrieval_matrix = joblib.load(
    os.path.join(
        MODEL_DIR,
        "retrieval_matrix.joblib"
    )
)

retrieval_pairs = pd.read_pickle(
    os.path.join(
        MODEL_DIR,
        "retrieval_pairs.pkl"
    )
)

print(
    f"Loaded {len(retrieval_pairs):,} "
    "historical conversations."
)


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(message):

    text = clean_text(message)

    # ========================================================
    # HIGH-PRECISION OVERRIDES
    # ========================================================

    # --------------------------------------------------------
    # 1. Very short / insufficient context
    # --------------------------------------------------------

    words = text.split()

    if len(words) <= 3:

        clear_problem_terms = [
            "battery",
            "wifi",
            "wi-fi",
            "bluetooth",
            "icloud",
            "password",
            "charging",
            "charged",
            "refund",
            "subscription",
            "screen",
            "speaker",
            "camera",
            "notifications",
            "notification",
            "facetime",
            "imessage",
            "update",
            "upgrade",
            "downgrade",
        ]

        if not any(
            term in text
            for term in clear_problem_terms
        ):
            return "general_support", 0.98

    # --------------------------------------------------------
    # 2. Device + iOS version but NO actual problem
    # --------------------------------------------------------

    has_iphone = bool(
        re.search(r"\biphone\s*\d*\b", text)
    )

    has_ios_version = bool(
        re.search(r"\bios\s*\d+(?:\.\d+)*\b", text)
    )

    actual_problem_words = [
        "problem",
        "issue",
        "error",
        "not",
        "can't",
        "cannot",
        "won't",
        "wont",
        "doesn't",
        "doesnt",
        "freeze",
        "freezes",
        "crash",
        "crashes",
        "slow",
        "lag",
        "battery",
        "wifi",
        "bluetooth",
        "update",
        "upgrade",
        "downgrade",
    ]

    if (
        has_iphone
        and has_ios_version
        and not any(
            word in text
            for word in actual_problem_words
        )
    ):
        return "general_support", 0.98

    # --------------------------------------------------------
    # 3. High-risk cases
    # --------------------------------------------------------
    # These are still classified normally, but escalation
    # will happen later.

    # --------------------------------------------------------
    # 4. iOS downgrade / rollback
    # --------------------------------------------------------
    # MUST happen before battery because a message can contain
    # both downgrade + battery.

    downgrade_request = (
        re.search(
            r"\b(downgrade|rollback|roll back)\b",
            text
        )
        or "go back to ios" in text
        or "go back from ios" in text
    )

    if downgrade_request:
        return "ios_update_issue", 0.99

    # --------------------------------------------------------
    # 5. Explicit iOS update problem
    # --------------------------------------------------------

    update_terms = [
        "ios update",
        "ios upgrade",
        "update ios",
        "upgrade ios",
        "updating ios",
        "update to ios",
        "upgrade to ios",
        "after updating ios",
        "after the update",
        "since the update",
        "new ios",
        "latest ios",
    ]

    update_symptoms = [
        "freeze",
        "freezes",
        "freezing",
        "crash",
        "crashes",
        "crashing",
        "slow",
        "lag",
        "lagging",
        "bug",
        "bugs",
        "glitch",
        "glitches",
        "not working",
        "stopped working",
    ]

    has_ios = bool(
        re.search(r"\bios\b", text)
    )

    has_update = any(
        phrase in text
        for phrase in update_terms
    )

    has_update_symptom = any(
        phrase in text
        for phrase in update_symptoms
    )

    if has_update:
        return "ios_update_issue", 0.98

    if has_ios and has_update_symptom:
        return "ios_update_issue", 0.98

    # Explicit numbered iOS version WITH a problem
    numbered_ios = bool(
        re.search(
            r"\bios\s*\d+(?:\.\d+)*\b",
            text
        )
    )

    if numbered_ios and has_update_symptom:
        return "ios_update_issue", 0.98

    # --------------------------------------------------------
    # 6. Apple services
    # --------------------------------------------------------
    # Service-specific issue gets priority over the generic
    # "subscription" keyword.
    #
    # Example:
    # "Apple Music subscription says choose a plan"
    # -> apple_services
    #
    # But:
    # "Apple Music subscription charged me twice"
    # -> payments

    service_terms = [
        "apple music",
        "itunes",
        "itunes store",
        "podcast",
        "podcasts",
        "facetime",
        "imessage",
        "i message",
        "siri",
        "homekit",
    ]

    has_service = any(
        phrase in text
        for phrase in service_terms
    )

    actual_payment_terms = [
        "charged",
        "charge",
        "billing",
        "bill",
        "payment",
        "paid",
        "refund",
        "purchase",
        "receipt",
        "money",
        "price",
        "cost",
        "credit card",
        "debit card",
        "unauthorized charge",
        "subscription charge",
        "charged twice",
        "charged me",
        "wrong charge",
    ]

    has_actual_payment = any(
        phrase in text
        for phrase in actual_payment_terms
    )

    if has_service and not has_actual_payment:
        return "apple_services", 0.99

    # --------------------------------------------------------
    # 7. Payments / billing
    # --------------------------------------------------------

    financial_charge = (
        has_actual_payment
        or "refund" in text
        or "billing" in text
        or "subscription payment" in text
        or "subscription charge" in text
        or "subscription charged" in text
        or "unauthorized" in text
    )

    if financial_charge:
        return "payments_and_billing", 0.99

    # --------------------------------------------------------
    # 8. Battery
    # --------------------------------------------------------
    # Explicit battery symptoms.

    battery_problem = any(
        phrase in text
        for phrase in [
            "battery",
            "battery life",
            "battery drain",
            "battery draining",
            "draining battery",
            "drain battery",
            "holding charge",
            "won't hold charge",
            "wont hold charge",
            "not holding charge",
            "slow charging",
            "won't charge",
            "wont charge",
            "not charging",
            "doesn't charge",
            "doesnt charge",
        ]
    )

    if battery_problem:
        return "battery_issue", 0.99

    # --------------------------------------------------------
    # 9. Bluetooth / Wi-Fi SETTINGS
    # --------------------------------------------------------
    # Specific settings behavior should be features/settings.

    bluetooth_setting = (
        "bluetooth" in text
        and any(
            word in text
            for word in [
                "setting",
                "settings",
                "discoverable",
                "discover",
                "turn on",
                "turn off",
                "automatically on",
                "auto on",
                "visible",
            ]
        )
    )

    wifi_setting = (
        ("wifi" in text or "wi-fi" in text)
        and any(
            word in text
            for word in [
                "setting",
                "settings",
                "automatically on",
                "auto on",
                "turns itself on",
                "turn itself on",
            ]
        )
    )

    if bluetooth_setting or wifi_setting:
        return "device_features_settings", 0.99

    # --------------------------------------------------------
    # 10. Actual connectivity failures
    # --------------------------------------------------------

    connectivity_failure = any(
        phrase in text
        for phrase in [
            "wifi keeps disconnecting",
            "wi-fi keeps disconnecting",
            "wifi disconnecting",
            "wi-fi disconnecting",
            "wifi disconnected",
            "wi-fi disconnected",
            "wifi disconnect",
            "wi-fi disconnect",
            "wifi won't connect",
            "wifi wont connect",
            "wi-fi won't connect",
            "wi-fi wont connect",
            "wifi not working",
            "wi-fi not working",
            "can't connect to wifi",
            "cant connect to wifi",
            "cannot connect to wifi",
            "bluetooth keeps disconnecting",
            "bluetooth disconnecting",
            "bluetooth disconnected",
            "bluetooth disconnect",
            "bluetooth won't connect",
            "bluetooth wont connect",
            "bluetooth not connecting",
            "can't connect bluetooth",
            "cant connect bluetooth",
            "cannot connect bluetooth",
            "connection keeps dropping",
            "connection dropped",
            "network connection",
            "network not working",
            "no service",
            "cellular not working",
            "mobile data not working",
            "sim card",
            "sim not working",
            "pairing",
            "can't pair",
            "cannot pair",
        ]
    )

    if connectivity_failure:
        return "connectivity_issue", 0.99

    # --------------------------------------------------------
    # 11. Apple account / iCloud
    # --------------------------------------------------------

    account_problem = (
        "icloud" in text
        or "apple id" in text
        or "appleid" in text
        or "apple account" in text
        or "can't log in" in text
        or "cannot log in" in text
        or "can't login" in text
        or "cannot login" in text
        or "sign in" in text
        or "signin" in text
        or "password" in text
        or "forgot password" in text
        or "account locked" in text
        or "locked out" in text
        or "keychain" in text
    )

    if account_problem:
        return "apple_account_icloud", 0.98

    # --------------------------------------------------------
    # 12. Apps / App Store
    # --------------------------------------------------------
    # Keep this specific. Do NOT classify every occurrence
    # of "app" as an app-support issue.

    app_problem = any(
        phrase in text
        for phrase in [
            "app store",
            "appstore",
            "download app",
            "download apps",
            "download an app",
            "downloading app",
            "downloading apps",
            "install app",
            "install apps",
            "installing app",
            "installing apps",
            "uninstall app",
            "uninstall apps",
            "delete app",
            "delete apps",
            "remove app",
            "remove apps",
            "app won't download",
            "app wont download",
            "apps won't download",
            "apps wont download",
        ]
    )

    if app_problem:
        return "apps_and_app_store", 0.98

    # --------------------------------------------------------
    # 13. Hardware
    # --------------------------------------------------------

    hardware_problem = any(
        phrase in text
        for phrase in [
            "cracked screen",
            "broken screen",
            "screen is broken",
            "screen broken",
            "touch screen broken",
            "touchscreen broken",
            "speaker broken",
            "speaker not working",
            "microphone",
            "microphone not working",
            "mic not working",
            "camera broken",
            "camera not working",
            "button broken",
            "button not working",
            "home button",
            "power button",
            "trackpad",
            "physical damage",
            "hardware",
        ]
    )

    if hardware_problem:
        return "device_hardware_issue", 0.98

    # --------------------------------------------------------
    # 14. Device features / settings
    # --------------------------------------------------------

    settings_problem = any(
        term in text
        for term in [
            "notification",
            "notifications",
            "keyboard",
            "screenshot",
            "airdrop",
            "air drop",
            "control center",
            "brightness",
            "gesture",
            "multitasking",
            "multitask",
            "voicemail",
            "settings",
            "setting",
            "home screen",
            "screen rotation",
            "dark mode",
            "do not disturb",
            "hotspot",
        ]
    )

    if settings_problem:
        return "device_features_settings", 0.97

    # --------------------------------------------------------
    # 15. Apple services fallback
    # --------------------------------------------------------

    if has_service:
        return "apple_services", 0.97

    # ========================================================
    # FALLBACK: STATISTICAL CLASSIFIER
    # ========================================================

    vector = intent_vectorizer.transform(
        [text]
    )

    probabilities = (
        intent_classifier
        .predict_proba(vector)[0]
    )

    best_index = np.argmax(probabilities)

    intent = (
        intent_classifier
        .classes_[best_index]
    )

    confidence = float(
        probabilities[best_index]
    )

    return intent, confidence


# ============================================================
# HISTORICAL RETRIEVAL
# ============================================================

def retrieve_examples(
    message,
    top_k=5
):

    query_vector = (
        retrieval_vectorizer
        .transform([clean_text(message)])
    )

    scores = cosine_similarity(
        query_vector,
        retrieval_matrix
    ).flatten()

    top_indices = (
        scores
        .argsort()[-top_k:][::-1]
    )

    results = []

    for idx in top_indices:

        results.append({
            "customer_message":
                retrieval_pairs.iloc[idx][
                    "customer_message"
                ],

            "brand_response":
                retrieval_pairs.iloc[idx][
                    "brand_response"
                ],

            "similarity":
                float(scores[idx]),
        })

    return results


# ============================================================
# HIGH-RISK DETECTION
# ============================================================

def high_risk(message):

    text = clean_text(message)

    risk_patterns = [

        r"\bhacked\b",
        r"\bhack(?:ed|ing)?\b",
        r"\bcompromised\b",
        r"\bfraud\b",
        r"\bscam\b",
        r"\bunauthorized\b",
        r"\bnot mine\b",
        r"\bdidn't make\b.*\bcharge\b",
        r"\bdid not make\b.*\bcharge\b",
        r"\bsue\b",
        r"\blawyer\b",
        r"\blawsuit\b",
        r"\blegal action\b",
        r"\bdata breach\b",
        r"\bpersonal data\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in risk_patterns
    )


# ============================================================
# RESPONSE GENERATION
# ============================================================

def generate_response(
    message,
    intent,
    confidence,
    examples
):

    text = clean_text(message)

    best_similarity = (
        examples[0]["similarity"]
        if examples
        else 0
    )

    # --------------------------------------------------------
    # Insufficient evidence
    # --------------------------------------------------------

    if best_similarity < 0.20:

        return (
            "Thanks for reaching out. We'd be happy to "
            "help with this. Please send us a DM with "
            "more details about the issue so we can "
            "look into it with you."
        )

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    if intent == "battery_issue":

        return (
            "We're sorry you're having trouble with "
            "your battery. We'd like to look into this "
            "with you. Could you let us know which "
            "device you're using and which iOS version "
            "it's running?"
        )

    # --------------------------------------------------------
    # Connectivity
    # --------------------------------------------------------

    if intent == "connectivity_issue":

        if "wifi" in text or "wi-fi" in text:

            return (
                "We'd like to help get you connected. "
                "Could you let us know which device "
                "you're using and which iOS version "
                "it's running? If possible, also let "
                "us know when the Wi-Fi disconnects."
            )

        if "bluetooth" in text:

            return (
                "We'd like to help with the Bluetooth "
                "connection. Could you let us know "
                "which device you're using, which "
                "iOS version it's running, and what "
                "you're trying to connect?"
            )

        return (
            "We're here to help with the connection "
            "issue. Could you let us know which "
            "device you're using and what happens "
            "when the connection fails?"
        )

    # --------------------------------------------------------
    # Apple account / iCloud
    # --------------------------------------------------------

    if intent == "apple_account_icloud":

        if (
            "log" in text
            or "sign" in text
            or "icloud" in text
            or "apple id" in text
        ):

            return (
                "We'd like to help you get back into "
                "your account. Could you tell us what "
                "happens when you try to sign in? "
                "If you see an error message, please "
                "include it."
            )

        return (
            "We'd be happy to help with your account. "
            "Please let us know what happens when you "
            "try to access it so we can look into this."
        )

    # --------------------------------------------------------
    # Apps / App Store
    # --------------------------------------------------------

    if intent == "apps_and_app_store":

        return (
            "We'd be happy to help with the app issue. "
            "Could you let us know which app you're "
            "having trouble with and what happens when "
            "you try to download, install, or use it?"
        )

    # --------------------------------------------------------
    # Apple services
    # --------------------------------------------------------

    if intent == "apple_services":

        return (
            "We'd be happy to help with this Apple "
            "service issue. Could you let us know "
            "which service and device you're using "
            "and describe what happens?"
        )

    # --------------------------------------------------------
    # Payments / billing
    # --------------------------------------------------------

    if intent == "payments_and_billing":

        return (
            "We'd like to help with your billing issue. "
            "Please send us a DM with the relevant "
            "purchase or charge details so the "
            "appropriate support team can look into it."
        )

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    if intent == "device_hardware_issue":

        return (
            "We're sorry you're experiencing this "
            "issue with your device. We'd like to "
            "look into it with you. Could you let us "
            "know which device you're using and "
            "describe what you're seeing or hearing?"
        )

    # --------------------------------------------------------
    # iOS update
    # --------------------------------------------------------

    if intent == "ios_update_issue":

        return (
            "We'd like to help with the issue you're "
            "experiencing with the iOS update. "
            "Could you let us know which device and "
            "iOS version you're using and describe "
            "what happens?"
        )

    # --------------------------------------------------------
    # Device features / settings
    # --------------------------------------------------------

    if intent == "device_features_settings":

        return (
            "We'd be happy to help with this issue. "
            "Could you let us know which device and "
            "iOS version you're using and describe "
            "what happens when you try it?"
        )

    # --------------------------------------------------------
    # General support
    # --------------------------------------------------------

    return (
        "Thanks for reaching out. We'd be happy to "
        "help. Could you provide a little more detail "
        "about the issue you're experiencing?"
    )


# ============================================================
# ESCALATION
# ============================================================

def escalation_decision(
    message,
    intent_confidence,
    examples
):

    best_similarity = (
        examples[0]["similarity"]
        if examples
        else 0
    )

    # --------------------------------------------------------
    # High-risk requests
    # --------------------------------------------------------

    if high_risk(message):

        return (
            True,
            "High-risk request detected; "
            "human review is required."
        )

    # --------------------------------------------------------
    # Low classifier confidence
    # --------------------------------------------------------

    if intent_confidence < 0.45:

        return (
            True,
            f"Low intent confidence "
            f"({intent_confidence:.2f})."
        )

    # --------------------------------------------------------
    # Little historical evidence
    # --------------------------------------------------------

    if best_similarity < 0.20:

        return (
            True,
            f"Insufficient historical evidence "
            f"(similarity {best_similarity:.2f})."
        )

    return (
        False,
        "Intent confidence and historical evidence "
        "are sufficient for an assisted response."
    )


# ============================================================
# COMPLETE AGENT
# ============================================================

def run_agent(message):

    intent, confidence = classify_intent(
        message
    )

    examples = retrieve_examples(
        message,
        top_k=5
    )

    response = generate_response(
        message,
        intent,
        confidence,
        examples
    )

    escalate, reason = escalation_decision(
        message,
        confidence,
        examples
    )

    return {
        "message": message,
        "intent": intent,
        "intent_confidence": confidence,
        "draft_response": response,
        "escalate": escalate,
        "escalation_reason": reason,
        "retrieved_examples": examples,
    }


# ============================================================
# INTERACTIVE DEMO
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("APPLE SUPPORT AI AGENT")
    print("=" * 70)

    while True:

        message = input(
            "\nCustomer message (or 'q' to quit): "
        ).strip()

        if message.lower() == "q":
            break

        if not message:
            continue

        result = run_agent(message)

        print("\n" + "-" * 70)
        print("INTENT")
        print("-" * 70)

        print(
            result["intent"]
        )

        print(
            f"Confidence: "
            f"{result['intent_confidence']:.2f}"
        )

        print("\n" + "-" * 70)
        print("DRAFT RESPONSE")
        print("-" * 70)

        print(
            result["draft_response"]
        )

        print("\n" + "-" * 70)
        print("DECISION")
        print("-" * 70)

        if result["escalate"]:
            print("ESCALATE")
        else:
            print("AUTO-HANDLE")

        print(
            result["escalation_reason"]
        )

        print("\n" + "-" * 70)
        print("HISTORICAL EVIDENCE")
        print("-" * 70)

        for i, example in enumerate(
            result["retrieved_examples"],
            start=1
        ):

            print(
                f"\n#{i} "
                f"(similarity="
                f"{example['similarity']:.3f})"
            )

            print(
                "Customer:",
                example["customer_message"]
            )

            print(
                "AppleSupport:",
                example["brand_response"]
            )
import streamlit as st
import sys
import os

# Allow importing agent.py from the src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import run_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🍎 Apple Support AI Agent")

st.markdown(
    """
    An AI-assisted support tool that classifies customer messages,
    drafts a response using historical AppleSupport conversations,
    and recommends whether the case can be auto-handled or should
    be escalated to a human.
    """
)


# ============================================================
# QUICK TEST CASES
# ============================================================

st.subheader("Quick test cases")

st.markdown(
    """
    ### 👆 Choose a sample below OR type your own customer message
    """
)

examples = {
    "Battery issue":
        "My iPhone battery is draining very quickly.",

    "iOS update issue":
        "After updating to iOS 11 my phone keeps freezing.",

    "Wi-Fi connectivity":
        "My WiFi keeps disconnecting from my iPhone.",

    "Bluetooth connectivity":
        "My Bluetooth keeps disconnecting from my headphones.",

    "Apple account / iCloud":
        "I can't sign in to my iCloud account.",

    "App Store":
        "I can't download apps from the App Store.",

    "Apple Music":
        "My Apple Music isn't working.",

    "Payment / billing":
        "I was charged twice for my subscription.",

    "Hardware":
        "My iPhone screen is broken.",

    "General support":
        "I need some help with my phone.",
}

selected_example = st.selectbox(
    "Choose a sample (optional)",
    ["None — enter a custom message"] + list(examples.keys())
)


# ============================================================
# CUSTOMER MESSAGE
# ============================================================

if selected_example == "None — enter a custom message":

    default_message = ""

else:

    default_message = examples[selected_example]


message = st.text_area(
    "Customer message",
    value=default_message,
    height=120,
    placeholder="Example: My iPhone battery is draining very quickly..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
)


if analyze:

    if not message.strip():

        st.warning(
            "Please enter a customer message before analyzing."
        )

    else:

        with st.spinner("Analyzing customer message..."):

            result = run_agent(
                message.strip()
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        st.divider()

        st.header("1. Intent Classification")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Intent",
                result["intent"].replace(
                    "_", " "
                ).title()
            )

        with col2:

            st.metric(
                "Confidence",
                f"{result['intent_confidence']:.2f}"
            )


        # ====================================================
        # AGENT DECISION
        # ====================================================

        st.header("2. Agent Decision")

        # ----------------------------------------------------
        # Suggested response
        # ----------------------------------------------------

        st.markdown("### Suggested Response")

        st.info(
            result["draft_response"]
        )


        # ----------------------------------------------------
        # Handling decision
        # ----------------------------------------------------

        st.markdown("### Recommended Handling")

        if result["escalate"]:

            st.error(
                "🚨 ESCALATE TO HUMAN"
            )

        else:

            st.success(
                "✅ AUTO-HANDLE"
            )


        st.caption(
            result["escalation_reason"]
        )


        # ====================================================
        # HISTORICAL EVIDENCE
        # ====================================================

        st.header("3. Historical Evidence")

        st.caption(
            "The agent retrieves similar historical "
            "customer-support conversations to ground "
            "the response."
        )


        for i, example in enumerate(
            result["retrieved_examples"],
            start=1
        ):

            similarity = example["similarity"]

            with st.expander(
                f"Historical example {i} "
                f"• similarity {similarity:.3f}"
            ):

                st.markdown(
                    "**Customer message**"
                )

                st.write(
                    example["customer_message"]
                )

                st.markdown(
                    "**Historical AppleSupport response**"
                )

                st.write(
                    example["brand_response"]
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("How it works")

    st.markdown(
        """
        **1. Classify**

        Predict the customer's support intent.

        **2. Retrieve**

        Find similar historical AppleSupport conversations.

        **3. Draft**

        Generate a support response using deterministic
        templates grounded in the support workflow.

        **4. Decide**

        Recommend either auto-handling or human escalation
        based on risk, classifier confidence, and historical
        evidence.
        """
    )

    st.divider()

    st.caption(
        "10 support intents • 200-example human-reviewed "
        "Golden Set • Historical AppleSupport data"
    )
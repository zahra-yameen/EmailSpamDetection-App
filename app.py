import streamlit as st
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Email Spam Detector",

    page_icon="📧",

    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/spam_model.pkl"
)


# ==========================================
# TITLE
# ==========================================

st.title("📧 Email Spam Detection System")


st.write(
    "Enter an email message below and the "
    "machine learning model will predict "
    "whether it is Spam or Not Spam."
)


# ==========================================
# TEXT INPUT
# ==========================================

message = st.text_area(

    "Enter your email/message:",

    height=200,

    placeholder="Type your email here..."
)


# ==========================================
# BUTTON
# ==========================================

if st.button("🔍 Check Message"):

    if message.strip() == "":

        st.warning(
            "Please enter a message."
        )

    else:

        # Make prediction

        prediction = model.predict(
            [message]
        )[0]


        # Probability

        probability = model.predict_proba(
            [message]
        )[0]


        # ==================================
        # SPAM
        # ==================================

        if prediction == 1:

            st.error(
                "🚨 This message is SPAM"
            )

            st.write(
                "Spam probability:",
                f"{probability[1] * 100:.2f}%"
            )


        # ==================================
        # NOT SPAM
        # ==================================

        else:

            st.success(
                "✅ This message is NOT SPAM"
            )

            st.write(
                "Not Spam probability:",
                f"{probability[0] * 100:.2f}%"
            )
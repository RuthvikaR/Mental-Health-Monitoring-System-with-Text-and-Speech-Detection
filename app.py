import streamlit as st
import pandas as pd
from datetime import datetime

# Modules
from modules.preprocessing import clean_text
from modules.keywords import keyword_scores
from modules.risk import compute_risk, risk_label
from modules.sentiment import get_sentiment

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Mental Health Monitor", layout="wide")
st.title("Mental Health Monitoring System")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "role" not in st.session_state:
    st.session_state.role = None

# -------------------------
# ANALYSIS FUNCTION
# -------------------------
def analyze(text):
    clean = clean_text(text)

    sentiment = get_sentiment(clean)
    dep, anx, density = keyword_scores(clean)

    risk = compute_risk(sentiment, dep, anx)
    label = risk_label(risk)

    return {
        "clean": clean,
        "sentiment": sentiment,
        "dep": dep,
        "anx": anx,
        "density": density,
        "risk": risk,
        "risk_label": label
    }

# =========================
# 🏠 LANDING PAGE
# =========================
if st.session_state.role is None:
    st.markdown("## Choose Your Role")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👤 Client"):
            st.session_state.role = "client"

    with col2:
        if st.button("🧑‍⚕️ Therapist"):
            st.session_state.role = "therapist"

# =========================
# 👤 CLIENT DASHBOARD
# =========================
elif st.session_state.role == "client":

    st.sidebar.title("Client Menu")
    mode = st.sidebar.selectbox("Select Mode", ["Text Chat", "Speech Input"])

    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.role = None
        st.rerun()

    # -------------------------
    # 💬 TEXT CHAT
    # -------------------------
    if mode == "Text Chat":
        st.header("💬 Mental Health Chat Analysis")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        input_col1, input_col2 = st.columns([6, 1])

        with input_col1:
            user_input = st.text_input(
                "Type your message",
                label_visibility="collapsed",
                placeholder="Type your message here..."
            )

        with input_col2:
            send_clicked = st.button("Send")

        if send_clicked and user_input:
            result = analyze(user_input)

            st.session_state.chat_history.append({
                "text": user_input,
                "analysis": result
            })

        for item in reversed(st.session_state.chat_history):
            st.markdown("---")

            col1, spacer, col2 = st.columns([2, 0.3, 3])

            with col1:
                st.markdown("### 💬 Message")
                st.success(item["text"])

            with col2:
                analysis = item["analysis"]

                st.markdown("### 📊 Analysis")

                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.markdown("#### 🧠 Sentiment")
                    st.write("Label:", analysis['sentiment']['label'])
                    st.write("Polarity:", analysis['sentiment']['polarity'])
                    st.write("Subjectivity:", analysis['sentiment']['subjectivity'])
                    st.write("Compound:", analysis['sentiment']['vader']['compound'])

                with sub_col2:
                    st.markdown("#### ⚠️ Mental Health")
                    st.write("Depression Score:", analysis['dep'])
                    st.write("Anxiety Score:", analysis['anx'])
                    st.write("Risk Score:", analysis['risk'])
                    st.write("Risk Level:", analysis['risk_label'])

    # -------------------------
    # 🎤 SPEECH INPUT
    # -------------------------
    elif mode == "Speech Input":
        import speech_recognition as sr

        st.header("🎤 Speech Analysis")

        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 2
        recognizer.non_speaking_duration = 1.5
        recognizer.phrase_threshold = 0.3

        if "speech_history" not in st.session_state:
            st.session_state.speech_history = []

        speak_clicked = st.button("🎙️ Speak")
        status_placeholder = st.empty()

        if speak_clicked:
            try:
                with sr.Microphone() as source:
                    status_placeholder.info("🎧 Listening...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=10)

                    status_placeholder.info("🧠 Processing...")
                    text = recognizer.recognize_google(audio)

                    result = analyze(text)
                    status_placeholder.empty()

                    st.session_state.speech_history.append({
                        "text": text,
                        "analysis": result
                    })

            except Exception as e:
                status_placeholder.empty()
                st.error(f"Error: {e}")

        for item in reversed(st.session_state.speech_history):
            st.markdown("---")

            col1, spacer, col2 = st.columns([2, 0.3, 3])

            with col1:
                st.markdown("### 📝 Speech")
                st.success(item["text"])

            with col2:
                analysis = item["analysis"]

                st.markdown("### 📊 Analysis")

                sub_col1, sub_col2 = st.columns(2)

                with sub_col1:
                    st.markdown("#### 🧠 Sentiment")
                    st.write("Label:", analysis['sentiment']['label'])
                    st.write("Polarity:", analysis['sentiment']['polarity'])
                    st.write("Subjectivity:", analysis['sentiment']['subjectivity'])
                    st.write("Compound:", analysis['sentiment']['vader']['compound'])

                with sub_col2:
                    st.markdown("#### ⚠️ Mental Health")
                    st.write("Depression Score:", analysis['dep'])
                    st.write("Anxiety Score:", analysis['anx'])
                    st.write("Risk Score:", analysis['risk'])
                    st.write("Risk Level:", analysis['risk_label'])

# =========================
# 🧑‍⚕️ THERAPIST DASHBOARD
# =========================
elif st.session_state.role == "therapist":

    st.sidebar.title("Therapist Menu")
    mode = st.sidebar.selectbox("Select Mode", ["Dataset Analysis"])

    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.role = None
        st.rerun()

    if mode == "Dataset Analysis":
        st.header("📊 Dataset Analysis")

        file = st.file_uploader("Upload Excel File", type=["xlsx"])

        if file:
            df = pd.read_excel(file)

            if "raw_text" not in df.columns:
                st.error("Dataset must contain 'raw_text'")
            else:
                if st.button("Run Analysis"):

                    results = df["raw_text"].apply(analyze)

                    df["polarity"] = results.apply(lambda x: x["sentiment"]["polarity"])
                    df["subjectivity"] = results.apply(lambda x: x["sentiment"]["subjectivity"])
                    df["sentiment_label"] = results.apply(lambda x: x["sentiment"]["label"])

                    df["compound"] = results.apply(lambda x: x["sentiment"]["vader"]["compound"])

                    df["depression_score"] = results.apply(lambda x: x["dep"])
                    df["anxiety_score"] = results.apply(lambda x: x["anx"])
                    df["risk_score"] = results.apply(lambda x: x["risk"])
                    df["risk_label"] = results.apply(lambda x: x["risk_label"])

                    st.success("Analysis Complete")
                    st.dataframe(df)

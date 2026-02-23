import streamlit as st
import pandas as pd

from modules.preprocessing import clean_text
from modules.sentiment import get_sentiment
from modules.keywords import keyword_scores
from modules.risk import compute_risk, risk_label
from modules.speech import speech_to_text

st.set_page_config(page_title="Mental Health Monitor", layout="wide")

st.title("🧠 Mental Health Monitoring System")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Dataset Analysis", "Manual Input", "Speech Input"]
)

# =========================
# 🔹 ANALYSIS FUNCTION
# =========================

def analyze(text):
    clean = clean_text(text)
    sentiment = get_sentiment(clean)
    dep, anx = keyword_scores(clean)
    risk = compute_risk(sentiment, dep, anx)
    label = risk_label(risk)

    return sentiment, dep, anx, risk, label


# =========================
# 📊 MODE 1: DATASET
# =========================

if mode == "Dataset Analysis":
    st.header("📊 Dataset Analysis")

    file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if file:
        df = pd.read_excel(file)

        st.write("Preview:", df.head())

        if "raw_text" not in df.columns:
            st.error("Dataset must contain 'raw_text' column")
        else:
            if st.button("Run Analysis"):
                results = df["raw_text"].apply(analyze)

                df["sentiment"] = results.apply(lambda x: x[0]["compound"])
                df["depression_score"] = results.apply(lambda x: x[1])
                df["anxiety_score"] = results.apply(lambda x: x[2])
                df["risk_score"] = results.apply(lambda x: x[3])
                df["risk_label"] = results.apply(lambda x: x[4])

                st.success("Analysis Complete")
                st.dataframe(df)

                # Trend analysis if timestamp exists
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    trend = df.groupby(df["timestamp"].dt.date)["risk_score"].mean()

                    st.subheader("📈 Risk Trend Over Time")
                    st.line_chart(trend)


# =========================
# ✍️ MODE 2: MANUAL INPUT
# =========================

elif mode == "Manual Input":
    st.header("✍️ Manual Text Analysis")

    text = st.text_area("Enter text here")

    if st.button("Analyze"):
        sentiment, dep, anx, risk, label = analyze(text)

        st.subheader("Results")

        st.write("**Sentiment Score:**", sentiment["compound"])
        st.write("**Depression Indicators:**", dep)
        st.write("**Anxiety Indicators:**", anx)
        st.write("**Risk Score:**", risk)
        st.write("**Risk Level:**", label)


# =========================
# 🎤 MODE 3: SPEECH INPUT
# =========================

elif mode == "Speech Input":
    st.header("🎤 Speech Analysis")

    audio_file = st.file_uploader("Upload WAV file", type=["wav"])

    if audio_file:
        text = speech_to_text(audio_file)

        st.write("**Transcribed Text:**", text)

        sentiment, dep, anx, risk, label = analyze(text)

        st.subheader("Results")

        st.write("**Sentiment Score:**", sentiment["compound"])
        st.write("**Depression Indicators:**", dep)
        st.write("**Anxiety Indicators:**", anx)
        st.write("**Risk Score:**", risk)
        st.write("**Risk Level:**", label)

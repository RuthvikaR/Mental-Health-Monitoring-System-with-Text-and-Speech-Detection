import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import random

# Modules
from modules.preprocessing import clean_text
from modules.keywords import keyword_scores
from modules.risk import compute_risk, risk_label
from modules.sentiment import get_sentiment


# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AURA - Mental Health Monitor", layout="wide")

st.markdown("""
<style>

/* ---------- GLOBAL BACKGROUND ---------- */
.stApp {
    background-color: #F2F4F6;
    color: #2E2E2E;
    font-family: 'Inter', sans-serif;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background-color: #6B8FA3;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar titles */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}

            
/* ---------- BUTTONS ---------- */
.stButton > button {
    background-color: #3E5C6E;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #2F4756;
    transform: scale(1.02);
}

/* ---------- TEXT INPUTS ---------- */
.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF;
    border-radius: 6px;
    border: 1px solid #AFC1CC;
}

/* ---------- SELECT BOX ---------- */
.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF;
    border-radius: 6px;
}

/* ---------- METRICS ---------- */
div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    border-left: 6px solid #3E5C6E;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame {
    background-color: #FFFFFF;
    border-radius: 10px;
}

/* ---------- INFO / SUCCESS / WARNING ---------- */
.stAlert {
    border-radius: 10px;
}

.stSuccess {
    background-color: #E6F4EC;
    border-left: 6px solid #6FAF8E;
}

.stInfo {
    background-color: #EAF2F6;
    border-left: 6px solid #6B8FA3;
}

.stWarning {
    background-color: #FFF4E1;
    border-left: 6px solid #E6B566;
}

.stError {
    background-color: #FBEAEA;
    border-left: 6px solid #C97A7A;
}

/* ---------- CHAT MESSAGE BOX ---------- */
div.stMarkdown > div {
    border-radius: 10px;
}

/* ---------- SECTION HEADERS ---------- */
h1, h2, h3, h4 {
    color: #3E5C6E;
}

/* ---------- CHART BACKGROUND ---------- */
svg {
    background-color: transparent !important;
}



            /* ---------- FILTER WIDGETS (MATCH BACK BUTTON) ---------- */

/* Selectbox & Multiselect container */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #3E5C6E !important;
    border-radius: 8px;
}

/* Text inside selectbox */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: white !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] svg {
    fill: white !important;
}

/* Multiselect selected pills */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #2F4756 !important;
    color: white !important;
    border-radius: 6px;
}

/* Hover state */
section[data-testid="stSidebar"] div[data-baseweb="select"]:hover {
    background-color: #2F4756 !important;
}

/* Date input */
section[data-testid="stSidebar"] input {
    background-color: #3E5C6E !important;
    color: white !important;
    border-radius: 8px;
    border: none;
}

/* Placeholder text */
section[data-testid="stSidebar"] input::placeholder {
    color: #E0E6EA !important;
}


            
            /* ---------- MESSAGE & ANALYSIS CARDS ---------- */
.blue-card {
    background-color: #EAF2F6;      /* light dusty blue */
    border-left: 6px solid #3E5C6E;  /* muted navy accent */
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

/* Card titles */
.blue-card h3 {
    color: #3E5C6E;
    margin-bottom: 10px;
}

/* Text inside cards */
.blue-card p,
.blue-card span,
.blue-card div {
    color: #000000;
}



/* ---------- ANALYSIS CONTAINER (REAL FIX) ---------- */
div[data-testid="analysis-card"] {
    background-color: #EAF2F6;
    border-left: 6px solid #3E5C6E;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 12px;
}

/* Section titles inside analysis */
div[data-testid="analysis-card"] h3 {
    color: #3E5C6E;
}

/* Text color */
div[data-testid="analysis-card"] * {
    color: #000000;
}    

.analysis-box {
    background-color: #EAF2F6;
    border-left: 6px solid #3E5C6E;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
    margin-bottom: 10px;
}

.analysis-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #000000;
    margin-bottom: 12px;
}

.analysis-item {
    font-size: 0.95rem;
    margin-bottom: 6px;
    color: #000000;
}

.analysis-value {
    color: #0A7D3A; /* green numeric emphasis */
    font-weight: 500;
}
            

            
            /* ---------- APP TITLE BOX ---------- */
.title-box {
    background-color: #EAF2F6;        /* dusty blue */
    border-left: 8px solid #3E5C6E;   /* muted navy */
    border-right: 8px solid #3E5C6E;
    
    border-radius: 16px;
    padding: 18px 0;
    margin: 20px auto 30px auto;
    width: 100%;
    max-width: 420px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* Title text */
.title-box h1 {
    margin: 0;
    color: #000000;
    font-size: 5rem;          /* BIGGER TEXT */
    letter-spacing: 0.15em;
    font-weight: 700;
}
            

            
            /* ---------- ROLE SELECTION CENTERING ---------- */
.role-container {
    max-width: 520px;
    margin: 60px auto;
    text-align: center;
}

/* Buttons spacing */
.role-container .stButton {
    width: 100%;
}
            
            
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="title-box">
    <h1>AURA</h1>
</div>
""", unsafe_allow_html=True)


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
    st.markdown("""
    <div class="role-container">
        <h2>Choose Your Role</h2>
    </div>
    """, unsafe_allow_html=True)

    # Center columns by adding side spacers
    left_spacer, col1, col2, right_spacer = st.columns([1, 2, 2, 1])

    with col1:
        st.markdown('<div class="role-button">', unsafe_allow_html=True)
        if st.button("👤 Client", use_container_width=True):
            st.session_state.role = "client"
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="role-button">', unsafe_allow_html=True)
        if st.button("🧑‍⚕️ Therapist", use_container_width=True):
            st.session_state.role = "therapist"
        st.markdown('</div>', unsafe_allow_html=True)



# =========================
# 👤 CLIENT DASHBOARD
# =========================
elif st.session_state.get("role") == "client":
    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.role = None
        st.rerun()
    st.sidebar.title("Client Menu")
    mode = st.sidebar.selectbox("Select Mode", ["Text Chat", "Speech Input"])
    

    # -------------------------
    # Initialize history
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "speech_history" not in st.session_state:
        st.session_state.speech_history = []

    # Combine histories into one DataFrame
    combined_data = st.session_state.chat_history + st.session_state.speech_history
    history_df = pd.DataFrame(combined_data)
    
    if not history_df.empty:
        # Ensure required columns exist
        for col in ["text", "analysis"]:
            if col not in history_df.columns:
                history_df[col] = None

        # Extract analysis fields
        history_df["polarity"] = history_df["analysis"].apply(lambda x: x['sentiment']['polarity'] if x else np.nan)
        history_df["subjectivity"] = history_df["analysis"].apply(lambda x: x['sentiment']['subjectivity'] if x else np.nan)
        history_df["sentiment_label"] = history_df["analysis"].apply(lambda x: x['sentiment']['label'] if x else "Unknown")
        history_df["compound"] = history_df["analysis"].apply(lambda x: x['sentiment']['vader']['compound'] if x else np.nan)
        history_df["depression_score"] = history_df["analysis"].apply(lambda x: x['dep'] if x else 0)
        history_df["anxiety_score"] = history_df["analysis"].apply(lambda x: x['anx'] if x else 0)
        history_df["risk_score"] = history_df["analysis"].apply(lambda x: x['risk'] if x else 0)
        history_df["risk_label"] = history_df["analysis"].apply(lambda x: x['risk_label'] if x else "Unknown")
        history_df["type"] = ["Text" if i < len(st.session_state.chat_history) else "Speech" for i in range(len(history_df))]
        

    
        def get_recommendations(filtered_df, n=2):
            """
            Generate exactly `n` recommendations per message, 
            each from a different category if available.
            """
            if filtered_df.empty:
                return ["No data to provide recommendations yet."]

            # Define all categories and statements
            categories = []

            # High risk
            high_risk = filtered_df[filtered_df["risk_score"] >= 6]
            if not high_risk.empty:
                categories.append([
                    "⚠️ Some messages indicate high risk. Consider contacting a mental health professional.",
                    "🚨 Your recent messages suggest heightened emotional stress. Seek support from a counselor.",
                    "⚠️ High-risk indicators detected. It might help to speak with a trusted friend or therapist.",
                    "🆘 Some responses show distress. Professional guidance is recommended.",
                    "⚠️ Alert: Risk scores are elevated. Taking immediate steps to manage stress is advised."
                ])

            # High depression
            high_dep = filtered_df[filtered_df["depression_score"] >= 3]
            if not high_dep.empty:
                categories.append([
                    "🧠 Your depression scores are elevated. Journaling can help clarify your feelings.",
                    "💭 Feeling low lately? Consider talking to someone you trust.",
                    "🧠 Elevated depression indicators detected. Exercise or mindfulness may help.",
                    "😔 Your recent messages suggest low mood. Small daily goals can improve outlook.",
                    "📝 Reflect on your emotions and consider seeking support if needed."
                ])

            # High anxiety
            high_anx = filtered_df[filtered_df["anxiety_score"] >= 3]
            if not high_anx.empty:
                categories.append([
                    "😰 Anxiety levels seem high. Deep breathing exercises may help.",
                    "🧘 Consider mindfulness or meditation to manage stress.",
                    "😟 Your messages suggest elevated anxiety. Take breaks and rest.",
                    "💆‍♀️ Relaxation exercises like stretching or yoga can reduce tension.",
                    "⚡ Try grounding techniques to calm racing thoughts."
                ])

            # Negative sentiment
            negative_sent = filtered_df[filtered_df["sentiment_label"] == "Negative"]
            if not negative_sent.empty:
                categories.append([
                    "💬 You have expressed negative feelings recently. Reflect on these feelings.",
                    "🗣️ Talking about your feelings can reduce emotional burden.",
                    "😔 Consider reframing negative thoughts to positive or neutral ones.",
                    "💡 Journaling or creative expression may help manage negative emotions.",
                    "🌱 Small acts of self-care can improve mood and resilience."
                ])

            # Balanced / general wellness (only if no other category triggered)
            if not categories:
                categories.append([
                    "✅ Your messages show a balanced emotional state. Keep maintaining healthy habits!",
                    "🌞 Continue practicing self-care and mindfulness.",
                    "💪 Stay consistent with positive routines and healthy activities.",
                    "🧠 Keep reflecting on emotions; self-awareness is key.",
                    "🎯 You are doing well. Continue your mental wellness practices."
                ])

            # Pick one recommendation from up to `n` different categories
            recommendations = []
            random.shuffle(categories)  # shuffle categories to randomize selection
            for cat in categories[:n]:
                recommendations.append(random.choice(cat))

            return recommendations

    # -------------------------
    # Mode-specific input
    # -------------------------
    if mode == "Text Chat":
        st.header("💬 Chat Review 💬")
        input_col1, input_col2 = st.columns([6,1])
        with input_col1:
            user_input = st.text_input("Type your message", label_visibility="collapsed", placeholder="Type your message here...")
        with input_col2:
            send_clicked = st.button("Send")
        if send_clicked and user_input:
            result = analyze(user_input)
            st.session_state.chat_history.append({"text": user_input, "analysis": result})
            st.rerun()

    elif mode == "Speech Input":
        import speech_recognition as sr
        st.header("🎤 Speech Review 🎤")
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 2
        recognizer.non_speaking_duration = 1.5
        recognizer.phrase_threshold = 0.3
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
                    st.session_state.speech_history.append({"text": text, "analysis": result})
                    st.rerun()
            except Exception as e:
                status_placeholder.empty()
                st.error(f"Error: {e}")


    # -------------------------
    # Sidebar Filters
    # -------------------------
    if not history_df.empty:
        st.sidebar.markdown("### Filters")
        
        # Message type
        types = history_df["type"].unique().tolist()
        selected_types = st.sidebar.multiselect("Message Type", types, default=types)

        # Sentiment filter
        sentiments = history_df["sentiment_label"].unique().tolist()
        selected_sentiments = st.sidebar.multiselect("Sentiment Label", sentiments, default=sentiments)

        # Risk filter
        risks = history_df["risk_label"].unique().tolist()
        selected_risks = st.sidebar.multiselect("Risk Label", risks, default=risks)

        # Apply filters
        filtered_df = history_df[
            history_df["type"].isin(selected_types) &
            history_df["sentiment_label"].isin(selected_sentiments) &
            history_df["risk_label"].isin(selected_risks)
        ]
    else:
        filtered_df = pd.DataFrame()  # empty


    # -------------------------
    # KPIs
    # -------------------------
    st.markdown("## Personal KPIs")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Messages", len(filtered_df))
    with col2:
        st.metric("Unique Sessions", 1)  # could be user_id if multi-user
    with col3:
        st.metric("Avg Polarity", round(filtered_df["polarity"].mean(), 2) if not filtered_df.empty else 0)
    with col4:
        st.metric("Avg Subjectivity", round(filtered_df["subjectivity"].mean(), 2) if not filtered_df.empty else 0)

    # -------------------------
    # Display Messages & Analysis
    # -------------------------
    if not history_df.empty:
        for item in reversed(history_df.to_dict(orient="records")):
            st.markdown("---")
            col1, spacer, col2 = st.columns([2, 0.3, 3])
                
            # ---- Message Column ----
            with col1:
                st.markdown("""
                <div class="blue-card">
                    <h3>📝 Message</h3>
                    <p>{}</p>
                </div>
                """.format(item["text"]), unsafe_allow_html=True)
                
            # ---- Analysis Column ----
            with col2:
                analysis = item["analysis"]
                st.markdown("""
                <div class="blue-card">
                    <h3>📊 Analysis</h3>
                """, unsafe_allow_html=True)

                sub_col1, sub_col2 = st.columns(2)

                # 🧠 SENTIMENT BOX
                with sub_col1:
                    st.markdown(f"""
                    <div class="analysis-box">
                        <div class="analysis-title">🧠 Sentiment</div>
                        <div class="analysis-item">Label: <span class="analysis-value">{analysis['sentiment']['label']}</span></div>
                        <div class="analysis-item">Polarity: <span class="analysis-value">{analysis['sentiment']['polarity']}</span></div>
                        <div class="analysis-item">Subjectivity: <span class="analysis-value">{analysis['sentiment']['subjectivity']}</span></div>
                        <div class="analysis-item">Compound: <span class="analysis-value">{analysis['sentiment']['vader']['compound']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                # ⚠️ MENTAL HEALTH BOX
                with sub_col2:
                    st.markdown(f"""
                    <div class="analysis-box">
                        <div class="analysis-title">⚠️ Mental Health</div>
                        <div class="analysis-item">Depression Score: <span class="analysis-value">{analysis['dep']}</span></div>
                        <div class="analysis-item">Anxiety Score: <span class="analysis-value">{analysis['anx']}</span></div>
                        <div class="analysis-item">Risk Score: <span class="analysis-value">{analysis['risk']}</span></div>
                        <div class="analysis-item">Risk Level: <span class="analysis-value">{analysis['risk_label']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
            

    
            # ---- Recommendations ----
            st.markdown("#### 💡 Recommendations")
            # Generate recommendations for this single message
            message_df = pd.DataFrame([item])  # make a mini DataFrame for one message
            recs = get_recommendations(message_df, n=2)  # show 2 recommendations per category
            for rec in recs:
                st.info(rec)

    



    
    





# =========================
# 🧑‍⚕️ THERAPIST DASHBOARD
# =========================
if st.session_state.get("role") == "therapist":

    if st.sidebar.button("⬅️ Back to Home"):
        st.session_state.role = None
        st.rerun()

    st.sidebar.title("Therapist Menu")
    mode = st.sidebar.selectbox("Select Mode", ["Dataset Analysis"])

    

    # -------------------------
    # SESSION STATE INIT
    # -------------------------
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None

    if "analyzed_df" not in st.session_state:
        st.session_state.analyzed_df = None

    # -------------------------
    # Dataset Analysis Mode
    # -------------------------
    if mode == "Dataset Analysis":
        st.header("📊 Chatlogs Data Analysis/Review 📊")

        # File upload
        file = st.file_uploader("Upload Excel File", type=["xlsx"])

        if file:
            try:
                df = pd.read_excel(file)
                st.session_state.uploaded_df = df
                st.subheader("Dataset Preview (First 10 Columns)")
                st.dataframe(df.iloc[:, :10])
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.stop()

            if "raw_text" not in df.columns:
                st.error("Dataset must contain 'raw_text' column")
                st.stop()

        # -------------------------
        # Run Analysis
        # -------------------------
        if st.session_state.uploaded_df is not None:

            if st.button("Run Analysis"):
                df = st.session_state.uploaded_df.copy()

                # Ensure 'analyze' function exists
                if "analyze" not in globals():
                    st.error("Function 'analyze' is not defined.")
                    st.stop()

                results = df["raw_text"].apply(analyze)

                # Add analysis columns safely
                df["polarity"] = results.apply(lambda x: x.get("sentiment", {}).get("polarity", np.nan))
                df["subjectivity"] = results.apply(lambda x: x.get("sentiment", {}).get("subjectivity", np.nan))
                df["sentiment_label"] = results.apply(lambda x: x.get("sentiment", {}).get("label", "Unknown"))
                df["compound"] = results.apply(lambda x: x.get("sentiment", {}).get("vader", {}).get("compound", np.nan))

                df["depression_score"] = results.apply(lambda x: x.get("dep", 0))
                df["anxiety_score"] = results.apply(lambda x: x.get("anx", 0))
                df["risk_score"] = results.apply(lambda x: x.get("risk", 0))
                df["risk_label"] = results.apply(lambda x: x.get("risk_label", "Unknown"))

                # Save in session state
                st.session_state.analyzed_df = df
                st.success("✅ Analysis completed!")

        # -------------------------
        # Sidebar Filters
        # -------------------------
        if st.session_state.analyzed_df is not None:
            df = st.session_state.analyzed_df.copy()

            st.sidebar.markdown("### Filters")

            # Filter by Date Range safely
            if "timestamp" in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                start_date = st.sidebar.date_input("Start Date", df['timestamp'].min().date())
                end_date = st.sidebar.date_input("End Date", df['timestamp'].max().date())
            else:
                start_date = None
                end_date = None

            # Filter by Source safely
            sources = df["Source"].unique().tolist() if "Source" in df.columns else []
            selected_sources = st.sidebar.multiselect("Select Source(s)", sources, default=sources)

            # Filter by Sentiment Label
            sentiments = df["sentiment_label"].unique().tolist() if "sentiment_label" in df.columns else []
            selected_sentiments = st.sidebar.multiselect("Select Sentiment(s)", sentiments, default=sentiments)

            # Filter by Risk Label
            risks = df["risk_label"].unique().tolist() if "risk_label" in df.columns else []
            selected_risks = st.sidebar.multiselect("Select Risk(s)", risks, default=risks)

            # Apply all filters
            filtered_df = df.copy()

            if "timestamp" in df.columns and start_date and end_date:
                filtered_df = filtered_df[
                    (filtered_df['timestamp'].dt.date >= start_date) &
                    (filtered_df['timestamp'].dt.date <= end_date)
                ]

            if "Source" in df.columns:
                filtered_df = filtered_df[filtered_df["Source"].isin(selected_sources)]
            
            if "sentiment_label" in df.columns:
                filtered_df = filtered_df[filtered_df["sentiment_label"].isin(selected_sentiments)]

            if "risk_label" in df.columns:
                filtered_df = filtered_df[filtered_df["risk_label"].isin(selected_risks)]


            # -------------------------
            # KPI Metrics
            # -------------------------
            st.markdown("## 📊 Dataset KPIs")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Messages", len(filtered_df))

            with col2:
                st.metric("Unique Users", filtered_df["user_id"].nunique() if "user_id" in filtered_df.columns else 0)

            with col3:
                st.metric("Avg Polarity", round(filtered_df["polarity"].mean(), 2) if "polarity" in filtered_df.columns else 0)

            with col4:
                st.metric("Avg Subjectivity", round(filtered_df["subjectivity"].mean(), 2) if "subjectivity" in filtered_df.columns else 0)

            # -------------------------
            # Visualizations (use filtered_df)
            # -------------------------
            if len(filtered_df) == 0:
                st.warning("No data available for the selected filters.")
            else:
                st.success("Analysis Complete")
                st.dataframe(filtered_df)

                st.subheader("📊 Insights & Visualizations 📊")
                sample_options = [10, 25, 50, 100, 250, 500, 750, 1000]

                # 1️⃣ Sentiment & Risk Side-by-Side
                if "sentiment_label" in filtered_df.columns and "risk_label" in filtered_df.columns:
                    col1, col2 = st.columns(2)

                    # LEFT → Sentiment Pie
                    with col1:
                        st.subheader("Sentiment Distribution")
                        sent_counts = filtered_df["sentiment_label"].value_counts()
                        fig1, ax1 = plt.subplots()
                        ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%', startangle=90)
                        st.pyplot(fig1)

                    # RIGHT → Risk Bar
                    with col2:
                        st.subheader("Risk Distribution")
                        risk_counts = filtered_df["risk_label"].value_counts()
                        st.bar_chart(risk_counts)

                
                # -------------------------
                # Source & Timestamp Charts (filtered)
                # -------------------------
                col1, col2 = st.columns(2)

                # --- Source Horizontal Bar Chart ---
                with col1:
                    st.markdown("#### Messages by Source")

                    if "Source" in filtered_df.columns:
                        source_counts = filtered_df["Source"].value_counts().reset_index()
                        source_counts.columns = ["Source", "Count"]

                        source_chart = (
                            alt.Chart(source_counts)
                            .mark_bar()
                            .encode(
                                x='Count',
                                y=alt.Y('Source', sort='-x')  # sort descending
                            )
                            .properties(
                                width=400,
                                height=380
                            )
                        )
                        st.altair_chart(source_chart, use_container_width=True)
                    else:
                        st.warning("No 'Source' column available for chart")

                # --- Timestamp Chart (Messages Over Time) ---
                with col2:
                    st.markdown("#### Messages by Time Slot")

                    if "timestamp" in filtered_df.columns:
                        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
                        filtered_df = filtered_df.dropna(subset=['timestamp'])

                        # Option 1: Messages per hour
                        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                        hourly_counts = filtered_df['hour'].value_counts().sort_index()

                        st.bar_chart(hourly_counts)
                    else:
                        st.warning("No 'timestamp' column available for chart")




                # depression vs anxiety bar chart
                st.markdown("### Depression vs Anxiety")
                sample_2 = st.selectbox("Sample Size", sample_options, index=3, key="s2")
                viz_df_2 = filtered_df.sample(min(sample_2, len(filtered_df)), random_state=42)
                st.bar_chart(viz_df_2[["depression_score", "anxiety_score"]])


                #coumpound score line chart
                st.markdown("### Compound Score")
                sample_3 = st.selectbox("Sample Size", sample_options, index=3, key="s3")
                viz_df_3 = filtered_df.sample(min(sample_3, len(filtered_df)), random_state=42)
                st.line_chart(viz_df_3["compound"])


                # polarity and subjectivity line charts side by side
                st.markdown("### Polarity & Subjectivity")
                col1, col2 = st.columns(2)

                with col1:
                    sample_4 = st.selectbox("Sample Size (Polarity)", sample_options, index=3, key="s4")
                    viz_df_4 = filtered_df.sample(min(sample_4, len(filtered_df)), random_state=42)
                    st.line_chart(viz_df_4["polarity"])

                with col2:
                    sample_5 = st.selectbox("Sample Size (Subjectivity)", sample_options, index=3, key="s5")
                    viz_df_5 = filtered_df.sample(min(sample_5, len(filtered_df)), random_state=42)
                    st.line_chart(viz_df_5["subjectivity"])


                    




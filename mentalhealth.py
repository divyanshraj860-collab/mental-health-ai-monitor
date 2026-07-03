import streamlit as st
from transformers import pipeline
import pandas as pd
import datetime
from pathlib import Path
import html

# 1. Config & File Paths
CSV_FILE = Path("mood_history.csv")

EMOTION_META = {
    "joy": {"emoji": "😊", "color": "#277c67", "soft": "rgba(39,124,103,.1)", "message": "Wonderful! Keep holding onto this positive energy today. ✨"},
    "love": {"emoji": "❤️", "color": "#b83b5e", "soft": "rgba(184,59,94,.1)", "message": "It sounds like you're feeling deeply connected and fulfilled. ❤️"},
    "sadness": {"emoji": "🫂", "color": "#2d4059", "soft": "rgba(45,64,89,.1)", "message": "It's completely okay to feel down. Give yourself some grace today. 🫂"},
    "anger": {"emoji": "🧭", "color": "#e056fd", "soft": "rgba(224,86,253,.1)", "message": "Take a deep breath. It's valid to feel frustrated; try taking a short break. 🧭"},
    "fear": {"emoji": "🌱", "color": "#ff7675", "soft": "rgba(255,118,117,.1)", "message": "Things feel uncertain, but you are safe. Take it one step at a time. 🌱"},
    "surprise": {"emoji": "⚡", "color": "#f1c40f", "soft": "rgba(241,196,15,.1)", "message": "Wow, sounds like an eventful day! Take time to process it. ⚡"}
}

st.set_page_config(page_title="MindCheck AI", page_icon="🧠", layout="centered")

# 2. Inject CSS Styles
def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&display=swap');
        
        .privacy-pill { background: rgba(48,104,87,.08); color: #306857; padding: .35rem .85rem; border-radius: 20px; font-size: .8rem; font-weight: 500; display: inline-block; margin-bottom: 1.5rem; }
        .brand { font-size: 1.8rem; font-weight: 800; font-family: 'Manrope'; color: #1d6655; margin-bottom: 2rem; }
        .brand-mark { margin-right: .3rem; }
        .eyebrow { color: #5fa48f; font-weight: 700; font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .5rem; }
        .hero-title { font-family: 'Manrope'; font-size: 3.2rem; font-weight: 800; line-height: 1.15; color: #112a23; margin-bottom: 1rem; }
        .hero-title span { color: #3f927b; }
        .hero-copy { color: #4f5e59; font-size: 1.1rem; line-height: 1.65; margin-bottom: 2.5rem; max-width: 620px; }
        
        .glass-card { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(48,104,87,.08); padding: 2rem; border-radius: 24px; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.04); margin-bottom: 1.5rem; }
        .mini-card { background: white; border: 1px solid rgba(48,104,87,.06); padding: 1.2rem; border-radius: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.02); text-align: center; }
        .mini-label { color: #7d8c87; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: .3rem; }
        .mini-value { font-family: 'Manrope'; font-size: 1.5rem; font-weight: 800; color: #1d6655; }
        
        .section-kicker { color:#40816e; font-weight:700; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; margin-top:2.7rem; }
        .section-title { font-family:'Manrope'; font-size:1.8rem; font-weight:800; margin:.25rem 0 1.15rem; }
        .result-card { border-radius:24px; padding:1.55rem; border:1px solid rgba(255,255,255,.8); }
        .result-emoji { width:56px; height:56px; display:grid; place-items:center; font-size:1.7rem; background:rgba(255,255,255,.72); border-radius:18px; }
        .result-label { color:#65716d; text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-weight:700; margin-top:1rem; }
        .result-emotion { font-family:'Manrope'; font-size:2rem; font-weight:800; text-transform:capitalize; margin:.1rem 0 .5rem; }
        .result-message { color:#4f5e59; line-height:1.6; }
        .confidence-track { height:8px; background:rgba(255,255,255,.58); border-radius:20px; overflow:hidden; margin-top:1.2rem; }
        .confidence-fill { height:100%; border-radius:20px; transform-origin:left; }
        .safety-note { color:#6a7672; font-size:.79rem; line-height:1.5; margin-top:1rem; }

        .stTextArea textarea { background:rgba(255,255,255,.72) !important; border:1px solid rgba(48,104,87,.14) !important; border-radius:18px !important; padding:1rem !important; font-size:1rem !important; line-height:1.6 !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.8); }
        .stTextArea textarea:focus { border-color:#5fa48f !important; box-shadow:0 0 0 4px rgba(95,164,143,.12) !important; }
        .stButton > button { width:100%; min-height:3.15rem; border:0; border-radius:15px; font-weight:700; background:linear-gradient(135deg,#277c67,#1d6655); color:white; box-shadow:0 10px 24px rgba(39,124,103,.22); transition:all .2s ease; }
        .stButton > button:hover { transform:translateY(-2px); box-shadow:0 14px 28px rgba(39,124,103,.28); color:white; }
        [data-testid='stMetric'] { background:rgba(255,255,255,.55); border:1px solid rgba(48,104,87,.09); padding:1rem; border-radius:18px; }
        [data-testid='stDataFrame'] { border-radius:16px; overflow:hidden; }
        details { background:rgba(255,255,255,.55) !important; border:1px solid rgba(48,104,87,.10) !important; border-radius:16px !important; }
        @media (max-width: 700px) { .privacy-pill { display:none; } .hero-title { font-size:2.65rem; } [data-testid='stMainBlockContainer'] { padding:1rem 1.05rem 3rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 3. Core ML Pipeline & Utilities
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
    )

def save_entry(text: str, emotion: str, score: float) -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="minutes")
    new_data = pd.DataFrame(
        [[timestamp, text, emotion, round(score, 4)]],
        columns=["Date", "Journal Entry", "Detected Emotion", "Confidence Score"],
    )
    if CSV_FILE.exists():
        current = pd.read_csv(CSV_FILE)
        new_data = pd.concat([current, new_data], ignore_index=True)
    new_data.to_csv(CSV_FILE, index=False)

def load_history() -> pd.DataFrame:
    columns = ["Date", "Journal Entry", "Detected Emotion", "Confidence Score"]
    if not CSV_FILE.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(CSV_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)

def render_result(emotion: str, confidence: float) -> None:
    meta = EMOTION_META.get(
        emotion,
        {"emoji": "💭", "color": "#688078", "soft": "rgba(104,128,120,.14)", "message": "Thank you for sharing."},
    )
    st.markdown(
        f"""
        <div class="result-card" style="background:{meta['soft']}">
            <div class="result-emoji">{meta['emoji']}</div>
            <div class="result-label">Your emotional tone</div>
            <div class="result-emotion" style="color:{meta['color']}">{html.escape(emotion)}</div>
            <div class="result-message">{meta['message']}</div>
            <div class="confidence-track"><div class="confidence-fill" style="width:{confidence * 100:.1f}%;background:{meta['color']}"></div></div>
            <div class="safety-note">{confidence * 100:.0f}% model confidence · A reflection, not a clinical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4. App Execution & Rendering Layout
inject_styles()

st.markdown(
    '<span class="privacy-pill">🔒 Stored only on this device</span><div class="brand"><span class="brand-mark">🧠</span> MindCheck</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">A quiet moment for yourself</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Understand what you feel.<br><span>One thought at a time.</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Write freely. MindCheck reflects the emotional tone in your words and helps you notice patterns—privately, gently, and without judgment.</p>',
    unsafe_allow_html=True,
)

history_df = load_history()
left, right = st.columns([1.55, 0.8], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### How are you feeling today?")
    user_input = st.text_area(
        "Journal entry",
        placeholder="There is no right way to write this. Start with what is on your mind…",
        height=190,
        label_visibility="collapsed",
    )
    analyze = st.button("Reflect on my mood  →", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    if "last_result" in st.session_state:
        render_result(**st.session_state.last_result)
    else:
        st.markdown(
            """
            <div class="glass-card">
                <div class="eyebrow">Your reflection</div>
                <h3 style="margin:.2rem 0 .7rem">A little clarity begins here.</h3>
                <p style="color:#6a7672;line-height:1.65;margin:0">Once you share an entry, your detected emotional tone and a gentle prompt will appear here.</p>
                <div style="font-size:2rem;margin-top:1.2rem">🌿</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if analyze:
    if not user_input.strip():
        st.warning("Write a few words first—your reflection needs something to listen to.")
    else:
        try:
            with st.spinner("Listening to the emotional tone in your words…"):
                prediction = load_model()(user_input.strip(), truncation=True)[0]
                emotion = prediction["label"].lower()
                confidence = float(prediction["score"])
                save_entry(user_input.strip(), emotion, confidence)
                st.session_state.last_result = {"emotion": emotion, "confidence": confidence}
            st.rerun()
        except Exception:
            st.error("The emotion model could not be loaded. Check your internet connection, then try again.")

history_df = load_history()
st.markdown('<div class="section-kicker">Patterns over time</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Your mood landscape</div>', unsafe_allow_html=True)

if history_df.empty:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;padding:2.5rem">
            <div style="font-size:2rem">🌱</div>
            <h3 style="margin:.7rem 0 .35rem">Your story starts with one check-in.</h3>
            <p style="color:#6a7672;margin:0">As you add reflections, your emotional patterns will grow here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    dominant = history_df["Detected Emotion"].mode().iloc[0]
    average_confidence = history_df["Confidence Score"].mean() * 100
    latest_date = history_df["Date"].max()

    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.markdown(f'<div class="mini-card"><div class="mini-label">Total reflections</div><div class="mini-value">{len(history_df)}</div></div>', unsafe_allow_html=True)
    with stat2:
        emoji = EMOTION_META.get(dominant, {}).get("emoji", "💭")
        st.markdown(f'<div class="mini-card"><div class="mini-label">Most felt</div><div class="mini-value">{emoji} {html.escape(dominant.title())}</div></div>', unsafe_allow_html=True)
    with stat3:
        date_text = latest_date.strftime("%d %b %Y") if pd.notna(latest_date) else "Today"
        st.markdown(f'<div class="mini-card"><div class="mini-label">Latest check-in</div><div class="mini-value" style="font-size:1.15rem">{date_text}</div></div>', unsafe_allow_html=True)

    chart_data = history_df["Detected Emotion"].value_counts().rename_axis("Emotion").reset_index(name="Check-ins")
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.bar_chart(chart_data, x="Emotion", y="Check-ins", color="#3f927b", horizontal=True)

    with st.expander("Read past journal entries"):
        display_df = history_df.sort_values("Date", ascending=False).copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y, %I:%M %p")
        display_df["Detected Emotion"] = display_df["Detected Emotion"].str.title()
        display_df["Confidence Score"] = (display_df["Confidence Score"] * 100).round(0).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"Average confidence across reflections: {average_confidence:.0f}%")

st.markdown(
    '<p class="safety-note" style="text-align:center;margin-top:3rem">MindCheck supports self-reflection and is not medical advice. If you are in immediate danger or crisis, contact local emergency services or a trusted person.</p>',
    unsafe_allow_html=True,
)
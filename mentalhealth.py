import streamlit as st
from transformers import pipeline
import pandas as pd
import datetime
import html
from supabase import create_client, Client

# 1. Page Configuration
st.set_page_config(page_title="MindCheck AI", page_icon="🧠", layout="centered")

# 2. Establish Secure Connection to Cloud Database
# (We fetch these credentials safely from Streamlit's environment settings)
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception:
    st.error("Database connection configuration missing. Please add secrets.")
    st.stop()

# --- CUSTOM CSS INJECTION ---
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
        .stTextArea textarea { background:rgba(255,255,255,.72) !important; border:1px solid rgba(48,104,87,.14) !important; border-radius:18px !important; padding:1rem !important; font-size:1rem !important; line-height:1.6 !important; }
        .stButton > button { width:100%; min-height:3.15rem; border:0; border-radius:15px; font-weight:700; background:linear-gradient(135deg,#277c67,#1d6655); color:white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

EMOTION_META = {
    "joy": {"emoji": "😊", "color": "#277c67", "soft": "rgba(39,124,103,.1)", "message": "Wonderful! Keep holding onto this positive energy today. ✨"},
    "love": {"emoji": "❤️", "color": "#b83b5e", "soft": "rgba(184,59,94,.1)", "message": "It sounds like you're feeling deeply connected and fulfilled. ❤️"},
    "sadness": {"emoji": "🫂", "color": "#2d4059", "soft": "rgba(45,64,89,.1)", "message": "It's completely okay to feel down. Give yourself some grace today. 🫂"},
    "anger": {"emoji": "🧭", "color": "#e056fd", "soft": "rgba(224,86,253,.1)", "message": "Take a deep breath. It's valid to feel frustrated; try taking a short break. 🧭"},
    "fear": {"emoji": "🌱", "color": "#ff7675", "soft": "rgba(255,118,117,.1)", "message": "Things feel uncertain, but you are safe. Take it one step at a time. 🌱"},
    "surprise": {"emoji": "⚡", "color": "#f1c40f", "soft": "rgba(241,196,15,.1)", "message": "Wow, sounds like an eventful day! Take time to process it. ⚡"}
}

# 3. Check Authentication State
if not st.experimental_user.is_logged_in:
    inject_styles()
    st.markdown('<div class="brand" style="text-align:center;"><span class="brand-mark">🧠</span> MindCheck AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; max-width:500px; margin:0 auto; padding:3rem 2rem;">
            <h2>Welcome to Your Safe Space</h2>
            <p style="color:#6a7672; margin-bottom:2rem;">To keep your daily journal entries completely private, secure, and isolated from other users, please sign in.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.login()
    st.stop()

# Grab the current user email identity
user_email = st.experimental_user.email

# 4. Core ML Pipeline
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# --- PERMANENT DATABASE STORAGE OPERATIONS ---
def save_entry(text: str, emotion: str, score: float) -> None:
    # Inserts a row into the permanent online Supabase table
    supabase.table("mood_entries").insert({
        "user_email": user_email,
        "journal_entry": text,
        "detected_emotion": emotion,
        "confidence_score": round(score, 4)
    }).execute()

def load_history() -> pd.DataFrame:
    # SECURE USER ISOLATION: Queries data that belongs ONLY to the logged-in user email
    response = supabase.table("mood_entries").select("*").eq("user_email", user_email).execute()
    records = response.data
    
    columns = ["Date", "Journal Entry", "Detected Emotion", "Confidence Score"]
    if not records:
        return pd.DataFrame(columns=columns)
    
    # Format incoming database dictionary items into a Pandas DataFrame
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "created_at": "Date",
        "journal_entry": "Journal Entry",
        "detected_emotion": "Detected Emotion",
        "confidence_score": "Confidence Score"
    })
    return df[columns]

def render_result(emotion: str, confidence: float) -> None:
    meta = EMOTION_META.get(emotion, {"emoji": "💭", "color": "#688078", "soft": "rgba(104,128,120,.14)", "message": "Thank you for sharing."})
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

# 5. UI Layout Execution
inject_styles()

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f'<span class="privacy-pill">🔒 Permanently secured for {st.experimental_user.name}</span>', unsafe_allow_html=True)
with col2:
    if st.button("Log out", key="logout_btn"):
        st.logout()

st.markdown('<div class="brand"><span class="brand-mark">🧠</span> MindCheck</div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">A quiet moment for yourself</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Understand what you feel.<br><span>One thought at a time.</span></h1>', unsafe_allow_html=True)

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
            st.error("The emotion model or database connection failed.")

history_df = load_history()
st.markdown('<div class="section-kicker">Patterns over time</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Your mood landscape</div>', unsafe_allow_html=True)

if history_df.empty:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;padding:2.5rem">
            <div style="font-size:2rem">🌱</div>
            <h3 style="margin:.7rem 0 .35rem">Your story starts with one check-in.</h3>
            <p style="color:#6a7672;margin:0">As you add reflections, your emotional patterns will grow here permanently.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    dominant = history_df["Detected Emotion"].mode().iloc[0]
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

st.markdown(
    '<p class="safety-note" style="text-align:center;margin-top:3rem">MindCheck supports self-reflection and is not medical advice.</p>',
    unsafe_allow_html=True,
)

"""
app.py  –  Streamlit Web App for Roman Urdu Sentiment Analysis
Run:  streamlit run app.py
"""

import os
import joblib
import streamlit as st
from preprocess import clean_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Roman Urdu Sentiment Analyzer",
    page_icon="🇵🇰",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .title  { text-align: center; font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub    { text-align: center; color: #666; margin-bottom: 2rem; }
    .pos-box { background: #d4edda; border-left: 6px solid #28a745;
               padding: 1rem 1.5rem; border-radius: 8px; margin-top: 1rem; }
    .neg-box { background: #f8d7da; border-left: 6px solid #dc3545;
               padding: 1rem 1.5rem; border-radius: 8px; margin-top: 1rem; }
    .metric-card { background: white; border-radius: 10px; padding: 1rem;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    tfidf  = joblib.load(f"{MODEL_DIR}/tfidf.pkl")
    models = {
        "Best Model (Auto)":   joblib.load(f"{MODEL_DIR}/best_model.pkl"),
        "Naive Bayes":         joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
        "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic_regression.pkl"),
        "SVM":                 joblib.load(f"{MODEL_DIR}/svm.pkl"),
    }
    return tfidf, models

# ── Check models exist ────────────────────────────────────────────────────────
if not os.path.exists(f"{MODEL_DIR}/best_model.pkl"):
    st.error("Models not found! Please run:  `python train.py`  first.")
    st.stop()

tfidf, models = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🇵🇰 Roman Urdu Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Paste any Roman Urdu text and detect whether it\'s Positive or Negative</p>', unsafe_allow_html=True)

# ── Model picker ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    selected_model = st.selectbox("Choose Model", list(models.keys()))
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    show_clean = st.checkbox("Show cleaned text")

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Enter Roman Urdu text:",
    placeholder="e.g.  yeh film bohat achi thi, mujhe bohat pasand aayi",
    height=130
)

# ── Examples ──────────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
ex_col1, ex_col2, ex_col3 = st.columns(3)
examples = {
    "😊 Positive":  "yeh film bohat achi thi mujhe bohat pasand aayi zabardast acting thi",
    "😞 Negative":  "bohat bura experience tha bilkul bekaar service thi dobara nahi jaunga",
    "📱 Product":   "yeh mobile bohat fast hai battery timing kamaal hai price bhi theek hai",
}
if ex_col1.button("😊 Positive"):
    user_input = examples["😊 Positive"]
if ex_col2.button("😞 Negative"):
    user_input = examples["😞 Negative"]
if ex_col3.button("📱 Product"):
    user_input = examples["📱 Product"]

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some Roman Urdu text first.")
    else:
        cleaned = clean_text(user_input)
        if not cleaned.strip():
            st.warning("Text became empty after cleaning. Try a longer sentence.")
        else:
            clf   = models[selected_model]
            vec   = tfidf.transform([cleaned])
            pred  = clf.predict(vec)[0]
            label = "Positive" if pred == 1 else "Negative"
            emoji = "😊" if pred == 1 else "😞"
            box   = "pos-box" if pred == 1 else "neg-box"

            st.markdown(f"""
            <div class="{box}">
                <h3 style="margin:0">{emoji} Sentiment: <strong>{label}</strong></h3>
                <p style="margin:0.4rem 0 0; color:#444">Model used: <em>{selected_model}</em></p>
            </div>
            """, unsafe_allow_html=True)

            if show_clean:
                st.info(f"**Cleaned text:** {cleaned}")

# ── Sidebar — Model performance ───────────────────────────────────────────────
st.sidebar.title("📊 Model Performance")
st.sidebar.markdown("Trained on **Roman Urdu Sentiment Dataset** — 11,000 social media comments")

perf = {
    "Naive Bayes":         {"Accuracy": "80.19%", "F1": "80.50%"},
    "Logistic Regression": {"Accuracy": "80.87%", "F1": "81.01%"},
    "SVM":                 {"Accuracy": "79.28%", "F1": "79.68%"},
}
for name, scores in perf.items():
    st.sidebar.markdown(f"**{name}**")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Accuracy", scores["Accuracy"])
    c2.metric("F1 Score", scores["F1"])
    st.sidebar.divider()

st.sidebar.markdown("**Dataset Stats**")
st.sidebar.markdown("- 11,000 samples\n- Source: Kaggle\n- Labels: Positive / Negative\n- Language: Roman Urdu")

# ── Plots tab ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📈 View Training Charts"):
    plot_dir = "plots"
    if os.path.exists(f"{plot_dir}/accuracy_comparison.png"):
        st.image(f"{plot_dir}/accuracy_comparison.png", caption="Model Accuracy Comparison")
        st.image(f"{plot_dir}/metrics_comparison.png",  caption="Full Metrics Comparison")
        st.image(f"{plot_dir}/confusion_matrices.png",  caption="Confusion Matrices")
    else:
        st.info("Run `python train.py` to generate charts.")

st.markdown("<br><center><small>Built with Python · Scikit-learn · Streamlit | Roman Urdu NLP</small></center>", unsafe_allow_html=True)

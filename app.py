import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Cards */
    .pred-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #2e75b6;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 10px 0;
    }
    .pred-number {
        font-size: 64px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .pred-label {
        font-size: 16px;
        color: #90b8d8;
        margin-top: 6px;
    }

    /* Category badge */
    .badge-excellent { background:#1a4731; color:#4ade80; border:1.5px solid #4ade80; border-radius:12px; padding:8px 20px; font-weight:700; font-size:18px; display:inline-block; }
    .badge-good      { background:#1a3a4f; color:#60a5fa; border:1.5px solid #60a5fa; border-radius:12px; padding:8px 20px; font-weight:700; font-size:18px; display:inline-block; }
    .badge-average   { background:#3a2e10; color:#fbbf24; border:1.5px solid #fbbf24; border-radius:12px; padding:8px 20px; font-weight:700; font-size:18px; display:inline-block; }
    .badge-poor      { background:#3a1a1a; color:#f87171; border:1.5px solid #f87171; border-radius:12px; padding:8px 20px; font-weight:700; font-size:18px; display:inline-block; }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #60a5fa;
        border-bottom: 2px solid #2e75b6;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Stat tiles */
    .stat-tile {
        background: #1a2332;
        border: 1px solid #2e4060;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-value { font-size: 28px; font-weight: 700; color: #60a5fa; }
    .stat-label { font-size: 13px; color: #7a9ab5; margin-top: 4px; }

    /* Slider labels */
    .feature-row {
        background: #161b27;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 8px;
    }

    /* Tip box */
    .tip-box {
        background: #1a2a1a;
        border-left: 4px solid #4ade80;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: #c0d8c0;
    }
    .warn-box {
        background: #2a1a1a;
        border-left: 4px solid #f87171;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: #d8c0c0;
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* Metric style */
    [data-testid="stMetric"] {
        background: #1a2332;
        border: 1px solid #2e4060;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student.csv")

model = load_model()
df    = load_data()

# Pre-compute model stats
X_all = df[["StudyHours","Attendance","PreviousMarks","PracticeTests","SleepHours","SocialMediaHours"]]
y_all = df["FinalMarks"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
y_pred_all = model.predict(X_te)
MODEL_R2  = round(r2_score(y_te, y_pred_all) * 100, 1)
MODEL_MAE = round(mean_absolute_error(y_te, y_pred_all), 2)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Marks Predictor")
    st.markdown("---")
    st.markdown("### 📥 Enter Student Details")
    st.markdown(" ")

    study      = st.slider("📚 Study Hours / Day",      min_value=0.0, max_value=12.0, value=6.0, step=0.1,
                           help="Average hours spent studying per day")
    attendance = st.slider("🏫 Attendance (%)",          min_value=40,  max_value=100,  value=75,  step=1,
                           help="Percentage of classes attended")
    previous   = st.slider("📝 Previous Marks",          min_value=0,   max_value=100,  value=60,  step=1,
                           help="Marks in the previous exam")
    practice   = st.slider("✏️ Practice Test Score",     min_value=0,   max_value=100,  value=60,  step=1,
                           help="Score in the most recent practice test")
    sleep      = st.slider("😴 Sleep Hours / Night",     min_value=3.0, max_value=10.0, value=7.0, step=0.5,
                           help="Average sleep hours per night")
    social     = st.slider("📱 Social Media Hours / Day",min_value=0.0, max_value=8.0,  value=2.0, step=0.5,
                           help="Hours spent on social media per day")

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Marks", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Model Accuracy**")
    st.progress(int(MODEL_R2), text=f"R² Score: {MODEL_R2}%")
    st.caption(f"Mean Absolute Error: ±{MODEL_MAE} marks")
    st.markdown("---")
    st.caption("Built with Streamlit · Linear Regression · 300 student records")


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("## 🎓 Student Marks Prediction System")
st.markdown("Predict final exam marks using Machine Learning based on 6 key academic factors.")
st.markdown("---")

# Always run prediction (also on slider change, not just button)
sample = pd.DataFrame([[study, attendance, previous, practice, sleep, social]],
    columns=["StudyHours","Attendance","PreviousMarks","PracticeTests","SleepHours","SocialMediaHours"])
prediction = round(float(model.predict(sample)[0]), 1)
prediction = max(0, min(100, prediction))  # Clamp 0-100

if   prediction >= 80: category, badge_class, emoji, cat_color = "Excellent",     "badge-excellent", "🏆", "#4ade80"
elif prediction >= 60: category, badge_class, emoji, cat_color = "Good",           "badge-good",      "👍", "#60a5fa"
elif prediction >= 40: category, badge_class, emoji, cat_color = "Below Average",  "badge-average",   "⚠️",  "#fbbf24"
else:                  category, badge_class, emoji, cat_color = "Poor",           "badge-poor",      "🚨", "#f87171"


# ── Result Cards Row ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-number">{prediction}</div>
        <div class="pred-label">Predicted Final Marks / 100</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="pred-card">
        <div style="font-size:52px; line-height:1">{emoji}</div>
        <div style="margin-top:10px">
            <span class="{badge_class}">{category}</span>
        </div>
        <div class="pred-label" style="margin-top:10px">Performance Category</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Score bar
    pct = prediction
    bar_color = cat_color
    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-label" style="margin-bottom:12px">Score Gauge</div>
        <div style="background:#1a2a3a; border-radius:8px; height:22px; width:100%; overflow:hidden">
            <div style="background:{bar_color}; width:{pct}%; height:100%; border-radius:8px;
                        transition:width 0.5s ease; display:flex; align-items:center;
                        justify-content:flex-end; padding-right:8px">
                <span style="font-size:12px; font-weight:700; color:#000">{pct}%</span>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:6px">
            <span style="color:#f87171; font-size:12px">0 — Poor</span>
            <span style="color:#4ade80; font-size:12px">100 — Excellent</span>
        </div>
        <div class="pred-label" style="margin-top:12px">Range: ±{MODEL_MAE} marks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ── Two-Column Layout ─────────────────────────────────────────────────────────
left, right = st.columns(2)

# ── Feature Impact Chart ──────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-header">📊 Feature Impact on Your Prediction</div>', unsafe_allow_html=True)

    features    = ["Study Hours","Attendance","Previous Marks","Practice Score","Sleep Hours","Social Media"]
    coeffs      = model.coef_
    user_vals   = [study, attendance, previous, practice, sleep, social]
    contributions = [c * v for c, v in zip(coeffs, user_vals)]

    colors = ["#4ade80" if c > 0 else "#f87171" for c in contributions]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161b27")
    bars = ax.barh(features, contributions, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#7a9ab5", linewidth=1, linestyle="--")
    ax.set_xlabel("Contribution to Predicted Score", color="#90b8d8", fontsize=11)
    ax.tick_params(colors="#c0ccdc", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e4060")
    ax.grid(axis="x", color="#2e4060", linestyle="--", alpha=0.5)
    # Value labels
    for bar, val in zip(bars, contributions):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right",
                color="#ffffff", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption("Green bars boost your marks. Red bars reduce them. Length = strength of impact.")


# ── Tips & Advice ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-header">💡 Personalized Improvement Tips</div>', unsafe_allow_html=True)

    tips  = []
    warns = []

    if study < 4:
        warns.append(f"📚 You study only <b>{study}h/day</b>. Increasing to 6h+ could add ~{round((6-study)*3.53,0):.0f} marks.")
    elif study >= 8:
        tips.append(f"📚 Great study habit! <b>{study}h/day</b> is giving you a strong boost.")

    if attendance < 65:
        warns.append(f"🏫 Attendance is low at <b>{attendance}%</b>. Aim for 80%+ to stay on track.")
    elif attendance >= 85:
        tips.append(f"🏫 Excellent attendance of <b>{attendance}%</b>! Keep it up.")

    if previous < 50:
        warns.append(f"📝 Previous marks were <b>{previous}</b>. Focus on revising weak topics from last time.")
    elif previous >= 75:
        tips.append(f"📝 Strong foundation! Previous score of <b>{previous}</b> is giving solid contribution.")

    if practice < 50:
        warns.append(f"✏️ Practice test score is <b>{practice}</b>. More mock tests will help build confidence.")
    elif practice >= 70:
        tips.append(f"✏️ Good practice score of <b>{practice}</b>! Regular mock tests are paying off.")

    if sleep < 6:
        warns.append(f"😴 Only <b>{sleep}h</b> of sleep. Lack of sleep hurts memory & focus. Aim for 7–8h.")
    elif sleep >= 7 and sleep <= 9:
        tips.append(f"😴 Healthy sleep of <b>{sleep}h</b>. Your brain is well-rested for learning.")

    if social > 5:
        warns.append(f"📱 <b>{social}h/day</b> on social media is high. Reducing to under 3h can recover marks.")
    elif social <= 2:
        tips.append(f"📱 Great discipline! Only <b>{social}h/day</b> on social media.")

    if not tips and not warns:
        tips.append("✅ Your values look balanced overall. Keep up the routine!")

    if tips:
        for t in tips:
            st.markdown(f'<div class="tip-box">✅ {t}</div>', unsafe_allow_html=True)
    if warns:
        for w in warns:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

    # Grade table
    st.markdown("**Grade Reference:**")
    grade_df = pd.DataFrame({
        "Marks Range": ["80 – 100", "60 – 79", "40 – 59", "Below 40"],
        "Category":    ["Excellent 🏆", "Good 👍", "Below Average ⚠️", "Poor 🚨"],
        "Advice":      ["Keep it up!", "Room to grow", "More effort needed", "Urgent attention"]
    })
    st.dataframe(grade_df, hide_index=True, use_container_width=True)


st.markdown("---")


# ── Model Analytics Section ───────────────────────────────────────────────────
with st.expander("📈  View Model Analytics & Dataset Insights", expanded=False):
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f'<div class="stat-tile"><div class="stat-value">{MODEL_R2}%</div><div class="stat-label">Model Accuracy (R² Score)</div></div>', unsafe_allow_html=True)
    with t2:
        st.markdown(f'<div class="stat-tile"><div class="stat-value">±{MODEL_MAE}</div><div class="stat-label">Mean Absolute Error (marks)</div></div>', unsafe_allow_html=True)
    with t3:
        st.markdown(f'<div class="stat-tile"><div class="stat-value">{len(df)}</div><div class="stat-label">Training Records</div></div>', unsafe_allow_html=True)

    st.markdown(" ")
    c1, c2 = st.columns(2)

    # Actual vs Predicted
    with c1:
        st.markdown("**Actual vs Predicted Marks (Test Set)**")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        fig2.patch.set_facecolor("#0f1117")
        ax2.set_facecolor("#161b27")
        ax2.scatter(y_te, y_pred_all, alpha=0.6, s=25, color="#60a5fa", edgecolors="none")
        lims = [min(y_te.min(), y_pred_all.min())-2, max(y_te.max(), y_pred_all.max())+2]
        ax2.plot(lims, lims, color="#f87171", linewidth=1.5, linestyle="--", label="Perfect Prediction")
        ax2.set_xlabel("Actual Marks", color="#90b8d8")
        ax2.set_ylabel("Predicted Marks", color="#90b8d8")
        ax2.tick_params(colors="#c0ccdc")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2e4060")
        ax2.legend(labelcolor="#c0ccdc", facecolor="#161b27", edgecolor="#2e4060", fontsize=9)
        ax2.grid(color="#2e4060", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        st.caption("Points close to the red line = accurate predictions.")

    # Feature importance
    with c2:
        st.markdown("**Feature Importance (Absolute Coefficients)**")
        abs_coef = np.abs(model.coef_)
        feat_names = ["Study Hrs","Attendance","Prev Marks","Practice","Sleep","Social Media"]
        sorted_idx = np.argsort(abs_coef)
        fig3, ax3 = plt.subplots(figsize=(5.5, 4))
        fig3.patch.set_facecolor("#0f1117")
        ax3.set_facecolor("#161b27")
        ax3.barh([feat_names[i] for i in sorted_idx], abs_coef[sorted_idx],
                 color="#60a5fa", edgecolor="none", height=0.6)
        ax3.set_xlabel("Absolute Coefficient", color="#90b8d8")
        ax3.tick_params(colors="#c0ccdc")
        for spine in ax3.spines.values(): spine.set_edgecolor("#2e4060")
        ax3.grid(axis="x", color="#2e4060", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()
        st.caption("Longer bar = feature has stronger effect on marks.")

    # Distribution
    st.markdown("**Distribution of Final Marks in Dataset**")
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    fig4.patch.set_facecolor("#0f1117")
    ax4.set_facecolor("#161b27")
    ax4.hist(df["FinalMarks"], bins=30, color="#60a5fa", edgecolor="#0f1117", alpha=0.9)
    ax4.axvline(prediction, color="#4ade80", linewidth=2, linestyle="--", label=f"Your Prediction: {prediction}")
    ax4.axvline(df["FinalMarks"].mean(), color="#fbbf24", linewidth=1.5, linestyle=":", label=f"Class Average: {df['FinalMarks'].mean():.1f}")
    ax4.set_xlabel("Final Marks", color="#90b8d8")
    ax4.set_ylabel("Number of Students", color="#90b8d8")
    ax4.tick_params(colors="#c0ccdc")
    for spine in ax4.spines.values(): spine.set_edgecolor("#2e4060")
    ax4.legend(labelcolor="#c0ccdc", facecolor="#161b27", edgecolor="#2e4060")
    ax4.grid(axis="y", color="#2e4060", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()
    st.caption("Green line = your predicted score. Yellow = class average.")

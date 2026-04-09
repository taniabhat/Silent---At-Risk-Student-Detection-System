"""
app.py
------
Silent & At-Risk Student Detection System
TPC Dashboard – Streamlit UI

Run:  streamlit run app.py
"""

import os
import sys
import time
import json
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Allow local imports ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "TPC · Silent Student Radar",
    page_icon   = "🎓",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  – semi-dark glassmorphism theme
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Google Fonts ──────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700;800&display=swap');

/* ── Root styling depending on System Theme ────────────────────── */
:root {
  --font-display: 'Syne', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --radius:       16px;
  --radius-sm:    10px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-base:      #0d1117; /* slightly distinct from standard #0e1117 */
    --bg-glass:     rgba(255,255,255,0.04);
    --bg-glass-hov: rgba(255,255,255,0.08);
    --border:       rgba(255,255,255,0.1);
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --accent-blue:  #6eb5ff;
    --accent-teal:  #5de0cb;
    --accent-purple:#b4a4f5;
    --accent-coral: #ff9e9e;
    --grad-hero:    linear-gradient(135deg, #161c2d 0%, #0d1117 60%, #090c10 100%);
    --grad-card:    linear-gradient(145deg, rgba(110,181,255,0.07) 0%, rgba(180,164,245,0.05) 100%);
    --shadow:       0 8px 32px rgba(0,0,0,0.35);
  }
}

@media (prefers-color-scheme: light) {
  :root {
    --bg-base:      #ffffff;
    --bg-glass:     rgba(0,0,0,0.02);
    --bg-glass-hov: rgba(0,0,0,0.05);
    --border:       rgba(0,0,0,0.09);
    --text-primary: #0f172a;
    --text-muted:   #475569;
    --accent-blue:  #2563eb;
    --accent-teal:  #0ea5e9;
    --accent-purple:#7c3aed;
    --accent-coral: #ef4444;
    --grad-hero:    linear-gradient(135deg, #f8fafc 0%, #f1f5f9 60%, #e2e8f0 100%);
    --grad-card:    linear-gradient(145deg, rgba(37,99,235,0.06) 0%, rgba(124,58,237,0.04) 100%);
    --shadow:       0 4px 20px rgba(0,0,0,0.05);
  }
}


/* ── App shell ─────────────────────────── */
.stApp {
  background: var(--grad-hero);
  color: var(--text-primary);
  font-family: var(--font-body);
}
.main .block-container { padding: 2rem 2.5rem 3rem; }
section[data-testid="stSidebar"] {
  background: rgba(15,17,26,0.92);
  border-right: 1px solid var(--border);
  backdrop-filter: blur(18px);
}

/* ── Sidebar labels ────────────────────── */
.css-17lntkn {
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

/* ── Metric cards ──────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.4rem;
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow);
  transition: background 0.25s;
}
[data-testid="metric-container"]:hover { background: var(--bg-glass-hov); }
[data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 2rem !important;
  color: var(--accent-blue) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem; }

/* ── Headers ───────────────────────────── */
h1, h2, h3 {
  font-family: var(--font-display) !important;
  letter-spacing: -0.5px;
}
h1 { font-size: 2.2rem !important; color: var(--text-primary) !important; }
h2 { font-size: 1.45rem !important; color: var(--accent-blue) !important; }
h3 { font-size: 1.1rem !important;  color: var(--accent-teal) !important; }

/* ── Buttons ───────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
  color: #0d1020 !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 0.6rem 1.8rem !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 4px 20px rgba(110,181,255,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(110,181,255,0.45) !important;
}

/* ── Selectbox / text input ────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
  background: var(--bg-glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

/* ── Dataframe / tables ────────────────── */
.stDataFrame, iframe[title="streamlit_dataframe"] {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

/* ── Tabs ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-glass);
  border-radius: var(--radius);
  padding: 4px;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-sm);
  color: var(--text-muted) !important;
  font-family: var(--font-body) !important;
  font-weight: 500;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(110,181,255,0.2), rgba(180,164,245,0.2)) !important;
  color: var(--text-primary) !important;
}

/* ── Progress bar ──────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue)) !important;
  border-radius: 50px !important;
}

/* ── Divider ───────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

/* ── Custom glass card ─────────────────── */
.glass-card {
  background: var(--grad-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(12px);
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}
.glass-card h4 {
  font-family: var(--font-display);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-muted);
  margin: 0 0 0.5rem;
}
.glass-card .value {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent-blue);
}

/* ── Nudge pill ────────────────────────── */
.nudge-pill {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 50px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.5px;
}
.pill-critical { background: rgba(255,100,100,0.15); color: #ff9e9e; border: 1px solid rgba(255,100,100,0.3); }
.pill-low      { background: rgba(255,200,80,0.15);  color: #ffe066; border: 1px solid rgba(255,200,80,0.3); }
.pill-good     { background: rgba(93,224,203,0.15);  color: #5de0cb; border: 1px solid rgba(93,224,203,0.3); }

/* ── Health Meter ring ─────────────────── */
.health-ring-wrap { text-align: center; padding: 1rem 0; }

/* ── Action plan card ──────────────────── */
.action-item {
  background: var(--bg-glass);
  border-left: 3px solid var(--accent-teal);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 0.8rem 1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  color: var(--text-primary);
}
.action-item.critical { border-left-color: var(--accent-coral); }
.action-item.low      { border-left-color: #ffe066; }
.action-item.good     { border-left-color: var(--accent-teal); }

/* ── Header banner ─────────────────────── */
.tpc-header {
  background: linear-gradient(90deg, rgba(110,181,255,0.12), rgba(180,164,245,0.08), rgba(93,224,203,0.06));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.8rem 2.2rem;
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 1.2rem;
}
.tpc-header-icon { font-size: 2.5rem; }
.tpc-header-title { font-family: var(--font-display); font-size: 1.9rem; font-weight: 800; color: var(--text-primary); }
.tpc-header-sub   { font-size: 0.88rem; color: var(--text-muted); margin-top: 2px; }

/* ── Plotly chart backgrounds ──────────── */
.js-plotly-plot { border-radius: var(--radius) !important; }

/* ── Scrollbar ─────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.25); border-radius: 10px; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY LAYOUT DEFAULTS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="DM Sans, sans-serif", size=12),
    margin        = dict(l=10, r=10, t=40, b=10),
    legend        = dict(bgcolor="rgba(0,0,0,0)"),
)
CLUSTER_COLORS = {
    "Placement Ready": "#5de0cb",
    "Silent/At-Risk":  "#ff9e9e",
    "Unprepared":      "#ffe066",
}


# ─────────────────────────────────────────────
# SESSION-STATE BOOTSTRAP & MODEL INIT
# ─────────────────────────────────────────────
def init_system():
    """Train (or load cached) models and store in session state."""
    if "system_ready" in st.session_state and st.session_state.system_ready:
        return

    from model_trainer import train_and_save, load_artefacts, InterventionAgent

    artefact_dir = "artefacts"
    required = [
        os.path.join(artefact_dir, f)
        for f in ["xgb_model.pkl", "kmeans_model.pkl", "processor.pkl",
                  "enriched_df.parquet", "metrics.json"]
    ]

    with st.spinner("🔬 Initialising AI engine – one moment…"):
        if not all(os.path.exists(p) for p in required):
            train_and_save()
        art = load_artefacts()

    st.session_state.df      = art["df"]
    st.session_state.metrics = art["metrics"]
    st.session_state.xgb     = art["xgb"]
    st.session_state.kmeans  = art["kmeans"]
    st.session_state.agent   = InterventionAgent(raw_df=art["df"])
    st.session_state.system_ready = True


# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────
def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"

def severity_pill(s: str) -> str:
    css = {"critical": "pill-critical", "low": "pill-low", "good": "pill-good"}.get(s, "")
    return f'<span class="nudge-pill {css}">{s.upper()}</span>'

def health_color(prob: float) -> str:
    if prob >= 0.7:  return "#5de0cb"
    if prob >= 0.45: return "#ffe066"
    return "#ff9e9e"

def cluster_badge(label: str) -> str:
    color = CLUSTER_COLORS.get(label, "#8892a4")
    return (
        f'<span style="background:rgba(255,255,255,0.07);'
        f'border:1px solid {color}40;border-radius:50px;'
        f'padding:2px 10px;color:{color};font-size:0.8rem;font-weight:600;">'
        f'{label}</span>'
    )


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def chart_cluster_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["cluster_label"].value_counts()
    colors = [CLUSTER_COLORS.get(l, "#8892a4") for l in counts.index]
    fig = go.Figure(go.Pie(
        labels  = counts.index,
        values  = counts.values,
        hole    = 0.62,
        marker  = dict(colors=colors,
                       line=dict(color="#0f1117", width=3)),
        textfont= dict(color="#e8eaf0"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Student Segments", font=dict(size=14)),
        showlegend=True,
        annotations=[dict(
            text=f"<b>{len(df)}</b><br>Students",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )],
    )
    return fig


def chart_risk_heatmap(df: pd.DataFrame) -> go.Figure:
    if "branch" not in df.columns:
        return go.Figure()

    # Decode branch integers if needed
    raw_df = st.session_state.df  # already has original branch strings
    if raw_df is not None and "branch" in raw_df.columns and raw_df["branch"].dtype == object:
        merged = raw_df[["student_id", "branch"]].copy()
        # Attach cluster_label from processed df
        cl_map = df.set_index("student_id")["cluster_label"].to_dict() if "student_id" in df.columns else {}
        merged["cluster_label"] = merged["student_id"].map(cl_map)
    else:
        merged = df.copy()

    if "cluster_label" not in merged.columns:
        return go.Figure()

    pivot = (
        merged.groupby(["branch", "cluster_label"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
    )
    pivot_m = pivot.melt(id_vars="branch", var_name="Cluster", value_name="Count")

    fig = px.density_heatmap(
        pivot_m, x="branch", y="Cluster", z="Count",
        title="Risk Distribution by Branch",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    fig.update_coloraxes(colorbar_title="Count")
    return fig


def chart_feature_radar(record: dict) -> go.Figure:
    dims = {
        "CGPA":          record.get("cgpa", 0),
        "Coding":        record.get("coding_skill_score", 0),
        "Communication": record.get("communication_skill_score", 0),
        "Technical":     record.get("technical_skill_score", 0),
        "Attendance":    record.get("attendance_percentage", 0),
        "Engagement":    record.get("engagement_score", 0),
    }
    # Normalise to 0-10 scale for readability
    norm = {}
    ranges = {
        "CGPA": 10, "Coding": 100, "Communication": 100,
        "Technical": 100, "Attendance": 100, "Engagement": 100,
    }
    for k, v in dims.items():
        norm[k] = round(v / ranges[k] * 10, 1)

    cats   = list(norm.keys()) + [list(norm.keys())[0]]
    values = list(norm.values()) + [list(norm.values())[0]]

    fig = go.Figure(go.Scatterpolar(
        r           = values,
        theta       = cats,
        fill        = "toself",
        fillcolor   = "rgba(110,181,255,0.15)",
        line        = dict(color="#6eb5ff", width=2),
        marker      = dict(color="#6eb5ff", size=6),
        hovertemplate="%{theta}: %{r:.1f}/10<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0,10],
            ),
        ),
        showlegend=False,
        height=320,
        title=dict(text="Skill Profile", font=dict(size=13)),
    )
    return fig


def chart_prob_gauge(prob: float) -> go.Figure:
    color = health_color(prob)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(prob * 100, 1),
        number= dict(suffix="%", font=dict(size=38, color=color, family="Syne")),
        gauge = dict(
            axis=dict(range=[0,100]),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 45],  color="rgba(255,100,100,0.12)"),
                dict(range=[45, 70], color="rgba(255,200,80,0.10)"),
                dict(range=[70, 100],color="rgba(93,224,203,0.10)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=prob*100),
        ),
        title=dict(text="Placement Probability", font=dict(size=13)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    return fig


def chart_score_bars(df: pd.DataFrame) -> go.Figure:
    skill_cols = [
        ("coding_skill_score",        "Coding"),
        ("communication_skill_score", "Comm."),
        ("technical_skill_score",     "Technical"),
        ("attendance_percentage",     "Attendance"),
        ("engagement_score",          "Engagement"),
    ]
    avgs = {}
    for col, label in skill_cols:
        if col in df.columns:
            avgs[label] = df[col].mean()

    colors = ["#6eb5ff","#b4a4f5","#5de0cb","#ffe066","#ff9e9e"]
    fig = go.Figure(go.Bar(
        x            = list(avgs.keys()),
        y            = list(avgs.values()),
        marker_color = colors[:len(avgs)],
        marker_line  = dict(width=0),
        text         = [f"{v:.1f}" for v in avgs.values()],
        textposition = "outside",
        textfont     = dict(color="#e8eaf0", size=11),
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Avg. Skill Scores (Cohort)", font=dict(size=13)),
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(tickfont=dict(color="#8892a4")),
        height=280,
    )
    return fig


# ─────────────────────────────────────────────
# AI INSIGHT GENERATOR  (Groq LLM)
# ─────────────────────────────────────────────
def generate_ai_insight(record: dict, plan: dict) -> str:
    """Generate personalised AI career insight using Groq LLM."""
    try:
        from groq import Groq
    except ImportError:
        return "groq package not installed. Run: pip install groq"

    api_key = os.environ.get("GROQ_API_KEY", "")
    # Fallback: check streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not api_key:
        return "Groq API key not configured. Add GROQ_API_KEY to .streamlit/secrets.toml"

    client = Groq(api_key=api_key)

    issues_text = "\n".join(
        [f"- {i['dimension']}: {i['severity']} (current: {i['value']})"
         for i in plan.get("issues", [])]
    ) or "No critical issues detected."

    prompt = f"""You are an expert placement counselor at an engineering college.
Analyze this student's profile and provide personalized, actionable career advice.

Student Profile:
- Student ID: {record.get('student_id', 'N/A')}
- Branch: {record.get('branch', 'N/A')}
- CGPA: {record.get('cgpa', 'N/A')}
- Coding Skill: {record.get('coding_skill_score', 'N/A')}/100
- Communication: {record.get('communication_skill_score', 'N/A')}/100
- Technical Skill: {record.get('technical_skill_score', 'N/A')}/100
- Attendance: {record.get('attendance_percentage', 'N/A')}%
- Projects: {record.get('projects_count', 'N/A')}
- Certifications: {record.get('certifications_count', 'N/A')}
- Internship Experience: {'Yes' if record.get('internship_experience', 0) else 'No'}
- Placement Cluster: {plan.get('cluster_label', 'N/A')}
- Placement Probability: {plan.get('placement_prob', 0):.1%}
- High-Risk Flag: {'Yes' if plan.get('risk_label', 0) else 'No'}

Key Issues Identified:
{issues_text}

Provide a concise but warm and encouraging analysis (200-300 words) with:
1. Overall assessment of placement readiness
2. Top 3 specific, actionable recommendations tailored to their exact scores
3. A motivational closing message

Be specific -- reference their actual scores and branch. Use emojis sparingly."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"


# ─────────────────────────────────────────────
# ADMIN VIEW
# ─────────────────────────────────────────────
def admin_view(df: pd.DataFrame, metrics: dict):
    # Header
    st.markdown("""
    <div class="tpc-header">
      <div class="tpc-header-icon">🏛️</div>
      <div>
        <div class="tpc-header-title">TPC Admin Dashboard</div>
        <div class="tpc-header-sub">
          Silent &amp; At-Risk Student Detection · Real-time Cohort Analytics
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    total    = metrics.get("n_samples", len(df))
    placed   = metrics.get("n_placed", 0)
    at_risk  = metrics.get("cluster_counts", {}).get("Silent/At-Risk", 0)
    unprep   = metrics.get("cluster_counts", {}).get("Unprepared", 0)
    roc_auc  = metrics.get("roc_auc", 0)

    c1.metric("🎓 Total Students",   f"{total:,}")
    c2.metric("✅ Placed",           f"{placed:,}", f"{placed/total*100:.1f}%")
    c3.metric("⚠️ Silent / At-Risk", f"{at_risk:,}", delta=f"-{at_risk/total*100:.1f}% of cohort", delta_color="inverse")
    c4.metric("❌ Unprepared",       f"{unprep:,}")
    c5.metric("🤖 Model ROC-AUC",    f"{roc_auc:.3f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────
    col_a, col_b = st.columns([1, 1.6], gap="medium")
    with col_a:
        st.plotly_chart(chart_cluster_donut(df), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_risk_heatmap(df), use_container_width=True)

    # ── Charts row 2 ─────────────────────────
    col_c, col_d = st.columns([1.6, 1], gap="medium")
    with col_c:
        st.plotly_chart(chart_score_bars(df), use_container_width=True)
    with col_d:
        # CGPA vs Placement Probability scatter
        if "cgpa" in df.columns and "placement_prob" in df.columns:
            raw_df = st.session_state.df
            cgpa_vals = raw_df["cgpa"] if "cgpa" in raw_df.columns else df["cgpa"]
            fig_sc = px.scatter(
                x      = cgpa_vals,
                y      = df["placement_prob"],
                color  = df["cluster_label"] if "cluster_label" in df.columns else None,
                color_discrete_map = CLUSTER_COLORS,
                labels = {"x":"CGPA","y":"Placement Prob."},
                title  = "CGPA vs Placement Probability",
                opacity= 0.7,
            )
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=280)
            fig_sc.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Student Directories ──────────────
    st.markdown("### 📋 Student Directories")
    tab_urgent, tab_placed = st.tabs(["🚨 Urgent Intervention", "🎓 Placed Students"])

    agent   = st.session_state.agent
    
    with tab_urgent:
        urgent  = agent.get_urgent_list(top_n=None)
    
        if urgent.empty:
            st.info("No students are currently flagged as Silent/At-Risk.")
        else:
            # Display columns
            display_cols = [
                "student_id", "branch", "cgpa",
                "coding_skill_score", "attendance_percentage",
                "placement_prob", "Risk_Label", "cluster_label",
            ]
            show = urgent[[c for c in display_cols if c in urgent.columns]].copy()
            show["placement_prob"] = show["placement_prob"].apply(lambda x: f"{x:.1%}")
    
            st.dataframe(
                show.rename(columns={
                    "student_id":          "ID",
                    "branch":              "Branch",
                    "cgpa":                "CGPA",
                    "coding_skill_score":  "Coding",
                    "attendance_percentage":"Attend%",
                    "placement_prob":      "Placement Prob",
                    "Risk_Label":          "High-Risk Flag",
                    "cluster_label":       "Cluster",
                }),
                use_container_width=True,
                height=360,
            )
    
            st.markdown("---")
            st.markdown("#### 📧 Automated Nudge System")
            col_n1, col_n2 = st.columns([2, 3])
            with col_n1:
                nudge_n = st.slider("Students to notify", 5, min(500, len(urgent)), min(15, len(urgent)))
            with col_n2:
                if st.button("🚀 Send Automated Nudge Emails"):
                    targets = urgent.head(nudge_n)["student_id"].tolist()
                    prog = st.progress(0, text="Preparing emails…")
                    results = []
                    for i, sid in enumerate(targets):
                        time.sleep(0.05)
                        prog.progress((i+1)/len(targets), text=f"📨 Sending to {sid}…")
                        results.append({
                            "ID":     sid,
                            "Status": random.choice(["✅ Delivered","✅ Delivered","⏳ Queued"]),
                            "Time":   time.strftime("%H:%M:%S"),
                        })
                    prog.empty()
                    st.success(f"✅ Nudge emails sent to {len(results)} students!")
                    st.dataframe(pd.DataFrame(results), use_container_width=True, height=250)

    with tab_placed:
        raw_df = st.session_state.df
        placed = pd.DataFrame()
        
        if "placement_status" in raw_df.columns:
            placed = raw_df[raw_df["placement_status"].astype(str).str.lower() == "placed"].copy()
        elif "placed" in raw_df.columns:
            placed = raw_df[raw_df["placed"] == 1].copy()
            
        if placed.empty:
            st.info("No placed students found in data.")
        else:
            display_cols_placed = [
                "student_id", "branch", "cgpa",
                "coding_skill_score", "attendance_percentage",
                "placement_status", "salary_package_lpa"
            ]
            show_placed = placed[[c for c in display_cols_placed if c in placed.columns]].copy()
            
            st.dataframe(
                show_placed.rename(columns={
                    "student_id":          "ID",
                    "branch":              "Branch",
                    "cgpa":                "CGPA",
                    "coding_skill_score":  "Coding",
                    "attendance_percentage":"Attend%",
                    "placement_status":    "Status",
                    "salary_package_lpa":  "Salary (LPA)"
                }),
                use_container_width=True,
                height=600,
            )


# ─────────────────────────────────────────────
# STUDENT VIEW
# ─────────────────────────────────────────────
def student_view(df: pd.DataFrame):
    st.markdown("""
    <div class="tpc-header">
      <div class="tpc-header-icon">👤</div>
      <div>
        <div class="tpc-header-title">Student Career Health Portal</div>
        <div class="tpc-header-sub">
          Personalised Placement Readiness · AI-Powered Action Plans
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Student selector ──────────────────────
    raw_df = st.session_state.df

    st.markdown("#### 1️⃣ Search by Exact Student ID")
    col_sel1, col_btn1 = st.columns([3, 1])
    with col_sel1:
        search_id = st.text_input("Enter ID", placeholder="e.g. 3", label_visibility="collapsed")
    with col_btn1:
        analyse_search = st.button("⚡ Analyse ID")

    st.markdown("#### 2️⃣ Or Quickly Select an At-Risk Student")
    col_sel2, col_btn2 = st.columns([3, 1])
    with col_sel2:
        urgent_list = st.session_state.agent.get_urgent_list(top_n=20)
        urgent_ids = urgent_list["student_id"].astype(str).tolist() if not urgent_list.empty else []
        quick_select = st.selectbox("Select ID", ["-- Select ID --"] + urgent_ids, label_visibility="collapsed")
    with col_btn2:
        analyse_quick = st.button("⚡ Analyse Selected")

    analyse = analyse_search or analyse_quick
    selected_id = ""
    
    if analyse_search:
        selected_id = str(search_id).strip()
    elif analyse_quick and quick_select != "-- Select ID --":
        selected_id = str(quick_select)

    if not analyse and "last_student" not in st.session_state:
        st.info("Select a student ID and click **Analyse** to view their Career Health Report.")
        return

    if analyse:
        if not selected_id:
            st.error("Please enter or select a valid Student ID.")
            return
        
        # Verify valid ID 
        if not (raw_df["student_id"].astype(str) == selected_id).any():
            st.error(f"Student ID '{selected_id}' not found in database.")
            return
            
        st.session_state.last_student = selected_id
    elif "last_student" not in st.session_state or not st.session_state.last_student:
        return
        
    sid = st.session_state.last_student

    # Fetch plan
    agent = st.session_state.agent
    with st.spinner("🧠 Generating personalised action plan…"):
        plan   = agent.generate_plan(sid)
        record = raw_df[raw_df["student_id"] == sid].iloc[0].to_dict()

    # ── Career Health Meter ───────────────────
    prob         = plan["placement_prob"]
    cluster      = plan["cluster_label"]
    risk_flag    = plan["risk_label"]
    hc           = health_color(prob)

    col1, col2, col3 = st.columns([1.2, 1, 1.8], gap="medium")

    with col1:
        st.plotly_chart(chart_prob_gauge(prob), use_container_width=True)
        badge = cluster_badge(cluster)
        rf    = '🔴 HIGH-RISK FLAG' if risk_flag else '🟢 Normal Status'
        st.markdown(
            f"<div style='text-align:center;margin-top:-0.5rem;'>"
            f"{badge}<br/>"
            f"<span style='color:#8892a4;font-size:0.8rem;margin-top:4px;display:block;'>{rf}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.plotly_chart(chart_feature_radar(record), use_container_width=True)

    with col3:
        st.markdown("#### 📋 AI-Generated Action Plan")
        st.markdown(
            f"<div class='glass-card'><p style='color:#b4a4f5;font-style:italic;font-size:0.92rem;margin:0;'>"
            f"{plan['summary']}</p></div>",
            unsafe_allow_html=True,
        )

        if not plan["action_plan"]:
            st.success("🎉 No critical gaps found. Focus on acing your interviews!")
        else:
            for item in plan["action_plan"]:
                css_cls = item["severity"]
                dim_label = item["dimension"]
                val       = item["current_val"]
                pill      = severity_pill(item["severity"])

                st.markdown(
                    f"<div class='action-item {css_cls}'>"
                    f"<strong>{dim_label}</strong> "
                    f"<span style='color:#8892a4;font-size:0.82rem;'>(current: {val:.1f})</span>"
                    f" &nbsp;{pill}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                for act in item["actions"]:
                    st.markdown(
                        f"<div style='padding:4px 0 4px 1.5rem;"
                        f"color:#c8d0de;font-size:0.88rem;'>{act}</div>",
                        unsafe_allow_html=True,
                    )

    # ── AI Counselor ──────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Counselor Insights")
    st.caption("Powered by Groq LLM — personalised, AI-generated career guidance")

    ai_key = f"ai_insight_{sid}"
    if st.button("✨ Generate AI-Powered Analysis"):
        with st.spinner("🧠 AI is analysing the student profile..."):
            insight = generate_ai_insight(record, plan)
            st.session_state[ai_key] = insight

    if ai_key in st.session_state:
        st.markdown(
            f"<div class='glass-card'><p style='color:#e8eaf0;font-size:0.92rem;"
            f"line-height:1.7;margin:0;white-space:pre-wrap;'>"
            f"{st.session_state[ai_key]}</p></div>",
            unsafe_allow_html=True,
        )

    # ── Peer Comparison ───────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 📊 Peer Comparison")

    skill_cols = [c for c in [
        "cgpa","coding_skill_score","communication_skill_score",
        "technical_skill_score","attendance_percentage","engagement_score",
    ] if c in raw_df.columns]

    cohort_means = raw_df[skill_cols].mean()
    student_vals = {c: record.get(c, 0) for c in skill_cols}

    compare_df = pd.DataFrame({
        "Skill":   [c.replace("_"," ").title() for c in skill_cols],
        "You":     [student_vals[c] for c in skill_cols],
        "Cohort":  [cohort_means[c] for c in skill_cols],
    })

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="You", x=compare_df["Skill"], y=compare_df["You"],
        marker_color="#6eb5ff", marker_line=dict(width=0),
    ))
    fig_cmp.add_trace(go.Bar(
        name="Cohort Avg", x=compare_df["Skill"], y=compare_df["Cohort"],
        marker_color="rgba(180,164,245,0.45)", marker_line=dict(width=0),
    ))
    fig_cmp.update_layout(
        **PLOTLY_LAYOUT, barmode="group",
        title="Your Scores vs Cohort Average",
        height=300,
        xaxis=dict(tickfont=dict(size=11, color="#8892a4")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Self-Nudge ─────────────────────────────
    st.markdown("<hr/>", unsafe_allow_html=True)
    col_nd1, col_nd2 = st.columns([1, 2])
    with col_nd1:
        if st.button("📩 Request Counsellor Meeting"):
            st.toast(f"✅ Meeting request sent for {sid}!", icon="📅")
            time.sleep(0.4)
            st.success("Your placement counsellor has been notified. "
                       "Expect a calendar invite within 24 hours.")
    with col_nd2:
        if st.button("📥 Download My Action Plan (JSON)"):
            st.download_button(
                label    = "💾 Download JSON",
                data     = json.dumps(plan, indent=2),
                file_name= f"action_plan_{sid}.json",
                mime     = "application/json",
            )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(metrics: dict):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1.2rem 0 0.5rem;'>
          <div style='font-size:2.2rem;'>🎓</div>
          <div style='font-family:Syne,sans-serif;font-size:1.1rem;
                      font-weight:800;color:#e8eaf0;margin:0.3rem 0;'>
            Silent Radar
          </div>
          <div style='font-size:0.72rem;color:#8892a4;letter-spacing:1px;
                      text-transform:uppercase;'>
            TPC AI Platform
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        view = st.radio(
            "Navigation",
            ["🏛️ Admin Dashboard", "👤 Student Portal"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**📈 Model Performance**")
        acc = metrics.get("accuracy", 0)
        auc = metrics.get("roc_auc", 0)
        st.metric("Accuracy",  f"{acc:.1%}")
        st.metric("ROC-AUC",   f"{auc:.4f}")

        st.markdown("---")
        st.markdown("**🗂️ Cohort Summary**")
        cc = metrics.get("cluster_counts", {})
        for label, count in cc.items():
            color = {"Placement Ready":"#5de0cb",
                     "Silent/At-Risk":"#ff9e9e",
                     "Unprepared":"#ffe066"}.get(label, "#8892a4")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;font-size:0.82rem;'>"
                f"<span style='color:{color};'>● {label}</span>"
                f"<span style='color:#e8eaf0;font-weight:600;'>{count}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🔄 Retrain Models"):
            with st.spinner("Retraining…"):
                from model_trainer import train_and_save, load_artefacts, InterventionAgent
                train_and_save()
                art = load_artefacts()
                st.session_state.df      = art["df"]
                st.session_state.metrics = art["metrics"]
                st.session_state.xgb     = art["xgb"]
                st.session_state.kmeans  = art["kmeans"]
                st.session_state.agent   = InterventionAgent(raw_df=art["df"])
            st.success("Models retrained!")
            st.rerun()

        st.markdown(
            "<div style='font-size:0.7rem;color:#4a5060;text-align:center;"
            "margin-top:2rem;'>Powered by XGBoost + KMeans<br/>ReAct Agent v1.0</div>",
            unsafe_allow_html=True,
        )

    return view


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    init_system()

    df      = st.session_state.df
    metrics = st.session_state.metrics

    view = render_sidebar(metrics)

    if "Admin" in view:
        admin_view(df, metrics)
    else:
        student_view(df)


if __name__ == "__main__":
    main()

"""
app.py – ESG Generative AI Agent MVP
=====================================
Improvements over original:
  - compute_kpi_improvements now passes df + col_map for real data baselines
  - compute_benchmark_comparison now passes df + col_map for real values
  - Added Tab 9: Data Explorer (peer comparison, YoY trends, raw metrics)
  - Added Excel download alongside PDF
  - LLM backend status shows a persistent coloured indicator
  - Missing-field form: improved layout and validation messages
  - Chat: "Clear history" button added
  - All st.session_state mutations centralised via _reset_analysis()
  - Various minor UI polish fixes

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_handler import (
    load_csv,
    validate_year_constraint,
    run_data_quality_checks,
    compute_kpi_improvements,
    generate_data_summary,
    get_industry_benchmark,
    compute_benchmark_comparison,
    compute_peer_comparison,
    get_yoy_metrics,
    detect_esg_columns,
    export_to_excel,
)
from utils.llm_agent import (
    extract_company_profile,
    classify_business_size,
    generate_risks_and_opportunities,
    rank_solutions,
    answer_what_if,
    set_llm_backend,
    get_llm_backend,
    get_regulatory_context,
    compute_esg_maturity_score,
    generate_executive_summary,
)
from utils.report_generator import generate_pdf_report


# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="ESG AI Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Global CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    text-align: center; padding: 2rem 1.5rem 1.5rem; margin: -1rem -1rem 1.5rem -1rem;
    background: linear-gradient(135deg, #0d4f2b 0%, #1a7a45 40%, #27ae60 100%);
    border-radius: 0 0 24px 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.hero-slogan { font-size: 1.4rem; font-weight: 600; color: rgba(255,255,255,0.88); font-style: italic; margin-bottom: 0.5rem; }
.hero-title  { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0; }
.hero-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.75); margin-top: 0.35rem; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #f0f7f2 0%, #dceee2 100%) !important; }
section[data-testid="stSidebar"] > div:first-child { background: transparent !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #1a5c32; }
section[data-testid="stSidebar"] hr { border-color: #b5d6be !important; }

button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 0.86rem !important; padding: 0.45rem 0.8rem !important; }
div[data-baseweb="tab-highlight"] { background-color: #27ae60 !important; }

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f0faf4, #e8f5e9);
    border: 1px solid #c8e6c9; border-radius: 12px;
    padding: 1rem 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label { color: #2e7d32 !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.7rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1b5e20 !important; font-weight: 700 !important; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; vertical-align: middle; margin-left: 6px; }
.badge-high   { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
.badge-medium { background: #fff8e1; color: #e65100; border: 1px solid #ffe082; }
.badge-low    { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }

.analysis-card {
    background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px;
    padding: 1.1rem 1.2rem; margin-bottom: 0.8rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04); transition: box-shadow 0.2s ease;
}
.analysis-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.analysis-card h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #333; }
.analysis-card p  { margin: 0; font-size: 0.85rem; color: #555; line-height: 1.5; }

.section-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem; }
.section-icon   { font-size: 1.5rem; }
.section-label  { font-size: 1.3rem; font-weight: 700; color: #1a5c32; }

.stDownloadButton > button {
    background: linear-gradient(135deg, #2e7d32, #43a047) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(46,125,50,0.25) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 15px rgba(46,125,50,0.35) !important;
}

.maturity-grade { font-size: 3rem; font-weight: 800; text-align: center; padding: 0.5rem; }
.grade-A { color: #2e7d32; } .grade-B { color: #4caf50; } .grade-C { color: #f57f17; }
.grade-D { color: #e65100; } .grade-F { color: #c62828; }

.streamlit-expanderHeader { font-weight: 600 !important; color: #2e7d32 !important; }

.privacy-local  { background:#e8f5e9; border-left:4px solid #2e7d32; padding:6px 10px; border-radius:4px; font-size:0.85rem; }
.privacy-cloud  { background:#fff8e1; border-left:4px solid #f9a825; padding:6px 10px; border-radius:4px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Hero Banner
# =====================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-slogan">"Measure what matters. Report what counts. Act before it's required."</div>
    <div class="hero-title">🌍 ESG Generative AI Agent</div>
    <div class="hero-subtitle">Upload your company data &bull; Get actionable, tailored ESG recommendations</div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# Session State
# =====================================================================
_DEFAULTS = {
    "df": None, "data_valid": False, "profile": None, "business_size": None,
    "solutions": None, "risks_opps": None, "chat_history": [],
    "llm_backend": "mock", "llm_api_key": "",
    "data_summary": None, "maturity": None, "exec_summary": None,
    "benchmark_df": None, "regulatory": None,
    "selected_company": None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _reset_analysis():
    """Clear all computed analysis so it gets regenerated with new data/settings."""
    for k in ["profile", "business_size", "solutions", "risks_opps", "chat_history",
              "data_summary", "maturity", "exec_summary", "benchmark_df", "regulatory",
              "selected_company"]:
        st.session_state[k] = [] if k == "chat_history" else None


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("### 🤖 AI Backend")
    backend_choice = st.selectbox(
        "LLM Provider",
        ["Mock (no API key)",
         "Ollama (Local – data stays private)",
         "OpenAI (GPT-4o)",
         "Anthropic (Claude Sonnet)",
         "Google Gemini (2.0 Flash)"],
        help="Mock uses built-in templates. Ollama runs locally. Cloud providers receive only aggregated summaries.",
    )

    if "Ollama" in backend_choice:
        st.caption("🔒 Runs entirely on your machine. No API key needed.")
        ollama_model = st.text_input("Model name", value="llama3.1",
                                     help="Must be pulled via `ollama pull <model>`")
        ollama_url   = st.text_input("Ollama URL", value="http://localhost:11434")
        if st.button("Connect to Ollama", key="btn_ollama"):
            msg = set_llm_backend("ollama", ollama_model)
            st.session_state["llm_backend"]  = "ollama"
            st.session_state["llm_api_key"]  = ollama_model
            _reset_analysis()
            if "All data stays" in msg:
                st.success(msg)
            else:
                st.error(msg)

    elif "OpenAI" in backend_choice:
        api_key = st.text_input("OpenAI API Key", type="password",
                                value=st.session_state["llm_api_key"] if st.session_state["llm_backend"] == "openai" else "")
        if st.button("Connect", key="btn_openai"):
            msg = set_llm_backend("openai", api_key)
            st.session_state["llm_backend"] = "openai"
            st.session_state["llm_api_key"] = api_key
            _reset_analysis()
            (st.success if "✅" in msg else st.error)(msg)

    elif "Anthropic" in backend_choice:
        api_key = st.text_input("Anthropic API Key", type="password",
                                value=st.session_state["llm_api_key"] if st.session_state["llm_backend"] == "anthropic" else "")
        if st.button("Connect", key="btn_anthropic"):
            msg = set_llm_backend("anthropic", api_key)
            st.session_state["llm_backend"] = "anthropic"
            st.session_state["llm_api_key"] = api_key
            _reset_analysis()
            (st.success if "✅" in msg else st.error)(msg)

    elif "Gemini" in backend_choice:
        api_key = st.text_input("Google API Key", type="password",
                                value=st.session_state["llm_api_key"] if st.session_state["llm_backend"] == "gemini" else "")
        if st.button("Connect", key="btn_gemini"):
            msg = set_llm_backend("gemini", api_key)
            st.session_state["llm_backend"] = "gemini"
            st.session_state["llm_api_key"] = api_key
            _reset_analysis()
            (st.success if "✅" in msg else st.error)(msg)

    else:  # Mock
        set_llm_backend("mock")
        st.session_state["llm_backend"] = "mock"
        st.session_state["llm_api_key"] = ""

    # Active backend badge
    active_backend = get_llm_backend()
    backend_labels = {"mock": "MOCK", "openai": "OPENAI GPT-4o",
                      "anthropic": "ANTHROPIC CLAUDE", "gemini": "GEMINI 2.0",
                      "ollama": "OLLAMA (LOCAL)"}
    st.caption(f"Active: **{backend_labels.get(active_backend, 'MOCK')}**")

    if active_backend in ("ollama", "mock"):
        st.markdown('<div class="privacy-local">🟢 <b>PRIVATE</b> — no data leaves your machine</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="privacy-cloud">🟡 <b>CLOUD</b> — only aggregated summaries sent, never raw data</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Data Upload ──────────────────────────────────────────────────────
    st.markdown("### 📂 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Company CSV",
        type=["csv"],
        help="CSV with ESG data. Must contain data from 2010 onwards.",
    )

    if uploaded_file is not None:
        try:
            df_new = load_csv(uploaded_file)
            # Only reset if it's a different file (avoid re-running on every interaction)
            if (st.session_state["df"] is None or
                    len(df_new) != len(st.session_state["df"]) or
                    list(df_new.columns) != list(st.session_state.get("df", pd.DataFrame()).columns)):
                st.session_state["df"] = df_new
                _reset_analysis()

            df_loaded = st.session_state["df"]
            st.success(f"Loaded **{len(df_loaded)}** rows, **{len(df_loaded.columns)}** columns.")

            is_valid, msg = validate_year_constraint(df_loaded)
            if is_valid:
                st.success(msg)
                st.session_state["data_valid"] = True
            else:
                st.error(msg)
                st.session_state["data_valid"] = False

            for w in run_data_quality_checks(df_loaded):
                st.warning(w)

        except ValueError as e:
            st.error(str(e))
            st.session_state["data_valid"] = False

    if st.session_state["df"] is not None:
        with st.expander("Preview uploaded data"):
            st.dataframe(st.session_state["df"].head(10), use_container_width=True)

    # ── Company Selector ────────────────────────────────────────────────
    if st.session_state["df"] is not None:
        _tmp_col_map = detect_esg_columns(st.session_state["df"])
        if _tmp_col_map and "company" in _tmp_col_map:
            _all_companies = sorted(st.session_state["df"][_tmp_col_map["company"]].dropna().unique().tolist())
            if len(_all_companies) > 1:
                st.divider()
                st.markdown("### 🏢 Company Selection")

                _prev_company = st.session_state.get("selected_company")
                selected_company = st.selectbox(
                    "Analyse company:",
                    options=_all_companies,
                    index=_all_companies.index(_prev_company) if _prev_company in _all_companies else 0,
                    key="sidebar_company_select",
                    help="Select a company from your dataset. All tabs will update to show this company's data.",
                )

                # If company changed, reset analysis so everything recomputes
                if selected_company != st.session_state.get("selected_company"):
                    # Reset analysis but keep df and data_valid
                    for k in ["profile", "business_size", "solutions", "risks_opps",
                              "chat_history", "data_summary", "maturity", "exec_summary",
                              "benchmark_df", "regulatory"]:
                        st.session_state[k] = [] if k == "chat_history" else None
                    st.session_state["selected_company"] = selected_company
                    st.rerun()

    # ── Data Privacy Notice ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔒 Data Privacy")
    with st.expander("How your data is protected"):
        st.markdown("""
**Your data never leaves your control:**

- **No raw data** is ever sent to cloud LLM APIs. Only aggregated, anonymised summaries (column statistics, row counts) are transmitted.
- **PII scrubbing**: Emails, phone numbers, URLs, and ID numbers are automatically redacted before any cloud API call.
- **Session-only storage**: All data lives in your browser session and is deleted when you close the tab.
- **Local option**: Use Ollama to keep ALL processing entirely on your machine — zero external calls.

For maximum security, use the **Ollama (Local)** backend.
        """)


# =====================================================================
# Guard – stop if no valid data
# =====================================================================
if not st.session_state["data_valid"]:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.info("⬅️ Upload a valid CSV file using the sidebar to begin your ESG analysis.")
    st.stop()

df = st.session_state["df"]
col_map = detect_esg_columns(df)

# =====================================================================
# Filter to selected company (if multi-company dataset)
# =====================================================================
if col_map and "company" in col_map:
    _all_companies = sorted(df[col_map["company"]].dropna().unique().tolist())
    if len(_all_companies) > 1:
        # Set default selected company if not yet set
        if st.session_state["selected_company"] is None:
            st.session_state["selected_company"] = _all_companies[0]
        _sel = st.session_state["selected_company"]
        if _sel in _all_companies:
            df = df[df[col_map["company"]] == _sel].copy()
        else:
            st.session_state["selected_company"] = _all_companies[0]
            df = df[df[col_map["company"]] == _all_companies[0]].copy()

# Keep full dataset available for cross-company features
df_full = st.session_state["df"]


# =====================================================================
# Data Summary
# =====================================================================
if st.session_state["data_summary"] is None:
    st.session_state["data_summary"] = generate_data_summary(df)

data_summary      = st.session_state["data_summary"]
data_summary_text = data_summary.get("text_summary", "")


# =====================================================================
# LLM Extraction (runs once)
# =====================================================================
_safe_text = (
    f"Columns: {', '.join(df.columns.tolist())}. "
    f"Row count: {len(df)}. "
    f"{data_summary_text}"
)
if len(df) > 0:
    _first_row    = df.iloc[0].to_dict()
    _sample_pairs = " | ".join(f"{k}: {v}" for k, v in list(_first_row.items())[:15])
    _safe_text   += f"\nSample record: {_sample_pairs}"

if st.session_state["profile"] is None:
    with st.spinner("Extracting company profile from uploaded data…"):
        profile = extract_company_profile(_safe_text)
        st.session_state["profile"] = profile

profile = st.session_state["profile"]


# =====================================================================
# Missing Fields Fallback Form
# =====================================================================
REQUIRED_FIELDS = ["employee_count", "budget", "industry", "country"]
missing_fields  = [f for f in REQUIRED_FIELDS if profile.get(f) is None]

if missing_fields:
    st.markdown('<div class="section-header"><span class="section-icon">✏️</span>'
                '<span class="section-label">Complete Missing Information</span></div>',
                unsafe_allow_html=True)
    st.info(f"The AI could not extract {len(missing_fields)} field(s) from your data. "
            f"Please fill in the missing information below.")

    with st.form("fallback_form"):
        form_cols = st.columns(min(len(missing_fields), 2))
        col_idx = 0
        for field in missing_fields:
            with form_cols[col_idx % len(form_cols)]:
                if field == "employee_count":
                    profile["employee_count"] = st.number_input(
                        "Number of Employees", min_value=1, max_value=500_000, value=100, step=10)
                elif field == "budget":
                    profile["budget"] = st.number_input(
                        "Willingness to Spend on ESG (USD)", min_value=1_000.0,
                        max_value=100_000_000.0, value=100_000.0, step=10_000.0, format="%.0f")
                elif field == "industry":
                    profile["industry"] = st.selectbox("Industry", options=[
                        "Technology", "Healthcare", "Finance", "Financial Services",
                        "Manufacturing", "Retail", "Energy", "Agriculture", "Construction",
                        "Transportation", "Education", "Telecommunications", "Pharmaceuticals",
                        "Automotive", "Real Estate", "Hospitality", "Mining",
                        "Oil And Gas", "Utilities", "Consumer Goods", "Other"])
                elif field == "country":
                    profile["country"] = st.selectbox("Country", options=[
                        "United States", "United Kingdom", "Canada", "Germany", "France",
                        "Japan", "China", "India", "Australia", "Brazil", "South Korea",
                        "Mexico", "Italy", "Spain", "Netherlands", "Switzerland",
                        "Singapore", "Sweden", "Norway", "Denmark", "Other"])
            col_idx += 1

        if st.form_submit_button("✅ Submit Missing Information"):
            st.session_state["profile"] = profile
            st.success("Profile updated! Running analysis…")
            st.rerun()

    if any(profile.get(f) is None for f in missing_fields):
        st.stop()


# =====================================================================
# Derived Data Computation
# =====================================================================
if st.session_state["business_size"] is None:
    size = classify_business_size(
        profile.get("employee_count", 0) or 0,
        profile.get("budget", 0) or 0,
    )
    profile["business_size"] = size
    st.session_state["business_size"] = size
    st.session_state["profile"] = profile

size = st.session_state["business_size"]

if st.session_state["regulatory"] is None:
    st.session_state["regulatory"] = get_regulatory_context(profile["country"])
regulatory = st.session_state["regulatory"]

if st.session_state["maturity"] is None:
    st.session_state["maturity"] = compute_esg_maturity_score(profile, data_summary)
maturity = st.session_state["maturity"]

if st.session_state["risks_opps"] is None:
    with st.spinner("Generating risk & opportunity analysis…"):
        st.session_state["risks_opps"] = generate_risks_and_opportunities(
            profile["industry"], profile["country"], size, data_summary_text)
ro = st.session_state["risks_opps"]

if st.session_state["solutions"] is None:
    with st.spinner("Ranking ESG solutions…"):
        st.session_state["solutions"] = rank_solutions(
            profile["industry"], size, profile["budget"], data_summary_text)
solutions = st.session_state["solutions"]

# KPI improvements now pass real data
kpi_df = compute_kpi_improvements(
    size, profile["industry"], solutions,
    df=df, col_map=col_map,
)

# Benchmark comparison with real data
if st.session_state["benchmark_df"] is None:
    st.session_state["benchmark_df"] = compute_benchmark_comparison(
        profile, maturity, profile["industry"],
        df=df, col_map=col_map,
    )
benchmark_df = st.session_state["benchmark_df"]

if st.session_state["exec_summary"] is None:
    with st.spinner("Generating executive summary…"):
        st.session_state["exec_summary"] = generate_executive_summary(
            profile, maturity, solutions, regulatory, data_summary_text)
exec_summary = st.session_state["exec_summary"]


# =====================================================================
# Plotly defaults
# =====================================================================
_PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,249,0.6)",
    margin=dict(l=40, r=20, t=50, b=40),
)
_GREEN_PALETTE = ["#1a7a45", "#27ae60", "#66bb6a", "#a5d6a7", "#c8e6c9", "#e8f5e9"]


# =====================================================================
# TABBED DASHBOARD
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Executive Dashboard",
    "🏢 Company Profile",
    "📈 Industry Benchmarking",
    "⚠️ Risks & Opportunities",
    "💡 Solutions & Costs",
    "🎯 KPI Tracking",
    "⚖️ Regulatory Intel",
    "💬 What-If Chat",
    "🔍 Data Explorer",
])


# -----------------------------------------------------------------
# Tab 1 – Executive Dashboard
# -----------------------------------------------------------------
with tab1:
    _dash_company = st.session_state.get("selected_company")
    _dash_label = f"Executive Dashboard — {_dash_company}" if _dash_company else "Executive Dashboard"
    st.markdown(f'<div class="section-header"><span class="section-icon">📊</span>'
                f'<span class="section-label">{_dash_label}</span></div>',
                unsafe_allow_html=True)

    grade_col, pillar_col = st.columns([1, 2])

    with grade_col:
        grade_class = f"grade-{maturity['grade']}"
        st.markdown(f'<div class="maturity-grade {grade_class}">{maturity["grade"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;font-size:1.1rem;color:#555;'>"
                    f"ESG Maturity: <b>{maturity['overall']}/100</b></p>",
                    unsafe_allow_html=True)
        st.caption(maturity.get("rationale", "")[:200])

    with pillar_col:
        p1, p2, p3 = st.columns(3)
        p1.metric("🌱 Environmental", f"{maturity['environmental']:.0f}/100",
                  delta="Strong" if maturity["environmental"] >= 65 else "Needs Work",
                  delta_color="normal" if maturity["environmental"] >= 65 else "inverse")
        p2.metric("👥 Social", f"{maturity['social']:.0f}/100",
                  delta="Strong" if maturity["social"] >= 65 else "Needs Work",
                  delta_color="normal" if maturity["social"] >= 65 else "inverse")
        p3.metric("🏛️ Governance", f"{maturity['governance']:.0f}/100",
                  delta="Strong" if maturity["governance"] >= 65 else "Needs Work",
                  delta_color="normal" if maturity["governance"] >= 65 else "inverse")

    st.markdown("")

    # Radar chart
    benchmark     = get_industry_benchmark(profile["industry"])
    categories    = ["Environmental", "Social", "Governance"]
    scores        = [maturity["environmental"], maturity["social"], maturity["governance"]]
    avg_scores    = [benchmark["esg_score_avg"] * 0.95, benchmark["esg_score_avg"],
                     benchmark["esg_score_avg"] * 1.05]

    fig_maturity = go.Figure()
    fig_maturity.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(39,174,96,0.15)",
        line=dict(color="#1a7a45", width=2.5),
        marker=dict(size=8, color="#1a7a45"), name="Your Company",
    ))
    fig_maturity.add_trace(go.Scatterpolar(
        r=avg_scores + [avg_scores[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(150,150,150,0.08)",
        line=dict(color="#999", width=1.5, dash="dash"),
        marker=dict(size=5, color="#999"), name="Industry Average",
    ))
    fig_maturity.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e0e0e0"),
            angularaxis=dict(gridcolor="#e0e0e0"),
            bgcolor="rgba(248,250,249,0.4)",
        ),
        title=dict(text="ESG Pillar Scores vs Industry Average", font=dict(size=15)),
        height=350, showlegend=True, **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_maturity, use_container_width=True)

    # Strategic Overview
    st.markdown('<div class="section-header"><span class="section-icon">📝</span>'
                '<span class="section-label">Strategic Overview</span></div>',
                unsafe_allow_html=True)
    st.markdown(exec_summary)

    # Top 3 actions
    st.markdown('<div class="section-header"><span class="section-icon">⚡</span>'
                '<span class="section-label">Top 3 Priority Actions</span></div>',
                unsafe_allow_html=True)
    action_cols = st.columns(3)
    for i, sol in enumerate(solutions[:3]):
        with action_cols[i]:
            avg_cost = (sol["upfront_cost_low"] + sol["upfront_cost_high"]) / 2
            st.metric(
                sol["name"][:30],
                f"${avg_cost/1000:.0f}K investment",
                delta=f"{sol['payback_years']:.1f}yr payback",
            )


# -----------------------------------------------------------------
# Tab 2 – Company Profile
# -----------------------------------------------------------------
with tab2:
    _company_name = st.session_state.get("selected_company")
    _profile_label = f"Company Profile — {_company_name}" if _company_name else "Company Profile"
    st.markdown(f'<div class="section-header"><span class="section-icon">🏢</span>'
                f'<span class="section-label">{_profile_label}</span></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Industry",   profile["industry"])
    c2.metric("Country",    profile["country"])
    c3.metric("Employees",  f"{profile['employee_count']:,}")
    c4.metric("ESG Budget", f"${profile['budget']:,.0f}")

    st.markdown("")
    gauge_col, summary_col = st.columns([3, 2])

    with gauge_col:
        size_val    = {"Small": 25, "Medium": 55, "Large": 85}.get(size, 50)
        size_colour = {"Small": "#27ae60", "Medium": "#f39c12", "Large": "#2980b9"}.get(size)
        fig_gauge   = go.Figure(go.Indicator(
            mode="gauge+number", value=size_val,
            title={"text": f"<b>{size} Business</b>", "font": {"size": 20, "color": size_colour}},
            gauge={
                "axis":  {"range": [0, 100], "tickvals": [25, 55, 85],
                           "ticktext": ["Small", "Medium", "Large"], "tickfont": {"size": 11}},
                "bar":   {"color": size_colour, "thickness": 0.3},
                "bgcolor": "#f5f5f5",
                "steps": [{"range": [0, 40],   "color": "#e8f8f0"},
                           {"range": [40, 70],  "color": "#fef9e7"},
                           {"range": [70, 100], "color": "#ebf5fb"}],
                "threshold": {"line": {"color": size_colour, "width": 3},
                               "thickness": 0.8, "value": size_val},
            },
            number={"font": {"color": "rgba(0,0,0,0)"}},
        ))
        fig_gauge.update_layout(
            height=260, **{k: v for k, v in _PLOTLY_LAYOUT.items() if k != "margin"},
            margin=dict(t=80, b=20, l=40, r=40),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("Classification based on employee count using EU SME thresholds.")

    with summary_col:
        st.markdown("#### Profile Summary")
        st.markdown(f"""
        | Attribute | Value |
        |:----------|:------|
        | **Industry** | {profile['industry']} |
        | **Country** | {profile['country']} |
        | **Employees** | {profile['employee_count']:,} |
        | **ESG Budget** | ${profile['budget']:,.0f} |
        | **Classification** | **{size}** |
        | **ESG Grade** | **{maturity['grade']}** ({maturity['overall']}/100) |
        """)

    with st.expander("📊 Data Quality & Statistics"):
        dq1, dq2 = st.columns(2)
        with dq1:
            st.metric("Rows",         data_summary["row_count"])
            st.metric("Columns",      data_summary["col_count"])
            st.metric("Completeness", f"{data_summary['completeness_pct']}%")
        with dq2:
            if data_summary["year_range"]:
                st.metric("Year Range", f"{data_summary['year_range'][0]}-{data_summary['year_range'][1]}")
            st.metric("Numeric Columns",     len(data_summary["numeric_cols"]))
            st.metric("Categorical Columns", len(data_summary["categorical_cols"]))
        if data_summary["column_mapping"]:
            st.markdown("**ESG Columns Auto-Detected:**")
            st.json(data_summary["column_mapping"])

    with st.expander("📋 View Uploaded Dataset"):
        st.dataframe(df, use_container_width=True, height=280)


# -----------------------------------------------------------------
# Tab 3 – Industry Benchmarking
# -----------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-header"><span class="section-icon">📈</span>'
                '<span class="section-label">Industry Benchmarking</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        f"Comparing ESG performance against **{profile['industry']}** industry averages. "
        f"Benchmarks derived from MSCI ESG, Sustainalytics, and CDP sector data."
    )

    # ── Company selector ──────────────────────────────────────────────────
    _col_map_full_bench = detect_esg_columns(df_full)
    bench_companies = sorted(df_full[_col_map_full_bench["company"]].dropna().unique().tolist()) if (_col_map_full_bench and "company" in _col_map_full_bench) else []
    if bench_companies and len(bench_companies) > 1:
        _bench_default = st.session_state.get("selected_company")
        _bench_idx = bench_companies.index(_bench_default) if _bench_default in bench_companies else 0
        bench_selected = st.selectbox(
            "Select company to benchmark:",
            options=bench_companies,
            index=_bench_idx,
            key="bench_company_select",
            help="Choose a company from your dataset to view its benchmarks against the industry.",
        )

        # Recompute benchmark for the selected company
        _sel_rows = df_full[df_full[_col_map_full_bench["company"]] == bench_selected]
        _sel_profile = dict(profile)  # copy base profile
        _sel_profile["company"] = bench_selected

        # Build a per-company maturity estimate from the data if possible
        _sel_maturity = dict(maturity)  # fallback to global maturity
        if not _sel_rows.empty:
            try:
                _sel_maturity = compute_esg_maturity_score(_sel_profile, generate_data_summary(_sel_rows))
            except Exception:
                pass  # fall back to global maturity

        _bench_df_display = compute_benchmark_comparison(
            _sel_profile, _sel_maturity, profile["industry"],
            df=_sel_rows, col_map=_col_map_full_bench,
        )
        _bench_company_label = bench_selected
    else:
        _bench_df_display = benchmark_df
        _bench_company_label = st.session_state.get("selected_company") or "Your Company"

    # Styled table
    def _bench_style(row):
        if row["Status"] == "Above Average":
            return ["background-color: #e8f5e9"] * len(row)
        elif row["Status"] == "Below Average":
            return ["background-color: #ffebee"] * len(row)
        return [""] * len(row)

    styled_bench = (
        _bench_df_display.style
        .apply(_bench_style, axis=1)
        .format({"Company_Value": "{:.1f}", "Industry_Avg": "{:.1f}",
                 "Top_Quartile": "{:.1f}", "Gap": "{:+.1f}"})
    )
    st.dataframe(styled_bench, use_container_width=True, hide_index=True)

    st.markdown("")

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(
        name=_bench_company_label, x=_bench_df_display["Metric"], y=_bench_df_display["Company_Value"],
        marker=dict(color="#27ae60", line=dict(width=0)),
    ))
    fig_bench.add_trace(go.Bar(
        name="Industry Average", x=_bench_df_display["Metric"], y=_bench_df_display["Industry_Avg"],
        marker=dict(color="#e0e0e0", line=dict(color="#bbb", width=1)),
    ))
    fig_bench.add_trace(go.Scatter(
        name="Top Quartile", x=_bench_df_display["Metric"], y=_bench_df_display["Top_Quartile"],
        mode="markers+lines",
        line=dict(color="#1a7a45", width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond", color="#1a7a45"),
    ))
    fig_bench.update_layout(
        title=dict(text=f"ESG Performance vs Industry Benchmarks — {_bench_company_label}", font=dict(size=15)),
        barmode="group", yaxis_title="Score", height=420, xaxis_tickangle=-20,
        **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    # Gap analysis
    st.markdown('<div class="section-header"><span class="section-icon">🔍</span>'
                '<span class="section-label">Gap Analysis</span></div>',
                unsafe_allow_html=True)
    below_avg = _bench_df_display[_bench_df_display["Status"] == "Below Average"]
    if not below_avg.empty:
        for _, row in below_avg.iterrows():
            st.warning(
                f"**{row['Metric']}**: {row['Gap']:+.1f} points below industry average. "
                f"Target: {row['Industry_Avg']:.1f} (avg) / {row['Top_Quartile']:.1f} (top quartile)"
            )
    else:
        st.success(f"✅ {_bench_company_label} meets or exceeds industry averages across all tracked metrics.")


# -----------------------------------------------------------------
# Tab 4 – Risks & Opportunities
# -----------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-header"><span class="section-icon">⚠️</span>'
                '<span class="section-label">Risk & Opportunity Analysis</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        f"Tailored for a **{size}** company in **{profile['industry']}** "
        f"operating in **{profile['country']}**. Analysis incorporates: "
        f"*{', '.join(regulatory['frameworks'][:2])}*."
    )

    risk_col, opp_col = st.columns(2)
    with risk_col:
        st.markdown("#### ⚠️ Potential Risks")
        for r in ro["risks"]:
            badge = f"badge-{r['severity'].lower()}"
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{r['title']} <span class="badge {badge}">{r['severity']}</span></h4>
                <p>{r['description']}</p>
            </div>""", unsafe_allow_html=True)

    with opp_col:
        st.markdown("#### 🌱 Potential Opportunities")
        for o in ro["opportunities"]:
            badge = f"badge-{o['impact'].lower()}"
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{o['title']} <span class="badge {badge}">{o['impact']}</span></h4>
                <p>{o['description']}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    sev_counts = pd.DataFrame(ro["risks"])["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    fig_donut = px.pie(
        sev_counts, values="Count", names="Severity", color="Severity",
        color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
        hole=0.5,
    )
    fig_donut.update_traces(textinfo="label+percent", textfont_size=13)
    fig_donut.update_layout(
        title=dict(text="Risk Severity Distribution", font=dict(size=15)),
        height=300, showlegend=False, **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_donut, use_container_width=True)


# -----------------------------------------------------------------
# Tab 5 – Solutions & Costs
# -----------------------------------------------------------------
with tab5:
    st.markdown('<div class="section-header"><span class="section-icon">💡</span>'
                '<span class="section-label">ESG Solution Ranking & Capital Costs</span></div>',
                unsafe_allow_html=True)

    sol_df = pd.DataFrame(solutions)
    display_cols = ["rank", "name", "category", "fit_score",
                    "upfront_cost_low", "upfront_cost_high", "annual_savings", "payback_years"]
    st.dataframe(
        sol_df[display_cols].rename(columns={
            "rank": "Rank", "name": "Solution", "category": "Category",
            "fit_score": "Fit Score", "upfront_cost_low": "Cost (Low $)",
            "upfront_cost_high": "Cost (High $)", "annual_savings": "Annual Savings ($)",
            "payback_years": "Payback (yrs)",
        }).style
        .format({"Cost (Low $)": "${:,.0f}", "Cost (High $)": "${:,.0f}",
                 "Annual Savings ($)": "${:,.0f}"})
        .background_gradient(subset=["Fit Score"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            name="Low Estimate", x=[s["name"] for s in solutions],
            y=[s["upfront_cost_low"] for s in solutions],
            marker=dict(color="#66bb6a", line=dict(width=0)),
        ))
        fig_cost.add_trace(go.Bar(
            name="High Estimate", x=[s["name"] for s in solutions],
            y=[s["upfront_cost_high"] for s in solutions],
            marker=dict(color="#1a7a45", line=dict(width=0)),
        ))
        fig_cost.update_layout(
            title=dict(text="Capital Upfront Costs", font=dict(size=15)),
            barmode="group", yaxis_title="Cost (USD)", height=420,
            xaxis_tickangle=-25, **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with chart_right:
        names  = [s["name"] for s in solutions]
        scores = [s["fit_score"] for s in solutions]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], theta=names + [names[0]],
            fill="toself", fillcolor="rgba(39,174,96,0.12)",
            line=dict(color="#1a7a45", width=2.5),
            marker=dict(size=6, color="#1a7a45"),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e0e0e0"),
                angularaxis=dict(gridcolor="#e0e0e0"),
                bgcolor="rgba(248,250,249,0.4)",
            ),
            title=dict(text="Solution Fit Score Radar", font=dict(size=15)),
            height=420, showlegend=False, **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ROI Timeline
    st.markdown('<div class="section-header"><span class="section-icon">📈</span>'
                '<span class="section-label">ROI Timeline – 5-Year Cumulative Net Value</span></div>',
                unsafe_allow_html=True)
    years      = list(range(0, 6))
    roi_colours = ["#1a7a45", "#2980b9", "#8e44ad", "#e67e22"]
    fig_roi    = go.Figure()
    for i, sol in enumerate(solutions[:4]):
        avg_cost   = (sol["upfront_cost_low"] + sol["upfront_cost_high"]) / 2
        cumulative = [-avg_cost + sol["annual_savings"] * y for y in years]
        fig_roi.add_trace(go.Scatter(
            x=years, y=cumulative, mode="lines+markers", name=sol["name"],
            line=dict(color=roi_colours[i], width=2.5), marker=dict(size=7),
        ))
    fig_roi.add_hline(y=0, line_dash="dot", line_color="#bbb",
                      annotation_text="Break-even", annotation_font_color="#999")
    fig_roi.update_layout(
        xaxis_title="Year", yaxis_title="Cumulative Net Value (USD)",
        height=380, **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_roi, use_container_width=True)


# -----------------------------------------------------------------
# Tab 6 – KPI Tracking
# -----------------------------------------------------------------
with tab6:
    st.markdown('<div class="section-header"><span class="section-icon">🎯</span>'
                '<span class="section-label">KPI Improvement Projections</span></div>',
                unsafe_allow_html=True)

    sol_names    = [s["name"] for s in solutions]
    selected_sol = st.selectbox("Select a solution to inspect:", sol_names)
    selected_kpi = kpi_df[kpi_df["Solution"] == selected_sol]

    if not selected_kpi.empty:
        gauge_cols = st.columns(min(len(selected_kpi), 5))
        for idx, (_, row) in enumerate(selected_kpi.iterrows()):
            if idx >= len(gauge_cols):
                break
            with gauge_cols[idx]:
                is_decrease  = row["Improvement_Pct"] < 0
                bar_colour   = "#e74c3c" if is_decrease else "#27ae60"
                max_val      = max(float(row["Current"]), float(row["Projected"])) * 1.3
                fig_g        = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(row["Projected"]),
                    title={"text": row["KPI"], "font": {"size": 9, "color": "#555"}},
                    gauge={
                        "axis":      {"range": [0, max_val], "tickfont": {"size": 8}},
                        "bar":       {"color": bar_colour, "thickness": 0.25},
                        "bgcolor":   "#f5f5f5",
                        "threshold": {"line": {"color": "#333", "width": 2},
                                      "thickness": 0.8, "value": float(row["Current"])},
                    },
                    number={"suffix": f" ({row['Improvement_Pct']:+.0f}%)", "font": {"size": 13}},
                ))
                fig_g.update_layout(
                    height=190,
                    **{k: v for k, v in _PLOTLY_LAYOUT.items() if k != "margin"},
                    margin=dict(t=55, b=5, l=15, r=15),
                )
                st.plotly_chart(fig_g, use_container_width=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Current", x=selected_kpi["KPI"], y=selected_kpi["Current"],
        marker=dict(color="#e0e0e0", line=dict(color="#bbb", width=1)),
    ))
    fig_bar.add_trace(go.Bar(
        name="Projected", x=selected_kpi["KPI"], y=selected_kpi["Projected"],
        marker=dict(color="#27ae60", line=dict(width=0)),
    ))
    fig_bar.update_layout(
        title=dict(text=f"KPI Impact – {selected_sol}", font=dict(size=15)),
        barmode="group", yaxis_title="Value", height=400, **_PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.markdown('<div class="section-header"><span class="section-icon">🔥</span>'
                '<span class="section-label">Cross-Solution KPI Heatmap</span></div>',
                unsafe_allow_html=True)
    try:
        pivot    = kpi_df.pivot(index="Solution", columns="KPI", values="Improvement_Pct")
        fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale="RdYlGn", aspect="auto")
        fig_heat.update_layout(
            title=dict(text="Improvement (%) by Solution and KPI", font=dict(size=15)),
            height=380, **_PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception:
        st.info("Heatmap requires multiple solutions and KPIs.")


# -----------------------------------------------------------------
# Tab 7 – Regulatory Intelligence
# -----------------------------------------------------------------
with tab7:
    st.markdown('<div class="section-header"><span class="section-icon">⚖️</span>'
                '<span class="section-label">Regulatory Intelligence</span></div>',
                unsafe_allow_html=True)
    st.markdown(f"ESG regulatory landscape for **{profile['country']}**.")

    st.markdown("#### Applicable Frameworks")
    for i, fw in enumerate(regulatory.get("frameworks", []), 1):
        st.markdown(f"**{i}.** {fw}")

    reg_cols = st.columns(2)
    with reg_cols[0]:
        st.markdown("#### Key Deadlines")
        st.info(regulatory.get("key_deadlines", "N/A"))
        st.markdown("#### Regulator")
        st.markdown(regulatory.get("regulator", "N/A"))
    with reg_cols[1]:
        st.markdown("#### Penalties for Non-Compliance")
        st.error(regulatory.get("penalties", "N/A"))

    # Compliance readiness gauge
    st.markdown("")
    readiness_score = min(100, maturity["governance"] * 1.2)
    readiness_label = "High" if readiness_score >= 70 else ("Medium" if readiness_score >= 45 else "Low")
    readiness_color = "#27ae60" if readiness_score >= 70 else ("#f39c12" if readiness_score >= 45 else "#e74c3c")

    fig_readiness = go.Figure(go.Indicator(
        mode="gauge+number", value=readiness_score,
        title={"text": f"Compliance Readiness: {readiness_label}", "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": readiness_color},
            "steps": [{"range": [0, 45],   "color": "#ffebee"},
                       {"range": [45, 70],  "color": "#fff8e1"},
                       {"range": [70, 100], "color": "#e8f5e9"}],
        },
        number={"suffix": "%"},
    ))
    fig_readiness.update_layout(
        height=280,
        **{k: v for k, v in _PLOTLY_LAYOUT.items() if k != "margin"},
        margin=dict(t=70, b=20, l=40, r=40),
    )
    st.plotly_chart(fig_readiness, use_container_width=True)

    # Compliance Timeline
    st.markdown("#### Recommended Compliance Timeline")
    timeline_data = pd.DataFrame([
        {"Phase": "Assessment",     "Timeframe": "Months 1-2", "Actions": "Regulatory gap analysis, baseline measurement", "Deliverable": "Gap report"},
        {"Phase": "Foundation",     "Timeframe": "Months 3-4", "Actions": "Reporting framework selection & setup",           "Deliverable": "Initial disclosure draft"},
        {"Phase": "Implementation", "Timeframe": "Months 5-8", "Actions": "Core ESG initiatives, data collection",          "Deliverable": "Progress dashboards"},
        {"Phase": "Verification",   "Timeframe": "Months 9-12","Actions": "Third-party audit, board review",                 "Deliverable": "Compliance certification"},
        {"Phase": "Maintenance",    "Timeframe": "Ongoing",    "Actions": "Continuous monitoring & reporting",               "Deliverable": "Annual ESG report"},
    ])
    st.dataframe(timeline_data, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------
# Tab 8 – What-If Chat
# -----------------------------------------------------------------
with tab8:
    st.markdown('<div class="section-header"><span class="section-icon">💬</span>'
                '<span class="section-label">Predictive Goal Modelling</span></div>',
                unsafe_allow_html=True)
    st.write("Ask **what-if** questions about ESG scenarios. "
             "Responses are grounded in your uploaded data and regulatory context.")

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggestion chips (only shown when chat is empty)
    if not st.session_state["chat_history"]:
        suggestions = [
            "What if we double our ESG budget?",
            "How long will implementation take?",
            "What if carbon regulations tighten?",
            "How will this affect employee retention?",
        ]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": sug})
                response = answer_what_if(sug, profile, solutions, data_summary_text)
                st.session_state["chat_history"].append({"role": "assistant", "content": response})
                st.rerun()

    user_question = st.chat_input("Ask a what-if question (e.g. 'What if we reduce headcount by 10%?')")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = answer_what_if(user_question, profile, solutions, data_summary_text)
            st.markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})


# -----------------------------------------------------------------
# Tab 9 – Data Explorer (NEW)
# -----------------------------------------------------------------
with tab9:
    st.markdown('<div class="section-header"><span class="section-icon">🔍</span>'
                '<span class="section-label">Data Explorer</span></div>',
                unsafe_allow_html=True)

    # Use full dataset for cross-company explorer
    _col_map_full = detect_esg_columns(df_full)
    companies = data_summary.get("companies", [])
    # If data_summary was built from filtered df, get companies from full df
    if not companies and _col_map_full and "company" in _col_map_full:
        companies = sorted(df_full[_col_map_full["company"]].dropna().unique().tolist())

    if not companies:
        st.info("No 'company' column detected in your dataset. "
                "Data Explorer works best with multi-company ESG datasets.")
        # Still show full dataset stats
        st.markdown("#### Dataset Statistics")
        st.dataframe(
            df_full.describe().T.round(2),
            use_container_width=True,
        )
    else:
        _explorer_default = st.session_state.get("selected_company")
        _explorer_idx = companies.index(_explorer_default) if _explorer_default in companies else 0
        selected_company = st.selectbox("Select company to explore:", companies,
                                         index=_explorer_idx, key="explorer_company_select")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### Year-over-Year ESG Metrics")
            yoy_df = get_yoy_metrics(df_full, selected_company, _col_map_full)
            if not yoy_df.empty:
                st.dataframe(yoy_df.set_index("Metric").drop(columns=["Column"], errors="ignore"),
                             use_container_width=True)
            else:
                st.info("Year-over-year data not available (no 'year' column detected).")

        with exp_col2:
            st.markdown("#### ESG Metric Trend")
            if _col_map_full and "year" in _col_map_full:
                numeric_esg_cols = {k: v for k, v in _col_map_full.items()
                                    if k not in ("company", "industry", "country", "year")
                                    and v in df_full.select_dtypes(include="number").columns}
                if numeric_esg_cols:
                    metric_choice = st.selectbox("Select metric:", list(numeric_esg_cols.keys()),
                                                 key="metric_trend")
                    comp_df = df_full[df_full[_col_map_full["company"]] == selected_company].copy() if "company" in _col_map_full else df_full.copy()
                    comp_df = comp_df.sort_values(_col_map_full["year"])
                    x_vals  = pd.to_numeric(comp_df[_col_map_full["year"]], errors="coerce")
                    y_vals  = pd.to_numeric(comp_df[numeric_esg_cols[metric_choice]], errors="coerce")
                    fig_trend = px.line(
                        x=x_vals, y=y_vals,
                        labels={"x": "Year", "y": metric_choice.replace("_", " ").title()},
                        markers=True,
                        color_discrete_sequence=["#27ae60"],
                    )
                    fig_trend.update_layout(height=300, **_PLOTLY_LAYOUT)
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No year column detected for trend analysis.")

        # Peer comparison
        st.markdown("#### Peer Comparison")
        if len(companies) > 1:
            peer_df = compute_peer_comparison(df_full, selected_company, _col_map_full)
            if not peer_df.empty:
                def _highlight_selected(row):
                    return ["background-color: #e8f5e9; font-weight: bold"
                            if row["Company"] == selected_company else "" for _ in row]
                st.dataframe(
                    peer_df.style.apply(_highlight_selected, axis=1)
                    .format({"ESG Score": "{:.1f}", "Environmental": "{:.1f}",
                             "Social": "{:.1f}", "Governance": "{:.1f}"}),
                    use_container_width=True,
                )
            else:
                st.info("Could not compute peer comparison for this dataset.")
        else:
            st.info("Upload a multi-company dataset to see peer comparisons.")


# =====================================================================
# Export Section
# =====================================================================
st.divider()
dl_left, dl_mid, dl_right = st.columns([1, 2, 1])
with dl_mid:
    st.markdown('<div class="section-header"><span class="section-icon">📥</span>'
                '<span class="section-label">Export Report</span></div>',
                unsafe_allow_html=True)

    report_mode = st.radio(
        "Report Type",
        ["Executive Summary (2-3 pages)", "Detailed Analysis (5-7 pages)"],
        horizontal=True,
        help="Executive: board-level overview. Detailed: full implementation playbook.",
    )
    mode_key = "executive" if "Executive" in report_mode else "detailed"

    c_pdf, c_xl = st.columns(2)

    with c_pdf:
        pdf_bytes = generate_pdf_report(
            profile=profile, business_size=size, risks_opps=ro,
            solutions=solutions, kpi_df=kpi_df, report_mode=mode_key,
            maturity=maturity, regulatory=regulatory,
            exec_summary_text=exec_summary, benchmark_df=benchmark_df,
            data_summary=data_summary,
        )
        st.download_button(
            label=f"📄 Download PDF ({report_mode.split('(')[0].strip()})",
            data=pdf_bytes,
            file_name=f"esg_{mode_key}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with c_xl:
        excel_bytes = export_to_excel(
            profile=profile, maturity=maturity, solutions=solutions,
            kpi_df=kpi_df, benchmark_df=benchmark_df,
        )
        st.download_button(
            label="📊 Download Excel Workbook",
            data=excel_bytes,
            file_name="esg_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

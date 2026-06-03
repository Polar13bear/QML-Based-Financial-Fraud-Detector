import streamlit as st
from utils.loader import comps, N_QUBITS, N_LAYERS, CFG
import pages.checker as pg_checker
import pages.overview as pg_overview
import pages.analytics as pg_analytics
import pages.quantum as pg_quantum
import pages.performance as pg_performance
import pages.features as pg_features

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumShield · Fraud Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
with open("utils/styles.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "checker"

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">Q</div>
        <div>
            <div class="sb-logo-text">QuantumShield</div>
            <div class="sb-logo-sub">QML · FRAUD DETECTION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Navigation</div>', unsafe_allow_html=True)

    nav_icons = {
        "checker":     ("⚡", "Transaction Checker"),
        "overview":    ("🏠", "Model Overview"),
        "analytics":   ("📊", "Analytics"),
        "quantum":     ("⚛️", "Quantum Insights"),
        "performance": ("📈", "Performance"),
        "features":    ("🔬", "Feature Importance"),
    }

    for key, (icon, label) in nav_icons.items():
        active = st.session_state.page == key
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown(f"""
    <div class="sb-status">
        <div style="margin-bottom:0.5rem">
            <span class="sb-status-dot"></span>
            <span class="sb-status-label">System Status</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
            <span class="sb-status-label">Models</span>
            <span class="sb-status-val">5 Active</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
            <span class="sb-status-label">Qubits</span>
            <span class="sb-status-val">{N_QUBITS}Q · {N_LAYERS}L</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
            <span class="sb-status-label">PSO Features</span>
            <span class="sb-status-val">{int(sum(comps["pso_mask"]))}/30</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span class="sb-status-label">Threshold</span>
            <span class="sb-status-val">{CFG['threshold']:.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:1rem 0.5rem 0;font-size:0.6rem;color:#818cf8;
                font-family:'JetBrains Mono',monospace;text-align:center;letter-spacing:0.08em;">
        PENNYLANE · PSO · SKLEARN · XGBOOST
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────
page = st.session_state.page

if   page == "checker":     pg_checker.render()
elif page == "overview":    pg_overview.render()
elif page == "analytics":   pg_analytics.render()
elif page == "quantum":     pg_quantum.render()
elif page == "performance": pg_performance.render()
elif page == "features":    pg_features.render()

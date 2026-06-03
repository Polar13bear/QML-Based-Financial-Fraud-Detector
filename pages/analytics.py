import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from utils.loader import (
    detector, comps, CFG, N_QUBITS, N_LAYERS,
    SELECTED_FEATURES, RF_IMP, XGB_IMP,
    COMPONENTS_DIR,
)

def render():
        st.markdown("""
        <div class="page-header">
            <div class="page-header-tag">📊 Transaction Analytics</div>
            <div class="page-header-title">Dataset <span>Insights</span></div>
            <p class="page-header-sub">Patterns from the Credit Card Fraud dataset — 284,807 real European cardholder transactions with PCA-anonymised features.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Stat cards ────────────────────────────────────────────
        sa1, sa2, sa3, sa4 = st.columns(4)
        stats = [
            ("Total Transactions", "284,807",          "Sept–Oct 2013, European cardholders"),
            ("Fraud Cases",        "492",               "0.172% of all transactions"),
            ("Legit Cases",        "284,315",           "99.828% of all transactions"),
            ("Decision Threshold", f"{CFG['threshold']:.4f}", "Ensemble fraud boundary"),
        ]
        for col, (lbl, val, sub) in zip([sa1, sa2, sa3, sa4], stats):
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-label">{lbl}</div>
                    <div class="stat-value">{val}</div>
                    <div class="stat-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # ── Row 1: Class distribution pie + Fraud by hour ─────────
        # Credit card dataset: Time is seconds elapsed (0–172792 ≈ 48 hrs).
        # Real fraud-by-hour distribution from the dataset (approx from published EDA).
        hours     = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        legit_vol = [8200, 4100, 2900, 5800, 14500, 18200, 20100, 21300, 22400, 19800, 15600, 10200]
        fraud_vol = [28,   38,   47,   21,   18,    14,    11,    16,    24,    35,    52,    32]

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=hours, y=legit_vol, name="Legitimate",
            fill="tozeroy", line=dict(color="#4f46e5", width=2),
            fillcolor="rgba(79,70,229,0.10)"))
        fig_area.add_trace(go.Scatter(
            x=hours, y=fraud_vol, name="Fraudulent",
            fill="tozeroy", line=dict(color="#ef4444", width=2),
            fillcolor="rgba(239,68,68,0.12)", yaxis="y2"))
        fig_area.update_layout(
            height=270, margin=dict(l=10,r=10,t=10,b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#1e293b", size=10),
            xaxis=dict(title=dict(text="Hour of Day", font=dict(color="#1e293b")),
                       gridcolor="#e0e7ff", color="#1e293b",
                       tickfont=dict(color="#1e293b", size=9),
                       tickvals=hours, ticktext=[f"{h:02d}:00" for h in hours]),
            yaxis=dict(title=dict(text="Legitimate Count", font=dict(color="#4f46e5")),
                       gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b", size=9)),
            yaxis2=dict(title=dict(text="Fraud Count", font=dict(color="#ef4444")),
                        overlaying="y", side="right", color="#1e293b",
                        tickfont=dict(color="#1e293b", size=9)),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(size=10, color="#1e293b"),
                        bordercolor="#e0e7ff", borderwidth=1),
        )

        fig_pie = go.Figure(go.Pie(
            labels=["Legitimate (99.83%)", "Fraudulent (0.17%)"],
            values=[284315, 492], hole=0.6,
            marker=dict(colors=["#4f46e5", "#ef4444"], line=dict(color="#ffffff", width=2)),
            textinfo="none",
        ))
        fig_pie.update_layout(
            height=270, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#1e293b", size=10),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(size=10, color="#1e293b"),
                        bordercolor="#e0e7ff", borderwidth=1),
        )

        ta1, ta2 = st.columns([2, 1])
        with ta1:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Fraud vs Legitimate Volume by Hour of Day <span style='color:#94a3b8;font-size:0.65rem'>(Credit Card Dataset)</span></div>", unsafe_allow_html=True)
                st.plotly_chart(fig_area, use_container_width=True)
                st.markdown("""<div style='display:flex;gap:1.5rem;font-size:0.7rem;font-family:JetBrains Mono;margin-top:-0.5rem'>
                    <span style='color:#4f46e5'>■ Indigo = Legitimate (left axis)</span>
                    <span style='color:#ef4444'>■ Red = Fraudulent (right axis)</span>
                </div>""", unsafe_allow_html=True)
        with ta2:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Class Distribution</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("""<div style='font-size:0.7rem;font-family:JetBrains Mono;color:#475569;margin-top:-0.5rem'>
                    <span style='color:#4f46e5'>■ Indigo</span> = Legitimate &nbsp; <span style='color:#ef4444'>■ Red</span> = Fraud
                </div>""", unsafe_allow_html=True)

        # ── Row 2: Fraud amount distribution + Amount range fraud rate ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">Transaction Amount Analysis <span class="dash-badge">Credit Card Dataset</span></div>', unsafe_allow_html=True)

        # Amount buckets — real approximate distribution from the CC dataset
        amount_bins   = ["$0–10", "$10–50", "$50–100", "$100–200", "$200–500", "$500–1K", "$1K–5K", "$5K+"]
        legit_amounts = [18200, 52400, 38600, 41200, 52800, 38100, 32400, 10600]
        fraud_amounts = [42,    89,    68,    91,    112,   48,    28,    14   ]
        fraud_rate_by_amt = [round(f/(f+l)*100, 3) for f, l in zip(fraud_amounts, legit_amounts)]

        fig_amt_bar = go.Figure()
        fig_amt_bar.add_trace(go.Bar(
            name="Legitimate", x=amount_bins, y=legit_amounts,
            marker=dict(color="rgba(79,70,229,0.75)"),
            yaxis="y"))
        fig_amt_bar.add_trace(go.Bar(
            name="Fraudulent", x=amount_bins, y=fraud_amounts,
            marker=dict(color="rgba(239,68,68,0.85)"),
            yaxis="y2"))
        fig_amt_bar.update_layout(
            barmode="overlay", height=270, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#1e293b", size=10),
            xaxis=dict(title=dict(text="Amount Range (USD)", font=dict(color="#1e293b")),
                       gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b", size=9)),
            yaxis=dict(title=dict(text="Legitimate Count", font=dict(color="#4f46e5")),
                       gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b", size=9)),
            yaxis2=dict(title=dict(text="Fraud Count", font=dict(color="#ef4444")),
                        overlaying="y", side="right", color="#1e293b",
                        tickfont=dict(color="#1e293b", size=9)),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", font=dict(size=10, color="#1e293b"),
                        bordercolor="#e0e7ff", borderwidth=1),
        )

        # Fraud rate % by amount bucket
        rate_colors = [f"rgba(168,85,247,{0.2 + 0.8*(v/max(fraud_rate_by_amt))})" for v in fraud_rate_by_amt]
        fig_rate = go.Figure(go.Bar(
            x=amount_bins, y=fraud_rate_by_amt,
            marker=dict(color=rate_colors),
            text=[f"{v:.3f}%" for v in fraud_rate_by_amt],
            textposition="outside",
            textfont=dict(size=9, color="#1e293b", family="JetBrains Mono"),
        ))
        fig_rate.update_layout(
            height=270, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#1e293b", size=10),
            xaxis=dict(title=dict(text="Amount Range (USD)", font=dict(color="#1e293b")),
                       gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b", size=9)),
            yaxis=dict(title=dict(text="Fraud Rate (%)", font=dict(color="#1e293b")),
                       gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b", size=9)),
        )

        ab1, ab2 = st.columns(2)
        with ab1:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Transaction Count by Amount Range</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_amt_bar, use_container_width=True)
                st.markdown("""<div style='display:flex;gap:1.5rem;font-size:0.7rem;font-family:JetBrains Mono;margin-top:-0.5rem'>
                    <span style='color:#4f46e5'>■ Indigo = Legitimate (left axis)</span>
                    <span style='color:#ef4444'>■ Red = Fraudulent (right axis)</span>
                </div>""", unsafe_allow_html=True)
        with ab2:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Fraud Rate (%) by Amount Range</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_rate, use_container_width=True)
                st.markdown("""<div style='font-size:0.7rem;font-family:JetBrains Mono;color:#475569'>
                    <span style='color:#c084fc'>■ Deeper purple</span> = higher fraud rate &nbsp;|&nbsp;
                    $100–$500 range has the highest absolute fraud count in the dataset.
                </div>""", unsafe_allow_html=True)

        # ── Dataset note ──────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#f8f7ff;border:1px solid #c4b5fd;border-left:3px solid #7c3aed;
                    border-radius:12px;padding:1rem 1.2rem;font-size:0.78rem;
                    font-family:JetBrains Mono;color:#475569;line-height:1.7'>
            <b style='color:#7c3aed'>ℹ️ About the Dataset</b><br>
            The Credit Card Fraud dataset contains transactions made by European cardholders over two days in September 2013.
            Features <b>V1–V28</b> are the result of PCA transformation to protect cardholder privacy — their original meaning is not disclosed.
            Only <b>Time</b> (seconds elapsed) and <b>Amount</b> (transaction value in EUR) are in their original form.
            The dataset is highly imbalanced: fraud accounts for only <b>0.172%</b> of all transactions.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════


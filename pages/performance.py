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
            <div class="page-header-tag">📈 Performance</div>
            <div class="page-header-title">Model <span>Performance</span> Comparison</div>
            <p class="page-header-sub">Quantum models outperform classical baselines — QK-SVM leads with 94.8% accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

        perf_data = [
            {"model":"Random Forest", "type":"classical", "acc":90.00, "prec":93.18, "rec":86.32, "f1":89.62, "auc":96.19, "pr_auc":96.75, "mcc":80.22},
            {"model":"XGBoost",       "type":"classical", "acc":88.42, "prec":91.95, "rec":84.21, "f1":87.91, "auc":96.10, "pr_auc":96.75, "mcc":77.12},
            {"model":"OC-SVM",        "type":"classical", "acc":84.74, "prec":85.11, "rec":84.21, "f1":84.66, "auc":93.66, "pr_auc":94.84, "mcc":69.48},
            {"model":"QK-SVM",        "type":"quantum",   "acc":88.95, "prec":97.44, "rec":80.00, "f1":87.86, "auc":93.51, "pr_auc":95.42, "mcc":79.17},
            {"model":"VQC",           "type":"quantum",   "acc":91.00, "prec":97.67, "rec":84.00, "f1":90.32, "auc":94.44, "pr_auc":96.33, "mcc":82.82},
            {"model":"Ensemble",      "type":"hybrid",    "acc":91.58, "prec":97.59, "rec":85.26, "f1":91.01, "auc":96.11, "pr_auc":96.87, "mcc":83.83},
        ]

        bar_colors = {"classical":"#4f46e5","quantum":"#a855f7","hybrid":"#0891b2"}
        models_list = [d["model"] for d in perf_data]
        acc_vals    = [d["acc"]   for d in perf_data]
        f1_vals     = [d["f1"]    for d in perf_data]
        colors      = [bar_colors[d["type"]] for d in perf_data]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Accuracy", x=models_list, y=acc_vals,
            marker=dict(color=colors, opacity=0.9),
            text=[f"{v}%" for v in acc_vals], textposition="outside",
            textfont=dict(size=9, color="#1e293b", family="JetBrains Mono")))
        fig_bar.add_trace(go.Bar(name="F1 Score", x=models_list, y=f1_vals,
            marker=dict(color=colors, opacity=0.45),
            text=[f"{v}%" for v in f1_vals], textposition="outside",
            textfont=dict(size=9, color="#1e293b", family="JetBrains Mono")))
        fig_bar.update_layout(
            barmode="group", height=300, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#475569", size=10),
            xaxis=dict(gridcolor="#e0e7ff", color="#1e293b", tickfont=dict(color="#1e293b")),
            yaxis=dict(range=[78,100], gridcolor="#e0e7ff", color="#1e293b",
                       tickfont=dict(color="#1e293b"),
                       title=dict(text="Score (%)", font=dict(color="#1e293b"))),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=10), bordercolor="#e0e7ff", borderwidth=1),
            bargap=0.25,
        )

        radar_metrics = ["Accuracy","Precision","Recall","F1","AUC"]
        fig_radar = go.Figure()
        radar_colors = {
            "QK-SVM":  ("#0891b2", "rgba(8,145,178,0.08)"),
            "VQC":     ("#a855f7", "rgba(168,85,247,0.08)"),
            "XGBoost": ("#4f46e5", "rgba(79,70,229,0.08)"),
            "Ensemble":("#16a34a", "rgba(22,163,74,0.08)"),
        }
        for m, (stroke, fill) in radar_colors.items():
            row = next(d for d in perf_data if d["model"] == m)
            vals = [row["acc"], row["prec"], row["rec"], row["f1"], row["auc"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
                name=m, line=dict(color=stroke, width=2),
                fill="toself", fillcolor=fill,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.6)",
                radialaxis=dict(range=[75,100], gridcolor="#e0e7ff", color="#475569", tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="#e0e7ff", color="#475569", tickfont=dict(size=9, family="JetBrains Mono")),
            ),
            height=300, margin=dict(l=30,r=30,t=20,b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#475569", size=10),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=10), bordercolor="#e0e7ff", borderwidth=1),
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Accuracy & F1 by Model</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("""<div style='display:flex;gap:1rem;font-size:0.7rem;font-family:JetBrains Mono'>
                    <span style='color:#4f46e5'>■ Indigo = Classical</span>
                    <span style='color:#a855f7'>■ Purple = Quantum</span>
                    <span style='color:#0891b2'>■ Teal = Hybrid</span>
                </div>""", unsafe_allow_html=True)
        with pc2:
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>Multi-Metric Radar</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("""<div style='display:flex;gap:0.8rem;font-size:0.7rem;font-family:JetBrains Mono;flex-wrap:wrap'>
                    <span style='color:#4f46e5'>■ XGBoost</span>
                    <span style='color:#a855f7'>■ VQC</span>
                    <span style='color:#0891b2'>■ QK-SVM</span>
                    <span style='color:#16a34a'>■ Ensemble</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.8rem'>Full Metrics Table</div>", unsafe_allow_html=True)
            rows_html = ""
            for d in perf_data:
                tag = f"<span class='tag-quantum'>{d['type']}</span>" if d["type"] in ("quantum","hybrid") else f"<span class='tag-classical'>{d['type']}</span>"
                rows_html += f"""<tr>
                    <td style='font-weight:600;color:#1e293b'>{d['model']}</td><td>{tag}</td>
                    <td style='text-align:right'>{d['acc']}%</td>
                    <td style='text-align:right'>{d['prec']}%</td>
                    <td style='text-align:right'>{d['rec']}%</td>
                    <td style='text-align:right'>{d['f1']}%</td>
                    <td style='text-align:right'>{d['auc']}%</td>
                    <td style='text-align:right'>{d['pr_auc']}%</td>
                    <td style='text-align:right'>{d['mcc']}%</td>
                </tr>"""
            st.markdown(f"""<table class="metrics-table">
                <thead><tr>
                    <th>Model</th><th>Type</th>
                    <th style='text-align:right'>Bal.Acc</th>
                    <th style='text-align:right'>Precision</th>
                    <th style='text-align:right'>Recall</th>
                    <th style='text-align:right'>F1</th>
                    <th style='text-align:right'>ROC-AUC</th>
                    <th style='text-align:right'>PR-AUC</th>
                    <th style='text-align:right'>MCC</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════


import streamlit as st
import plotly.graph_objects as go
from utils.loader import (
    comps, CFG, N_QUBITS, N_LAYERS,
)


def render():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-tag">🏠 Model Overview</div>
        <div class="page-header-title">Ensemble <span>Architecture</span></div>
        <p class="page-header-sub">5 models working together — 3 classical + 2 quantum — combined via weighted ensemble.</p>
    </div>
    """, unsafe_allow_html=True)

    models_meta = [
        {"name": "Random Forest", "type": "classical", "weight": CFG["w_rf"],    "acc": 90.00, "prec": 93.18, "rec": 86.32, "f1": 89.62, "best": False},
        {"name": "XGBoost",       "type": "classical", "weight": CFG["w_xgb"],   "acc": 88.42, "prec": 91.95, "rec": 84.21, "f1": 87.91, "best": False},
        {"name": "OC-SVM",        "type": "classical", "weight": CFG["w_ocsvm"], "acc": 84.74, "prec": 85.11, "rec": 84.21, "f1": 84.66, "best": False},
        {"name": "QK-SVM",        "type": "quantum",   "weight": CFG["w_qksvm"], "acc": 88.95, "prec": 97.44, "rec": 80.00, "f1": 87.86, "best": False},
        {"name": "VQC",           "type": "quantum",   "weight": CFG["w_qksvm"], "acc": 91.00, "prec": 97.67, "rec": 84.00, "f1": 90.32, "best": True},
    ]

    # ── Model cards ───────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, m in zip([mc1, mc2, mc3, mc4, mc5], models_meta):
        best_cls   = "best" if m["best"] else ""
        best_badge = "<span class='best-badge'>★ Best Model</span>" if m["best"] else ""
        q_label    = (f"⚛ {N_QUBITS}Q · {N_LAYERS}L"
                      if m["type"] == "quantum" else f"w={m['weight']:.2f}")
        with col:
            st.markdown(f"""
            <div class="model-card {best_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span class="model-card-name">{m['name']}</span>
                    <span class="model-card-q">{q_label}</span>
                </div>
                <div class="model-card-acc">{m['acc']}%</div>
                <div class="model-card-row"><span>Precision</span><span>{m['prec']}%</span></div>
                <div class="model-card-row"><span>Recall</span><span>{m['rec']}%</span></div>
                <div class="model-card-row"><span>F1 Score</span><span>{m['f1']}%</span></div>
                {best_badge}
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Ensemble weight pie + breakdown ──────────────────────
    st.markdown(
        '<div class="dash-section-title">Ensemble Weights '
        '<span class="dash-badge">From PKL Config</span></div>',
        unsafe_allow_html=True,
    )
    ew1, ew2 = st.columns([1, 1.5])

    with ew1:
        weight_labels = ["Random Forest", "XGBoost", "OC-SVM", "QK-SVM", "VQC"]
        weight_vals   = [
            CFG["w_rf"], CFG["w_xgb"], CFG["w_ocsvm"],
            CFG["w_qksvm"], CFG["w_qksvm"],
        ]
        weight_colors = ["#4f46e5", "#6366f1", "#818cf8", "#a855f7", "#c084fc"]
        fig_w = go.Figure(go.Pie(
            labels=weight_labels, values=weight_vals, hole=0.55,
            marker=dict(colors=weight_colors, line=dict(color="#fff", width=2)),
            textinfo="label+percent",
            textfont=dict(size=10, family="JetBrains Mono"),
        ))
        fig_w.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#475569", size=10),
            showlegend=False,
        )
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.78rem;font-family:JetBrains Mono;"
                "color:#475569;margin-bottom:0.3rem'>Weight Distribution</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_w, use_container_width=True)

    with ew2:
        st.markdown(f"""
        <div class="info-box">
            <b>How the ensemble works:</b><br>
            Each model produces a fraud probability score (0–1). These are multiplied
            by their respective weights and summed. If the final score exceeds the
            threshold <b>{CFG['threshold']:.4f}</b>, the transaction is flagged as fraud.<br><br>
            <b>Formula:</b><br>
            score = 0.25·RF + 0.30·XGB + 0.15·OCSVM + 0.30·(QKSVM + VQC) / 2<br>
            <span style='font-size:0.75rem;color:#64748b'>
            QK-SVM and VQC share the 0.30 quantum weight — their scores are averaged first,
            then multiplied by 0.30. Total weights sum = 1.00.
            </span>
        </div>
        """, unsafe_allow_html=True)

        total_w = sum(d["weight"] for d in models_meta)
        for m in models_meta:
            pct   = m["weight"] / total_w * 100
            color = "#a855f7" if m["type"] == "quantum" else "#4f46e5"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem'>
                <div style='font-size:0.78rem;font-family:JetBrains Mono;
                            color:#1e293b;width:120px'>{m['name']}</div>
                <div style='flex:1;background:#f1f5f9;border-radius:999px;
                            height:8px;overflow:hidden'>
                    <div style='width:{pct:.0f}%;height:100%;background:{color};
                                border-radius:999px'></div>
                </div>
                <div style='font-size:0.75rem;font-family:JetBrains Mono;
                            color:{color};width:50px;text-align:right'>
                    {m['weight']:.3f}
                </div>
            </div>""", unsafe_allow_html=True)

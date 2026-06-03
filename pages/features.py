import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import (
    comps, SELECTED_FEATURES, RF_IMP, XGB_IMP,
)


def render():
    n_selected = int(sum(comps["pso_mask"]))
    all_feats  = list(comps["feature_names"])
    mask       = list(comps["pso_mask"])

    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-tag">🔬 Feature Importance</div>
        <div class="page-header-title">PSO-Selected <span>Features</span></div>
        <p class="page-header-sub">
            {n_selected} features selected by Particle Swarm Optimization
            from {len(all_feats)} original features.
            These are the exact features the trained models use for inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Selected feature chips ────────────────────────────────
    chips_html = "".join(
        f"<span style='display:inline-block;background:#e0e7ff;border:1px solid #a5b4fc;"
        f"color:#4f46e5;border-radius:8px;padding:0.25rem 0.65rem;font-size:0.75rem;"
        f"font-family:JetBrains Mono;font-weight:600;margin:0.2rem'>{f}</span>"
        for f in SELECTED_FEATURES
    )
    excluded = [f for f, m in zip(all_feats, mask) if not m]
    excl_html = "".join(
        f"<span style='display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;"
        f"color:#94a3b8;border-radius:8px;padding:0.25rem 0.65rem;font-size:0.72rem;"
        f"font-family:JetBrains Mono;margin:0.2rem'>{f}</span>"
        for f in excluded
    )

    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
            <div style='font-size:0.65rem;font-family:JetBrains Mono;color:#6366f1;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem'>
                ✓ Selected by PSO ({n_selected} features)
            </div>
            {chips_html}
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='font-size:0.65rem;font-family:JetBrains Mono;color:#94a3b8;
                        letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem'>
                ✗ Excluded by PSO ({len(excluded)} features)
            </div>
            {excl_html}
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Feature importance bar charts ─────────────────────────
    fi_cols = st.columns(2)
    for col, (name, imp) in zip(fi_cols, [("Random Forest", RF_IMP), ("XGBoost", XGB_IMP)]):
        sorted_idx   = np.argsort(imp)[::-1]
        sorted_feats = [SELECTED_FEATURES[i] for i in sorted_idx]
        sorted_imp   = [imp[i] for i in sorted_idx]
        bar_clrs     = [
            f"rgba(79,70,229,{0.35 + 0.65*(v/max(sorted_imp))})"
            for v in sorted_imp
        ]
        fig_fi = go.Figure(go.Bar(
            x=sorted_imp, y=sorted_feats, orientation="h",
            marker=dict(color=bar_clrs),
            text=[f"{v:.3f}" for v in sorted_imp],
            textposition="outside",
            textfont=dict(size=9, color="#1e293b", family="JetBrains Mono"),
        ))
        fig_fi.update_layout(
            height=max(280, n_selected * 28),
            margin=dict(l=10, r=60, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#475569", size=10),
            xaxis=dict(
                gridcolor="#e0e7ff", color="#1e293b",
                title=dict(text="Importance Score", font=dict(color="#1e293b")),
                tickfont=dict(color="#1e293b", size=10),
            ),
            yaxis=dict(
                color="#1e293b", autorange="reversed",
                tickfont=dict(color="#1e293b", size=10),
            ),
            title=dict(
                text=name,
                font=dict(size=11, color="#1e293b", family="JetBrains Mono"),
                x=0,
            ),
        )
        with col:
            with st.container(border=True):
                st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div style='font-size:0.7rem;font-family:JetBrains Mono;color:#475569;
                padding:0.6rem 0.8rem;background:#f0f4ff;border-radius:8px;
                border:1px solid #e0e7ff;margin-top:0.5rem'>
        <b>Color:</b> &nbsp;
        <span style='color:#4f46e5'>■ Deep indigo</span> = highest importance &nbsp;
        <span style='color:#a5b4fc'>■ Light indigo</span> = lower importance
    </div>""", unsafe_allow_html=True)

    # ── PSO mask bar ──────────────────────────────────────────
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-section-title">PSO Selection Mask '
        '<span class="dash-badge">All 30 Features</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        mask_colors = ["#4f46e5" if m else "#e2e8f0" for m in mask]
        mask_text   = ["✓" if m else "✗" for m in mask]
        fig_mask = go.Figure(go.Bar(
            x=all_feats,
            y=[1] * len(all_feats),
            marker=dict(color=mask_colors, line=dict(color="#fff", width=1)),
            text=mask_text,
            textposition="inside",
            textfont=dict(size=9, color="#ffffff", family="JetBrains Mono"),
            hovertext=[
                f"{f}: {'✓ Selected' if m else '✗ Excluded'}"
                for f, m in zip(all_feats, mask)
            ],
            hoverinfo="text",
        ))
        fig_mask.update_layout(
            height=160,
            margin=dict(l=10, r=10, t=10, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.6)",
            font=dict(family="JetBrains Mono", color="#1e293b", size=9),
            xaxis=dict(
                tickangle=-45, color="#1e293b",
                tickfont=dict(color="#1e293b", size=9),
                showgrid=False, linecolor="#c7d2fe",
            ),
            yaxis=dict(showticklabels=False, showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_mask, use_container_width=True)
        st.markdown(f"""
        <div style='font-size:0.7rem;font-family:JetBrains Mono;color:#475569'>
            <span style='color:#4f46e5'>■ Indigo ({n_selected})</span> = PSO selected &nbsp;
            <span style='color:#94a3b8'>■ Light ({len(excluded)})</span> = excluded by PSO &nbsp;|&nbsp;
            Selected: <b style='color:#4f46e5'>{', '.join(SELECTED_FEATURES)}</b>
        </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;color:#6366f1;font-size:0.65rem;padding:2rem 0 1rem;
                letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;">
        ⚡ QUANTUMSHIELD · QML FRAUD DETECTION · PENNYLANE + PSO ENSEMBLE
    </div>
    """, unsafe_allow_html=True)

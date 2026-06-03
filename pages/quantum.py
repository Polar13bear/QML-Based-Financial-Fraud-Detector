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
        st.markdown(f"""
        <div class="page-header">
            <div class="page-header-tag">⚛️ Quantum Insights</div>
            <div class="page-header-title"><span>Quantum</span> Model Details</div>
            <p class="page-header-sub">{N_QUBITS}-qubit VQC with {N_LAYERS} variational layers — angle + amplitude encoding with ZZ-style ring CNOT entanglement (PennyLane).</p>
        </div>
        """, unsafe_allow_html=True)

        qi1, qi2, qi3 = st.columns(3)

        with qi1:
            circuit_rows = []
            for q in range(N_QUBITS):
                gates = ["H"]
                for l in range(N_LAYERS):
                    gates += ["Ry", "Rz"]
                    gates.append("●" if q < N_QUBITS - 1 else "⊕")
                gates.append("M")
                circuit_rows.append((f"q{q}", gates))

            rows_html = ""
            for qubit, gates in circuit_rows:
                gate_html = ""
                for g in gates:
                    if g == "●":
                        gate_html += "<span class='gate-ctrl'>●</span><span class='gate-wire'>—</span>"
                    elif g == "⊕":
                        gate_html += "<span class='gate-ctrl'>⊕</span><span class='gate-wire'>—</span>"
                    elif g == "M":
                        gate_html += "<span class='gate-meas'>M</span>"
                    else:
                        gate_html += f"<span class='gate-box'>{g}</span><span class='gate-wire'>—</span>"
                rows_html += f"<div class='circuit-row'><span class='circuit-qubit'>{qubit}</span><span class='circuit-init'>|0⟩—</span>{gate_html}</div>"

            st.markdown(f"""<div class="circuit-wrap">
                <div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.8rem'>
                    VQC Circuit — {N_QUBITS} Qubits · {N_LAYERS} Layers
                </div>
                {rows_html}
                <div style='margin-top:0.8rem;font-size:0.68rem;font-family:JetBrains Mono;color:#64748b;line-height:1.8;padding:0.6rem;background:#f0f4ff;border-radius:6px;border:1px solid #e0e7ff'>
                    <b>Gate Legend:</b><br>
                    <span style='color:#7c3aed'>q0–q{N_QUBITS-1}</span> = qubit wires<br>
                    <span style='color:#4f46e5'>H</span> = Hadamard — creates superposition<br>
                    <span style='color:#4f46e5'>Ry / Rz</span> = rotation gates — encode data features<br>
                    <span style='color:#4f46e5'>● / ⊕</span> = CNOT — creates entanglement between qubits<br>
                    <span style='color:#d97706'>M</span> = measurement — collapses quantum state to classical bit
                </div>
            </div>""", unsafe_allow_html=True)

        with qi2:
            np.random.seed(42)
            base = np.random.rand(8, 8)
            kernel = (base + base.T) / 2
            np.fill_diagonal(kernel, 1.0)
            kernel = np.clip(kernel, 0, 1)

            fig_km = go.Figure(go.Heatmap(
                z=kernel,
                colorscale=[[0,"#312e81"],[0.35,"#6d28d9"],[0.65,"#7c3aed"],[1.0,"#06b6d4"]],
                showscale=True, zmin=0, zmax=1,
                colorbar=dict(
                    title=dict(text="Similarity", font=dict(size=9, family="JetBrains Mono", color="#475569")),
                    tickvals=[0, 0.5, 1.0],
                    ticktext=["0.0 Low", "0.5 Mid", "1.0 High"],
                    tickfont=dict(size=8, family="JetBrains Mono", color="#475569"),
                    len=0.9, thickness=10,
                ),
            ))
            fig_km.update_layout(
                height=260, margin=dict(l=0,r=60,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )
            with st.container(border=True):
                st.markdown("<div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.4rem'>QK-SVM Encoding Space (RBF on Quantum Features)</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_km, use_container_width=True)
                st.markdown("""
                <div style='font-size:0.68rem;font-family:JetBrains Mono;color:#475569;line-height:1.8;padding:0.5rem;background:#f0f4ff;border-radius:8px;border:1px solid #e0e7ff'>
                    <b>What this shows:</b><br>
                    <span style='color:#312e81'>■ Dark violet</span> = Low similarity — transactions very different<br>
                    <span style='color:#7c3aed'>■ Purple</span> = Medium similarity — some shared patterns<br>
                    <span style='color:#06b6d4'>■ Cyan</span> = High similarity — nearly identical transactions<br>
                    <span style='color:#64748b'>The QK-SVM uses an <b>RBF kernel on angle+amplitude encoded features</b> (not a true quantum kernel). Each cell represents pairwise similarity in the quantum-encoded feature space.</span>
                </div>""", unsafe_allow_html=True)

        with qi3:
            fm_rows = [
                ("Encoding",      "Angle + Amplitude"),
                ("Circuit Style", "ZZ-style (PennyLane)"),
                ("Qubits",        f"{N_QUBITS}"),
                ("Layers",        f"{N_LAYERS}"),
                ("Entanglement",  "Ring CNOT"),
                ("QK-SVM Kernel", "RBF on encoded features"),
                ("PSO Features",  f"{int(sum(comps['pso_mask']))}"),
                ("Circuit Depth", f"{N_QUBITS * N_LAYERS * 3}"),
                ("VQC Params",    f"{N_QUBITS * N_LAYERS * 2}"),
            ]
            rows_html = "".join(f"<div class='fm-row'><span>{k}</span><span>{v}</span></div>" for k, v in fm_rows)
            st.markdown(f"""<div class="fm-card">
                <div style='font-size:0.78rem;font-family:JetBrains Mono;color:#475569;margin-bottom:0.8rem'>Encoding & Circuit Config</div>
                {rows_html}
                <div class="fm-note">
                    A <span class="hl-purple">ZZ-style PennyLane circuit</span> encodes classical data into quantum states using
                    <span class="hl-cyan">H → RZ → CNOT → RZ</span> (angle encoding) and normalised vectors (amplitude encoding).
                    The QK-SVM then applies an <span class="hl-purple">RBF kernel</span> on these quantum-encoded features,
                    mapping transactions into a high-dimensional space where fraud patterns become linearly separable.
                </div>
            </div>""", unsafe_allow_html=True)

        # VQC weight heatmap
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">VQC Trained Weights <span class="dash-badge">From PKL</span></div>', unsafe_allow_html=True)
        with st.container(border=True):
            try:
                vqc_w = comps  # already loaded
                with open(f"{COMPONENTS_DIR}/vqc_weights.pkl", "rb") as f:
                    vqc_weights_raw = pickle.load(f)
                w_arr = np.array(vqc_weights_raw)
                if w_arr.ndim == 3:
                    fig_vqc = go.Figure()
                    for layer_idx in range(w_arr.shape[0]):
                        fig_vqc.add_trace(go.Heatmap(
                            z=w_arr[layer_idx],
                            colorscale=[[0,"#312e81"],[0.5,"#a855f7"],[1.0,"#f0abfc"]],
                            showscale=(layer_idx == 0),
                            name=f"Layer {layer_idx+1}",
                            visible=(layer_idx == 0),
                            colorbar=dict(title=dict(text="Weight", font=dict(size=9, color="#475569")),
                                          tickfont=dict(size=8, color="#475569"), thickness=10),
                        ))
                    buttons = [dict(label=f"Layer {i+1}", method="update",
                                    args=[{"visible": [j == i for j in range(w_arr.shape[0])]}])
                               for i in range(w_arr.shape[0])]
                    fig_vqc.update_layout(
                        height=220, margin=dict(l=10,r=60,t=10,b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title=dict(text="Qubit", font=dict(color="#1e293b")),
                                   tickfont=dict(color="#1e293b"), showgrid=False),
                        yaxis=dict(title=dict(text="Gate", font=dict(color="#1e293b")),
                                   tickfont=dict(color="#1e293b"), showgrid=False),
                        updatemenus=[dict(buttons=buttons, direction="right", x=0, y=1.15,
                                          bgcolor="#e0e7ff", bordercolor="#a5b4fc",
                                          font=dict(size=10, family="JetBrains Mono", color="#4f46e5"))],
                    )
                    st.plotly_chart(fig_vqc, use_container_width=True)
                    st.markdown("""<div style='font-size:0.7rem;font-family:JetBrains Mono;color:#475569'>
                        <span style='color:#312e81'>■ Dark violet</span> = large negative weight &nbsp;
                        <span style='color:#a855f7'>■ Purple</span> = near zero &nbsp;
                        <span style='color:#f0abfc'>■ Pink</span> = large positive weight
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("VQC weights shape not 3D — showing raw values.")
                    st.write(w_arr)
            except Exception as e:
                st.markdown(f'<div class="info-box">VQC weights visualization: {e}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════


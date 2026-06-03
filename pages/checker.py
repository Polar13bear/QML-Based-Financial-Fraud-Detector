import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.loader import paysim_detector, CFG
from vqc_paysim_model import TYPE_ENCODING


def render():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-tag">⚡ PaySim Dataset · VQC Quantum Model</div>
        <div class="page-header-title"><span>Transaction Checker</span></div>
        <p class="page-header-sub">
            Enter PaySim transaction fields. The dedicated PaySim VQC model
            (3 qubits, trained on real PaySim features) will assess fraud risk instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("txn_form"):
        st.markdown('<div class="sec-head">💳  Transaction Details</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                txn_type = st.selectbox(
                    "Transaction Type  (type)",
                    options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                    format_func=lambda x: {
                        "PAYMENT":  "💳 PAYMENT — merchant payment",
                        "TRANSFER": "🔁 TRANSFER — account transfer",
                        "CASH_OUT": "💸 CASH_OUT — cash withdrawal",
                        "DEBIT":    "🏦 DEBIT — debit transaction",
                        "CASH_IN":  "📥 CASH_IN — cash deposit",
                    }[x])
            with c2:
                amount = st.number_input(
                    "Transaction Amount  (amount)",
                    min_value=0.01, max_value=100_000_000.0,
                    value=9839.64, step=0.01,
                    help="Amount in local currency")
            with c3:
                step = st.number_input(
                    "Time Step  (step)", min_value=1, max_value=744,
                    value=1, step=1,
                    help="1 step = 1 hour. 744 steps = 30 days")
                hour_of_day = step % 24
                st.caption(f"Hour of day: {hour_of_day:02d}:00 "
                           f"{'🌙 Night' if hour_of_day <= 5 or hour_of_day >= 22 else '☀️ Day'}")

        st.markdown('<div class="sec-head">🏦  Origin Account Balances</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            ob1, ob2 = st.columns(2)
            with ob1:
                old_balance_orig = st.number_input(
                    "Balance Before  (oldbalanceOrg)",
                    min_value=0.0, max_value=100_000_000.0,
                    value=170136.0, step=0.01)
            with ob2:
                new_balance_orig = st.number_input(
                    "Balance After  (newbalanceOrig)",
                    min_value=0.0, max_value=100_000_000.0,
                    value=160296.36, step=0.01)
            if old_balance_orig > 0:
                drain_pct = max(0, (old_balance_orig - new_balance_orig)
                                / old_balance_orig * 100)
                color = ("#dc2626" if drain_pct > 80
                         else "#d97706" if drain_pct > 40 else "#16a34a")
                st.markdown(
                    f"<div style='font-size:0.75rem;font-family:JetBrains Mono;"
                    f"color:{color}'>Account drained: {drain_pct:.1f}%</div>",
                    unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🏦  Destination Account Balances</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            db1, db2 = st.columns(2)
            with db1:
                old_balance_dest = st.number_input(
                    "Dest. Balance Before  (oldbalanceDest)",
                    min_value=0.0, max_value=100_000_000.0,
                    value=0.0, step=0.01)
            with db2:
                new_balance_dest = st.number_input(
                    "Dest. Balance After  (newbalanceDest)",
                    min_value=0.0, max_value=100_000_000.0,
                    value=0.0, step=0.01)
            if old_balance_dest == 0 and new_balance_dest == 0 and amount > 0:
                st.markdown(
                    "<div style='font-size:0.75rem;font-family:JetBrains Mono;"
                    "color:#dc2626'>⚠️ Destination balance unchanged — common fraud pattern</div>",
                    unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🔍  Analyze with PaySim VQC Model", use_container_width=True)

    if submitted:
        # Build the 8-feature vector in training order
        type_enc         = TYPE_ENCODING.get(txn_type, 3)
        balance_diff_orig = new_balance_orig - old_balance_orig
        balance_diff_dest = new_balance_dest - old_balance_dest

        x_raw = np.array([
            type_enc,
            amount,
            old_balance_orig,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest,
            balance_diff_orig,
            balance_diff_dest,
        ])

        with st.spinner("Running PaySim VQC analysis..."):
            result = paysim_detector.predict_single(x_raw)

        is_fraud   = result["prediction"] == 1
        fraud_pct  = int(round(result["fraud_score"] * 100))
        safety_pct = 100 - fraud_pct
        tier       = "danger" if fraud_pct >= 60 else ("warn" if fraud_pct >= 30 else "safe")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.1, 1, 1])

        with r1:
            if not is_fraud:
                st.markdown("""
                <div class="result-safe">
                    <div class="result-icon">✅</div>
                    <div class="result-title-safe">Looks Legitimate</div>
                    <div class="result-msg">No significant fraud patterns detected.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-fraud">
                    <div class="result-icon">🚨</div>
                    <div class="result-title-fraud">Fraud Detected!</div>
                    <div class="result-msg">Transaction matches known PaySim fraud patterns.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #e0e7ff;border-radius:14px;
                        padding:1rem 1.2rem;box-shadow:0 2px 12px #6366f110'>
                <div style='font-size:0.65rem;font-family:JetBrains Mono;color:#6366f1;
                            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem'>
                    PaySim VQC Score
                </div>
                <div class="score-row"><span>VQC Fraud Score</span>
                    <span style='color:{"#dc2626" if is_fraud else "#16a34a"};font-size:1.1rem'>
                        {result["fraud_score"]:.4f}
                    </span>
                </div>
                <div class="score-row"><span>Threshold</span>
                    <span>{paysim_detector.threshold:.2f}</span>
                </div>
                <div class="score-row"><span>Verdict</span>
                    <span style='color:{"#dc2626" if is_fraud else "#16a34a"};font-weight:700'>
                        {result["label"]}
                    </span>
                </div>
                <div style='font-size:0.65rem;color:#94a3b8;font-family:JetBrains Mono;
                            margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #e0e7ff'>
                    Model: PaySim VQC · {paysim_detector.n_qubits} qubits · 1 layer
                </div>
            </div>""", unsafe_allow_html=True)

        with r2:
            flags = []
            if txn_type in ("CASH_OUT", "TRANSFER"):
                flags.append(f"🔴 High-risk type: {txn_type}")
            if old_balance_orig > 0:
                drain = (old_balance_orig - new_balance_orig) / old_balance_orig
                if drain > 0.8:
                    flags.append("🔴 Origin account nearly drained (>80%)")
            if old_balance_dest == 0 and new_balance_dest == 0 and amount > 0:
                flags.append("🔴 Destination balance unchanged")
            if amount > old_balance_orig > 0:
                flags.append("🟠 Amount exceeds origin balance")
            if amount > 1_000_000:
                flags.append(f"🟠 Very large amount ({amount:,.0f})")
            if new_balance_orig == 0 and old_balance_orig > 0:
                flags.append("🔴 Origin account fully emptied")
            if hour_of_day <= 4 or hour_of_day >= 22:
                flags.append(f"🟡 Off-hours transaction ({hour_of_day:02d}:00)")

            flags_html = "".join(f"<div class='flag-item'>{f}</div>" for f in flags) \
                if flags else "<div class='flag-item' style='color:#16a34a'>✨ No suspicious signals.</div>"

            st.markdown(f"""
            <div class="meter-wrap">
                <div class="meter-lbl">Safety Rating</div>
                <div class="meter-num {tier}">{safety_pct}%</div>
                <div class="meter-sub">transaction safety score</div>
                <div class="bar-bg">
                    <div class="bar-fill-{tier}" style="width:{safety_pct}%"></div>
                </div>
                <div class="flags-wrap">
                    <div class="flags-lbl">Risk Signals</div>
                    {flags_html}
                </div>
            </div>""", unsafe_allow_html=True)

        with r3:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_pct,
                number=dict(suffix="%",
                            font=dict(size=28, color="#1e293b", family="JetBrains Mono")),
                gauge=dict(
                    axis=dict(range=[0, 100],
                              tickfont=dict(color="#475569", size=9)),
                    bar=dict(
                        color="#dc2626" if fraud_pct >= 60
                        else "#d97706" if fraud_pct >= 30 else "#16a34a",
                        thickness=0.25),
                    bgcolor="#f1f5f9", borderwidth=0,
                    steps=[
                        dict(range=[0, 30],   color="#dcfce7"),
                        dict(range=[30, 60],  color="#fef9c3"),
                        dict(range=[60, 100], color="#fee2e2"),
                    ],
                    threshold=dict(
                        line=dict(color="#6366f1", width=2),
                        thickness=0.75,
                        value=paysim_detector.threshold * 100),
                ),
                title=dict(text="Fraud Risk Score",
                           font=dict(size=11, color="#475569",
                                     family="JetBrains Mono")),
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono"),
            )
            with st.container(border=True):
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(f"""
                <div style='font-size:0.68rem;font-family:JetBrains Mono;
                            color:#475569;text-align:center'>
                    <span style='color:#16a34a'>■ Green</span> = Safe &nbsp;
                    <span style='color:#d97706'>■ Yellow</span> = Caution &nbsp;
                    <span style='color:#dc2626'>■ Red</span> = Fraud<br>
                    <span style='color:#6366f1'>│</span> = threshold
                    ({paysim_detector.threshold * 100:.0f}%)
                </div>""", unsafe_allow_html=True)

            # Input summary
            st.markdown(f"""
            <div style='background:#f8f7ff;border:1px solid #c4b5fd;border-radius:12px;
                        padding:0.9rem 1rem;margin-top:0.8rem;font-size:0.72rem;
                        font-family:JetBrains Mono'>
                <div style='color:#6366f1;font-weight:700;margin-bottom:0.5rem;
                            font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase'>
                    Input Features Used
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>type</span><span style='color:#1e293b;font-weight:600'>{txn_type}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>amount</span><span style='color:#1e293b;font-weight:600'>{amount:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>oldbalanceOrg</span><span style='color:#1e293b;font-weight:600'>{old_balance_orig:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>newbalanceOrig</span><span style='color:#1e293b;font-weight:600'>{new_balance_orig:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>oldbalanceDest</span><span style='color:#1e293b;font-weight:600'>{old_balance_dest:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>newbalanceDest</span><span style='color:#1e293b;font-weight:600'>{new_balance_dest:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;
                            padding:0.2rem 0;border-top:1px solid #e0e7ff;margin-top:0.3rem'>
                    <span>balance_diff_orig</span>
                    <span style='color:#7c3aed;font-weight:600'>{balance_diff_orig:,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;color:#475569;padding:0.2rem 0'>
                    <span>balance_diff_dest</span>
                    <span style='color:#7c3aed;font-weight:600'>{balance_diff_dest:,.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        tip = ("""<div class="tip-box"><strong>💡 Action required:</strong>
            This transaction matches PaySim fraud patterns. Block and investigate
            the origin/destination accounts immediately.</div>"""
               if is_fraud else
               """<div class="tip-box"><strong>💡 Transaction looks clean:</strong>
            Balance flows and transaction type are consistent with legitimate activity.</div>""")
        st.markdown(tip, unsafe_allow_html=True)

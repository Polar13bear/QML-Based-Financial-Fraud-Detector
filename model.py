"""
HybridFraudDetector class — must be defined here so pickle can load the model.
"""
import numpy as np
import pennylane as qml

# These will be set when the model is loaded (read from the pkl's stored values)
_N_QUBITS = 4
_N_LAYERS = 2
_dev = qml.device("default.qubit", wires=_N_QUBITS)

@qml.qnode(_dev)
def _vqc_circuit_global(weights, x):
    for l in range(_N_LAYERS):
        for i in range(_N_QUBITS):
            qml.RY(x[i], wires=i)
        for i in range(_N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(_N_QUBITS - 1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[_N_QUBITS-1, 0])
    return qml.expval(qml.PauliZ(0))


class HybridFraudDetector:
    def __init__(self, minmax_scaler, pso_mask, feature_names,
                 rf, xgb, ocsvm,
                 q_scaler, n_qubits, n_layers, vqc_weights, qksvm,
                 hybrid_config):
        self.minmax_scaler = minmax_scaler
        self.pso_mask = pso_mask
        self.feature_names = feature_names
        self.rf = rf
        self.xgb = xgb
        self.ocsvm = ocsvm
        self.q_scaler = q_scaler
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.vqc_weights = vqc_weights
        self.qksvm = qksvm
        self.w_rf = hybrid_config['w_rf']
        self.w_xgb = hybrid_config['w_xgb']
        self.w_ocsvm = hybrid_config['w_ocsvm']
        self.w_qksvm = hybrid_config['w_qksvm']
        self.threshold = hybrid_config['threshold']

    def _angle_encode(self, X):
        return np.hstack([np.cos(X), np.sin(X)])

    def _amplitude_encode(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms

    def _normalize_score(self, scores):
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def _run_vqc(self, X_angle):
        raw = np.array([
            float(_vqc_circuit_global(self.vqc_weights, x))
            for x in X_angle
        ])
        return (raw + 1) / 2

    def predict_single(self, x_raw):
        X = x_raw.reshape(1, -1).copy().astype(float)
        time_idx = self.feature_names.index('Time')
        amount_idx = self.feature_names.index('Amount')
        X[:, [time_idx, amount_idx]] = self.minmax_scaler.transform(
            X[:, [time_idx, amount_idx]]
        )
        X_pso = X[:, self.pso_mask]

        s_rf = float(self.rf.predict_proba(X_pso)[0, 1])
        s_xgb = float(self.xgb.predict_proba(X_pso)[0, 1])
        s_ocsvm = float(self._normalize_score(
            -self.ocsvm.decision_function(X_pso))[0])

        X_angle = self.q_scaler.transform(X_pso)[:, :self.n_qubits]
        X_qi = np.hstack([self._angle_encode(X_angle),
                          self._amplitude_encode(X_angle)])
        s_qksvm = float(self.qksvm.predict_proba(X_qi)[0, 1])
        s_vqc = float(self._run_vqc(X_angle)[0])
        s_quantum = (s_qksvm + s_vqc) / 2

        fraud_score = (self.w_rf * s_rf +
                       self.w_xgb * s_xgb +
                       self.w_ocsvm * s_ocsvm +
                       self.w_qksvm * s_quantum)

        pred = int(fraud_score >= self.threshold)
        return {
            'prediction': pred,
            'label': "FRAUD" if pred == 1 else "LEGITIMATE",
            'fraud_score': round(fraud_score, 4),
            'threshold': round(self.threshold, 4),
            'rf_score': round(s_rf, 4),
            'xgb_score': round(s_xgb, 4),
            'ocsvm_score': round(s_ocsvm, 4),
            'qksvm_score': round(s_qksvm, 4),
            'vqc_score': round(s_vqc, 4),
            'quantum_score': round(s_quantum, 4),
        }

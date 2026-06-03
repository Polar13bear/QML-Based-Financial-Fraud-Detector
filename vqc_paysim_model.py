"""
VQCFraudDetector — PaySim-trained VQC model.
Must be defined at module level so pickle can load the saved model.
"""
import numpy as np
import pennylane as qml

_N_QUBITS_PS = 3
_dev_ps = qml.device("default.qubit", wires=_N_QUBITS_PS)


@qml.qnode(_dev_ps)
def _vqc_paysim_circuit(weights, x):
    for i in range(_N_QUBITS_PS):
        qml.RY(x[i], wires=i)
    for i in range(_N_QUBITS_PS):
        qml.RY(weights[i, 0], wires=i)
        qml.RZ(weights[i, 1], wires=i)
    for i in range(_N_QUBITS_PS - 1):
        qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))


# PaySim feature order (must match training)
PAYSIM_FEATURES = [
    "type_enc",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
]

TYPE_ENCODING = {
    "CASH_IN":  0,
    "CASH_OUT": 1,
    "DEBIT":    2,
    "PAYMENT":  3,
    "TRANSFER": 4,
}


class VQCFraudDetector:
    """
    PaySim VQC fraud detector.
    Trained on: type_enc, amount, oldbalanceOrg, newbalanceOrig,
                oldbalanceDest, newbalanceDest, balance_diff_orig, balance_diff_dest
    """
    def __init__(self, scaler, q_scaler, weights, n_qubits, threshold=0.5):
        self.scaler    = scaler
        self.q_scaler  = q_scaler
        self.weights   = weights
        self.n_qubits  = n_qubits
        self.threshold = threshold

    def predict_single(self, x_raw):
        """
        x_raw: array of 8 features in PAYSIM_FEATURES order.
        Returns dict with fraud_score and prediction.
        """
        x = np.array(x_raw).reshape(1, -1).astype(float)
        x_scaled   = self.scaler.transform(x)
        x_angle    = self.q_scaler.transform(x_scaled)[:, :self.n_qubits]
        raw        = float(_vqc_paysim_circuit(self.weights, x_angle[0]))
        fraud_prob = (raw + 1) / 2
        return {
            "fraud_score": round(fraud_prob, 4),
            "prediction":  int(fraud_prob >= self.threshold),
            "label":       "FRAUD" if fraud_prob >= self.threshold else "LEGITIMATE",
        }

import streamlit as st
import numpy as np
import pickle
import sys

import model
from model import HybridFraudDetector  # noqa: F401
import vqc_paysim_model
from vqc_paysim_model import VQCFraudDetector, TYPE_ENCODING, PAYSIM_FEATURES  # noqa: F401

# Pickle saved VQCFraudDetector under __main__, so we inject it there
sys.modules['__main__'].VQCFraudDetector = VQCFraudDetector

COMPONENTS_DIR = "fraud_detection_final/components"


@st.cache_resource
def load_model():
    def load(name):
        with open(f"{COMPONENTS_DIR}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    return HybridFraudDetector(
        minmax_scaler=load("minmax_scaler"),
        pso_mask=load("pso_mask"),
        feature_names=load("feature_names"),
        rf=load("random_forest"),
        xgb=load("xgboost"),
        ocsvm=load("ocsvm"),
        q_scaler=load("q_scaler"),
        n_qubits=load("n_qubits"),
        n_layers=load("n_layers"),
        vqc_weights=load("vqc_weights"),
        qksvm=load("qksvm"),
        hybrid_config=load("hybrid_config"),
    )


@st.cache_resource
def load_components():
    def load(name):
        with open(f"{COMPONENTS_DIR}/{name}.pkl", "rb") as f:
            return pickle.load(f)
    return {
        "pso_mask":      load("pso_mask"),
        "feature_names": load("feature_names"),
        "n_qubits":      load("n_qubits"),
        "n_layers":      load("n_layers"),
        "hybrid_config": load("hybrid_config"),
        "rf":            load("random_forest"),
        "xgb":           load("xgboost"),
    }


detector = load_model()
comps    = load_components()


@st.cache_resource
def load_paysim_model():
    with open(f"{COMPONENTS_DIR}/vqc_paysim.pkl", "rb") as f:
        return pickle.load(f)


paysim_detector = load_paysim_model()

SELECTED_FEATURES = [
    comps["feature_names"][i]
    for i in range(len(comps["feature_names"]))
    if comps["pso_mask"][i]
]
N_QUBITS = comps["n_qubits"]
N_LAYERS = comps["n_layers"]
CFG      = comps["hybrid_config"]
RF_IMP   = comps["rf"].feature_importances_
XGB_IMP  = comps["xgb"].feature_importances_


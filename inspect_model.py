import pickle

with open('hybrid_fraud_detector.pkl', 'rb') as f:
    model = pickle.load(f)

print("Type:", type(model))
print("Features:", getattr(model, 'feature_names_in_', None))
print("N features:", getattr(model, 'n_features_in_', None))
print("Attrs:", [a for a in dir(model) if not a.startswith('_')])

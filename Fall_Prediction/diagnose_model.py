import joblib

print("Loading model...")

data = joblib.load("thermora_event_model_v2.pkl")

print("Model object type:", type(data))
print("Keys:", data.keys())

model = data["model"]
threshold = data["threshold"]

print()
print("Model:", type(model))
print("Threshold:", threshold)

if hasattr(model, "n_estimators"):
    print("Number of trees:", model.n_estimators)

if hasattr(model, "n_features_in_"):
    print("Expected features:", model.n_features_in_)

print()
print("DIAGNOSTIC COMPLETE")
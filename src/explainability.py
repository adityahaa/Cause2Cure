"""SHAP support for the persisted best predictive model; SHAP is not causal."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def _effective_values(explainer, frame: pd.DataFrame):
    values = np.asarray(explainer.shap_values(frame))
    base = np.asarray(explainer.expected_value)
    # Tree classifiers may return (rows, features, classes) or class-indexed lists.
    if values.ndim == 3:
        return values[:, :, 0], float(base[0])
    if values.ndim == 2:
        return values, float(base[0] if base.ndim else base)
    return np.asarray(values[0]), float(base[0] if base.ndim else base)


def train_shap_metadata() -> dict:
    model = joblib.load("models/best_model.pkl")
    meta = json.load(open("models/metadata.json", encoding="utf-8"))
    frame = pd.read_csv("data/diabetes_drug_effectiveness.csv")[meta["features"]]
    explainer = shap.TreeExplainer(model)
    values, _ = _effective_values(explainer, frame)
    global_importance = dict(sorted(zip(meta["features"], np.abs(values).mean(axis=0)), key=lambda item: item[1], reverse=True))
    result = {"model": meta["best_model"], "explanation_type": "SHAP values for predictive effectiveness probability; not causal evidence.",
              "global_mean_abs_shap": {k: round(float(v), 6) for k, v in global_importance.items()}}
    Path("models/shap_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def explain_patient(model, features: list[str], patient_values: dict) -> dict:
    frame = pd.DataFrame([[patient_values[name] for name in features]], columns=features)
    explainer = shap.TreeExplainer(model)
    values, base = _effective_values(explainer, frame)
    contributions = [{"feature": name, "value": float(frame.iloc[0][name]), "shap_value": round(float(value), 6),
                      "direction": "increases predicted effectiveness" if value > 0 else "decreases predicted effectiveness" if value < 0 else "no contribution"}
                     for name, value in zip(features, values[0])]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    label = int(model.predict(frame)[0])
    # API probability is always for the returned predicted class.  SHAP values
    # remain intentionally class-0/effectiveness framed above.
    probability = float(model.predict_proba(frame)[0, label])
    return {"prediction": label, "predicted_probability": round(probability, 6), "base_value": round(base, 6),
            "feature_contributions": contributions,
            "disclaimer": "SHAP explains this predictive model; it does not establish causality."}

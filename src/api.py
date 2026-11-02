"""
CDSS Backend API — FastAPI
Endpoints: /predict  /counterfactual  /clusters  /insights
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib, json, os, base64
import numpy as np
import pandas as pd
from causal import DRUGS, causal_adjustment_features
from explainability import explain_patient

try:
    from recommendations import generate_recommendations
    RECOMMENDATIONS_IMPORT_ERROR = None
except Exception as exc:
    # The predictive and causal API remains available if optional Gemini
    # dependencies are not installed; /recommendations reports the issue.
    generate_recommendations = None
    RECOMMENDATIONS_IMPORT_ERROR = str(exc)

# ── Load artifacts ─────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

best_model  = joblib.load(f"{MODELS}/best_model.pkl")
rf_model    = joblib.load(f"{MODELS}/RandomForest.pkl")
gb_model    = joblib.load(f"{MODELS}/GradientBoosting.pkl")
cluster_scaler = joblib.load(f"{MODELS}/cluster_scaler.pkl")
kmeans      = joblib.load(f"{MODELS}/kmeans.pkl")

with open(f"{MODELS}/metadata.json") as f:
    META = json.load(f)
CAUSAL_DIR = os.path.join(MODELS, "causal")
CAUSAL_META = None
if os.path.exists(os.path.join(CAUSAL_DIR, "causal_metadata.json")):
    with open(os.path.join(CAUSAL_DIR, "causal_metadata.json"), encoding="utf-8") as f:
        CAUSAL_META = json.load(f)
SHAP_META = None
if os.path.exists(os.path.join(MODELS, "shap_metadata.json")):
    with open(os.path.join(MODELS, "shap_metadata.json"), encoding="utf-8") as f:
        SHAP_META = json.load(f)

FEATURES = META["features"]
CLUSTER_FEATURES = META.get("cluster_features", ["age", "bmi", "hba1c_v1", "glucose_v1", "bp_v1", "adherence"])

app = FastAPI(title="CDSS API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age: float = Field(ge=0, le=120)
    gender: int = Field(ge=0, le=1)
    bmi: float = Field(gt=0, le=100)
    hba1c_v1: float = Field(gt=0, le=30)
    glucose_v1: float = Field(gt=0, le=1000)
    bp_v1: float = Field(gt=0, le=400)
    metformin_v1: int = Field(ge=0, le=3, description="Baseline regimen-state code; not dose intensity")
    sulfonylurea_v1: int = Field(ge=0, le=3, description="Baseline regimen-state code; not dose intensity")
    dpp4_v1: int = Field(ge=0, le=3, description="Baseline regimen-state code; not dose intensity")
    sglt2_v1: int = Field(ge=0, le=3, description="Baseline regimen-state code; not dose intensity")
    insulin_v1: int = Field(ge=0, le=3, description="Baseline regimen-state code; not dose intensity")
    adherence: int = Field(ge=0, le=1)
    diet: int = Field(ge=0, le=1)
    exercise: int = Field(ge=0, le=1)

# ── Helpers ────────────────────────────────────────────────────────────────
def to_array(p: PatientInput):
    return pd.DataFrame([[
        p.age, p.gender, p.bmi,
        p.hba1c_v1, p.glucose_v1, p.bp_v1,
        p.metformin_v1, p.sulfonylurea_v1, p.dpp4_v1, p.sglt2_v1, p.insulin_v1,
        p.adherence, p.diet, p.exercise
    ]], columns=FEATURES)

def get_cluster(p: PatientInput):
    cf = pd.DataFrame([[p.age, p.bmi, p.hba1c_v1, p.glucose_v1, p.bp_v1, p.adherence]], columns=CLUSTER_FEATURES)
    if cluster_scaler.n_features_in_ != len(CLUSTER_FEATURES) or kmeans.n_features_in_ != len(CLUSTER_FEATURES):
        raise RuntimeError("Cluster artifacts are incompatible with configured cluster features")
    cf_sc = cluster_scaler.transform(cf)
    return int(kmeans.predict(cf_sc)[0])

def predict_one(arr):
    prob = best_model.predict_proba(arr)[0]
    label = int(best_model.predict(arr)[0])
    return {
        "label": label,
        "effective": label == 0,
        "confidence": round(float(prob[0] if label == 0 else prob[1]) * 100, 1),
        "prob_effective": round(float(prob[0]) * 100, 1),
        "prob_not_effective": round(float(prob[1]) * 100, 1),
    }

class TreatmentRequest(BaseModel):
    treatment: str

class CausalPatientRequest(PatientInput):
    treatment: str

class CausalCounterfactualRequest(CausalPatientRequest):
    alternative_state: int = Field(ge=0, le=1)

def causal_artifact(treatment: str):
    if treatment not in DRUGS:
        raise HTTPException(422, f"Unknown treatment '{treatment}'. Choose one of: {', '.join(DRUGS)}")
    if CAUSAL_META is None:
        raise HTTPException(503, "Causal artifacts are unavailable. Run train_causal.py.")
    path = os.path.join(CAUSAL_DIR, f"{treatment}_causal.pkl")
    if not os.path.exists(path):
        raise HTTPException(503, f"Causal artifact for {treatment} is unavailable.")
    return joblib.load(path)

def causal_patient_values(p: PatientInput, artifact: dict) -> pd.DataFrame:
    values = {"age": p.age, "gender": p.gender, "bmi": p.bmi, "hba1c_v1": p.hba1c_v1,
              "glucose_v1": p.glucose_v1, "bp_v1": p.bp_v1, "adherence": p.adherence,
              "diet": p.diet, "exercise": p.exercise}
    for drug in DRUGS:
        if f"{drug}_present" in artifact["features"]:
            values[f"{drug}_present"] = int(getattr(p, f"{drug}_v1") > 0)
    return pd.DataFrame([[values[name] for name in artifact["features"]]], columns=artifact["features"])

def causal_patient_result(p: PatientInput, treatment: str) -> dict:
    artifact = causal_artifact(treatment)
    x = causal_patient_values(p, artifact)
    propensity = float(artifact["propensity_model"].predict_proba(x)[0, 1])
    lower, upper = artifact["overlap"]["overlap_region"]
    positivity = bool(PROPENSITY_OK(propensity, lower, upper))
    cate = float(artifact["cate_model"].effect(x)[0])
    cluster = get_cluster(p)
    return {"artifact": artifact, "patient_cate": cate, "propensity": propensity, "positivity_flag": positivity, "cluster": cluster}

def PROPENSITY_OK(propensity: float, lower: float, upper: float) -> bool:
    return lower <= propensity <= upper and 0.05 <= propensity <= 0.95

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "CDSS API running", "version": "1.0"}

@app.post("/predict")
def predict(p: PatientInput):
    arr = to_array(p)
    pred = predict_one(arr)
    cluster = get_cluster(p)
    cluster_info = META["cluster_summary"][str(cluster)]
    return {
        **pred,
        "cluster": cluster,
        "cluster_info": cluster_info,
        "model_used": META["best_model"]
    }

@app.post("/counterfactual")
def counterfactual(p: PatientInput):
    original_arr  = to_array(p)
    original_pred = predict_one(original_arr)

    scenarios = []

    # Scenario 1 — Improve adherence
    p1 = p.model_copy()
    p1.adherence = 1
    p1.diet      = 1
    p1.exercise  = 1
    arr1  = to_array(p1)
    pred1 = predict_one(arr1)
    scenarios.append({
        "name": "Scenario 1: Improve Lifestyle Adherence",
        "description": "Predictive what-if analysis: set lifestyle/adherence fields to observed positive states; this is not a causal treatment effect or dosing recommendation.",
        "changes": {"adherence": 1, "diet": 1, "exercise": 1},
        "prediction": pred1,
        "delta_confidence": round(pred1["prob_effective"] - original_pred["prob_effective"], 1),
        "recommendation": "Predictive what-if analysis only; this is not a causal treatment effect or dosing recommendation."
    })

    # Scenario 2 — predictive regimen-state activation (not a dose escalation)
    p2 = p.model_copy()
    p2.metformin_v1    = 1 if p.metformin_v1 == 0 else p.metformin_v1
    p2.sulfonylurea_v1 = 1 if p.sulfonylurea_v1 == 0 else p.sulfonylurea_v1
    arr2  = to_array(p2)
    pred2 = predict_one(arr2)
    scenarios.append({
        "name": "Scenario 2: Activate Baseline Regimen States",
        "description": "Set absent Metformin and Sulfonylurea regimen states to present for a predictive what-if comparison",
        "changes": {
            "metformin_v1": p2.metformin_v1,
            "sulfonylurea_v1": p2.sulfonylurea_v1
        },
        "prediction": pred2,
        "delta_confidence": round(pred2["prob_effective"] - original_pred["prob_effective"], 1),
        "recommendation": "Predictive what-if comparison only; this is not a causal treatment effect or dosing recommendation."
    })

    # Scenario 3 — Combination therapy
    p3 = p.model_copy()
    p3.metformin_v1    = 1 if p.metformin_v1 == 0 else p.metformin_v1
    p3.dpp4_v1         = max(p.dpp4_v1, 1)
    p3.sglt2_v1        = max(p.sglt2_v1, 1)
    p3.adherence       = 1
    arr3  = to_array(p3)
    pred3 = predict_one(arr3)
    scenarios.append({
        "name": "Scenario 3: Combination Therapy + Full Adherence",
        "description": "Baseline regimen-state activation combined with DPP-4, SGLT-2, and adherence changes for a predictive what-if comparison",
        "changes": {
            "metformin_v1": p3.metformin_v1,
            "dpp4_v1": p3.dpp4_v1,
            "sglt2_v1": p3.sglt2_v1,
            "adherence": 1
        },
        "prediction": pred3,
        "delta_confidence": round(pred3["prob_effective"] - original_pred["prob_effective"], 1),
        "recommendation": "Predictive what-if comparison only; it is not a causal effect estimate or treatment recommendation."
    })

    return {
        "original": original_pred,
        "scenarios": scenarios,
        "best_scenario": max(scenarios, key=lambda s: s["prediction"]["prob_effective"])["name"]
    }

@app.get("/clusters")
def clusters():
    return {
        "optimal_k": META["optimal_k"],
        "cluster_summary": META["cluster_summary"]
    }

@app.get("/insights")
def insights():
    return {
        "model_results": META["results"],
        "best_model": META["best_model"],
        "feature_importance": META["feature_importance"],
        "dataset_size": META["dataset_size"],
        "class_distribution": META["class_distribution"]
    }

@app.post("/explain")
def explain(p: PatientInput):
    if SHAP_META is None:
        raise HTTPException(503, "SHAP artifacts are unavailable. Regenerate shap_metadata.json.")
    return {**explain_patient(best_model, FEATURES, p.model_dump()), "global_feature_importance": SHAP_META["global_mean_abs_shap"]}

@app.post("/predictive-what-if")
def predictive_what_if(p: PatientInput):
    """Alias retained separately from causal counterfactual analysis."""
    return counterfactual(p)

@app.post("/causal/ate")
def causal_ate(request: TreatmentRequest):
    artifact = causal_artifact(request.treatment)
    ci_lower, ci_upper = artifact["ci"]
    return {
        "treatment": request.treatment, "ate_estimate": round(float(artifact["ate_estimate"]), 6),
        "ci_lower": round(float(ci_lower), 6), "ci_upper": round(float(ci_upper), 6),
        "treated_count": artifact["treated_count"], "untreated_count": artifact["untreated_count"],
        "overlap_status": artifact["overlap"]["status"], "overlap": artifact["overlap"],
        "estimator": CAUSAL_META["estimator"],
        "interpretation": "Estimated add-on effect on probability of effectiveness for drug presence versus absence, conditional on baseline covariates and other baseline drug presence.",
        "limitations": CAUSAL_META["warnings"],
    }

@app.post("/causal/cate")
def causal_cate(request: CausalPatientRequest):
    result = causal_patient_result(request, request.treatment)
    artifact = result["artifact"]
    cluster_stats = artifact["cluster_cate"][str(result["cluster"])]
    return {
        "treatment": request.treatment, "patient_cate": round(result["patient_cate"], 6),
        "cluster_id": result["cluster"], "cluster_mean_cate": cluster_stats["mean_cate"],
        "ci_lower": None, "ci_upper": None, "positivity_flag": result["positivity_flag"],
        "propensity_score": round(result["propensity"], 6), "overlap_region": artifact["overlap"]["overlap_region"],
        "interpretation": "Estimated conditional add-on effect for patients with similar baseline characteristics; no individual confidence interval is supplied by this X-Learner implementation.",
        "limitations": CAUSAL_META["warnings"],
    }

def recommendation_evidence(p: PatientInput) -> dict:
    """Build a compact, calculated-evidence-only payload for Gemini."""
    prediction = predict(p)
    explanation = explain(p)
    causal_evidence = []
    for treatment in DRUGS:
        ate = causal_ate(TreatmentRequest(treatment=treatment))
        cate = causal_cate(CausalPatientRequest(**p.model_dump(), treatment=treatment))
        causal_evidence.append({
            "drug": treatment,
            "patient_cate": cate["patient_cate"],
            "population_ate": ate["ate_estimate"],
            "ci": [ate["ci_lower"], ate["ci_upper"]],
            "positivity_flag": cate["positivity_flag"],
            "interpretation": cate["interpretation"],
        })
    return {
        "patient_profile": {
            "age": p.age, "bmi": p.bmi, "hba1c_v1": p.hba1c_v1,
            "glucose_v1": p.glucose_v1, "bp_v1": p.bp_v1,
            "adherence": p.adherence, "diet": p.diet, "exercise": p.exercise,
        },
        "prediction": {
            "label": "Effective" if prediction["effective"] else "Not Effective",
            "effectiveness_probability": prediction["prob_effective"] / 100,
            "model": prediction["model_used"],
        },
        "shap": [
            {"feature": item["feature"], "value": item["value"],
             "shap_value": item["shap_value"], "direction": item["direction"]}
            for item in explanation["feature_contributions"][:5]
        ],
        "cluster": {"id": prediction["cluster"], "summary": prediction["cluster_info"]},
        "causal_evidence": causal_evidence,
        "limitations": [
            "Cause2Cure is a research prototype using synthetic data.",
            "Predictive and SHAP outputs are associations, not causal effects.",
            "Causal estimates are model-based and do not encode safety, contraindications, or prescribing rules.",
        ],
    }

@app.post("/recommendations")
def ai_recommendations(p: PatientInput):
    """Return five validated, evidence-grounded Gemini decision-support items."""
    if generate_recommendations is None:
        raise HTTPException(503, "AI recommendations are unavailable because the Gemini integration is not installed.")
    try:
        result = generate_recommendations(recommendation_evidence(p))
    except RuntimeError as exc:
        raise HTTPException(503, "AI recommendations are currently unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(502, "AI recommendations returned an invalid response.") from exc
    recommendations = result.get("recommendations") if isinstance(result, dict) else None
    if not isinstance(recommendations, list) or len(recommendations) != 5:
        raise HTTPException(502, "AI recommendations returned an invalid response.")
    return {
        "recommendations": recommendations,
        "disclaimer": "AI-assisted decision support for a synthetic-data research prototype; not a prescription or treatment recommendation.",
    }

@app.post("/causal/counterfactual")
def causal_counterfactual(request: CausalCounterfactualRequest):
    result = causal_patient_result(request, request.treatment)
    current_state = int(getattr(request, f"{request.treatment}_v1") > 0)
    effect = result["patient_cate"] * (request.alternative_state - current_state)
    return {
        "treatment": request.treatment, "current_state": current_state, "alternative_state": request.alternative_state,
        "estimated_effect": round(float(effect), 6), "ci_lower": None, "ci_upper": None,
        "positivity_flag": result["positivity_flag"], "cluster_id": result["cluster"],
        "interpretation": "Estimated conditional treatment effect for patients with similar baseline characteristics.",
        "disclaimer": "This is not the exact outcome this patient would have experienced. It is a model-based conditional causal estimate under standard causal assumptions.",
    }

@app.post("/causal/recommendation")
def causal_recommendation(p: PatientInput):
    """Evidence display for all candidate add-on effects; deliberately not a drug ranking."""
    predictive = predict_one(to_array(p))
    candidates = []
    for treatment in DRUGS:
        result = causal_patient_result(p, treatment); artifact = result["artifact"]
        cluster_stats = artifact["cluster_cate"][str(result["cluster"])]
        candidates.append({"type": "causal_recommendation", "treatment": treatment,
                           "estimated_effect": round(result["patient_cate"], 6),
                           "ate_estimate": round(float(artifact["ate_estimate"]), 6),
                           "confidence_interval": [round(float(v), 6) for v in artifact["ci"]],
                           "positivity_flag": result["positivity_flag"], "cluster": result["cluster"],
                           "cluster_mean_cate": cluster_stats["mean_cate"],
                           "basis": "Baseline-only AIPW ATE plus EconML X-Learner conditional estimate; other baseline drug presence is adjusted for.",
                           "limitations": "Research prototype: no safety, contraindication, dose, or prescribing rules are encoded."})
    return {"predictive_model_output": predictive, "candidates": candidates,
            "disclaimer": "This clinical decision-support prototype presents estimated effects and uncertainty; it does not prescribe treatment or guarantee benefit."}

@app.get("/causal/summary")
def causal_summary():
    if CAUSAL_META is None:
        raise HTTPException(503, "Causal artifacts are unavailable. Run train_causal.py.")
    return CAUSAL_META

@app.get("/causal/clusters")
def causal_clusters():
    if CAUSAL_META is None:
        raise HTTPException(503, "Causal artifacts are unavailable. Run train_causal.py.")
    return {drug: info["cluster_cate"] for drug, info in CAUSAL_META["treatments"].items()}

@app.get("/causal/refutation")
def causal_refutation():
    if CAUSAL_META is None:
        raise HTTPException(503, "Causal artifacts are unavailable. Run train_causal.py.")
    return CAUSAL_META.get("refutations", {"status": "unavailable", "reason": "Refutation artifacts were not generated."})

@app.get("/plot/{name}")
def get_plot(name: str):
    path = f"{MODELS}/plots/{name}.png"
    if not os.path.exists(path):
        raise HTTPException(404, "Plot not found")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image": f"data:image/png;base64,{data}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

"""Causal/explainability artifacts for the Cause2Cure research prototype.

Drug codes are regimen-state codes.  Treatments are deliberately binary
"drug present at baseline" indicators (drug_v1 > 0), never ordinal doses.
All static adjustment variables are baseline variables; v2/v3 values are
downstream and are intentionally excluded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes_drug_effectiveness.csv"
CAUSAL_DIR = ROOT / "models" / "causal"
DRUGS = ("metformin", "sulfonylurea", "dpp4", "sglt2", "insulin")
BASELINE_COVARIATES = (
    "age", "gender", "bmi", "hba1c_v1", "glucose_v1", "bp_v1",
    "adherence", "diet", "exercise",
)
POST_TREATMENT_FEATURES = tuple(
    f"{name}_v{visit}" for name in ("hba1c", "glucose", "bp", *DRUGS)
    for visit in (2, 3)
)
PROPENSITY_CLIP = 0.05
OVERLAP_QUANTILE = 0.01
BOOTSTRAP_REPS = 400
RANDOM_STATE = 42


def treatment_column(drug: str) -> str:
    if drug not in DRUGS:
        raise ValueError(f"Unknown treatment: {drug}")
    return f"{drug}_v1"


def treatment_indicator(df: pd.DataFrame, drug: str) -> pd.Series:
    """Binary add-on treatment: 1 iff the baseline drug regimen is present."""
    return (df[treatment_column(drug)] > 0).astype(int)


def causal_adjustment_features(drug: str) -> list[str]:
    """Baseline-only covariates plus other-drug binary presence indicators."""
    treatment_column(drug)
    return list(BASELINE_COVARIATES) + [f"{other}_present" for other in DRUGS if other != drug]


def causal_frame(df: pd.DataFrame, drug: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = pd.DataFrame(index=df.index)
    for col in BASELINE_COVARIATES:
        work[col] = df[col]
    for other in DRUGS:
        if other != drug:
            work[f"{other}_present"] = treatment_indicator(df, other)
    # Raw target 0 means effective in the canonical dataset; reverse only in memory.
    outcome = (df["treatment_effectiveness"] == 0).astype(int).to_numpy()
    treatment = treatment_indicator(df, drug).to_numpy()
    return work[causal_adjustment_features(drug)], treatment, outcome


def _rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=160, min_samples_leaf=12, random_state=seed, n_jobs=-1)


def _propensity_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def _crossfit_aipw(x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five-fold out-of-fold doubly robust scores and propensity estimates."""
    score = np.zeros(len(y)); propensity = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train, test) in enumerate(folds.split(x, t)):
        xtr, xte, ttr, ytr = x.iloc[train], x.iloc[test], t[train], y[train]
        prop = _propensity_model().fit(xtr, ttr)
        e = np.clip(prop.predict_proba(xte)[:, 1], PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)
        m0 = _rf(RANDOM_STATE + fold).fit(xtr[ttr == 0], ytr[ttr == 0])
        m1 = _rf(RANDOM_STATE + 10 + fold).fit(xtr[ttr == 1], ytr[ttr == 1])
        mu0, mu1 = m0.predict(xte), m1.predict(xte)
        tt, yy = t[test], y[test]
        score[test] = mu1 - mu0 + tt * (yy - mu1) / e - (1 - tt) * (yy - mu0) / (1 - e)
        propensity[test] = e
    return score, propensity


def _bootstrap_ci(scores: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    means = [scores[rng.integers(0, len(scores), len(scores))].mean() for _ in range(BOOTSTRAP_REPS)]
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def _balance(x: pd.DataFrame, t: np.ndarray, e: np.ndarray) -> dict:
    result = {}
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        pooled = np.sqrt((values[t == 1].var() + values[t == 0].var()) / 2) or 1.0
        before = (values[t == 1].mean() - values[t == 0].mean()) / pooled
        wt, wc = 1 / e[t == 1], 1 / (1 - e[t == 0])
        after = (np.average(values[t == 1], weights=wt) - np.average(values[t == 0], weights=wc)) / pooled
        result[col] = {"smd_before": round(float(before), 4), "smd_ipw": round(float(after), 4)}
    return result


def _overlap(t: np.ndarray, e: np.ndarray) -> dict:
    treated, control = e[t == 1], e[t == 0]
    lower = max(float(np.quantile(treated, OVERLAP_QUANTILE)), float(np.quantile(control, OVERLAP_QUANTILE)))
    upper = min(float(np.quantile(treated, 1 - OVERLAP_QUANTILE)), float(np.quantile(control, 1 - OVERLAP_QUANTILE)))
    return {
        "treatment_prevalence": round(float(t.mean()), 4),
        "propensity_range": [round(float(e.min()), 4), round(float(e.max()), 4)],
        "overlap_region": [round(lower, 4), round(upper, 4)],
        "status": "adequate" if lower < upper else "weak",
        "positivity_threshold": PROPENSITY_CLIP,
    }


def _dowhy_refutations(x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> dict:
    """Run DoWhy refuters against a DoWhy PS-weighting estimate, never primary AIPW."""
    try:
        from dowhy import CausalModel
        data = x.copy(); data["T"] = t; data["Y"] = y
        model = CausalModel(data=data, treatment="T", outcome="Y", common_causes=list(x.columns))
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting")
        results = {"status": "available", "estimator_for_refutation_only": "DoWhy backdoor.propensity_score_weighting", "original_estimate": float(estimate.value), "checks": {}}
        for method in ("placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"):
            try:
                refuted = model.refute_estimate(estimand, estimate, method_name=method)
                results["checks"][method] = {"status": "available", "new_effect": float(refuted.new_effect)}
            except Exception as exc:
                results["checks"][method] = {"status": "unavailable", "reason": str(exc)}
        return results
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def train_causal_artifacts() -> dict:
    df = pd.read_csv(DATA_PATH)
    required = {"patient_id", "treatment_effectiveness", *BASELINE_COVARIATES, *(f"{d}_v1" for d in DRUGS)}
    if df.shape != (4200, 32) or not required.issubset(df.columns):
        raise ValueError(f"Canonical dataset validation failed: shape={df.shape}")
    if any(set(df[f"{drug}_v1"].unique()) - {0, 1, 2, 3} for drug in DRUGS):
        raise ValueError("Unexpected baseline drug regimen-state code")
    if set(df.treatment_effectiveness.unique()) != {0, 1}:
        raise ValueError("Unexpected treatment_effectiveness categories")
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)

    cluster_scaler = joblib.load(ROOT / "models" / "cluster_scaler.pkl")
    kmeans = joblib.load(ROOT / "models" / "kmeans.pkl")
    cluster_x = df[["age", "bmi", "hba1c_v1", "glucose_v1", "bp_v1", "adherence"]]
    cluster_ids = kmeans.predict(cluster_scaler.transform(cluster_x))
    metadata = {
        "dataset_path": str(DATA_PATH), "dataset_shape": list(df.shape),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "treatment_definition": "T_drug = 1 if drug_v1 > 0, else 0; add-on presence effect, not dose intensity.",
        "outcome_definition": "effective = 1 when raw treatment_effectiveness == 0; raw data is unchanged.",
        "adjustment_features": {drug: causal_adjustment_features(drug) for drug in DRUGS},
        "excluded_post_treatment_features": list(POST_TREATMENT_FEATURES),
        "estimator": "5-fold cross-fitted augmented inverse probability weighting (AIPW/doubly robust)",
        "cate_method": "EconML XLearner with random-forest outcome and CATE models",
        "treatments": {},
        "warnings": ["Synthetic dataset; data-generation process is not verified.", "Effects require consistency, exchangeability/no unmeasured confounding, and positivity.", "Individual conditional effects are not ground-truth individual outcomes."],
        "refutations": {},
    }
    for drug in DRUGS:
        x, t, y = causal_frame(df, drug)
        scores, e_oof = _crossfit_aipw(x, t, y)
        ate = float(scores.mean()); ci = _bootstrap_ci(scores)
        prop_full = _propensity_model().fit(x, t)
        # One CATE model per drug, trained on all patients; never per cluster.
        xlearner = XLearner(models=[_rf(100), _rf(101)], propensity_model=_propensity_model(), cate_models=[_rf(102), _rf(103)])
        xlearner.fit(y, t, X=x)
        cate = xlearner.effect(x)
        overlap = _overlap(t, e_oof)
        cluster_summary = {}
        for cluster in sorted(np.unique(cluster_ids)):
            values = cate[cluster_ids == cluster]
            cluster_summary[str(int(cluster))] = {"mean_cate": round(float(values.mean()), 4), "median_cate": round(float(np.median(values)), 4), "sample_count": int(len(values))}
        artifact = {"drug": drug, "features": list(x.columns), "propensity_model": prop_full, "cate_model": xlearner,
                    "ate_estimate": ate, "ci": ci, "overlap": overlap, "cluster_cate": cluster_summary,
                    "treated_count": int(t.sum()), "untreated_count": int((1 - t).sum()), "balance": _balance(x, t, e_oof)}
        joblib.dump(artifact, CAUSAL_DIR / f"{drug}_causal.pkl")
        metadata["treatments"][drug] = {k: v for k, v in artifact.items() if k not in {"propensity_model", "cate_model", "features", "balance"}}
        metadata["treatments"][drug]["adjustment_features"] = list(x.columns)
        metadata["treatments"][drug]["balance"] = artifact["balance"]
        metadata["refutations"][drug] = _dowhy_refutations(x, t, y)
    with open(CAUSAL_DIR / "causal_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata

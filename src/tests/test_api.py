import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from api import (PatientInput, CausalCounterfactualRequest, CausalPatientRequest,
                 TreatmentRequest, causal_ate, causal_cate, causal_counterfactual,
                 causal_recommendation, causal_refutation, clusters, counterfactual, explain, insights, predict, to_array, best_model,
                 ai_recommendations)
from causal import CAUSAL_DIR, DRUGS, POST_TREATMENT_FEATURES, causal_adjustment_features, treatment_indicator


PATIENT = {
    "age": 57, "gender": 1, "bmi": 26.5,
    "hba1c_v1": 7.4, "glucose_v1": 195, "bp_v1": 132,
    "metformin_v1": 1, "sulfonylurea_v1": 1,
    "dpp4_v1": 1, "sglt2_v1": 0, "insulin_v1": 0,
    "adherence": 0, "diet": 0, "exercise": 0,
}


class ApiTests(unittest.TestCase):
    def test_predict_returns_model_and_cluster_result(self):
        body = predict(PatientInput(**PATIENT))
        self.assertTrue({"label", "prob_effective", "cluster", "cluster_info"} <= body.keys())
        self.assertTrue(0 <= body["cluster"] < 4)

    def test_existing_analysis_endpoints_return_data(self):
        self.assertIn("scenarios", counterfactual(PatientInput(**PATIENT)))
        self.assertIn("cluster_summary", clusters())
        self.assertIn("model_results", insights())

    def test_invalid_categories_are_rejected(self):
        invalid = {**PATIENT, "adherence": 2}
        with self.assertRaises(ValidationError):
            PatientInput(**invalid)
        with self.assertRaises(ValidationError):
            PatientInput()

    def test_explainability_and_all_causal_drugs(self):
        explanation = explain(PatientInput(**PATIENT))
        self.assertIn("feature_contributions", explanation)
        for drug in DRUGS:
            self.assertEqual(causal_ate(TreatmentRequest(treatment=drug))["treatment"], drug)
            cate = causal_cate(CausalPatientRequest(**PATIENT, treatment=drug))
            self.assertIn("cluster_mean_cate", cate)
            counter = causal_counterfactual(CausalCounterfactualRequest(**PATIENT, treatment=drug, alternative_state=1))
            self.assertIn("estimated_effect", counter)
        self.assertEqual(len(causal_recommendation(PatientInput(**PATIENT))["candidates"]), len(DRUGS))
        self.assertEqual(set(causal_refutation()), set(DRUGS))

    def test_explain_probability_matches_the_predicted_class(self):
        patient = PatientInput(**PATIENT)
        prediction = predict(patient)
        explanation = explain(patient)
        expected = float(best_model.predict_proba(to_array(patient))[0, prediction["label"]])
        self.assertEqual(prediction["label"], 1)  # Regression patient: catches the former class-0-only bug.
        self.assertAlmostEqual(explanation["predicted_probability"], expected, places=6)

    def test_causal_leakage_and_binary_treatment_definition(self):
        features = [feature for drug in DRUGS for feature in causal_adjustment_features(drug)]
        self.assertFalse(any(feature.endswith("_v2") or feature.endswith("_v3") for feature in features))
        self.assertTrue(set(POST_TREATMENT_FEATURES))
        import pandas as pd
        frame = pd.DataFrame({"metformin_v1": [0, 1, 2, 3]})
        self.assertEqual(treatment_indicator(frame, "metformin").tolist(), [0, 1, 1, 1])

    @patch("api.generate_recommendations")
    def test_ai_recommendations_are_five_mocked_grounded_items(self, mocked_generate):
        items = [{"title": f"Item {i}", "recommendation": "Review supplied evidence.",
                  "evidence": "A supplied model result.", "evidence_type": "Integrated"} for i in range(1, 6)]
        mocked_generate.return_value = {"recommendations": items}
        result = ai_recommendations(PatientInput(**PATIENT))
        self.assertEqual(result["recommendations"], items)
        evidence = mocked_generate.call_args.args[0]
        self.assertEqual(len(evidence["causal_evidence"]), len(DRUGS))
        self.assertEqual(evidence["prediction"]["label"], "Not Effective")
        self.assertNotIn("dataframe", evidence)

    @patch("api.generate_recommendations", side_effect=RuntimeError("Gemini unavailable"))
    def test_ai_recommendation_service_failure_is_clean(self, _mocked_generate):
        with self.assertRaises(HTTPException) as context:
            ai_recommendations(PatientInput(**PATIENT))
        self.assertEqual(context.exception.status_code, 503)

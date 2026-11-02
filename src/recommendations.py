
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def _get_client():
    """Create the Gemini client only when the recommendation endpoint is used."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


# ============================================================
# LLM Prompt
# ============================================================

SYSTEM_PROMPT = """
You are the AI-assisted recommendation interpretation layer
of Cause2Cure, a research prototype for analyzing diabetes
treatment effectiveness.

Your task is to generate EXACTLY 5 explicit, evidence-grounded
decision-support recommendations from the supplied patient evidence.

The predictive model, SHAP analysis, and causal analysis have
already been performed. You are NOT responsible for performing
new calculations or generating new clinical evidence.

STRICT RULES:

1. Use ONLY information explicitly supplied in PATIENT EVIDENCE.

2. Do not invent patient characteristics, laboratory values,
   diagnoses, symptoms, medications, contraindications, risks,
   or clinical history.

3. Do not calculate new statistics or modify supplied statistics.

4. Do not prescribe medication.

5. Do not recommend starting, stopping, switching, increasing,
   or decreasing a medication.

6. Do not recommend dosage or dosage changes.

7. Do not invent medical guidelines or safety rules.

8. Do not claim that any drug is clinically superior or "best".

9. Do not rank the drugs.

10. Do not treat predictive associations as causal effects.

11. Clearly distinguish predictive, SHAP, patient-profile,
    and causal evidence.

12. Causal estimates are model-based estimates obtained from
    synthetic data and must not be presented as established
    clinical efficacy.

13. A drug may be mentioned only in the context of reviewing
    the supplied causal evidence for that evaluated option.

14. Recommendations must be actionable as items for clinician
    review, investigation, or interpretation.

15. Every recommendation must cite the specific evidence that
    supports it.

16. Do not make claims about information that is not present
    in the supplied evidence.

17. Generate exactly 5 recommendations.

18. Make the five recommendations meaningfully different.
    Do not repeat the same recommendation using different wording.

GOOD recommendation language includes:

"Review the patient's predicted treatment response..."

"Review adherence as a contributor to the model's prediction..."

"Review the patient's baseline glycaemic profile..."

"Review the causal evidence for metformin..."

"Consider the predictive and causal evidence together during
clinical review..."

AVOID:

"Prescribe metformin."

"Start metformin."

"Switch the patient to metformin."

"Metformin is the best treatment."

"Metformin will improve the patient's condition."

"Increase the dose."

"Metformin is clinically proven to work for this patient."

Remember:

Cause2Cure is a research prototype using synthetic data.

Your output is decision-support interpretation,
NOT a prescription.

Return ONLY valid JSON using exactly this structure:

{
  "recommendations": [
    {
      "title": "short descriptive title",
      "recommendation": "explicit action for clinical review",
      "evidence": "specific supplied evidence supporting the recommendation",
      "evidence_type": "Predictive | SHAP | Causal | Patient Profile | Integrated"
    }
  ]
}
"""


# ============================================================
# Recommendation Generator
# ============================================================

def generate_recommendations(evidence):
    """
    Send structured Cause2Cure evidence to Gemini and return
    exactly five validated recommendation objects.
    """

    client = _get_client()
    prompt = (
        SYSTEM_PROMPT
        + "\n\nPATIENT EVIDENCE:\n"
        + json.dumps(
            evidence,
            indent=2,
            default=str
        )
    )

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
    ]

    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"Trying {model_name}...")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            print(f"Using {model_name}")
            break

        except Exception as exc:
            last_error = exc
            print(f"{model_name} unavailable: {exc}")

    if response is None:
        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_error}"
        )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    text = response.text.strip()

    # Gemini may occasionally wrap JSON in markdown fences.
    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "", 1).strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned invalid JSON:\n{text}"
        ) from exc

    recommendations = result.get("recommendations")

    if not isinstance(recommendations, list):
        raise ValueError(
            "Gemini response does not contain a recommendations list."
        )

    if len(recommendations) != 5:
        raise ValueError(
            f"Expected exactly 5 recommendations, "
            f"got {len(recommendations)}."
        )

    required_fields = {
        "title",
        "recommendation",
        "evidence",
        "evidence_type"
    }

    allowed_evidence_types = {
        "Predictive",
        "SHAP",
        "Causal",
        "Patient Profile",
        "Integrated"
    }

    for index, item in enumerate(recommendations, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Recommendation {index} is not a JSON object."
            )

        missing_fields = required_fields - set(item.keys())

        if missing_fields:
            raise ValueError(
                f"Recommendation {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if item["evidence_type"] not in allowed_evidence_types:
            raise ValueError(
                f"Recommendation {index} has invalid evidence_type: "
                f"{item['evidence_type']}"
            )

        for field in required_fields:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(
                    f"Recommendation {index} has an empty/invalid "
                    f"'{field}' field."
                )

    return result


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    # This is temporary test evidence.
    # Later this will be generated dynamically from the
    # actual Cause2Cure API outputs.

    test_evidence = {

        "patient": {
            "age": 57,
            "bmi": 26.5,
            "hba1c": 7.4,
            "glucose": 195,
            "bp": 132
        },

        "prediction": {
            "outcome": "Not Effective",
            "effectiveness_probability": 0.273,
            "model": "RandomForest"
        },

        "shap": [
            {
                "feature": "adherence",
                "effect": -0.1007
            },
            {
                "feature": "hba1c_v1",
                "effect": 0.0978
            },
            {
                "feature": "metformin_v1",
                "effect": -0.0837
            },
            {
                "feature": "diet",
                "effect": -0.0591
            },
            {
                "feature": "exercise",
                "effect": -0.0415
            }
        ],

        "causal_evidence": [

            {
                "drug": "metformin",
                "patient_cate_pp": 28.7,
                "population_ate_pp": 19.1,
                "ci_pp": [14.2, 23.7],
                "positivity": "within_support"
            },

            {
                "drug": "sulfonylurea",
                "patient_cate_pp": 5.9,
                "population_ate_pp": 7.0,
                "ci_pp": [4.4, 9.7],
                "positivity": "within_support"
            },

            {
                "drug": "dpp4",
                "patient_cate_pp": 8.3,
                "population_ate_pp": 7.5,
                "ci_pp": [5.1, 10.2],
                "positivity": "within_support"
            },

            {
                "drug": "sglt2",
                "patient_cate_pp": 10.9,
                "population_ate_pp": 7.4,
                "ci_pp": [4.8, 10.0],
                "positivity": "within_support"
            },

            {
                "drug": "insulin",
                "patient_cate_pp": 17.6,
                "population_ate_pp": 4.3,
                "ci_pp": [-0.1, 9.4],
                "positivity": "within_support"
            }
        ]
    }

    print("\nGenerating recommendations...\n")

    result = generate_recommendations(test_evidence)

    print(json.dumps(result, indent=2))


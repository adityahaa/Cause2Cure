# Synthetic dataset

**File:** [diabetes_drug_effectiveness.csv](../src/data/diabetes_drug_effectiveness.csv). Original bytes are retained; 4,200 rows and 32 columns. The manuscript identifies the cohort as synthetic, and the repository owner describes it as synthetic. The generation code, external origin and generation parameters were not supplied.

Static review found numeric columns only and sequential record IDs 1–4,200. There are no name, contact, address or free-text fields. These are synthetic record identifiers, not verified real patient identifiers. No separate clinician registry or real patient records are included.

| Fields | Meaning in supplied files |
| --- | --- |
| `patient_id` | Sequential synthetic record ID; excluded from modelling |
| `age`, `gender`, `bmi` | Demographics; original gender encoding 0=female, 1=male |
| `adherence`, `diet`, `exercise` | Binary behavioural indicators |
| `hba1c_v1` … `hba1c_v3` | HbA1c measurements across three visits |
| `glucose_v1` … `glucose_v3` | Glucose measurements across three visits |
| `bp_v1` … `bp_v3` | Blood-pressure measurements across three visits |
| `{metformin,sulfonylurea,dpp4,sglt2,insulin}_v1` … `_v3` | Drug-class regimen-state codes, 0–3; not doses |
| `treatment_effectiveness` | Supplied target: 0=Effective, 1=Not Effective |

The manuscript describes a third-visit target glycaemic outcome; the exact synthetic label-generation rule is unavailable. Do not infer a validated clinical endpoint from the column name.

Prediction uses age, gender, BMI, first-visit HbA1c/glucose/BP, five first-visit regimen fields and the three behavioural indicators (14 features). Clustering uses age, BMI, first-visit HbA1c/glucose/BP and adherence (six features). Causal estimation reverses the raw target in memory so that effective=1, and represents each drug by binary baseline presence while adjusting for other baseline drugs.

The manuscript's 840-record holdout is not supplied as a separate index file. The ZIP's predictive code uses `test_size=0.2`, `random_state=42`, stratified by target. No split was executed or regenerated during publication. The 11 additional pending registry entries mentioned in the manuscript are absent from the ZIP and this repository.

No dataset licence was supplied. Publication at the project owner's request does not grant a new data reuse licence; see [RIGHTS.md](../RIGHTS.md).

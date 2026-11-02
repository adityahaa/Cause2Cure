# Cause2Cure: short technical report

## Purpose and evidence

This report summarises the supplied seven-page IEEE-format manuscript, final 27-slide presentation and archived implementation. It is an archival synthesis, not a new experiment. Publication status is conference accepted and presented; proceedings publication in progress, according to the team. The manuscript PDF is withheld pending redistribution permission.

The research examines an interface that separates treatment-response prediction, phenotype context, model attribution, predictive scenarios and observational causal estimation for Type 2 diabetes. All quantitative findings below arise from synthetic data.

## Data and architecture

The development cohort comprises 4,200 synthetic records. The manuscript reports a stratified frozen holdout of 840 and a development partition of 3,360. The source CSV contains 32 numeric columns covering synthetic IDs, demographics, behavioural indicators, three visits of measurements and regimen states, and a binary response label. Fourteen first-visit/baseline features feed the predictive classifier. The paper also describes 11 pending registry entries; those records are not supplied or published here.

A React/Chart.js dashboard communicates with FastAPI. Python modules provide prediction, K-Means assignment, SHAP explanations, rule-based scenarios, AIPW and EconML X-Learner estimation. The original conceptual architecture is reproduced from presentation slide 13, with its original wording retained. It should be interpreted alongside the implementation boundaries rather than as evidence of clinical validation.

## Methods and evaluation boundaries

The paper describes a calibrated Random Forest, Borderline-SMOTE within development folds, randomised five-fold search, an F1-selected threshold, an XGBoost challenger, DiCE-supported scenarios and an outcome-verified learning registry. The ZIP instead contains fixed Random Forest/Gradient Boosting models, ordinary SMOTE, raw tree inputs, standard classifier decisions and rule-based what-if changes. It does not contain the described DiCE or registry implementation. These differences are preserved and documented rather than silently reconciled.

K-Means uses four clusters over scaled age, BMI, baseline HbA1c, glucose, blood pressure and adherence. SHAP TreeExplainer attributes the predictive model's output; forest feature importance is a separate, global measure.

For causal estimation, baseline presence of each drug class defines a binary add-on contrast, adjusting for measured baseline patient factors and other drug-class presence. The code uses five-fold cross-fitted augmented inverse-probability weighting for the population benchmark, a bootstrap interval, and one EconML X-Learner per drug for conditional effects. Follow-up visit variables are excluded. DoWhy refutations target a separate propensity-weighting estimate, not the primary AIPW estimator. Positivity checks and refutations do not prove absence of unmeasured confounding.

## Existing results

| Measure | Random Forest, manuscript Table III |
| --- | ---: |
| ROC-AUC | 0.7341 |
| Accuracy | 69.9% |
| Class-1 precision | 52.9% |
| Class-1 recall | 55.0% |
| Class-1 F1 | 0.539 |
| Specificity | 76.9% |
| Balanced accuracy | 66.0% |
| MCC | 0.316 |
| Majority-class accuracy baseline | 68.0% |

The class-1 label in the supplied predictive metadata is “Not Effective.” The saved Random Forest matrix is TN=439, FP=132, FN=121, TP=148 relative to that label. Gradient Boosting metadata records accuracy 0.7036 and ROC-AUC 0.7302; presentation slide 14 reports rounded accuracy 70.4%. No supplied model evaluation supports a claim of 71% accuracy.

The manuscript's index-patient example reports baseline predicted effectiveness 22.9%, lifestyle/adherence scenario 71.0% (+48.1 percentage points), regimen-state scenario 23.3% (+0.4 points), and combination/adherence scenario 51.7% (+28.8 points). These are individual model probabilities, not accuracy or measured intervention outcomes.

For that example, the manuscript reports insulin CATE +9.7 percentage points and population AIPW +4.3 points with interval [0.1, 9.4] points. The ZIP metadata records AIPW 0.0430305 with interval [-0.0012572, 0.0937079], approximately **[-0.13, 9.37] percentage points**, which includes zero. The lower-bound discrepancy is unresolved. Preserve the distinction; do not treat the paper's interval as independently reproduced. The population interval is not an uncertainty interval for the individual CATE.

## Interpretation and limitations

The results demonstrate a synthetic research workflow with modest predictive discrimination. They do not establish clinical effectiveness or transportability to real populations. The generation procedure is unavailable; causal estimates require consistency, overlap and no unmeasured confounding. The single worked patient does not validate heterogeneous effects against causal ground truth. Comprehensive fairness, prospective evaluation and real-world treatment-impact evidence are unavailable.

The ZIP chooses its best model using holdout ROC-AUC, unlike the manuscript's description of an untouched evaluation set. Clustering and causal estimation use the full supplied cohort; the predictive holdout does not establish independent validation for those modules. Calibration and broader lifecycle claims cannot be confirmed from this source snapshot.

Only static document, source, metadata and figure review was performed for publication. No application code, model loading, training, tests, experiments or deployment was performed.

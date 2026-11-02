# Implementation boundaries and archival changes

The manuscript and presentation describe a broader system than the supplied ZIP. This repository retains the supplied snapshot rather than developing missing components.

| Topic | Manuscript / presentation | Supplied ZIP |
| --- | --- | --- |
| Challenger | Manuscript: XGBoost; presentation: Gradient Boosting | scikit-learn GradientBoostingClassifier |
| Oversampling | Manuscript: Borderline-SMOTE in development folds | Ordinary SMOTE after one stratified split |
| Model selection | Manuscript: five-fold randomised search; holdout reserved | Fixed parameters; winner chosen using holdout ROC-AUC |
| Calibration / threshold | Manuscript: post-hoc calibration and F1-selected threshold | No calibration or threshold-search implementation found |
| Tree scaling | Manuscript describes training-fit scaling | Trees use raw inputs; separate scaler is retained as a legacy artifact |
| Counterfactuals | Manuscript: DiCE-supported search plus scenarios | Rule-based input edits; no DiCE import or requirement |
| Learning registry | Manuscript: pending outcomes and 50-outcome update gate | No persistent registry or continuous-learning implementation found |
| Fairness | Manuscript mentions Fairlearn hooks; slides call auditing future work | No Fairlearn dependency or subgroup audit implementation found |
| Cohort size | Manuscript and CSV: 4,200 | Dashboard contains a stale static “12,000” label; not evidence of dataset size |
| API port | API entry point: 8000 | Dashboard API constant: localhost:8001; pre-existing mismatch |

## Archival handling

- Original Python source, test source, requirements, numeric synthetic CSV, selected saved plots and JSON metadata are copied byte-for-byte.
- Source and its relative data/model paths are grouped under `src/`. Training and causal entry points remain beside their imported modules.
- The dashboard's hard-coded demonstration username/password and accompanying hints are removed. Its demonstration login is intentionally disabled in this public archive. No research calculations were changed.
- The old README is replaced because it contains demonstration credentials, deployment directions, unsupported wording and outdated file references.
- `.env`, virtual environment, bytecode, logs, pickles, redundant saved models and unselected generated plots are omitted from the publication tree. Original inputs remain outside the repository.
- The complete slide deck is not redistributed: selected source artwork is sufficient, and browser screenshots include incidental personal browsing context. The manuscript PDF is withheld at the owner's request.

No application, tests, model artifacts, training process or experiments were executed. Static inspection cannot establish runtime correctness. See [SETUP.md](SETUP.md) for artifact requirements and existing environment limitations.

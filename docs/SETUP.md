# Environment and existing entry points

These instructions are derived from existing files and were not executed. This repository is an archival source publication, not a working model release.

## Environment preparation

The ZIP includes a Windows Python 3.11 environment, inferred from its environment metadata and installed-package names. The original requirements pin scikit-learn 1.7.2 and SciPy 1.15.3; other requirements mostly use lower bounds. Compatibility and installation have not been checked.

For a future authorised local review, from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

These commands prepare dependencies only. They do not start the application or train models. No lockfile or fully validated environment is supplied. The frontend loads React 18.2.0, ReactDOM 18.2.0, Babel Standalone 7.23.2 and Chart.js 4.4.0 from CDN links; Google Fonts are also referenced.

## Missing fitted artifacts

The original API loads these files at import. They are intentionally excluded as generated binaries:

```text
src/models/best_model.pkl
src/models/RandomForest.pkl
src/models/GradientBoosting.pkl
src/models/cluster_scaler.pkl
src/models/kmeans.pkl
src/models/causal/{metformin,sulfonylurea,dpp4,sglt2,insulin}_causal.pkl
```

The original ZIP also contains a legacy `models/scaler.pkl` that predictive inference does not use. Restoring trusted original fitted artifacts would be necessary for an independent local runtime review; metadata alone cannot replace them. This publication task does not regenerate or load them. Serialized Python artifacts should only be loaded from a trusted source.

## Existing entry points and limitations

| Existing file | Purpose / limitation |
| --- | --- |
| `src/api.py` | FastAPI service; import immediately reads fitted artifacts; main block binds port 8000 |
| `src/dashboard.html` | React frontend; API constant points to port 8001; demonstration login disabled during credential removal |
| `src/train_models.py` | Training and plot generation execute at module top level; do not import for inspection |
| `src/train_causal.py` | Calls causal training immediately; do not import for inspection |
| `src/explainability.py` | SHAP code uses paths relative to the working directory for some operations |
| `src/tests/test_api.py` | Existing unittest source; imports the API and therefore requires fitted artifacts |

The original working-directory convention was the source directory, now `src/`. No startup, training, test or deployment commands are provided as part of this portfolio publication.

## Optional external interpretation service

`recommendations.py` reads `GEMINI_API_KEY` via python-dotenv and the Google GenAI client. The original `.env` is excluded, and no key is provided. This optional path calls an external service; it was not invoked. Patient data must not be submitted through this repository. See [dependency notes](DEPENDENCIES.md).

# Dependency documentation

The root `requirements.txt` is the original declared dependency list, copied unchanged. It is not a lockfile and was not installed or resolved during publication.

| Role | Declared packages |
| --- | --- |
| Numerical and tabular processing | NumPy, pandas, SciPy (1.15.3) |
| Prediction, preprocessing and clustering | scikit-learn (1.7.2), imbalanced-learn |
| Plotting and serialization | Matplotlib, seaborn, joblib |
| API | FastAPI, Uvicorn, python-multipart |
| Explainability and causal methods | SHAP, EconML, DoWhy |
| Optional interpretation service | google-genai, python-dotenv |

The API and tests import Pydantic directly, although it is supplied transitively through FastAPI rather than listed separately. The original tests use the standard-library unittest module. This list has not been upgraded or corrected by this publication task.

The frontend embeds JSX in HTML and references CDN-hosted React 18.2.0, ReactDOM 18.2.0, Babel Standalone 7.23.2, Chart.js 4.4.0 and Google Fonts. There is no supplied npm manifest or local JavaScript build pipeline.

DiCE (`dice-ml`), XGBoost and Fairlearn are mentioned in the manuscript but do not appear in this ZIP's imports or requirements. They are not silently added as dependencies for absent implementations.

Third-party packages, hosted libraries and fonts retain their respective licences. Their code and the bundled virtual environment are not vendored in this repository. No permission to relicense those dependencies is asserted.

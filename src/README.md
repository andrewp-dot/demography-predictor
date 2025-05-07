# Brief description of project structure

This document provides a high-level overview of the project's organization and the purpose of each core component.

## Project Structure Overview

This table summarizes the purpose of each directory and key file in the project.

| Path                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `README.md`             | Project-level documentation.                                                |
| `api/`                  | Implemnation of simple API via FastAPI                                      |
| `cli/`                  | CLI commands and entry points using Click.                                  |
| `compare_models/`       | Logic for comparing multiple models based on evaluation metrics.            |
| `feature_model/`        | Implementation of feature prediction models.                                |
| `target_model/`         | Implementation of target prediction models.                                 |
| `preprocessors/`        | Custom data transformation and preprocessing pipelines and data loaders.    |
| `shap_explainer/`       | SHAP-based model explainability and visualization tools.                    |
| `statistical_models/`   | Classical models (e.g., linear regression, ARIMA, etc.).                    |
| `train_scripts/`        | Training scripts for pipelines (pipeline = Datapreprocessing + Model).      |
| `utils/`                | General utility functions shared across modules.                            |
| `base.py`               | Base classes or interfaces shared by multiple modules.                      |
| `pipeline.py`           | Main logic for definition, creating, saving, and executing ML pipelines.    |
| `evaluation.py`         | Common evaluation metrics and plotting logic.                               |
| `state_groups.py`       | Logic for organizing or grouping data by states (geographic or otherwise).  |

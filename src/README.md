# Brief description of project structure

This document provides a high-level overview of the project's organization and the purpose of each core component.

## Project Structure Overview

This table summarizes the purpose of each directory and key file in the project.

| Path                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `README.md`             | Project-level documentation.                                                |
| `api/`                  | Contains code related to exposing model functionality via API (e.g. FastAPI). |
| `cli/`                  | CLI commands and entry points using Click or argparse.                     |
| `compare_models/`       | Logic for comparing multiple models based on evaluation metrics.            |
| `feature_model/`        | Training and handling of feature prediction models.                         |
| `target_model/`         | Training and handling of target prediction models.                          |
| `preprocessors/`        | Custom data transformation and preprocessing pipelines.                     |
| `shap_explainer/`       | SHAP-based model explainability and visualization tools.                    |
| `statistical_models/`   | Classical models (e.g., linear regression, ARIMA, etc.).                    |
| `train_scripts/`        | One-off or orchestrated training scripts for different model types.         |
| `utils/`                | General utility functions shared across modules.                            |
| `base.py`               | Base classes or interfaces shared by multiple modules.                      |
| `pipeline.py`           | Main logic for creating, saving, and executing ML pipelines.                |
| `evaluation.py`         | Common evaluation metrics and plotting logic.                               |
| `state_groups.py`       | Logic for organizing or grouping data by states (geographic or otherwise).  |

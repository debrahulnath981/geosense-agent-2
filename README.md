# GeoSense Agent 2.0

GeoSense Agent 2.0 is a geospatial machine-learning workflow for identifying and scoring potential sites in the Mumbai Municipal Corporation (MCGM) study area.

## Project Overview

The project integrates spatial data processing, feature engineering, rule-based site labelling, machine learning, SHAP explainability, automated testing, and a reusable site-scoring function.

## Study Area

Mumbai Municipal Corporation (MCGM), Maharashtra, India.

## Technologies Used

- Python 3.11
- PostgreSQL
- PostGIS
- GeoPandas
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Rasterio
- Pytest
- QGIS
- Git and GitHub

## Workflow

1. Connect spatial data using PostgreSQL/PostGIS.
2. Generate candidate locations.
3. Calculate spatial features such as distance to roads and hospitals.
4. Create rule-based Good/Poor site labels.
5. Train Random Forest and XGBoost models.
6. Compare model performance.
7. Generate SHAP explanations.
8. Package the model into a reusable `site_scorer.py`.
9. Test the scorer using Pytest.

## Main Outputs

- Candidate location shapefile
- Feature dataset
- Labelled site dataset
- Machine-learning training workflow
- SHAP feature-importance plot
- Reusable site scorer
- Automated tests

## How to Run

Activate the GeoSense environment:

```powershell
conda activate geosense
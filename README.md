# AYUSH: ML-Based Remaining Useful Life Prediction for Defence Equipment

> An explainable machine-learning prototype for Remaining Useful Life (RUL) prediction using equipment degradation data, developed as a reproducible predictive-maintenance benchmark with potential applications to defence equipment health monitoring.

## Live Demo

**Streamlit App:** https://ayush-rul.streamlit.app

## Project Overview

AYUSH is a machine-learning-based predictive-maintenance project that investigates whether historical sensor measurements can be used to estimate the Remaining Useful Life (RUL) of equipment.

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001 as a publicly available and reproducible benchmark.

The project focuses on:

- Remaining Useful Life prediction
- Time-series sensor analysis
- Predictive maintenance
- Explainable machine learning
- Maintenance decision support

The complete pipeline covers data preprocessing, exploratory data analysis, RUL target generation, feature engineering, model comparison, final model selection, official test evaluation, explainability analysis, and maintenance-risk classification.

## Problem Statement

Equipment can undergo gradual degradation before eventual failure. Unexpected failures can result in:

- Equipment downtime
- Increased maintenance requirements
- Reduced equipment availability
- Unplanned maintenance interventions
- Increased operational costs

Traditional maintenance approaches may rely heavily on predefined maintenance schedules rather than continuously estimated equipment health.

AYUSH investigates whether historical sensor measurements can be used by machine-learning models to estimate Remaining Useful Life and provide maintenance-oriented decision support.

## Research Question

> Can machine-learning models accurately estimate the Remaining Useful Life of equipment from historical sensor measurements while providing interpretable information to support maintenance decision-making?

## Dataset

### NASA C-MAPSS — FD001

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset, specifically the FD001 subset.

FD001 contains:

- 100 training engine trajectories
- 100 test engine trajectories
- One operating condition
- One fault mode
- Multiple sensor measurements
- Sequential operating-cycle observations

The dataset represents simulated turbofan engine degradation and is used as a reproducible benchmark for Remaining Useful Life prediction.

### Dataset Source

NASA Prognostics Center of Excellence Data Repository:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

> The dataset represents simulated aircraft-engine degradation and is not classified or proprietary defence-equipment data. AYUSH uses it as a reproducible benchmark to investigate predictive-maintenance methodology with potential applications to defence equipment health monitoring.

## Methodology

### 1. Data Preprocessing

The dataset was processed through:

- Data quality checks
- Sensor analysis
- Removal of non-informative sensor measurements
- Temporal data preparation
- RUL target generation
- Engine-level train/validation separation
- Test-data preparation

Engine trajectories were kept separate during validation to reduce the risk of information leakage between equipment units.

### 2. RUL Target Generation

For each training engine, Remaining Useful Life was calculated from the final operating cycle of that engine.

```text
RUL = Final Cycle - Current Cycle

For example:

Final cycle = 200
Current cycle = 150

RUL = 50 cycles
```

### 3. Feature Engineering

Temporal features were investigated for selected degradation-related sensors.

The engineered features included:

- One-cycle sensor differences
- Five-cycle rolling means
- Five-cycle rolling standard deviations

The selected temporal sensors were:

- `sensor_11`
- `sensor_4`
- `sensor_9`
- `sensor_12`
- `sensor_7`
- `sensor_14`
- `sensor_15`

The feature-engineered dataset contained 21 additional temporal features.

The feature-engineered XGBoost model was compared against the baseline XGBoost model to determine whether the additional temporal features improved validation performance.

## Model Development

Three baseline regression approaches were evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The final model was selected based on validation performance.

## Validation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25.155 | 31.656 | 0.768 |
| Random Forest | 23.939 | 31.606 | 0.768 |
| XGBoost | **23.678** | **31.189** | **0.774** |
| XGBoost + Temporal Features | 23.778 | 32.045 | 0.762 |

The baseline XGBoost model achieved the strongest validation performance and was therefore selected as the final model.

## Final Model

The final model is an XGBoost Regressor trained using the original 18 input features.

The final feature set contains:

```text
cycle
op_setting_1
op_setting_2
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

The trained model is saved as:

```text
models/fd001_final_model.pkl
```

## Official FD001 Test Evaluation

The selected final XGBoost model was evaluated on the official FD001 test trajectories.

### Test Results

| Metric | Result |
|---|---:|
| MAE | **19.627** |
| RMSE | **26.767** |
| R² | **0.585** |

The test evaluation was performed on 100 test engine observations, with one final observation selected for each engine trajectory.

## Explainability

AYUSH includes model explainability analysis using:

- XGBoost feature importance
- SHAP global feature importance
- Individual prediction explanation

### Feature Importance

The final XGBoost model identified the following as the strongest features:

| Rank | Feature |
|---:|---|
| 1 | `cycle` |
| 2 | `sensor_11` |
| 3 | `sensor_4` |
| 4 | `sensor_9` |
| 5 | `sensor_12` |
| 6 | `sensor_7` |
| 7 | `sensor_14` |
| 8 | `sensor_15` |

### SHAP Analysis

SHAP analysis confirmed that `cycle` had the largest average contribution to model predictions, followed by `sensor_11`, `sensor_4`, and `sensor_9`.

This analysis provides insight into which measurements contribute most strongly to the model's RUL predictions.

Individual prediction explanations were also generated for selected test-engine predictions.

## Maintenance Decision Support

Predicted RUL values were converted into prototype maintenance-risk categories.

```text
Predicted RUL > 100
        |
        v
     NORMAL

50 < Predicted RUL <= 100
        |
        v
     MONITOR

Predicted RUL <= 50
        |
        v
HIGH MAINTENANCE PRIORITY
```

For the 100 FD001 test engines, the prototype classification produced:

| Maintenance Risk | Engines |
|---|---:|
| NORMAL | 38 |
| MONITOR | 32 |
| HIGH MAINTENANCE PRIORITY | 30 |

> These thresholds are experimental prototype decision rules and are not real-world defence maintenance standards.

## Streamlit Prototype

AYUSH includes an interactive Streamlit dashboard for exploring the official FD001 test predictions.

The dashboard provides:

- Test-engine selection
- Current operating cycle
- Predicted RUL
- Maintenance-risk classification
- Actual RUL
- Prediction error
- Absolute error
- Fleet-level maintenance-risk overview

The application is intended as a demonstration of how RUL predictions can be presented as maintenance-oriented decision-support information.

## Project Structure

The repository currently contains the following major components:

```text
AYUSH/
│
├── data/
│   ├── raw/
│   │   └── FD001/
│   │       ├── train_FD001.txt
│   │       ├── test_FD001.txt
│   │       └── RUL_FD001.txt
│   │
│   └── processed/
│       └── FD001/
│           ├── train_FD001_clean.csv
│           ├── train_FD001_with_rul.csv
│           └── train_FD001_feature_engineered.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_rul_target.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_final_model.ipynb
│   └── 08_test_evaluation.ipynb
│
├── models/
│   └── fd001_final_model.pkl
│
├── reports/
│   └── FD001/
│       ├── eda/
│       ├── figures/
│       │   ├── comparison/
│       │   ├── final/
│       │   ├── fe_xgboost/
│       │   ├── linear_regression/
│       │   ├── random_forest/
│       │   └── xgboost/
│       │
│       ├── baseline_results.csv
│       ├── feature_engineering_comparison.csv
│       ├── final_features.csv
│       ├── model_comparison.csv
│       ├── fd001_test_predictions.csv
│       └── fd001_test_results.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The repository structure may continue to evolve as additional project components are developed.

## Notebook Workflow

The current experimental workflow is organized across the project notebooks.

### Data and EDA

#### `01_data_audit.ipynb`

#### `02_eda.ipynb`

#### `03_rul_target.ipynb`

These notebooks cover data inspection, exploratory analysis, and RUL preparation.

### Model Development

#### `05_baseline_models.ipynb`

Contains the baseline model experiments:

- Linear Regression
- Random Forest
- XGBoost
- Model diagnostics
- Feature importance

### Feature Engineering

#### `06_feature_engineering.ipynb`

Contains:

- Temporal feature construction
- Rolling statistics
- Feature-engineered XGBoost
- Baseline vs. feature-engineered comparison

### Final Model

#### `07_final_model.ipynb`

Contains:

- Final feature selection
- Final XGBoost training
- Model serialization
- Final feature list

### Official Test Evaluation

#### `08_test_evaluation.ipynb`

Contains:

- Official FD001 test-set preparation
- Final-cycle extraction
- Final-model prediction
- Official RUL comparison
- Test metrics
- Final prediction visualizations

## Technology Stack

The project is implemented primarily in Python using:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Joblib

Additional explainability and application technologies such as SHAP and Streamlit may be incorporated in future development.

## Current Project Status

### Completed

- Project concept and problem definition
- NASA C-MAPSS FD001 dataset preparation
- Data audit
- Exploratory data analysis
- RUL target generation
- Engine-level train/validation splitting
- Linear Regression baseline
- Random Forest baseline
- XGBoost baseline
- Baseline model comparison
- XGBoost feature-importance analysis
- Temporal feature-engineering experiment
- Feature-engineered XGBoost comparison
- Final model selection
- Final XGBoost training
- Official FD001 test evaluation
- Test prediction generation
- Test performance reporting
- Prediction visualization
- Error-distribution visualization

## Current Best Model

**Model:** XGBoost  
**Features:** 18 baseline features

### Validation Performance

```text
MAE  = 23.678
RMSE = 31.189
R²   = 0.774
```

### Official FD001 Test Performance

```text
MAE  = 19.627
RMSE = 26.767
R²   = 0.585
```

## Limitations

The current implementation has several limitations.

### Simulated Benchmark

The model is trained and evaluated on the NASA C-MAPSS simulated turbofan-engine dataset rather than real defence-equipment data.

### Domain Transfer

Performance on C-MAPSS does not guarantee equivalent performance on real defence equipment.

Different equipment types may exhibit substantially different degradation mechanisms, sensor characteristics, operating conditions, and failure modes.

### Limited FD001 Scope

The current experiment focuses on FD001:

- One operating condition
- One fault mode
- 100 training engines
- 100 test engines

The results should therefore not be generalized to all equipment types or operating environments.

### Model Limitations

The current XGBoost model provides a point estimate of RUL. It does not currently provide a calibrated uncertainty interval for every prediction.

### Explainability Limitations

Feature importance describes how the trained model uses input variables. It should not be interpreted as proof that a particular sensor is causally responsible for equipment degradation.

### Operational Limitations

The project does not provide real-world maintenance authorization or operational recommendations.

Any maintenance-risk thresholds developed in the future would require domain-expert validation and equipment-specific calibration.

## Future Scope

Potential future extensions include:

- Evaluation across C-MAPSS FD002, FD003, and FD004
- Multiple operating conditions
- Multiple fault modes
- Additional temporal-feature experiments
- Hyperparameter optimization
- Cross-validation strategies suitable for equipment trajectories
- Uncertainty estimation
- SHAP-based local explanations
- Real-time health monitoring
- Prototype maintenance-risk dashboards
- Edge or embedded deployment
- Integration with maintenance-management systems
- Equipment-specific model adaptation
- Real-world equipment sensor data
- Digital-twin integration
- Continuous equipment health-state monitoring

## Competition

**ML Bubble 2026 — Machine Learning Awareness & Skill Building Challenge**

**Organized by:**  
Army Institute of Technology (AIT), Pune

**Domain:**  
Defense & National Security

## Disclaimer

AYUSH is an academic/research prototype developed using publicly available simulated equipment-degradation data.

It is intended to demonstrate machine-learning techniques for predictive maintenance and decision-support research.

The project:

- Does not use classified defence data
- Does not use proprietary military equipment data
- Does not represent a deployed military system
- Does not provide operational maintenance instructions
- Does not establish real-world defence maintenance standards

Any future application to defence equipment would require equipment-specific data, domain-expert validation, safety evaluation, and appropriate operational authorization.

## Author

**Tisya Ahuja**

B.Tech — Computer Science  
B.S. — Data Science

**GitHub:**  
https://github.com/tisya-ahuja

**Project Repository:**  
https://github.com/tisya-ahuja/AYUSH

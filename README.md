# AYUSH: ML-Based Remaining Useful Life Prediction for Defence Equipment

«An explainable machine-learning prototype for Remaining Useful Life (RUL) prediction using equipment degradation data, developed as a reproducible predictive-maintenance benchmark with potential applications to defence equipment health monitoring.»

## Live Demo

Streamlit App: https://ayush-rul.streamlit.app

## Project Overview

AYUSH is a machine-learning-based predictive-maintenance project that investigates whether historical sensor measurements can be used to estimate the Remaining Useful Life (RUL) of equipment.

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset - FD001 as a publicly available and reproducible benchmark.

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

«Can machine-learning models accurately estimate the Remaining Useful Life of equipment from historical sensor measurements while providing interpretable information to support maintenance decision-making?»

## Dataset

NASA C-MAPSS - FD001

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

NASA Prognostics Center of Excellence Data Repository

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

«The dataset represents simulated aircraft-engine degradation and is not classified or proprietary defence-equipment data. AYUSH uses it as a reproducible benchmark to investigate predictive-maintenance methodology with potential applications to defence equipment health monitoring.»

## Methodology

1. Data Preprocessing

The dataset was processed through:

- Data quality checks

- Sensor analysis

- Removal of non-informative sensor measurements

- Temporal data preparation

- RUL target generation

- Engine-level train/validation separation

- Test-data preparation

Engine trajectories were kept separate during validation to reduce the risk of information leakage between equipment units.

2. RUL Target Generation

For each training engine, Remaining Useful Life was calculated from the final operating cycle of that engine.

RUL = Final Cycle - Current Cycle

For example:

Final cycle = 200
Current cycle = 150

RUL = 50 cycles

3. Feature Engineering

Temporal features were investigated for selected degradation-related sensors.

The engineered features included:

- One-cycle sensor differences

- Five-cycle rolling means

- Five-cycle rolling standard deviations

The selected temporal sensors were:

sensor_11
sensor_4
sensor_9
sensor_12
sensor_7
sensor_14
sensor_15

The feature-engineered dataset contained 21 additional temporal features.

The feature-engineered XGBoost model was compared against the baseline XGBoost model to determine whether the additional temporal features improved validation performance.

### Model Development

Three baseline regression approaches were evaluated:

- Linear Regression

- Random Forest Regressor

- XGBoost Regressor

The final model was selected based on validation performance.

Validation Results

Model| MAE| RMSE| R²
Linear Regression| 25.155| 31.656| 0.768
Random Forest| 23.939| 31.606| 0.768
XGBoost| 23.678| 31.189| 0.774
XGBoost + Temporal Features| 23.778| 32.045| 0.762

The baseline XGBoost model achieved the strongest validation performance and was therefore selected as the final model.

#### Final Model

The final model is an XGBoost Regressor trained using the original 18 input features.

The final feature set contains:
```
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

```models/fd001_final_model.pkl```

#### Official FD001 Test Evaluation

The selected final XGBoost model was evaluated on the official FD001 test trajectories.

### Test Results

Metric	Result

MAE	19.627
RMSE	26.767
R²	0.585

The test evaluation was performed on 100 test engine observations, with one final observation selected for each engine trajectory.

## Explainability

AYUSH includes model explainability analysis using:

XGBoost feature importance

SHAP global feature importance

Individual prediction explanation

Feature Importance

The final XGBoost model identified the following as the strongest features:

Rank	Feature

1	cycle
2	sensor_11
3	sensor_4
4	sensor_9
5	sensor_12
6	sensor_7
7	sensor_14
8	sensor_15

### SHAP Analysis

SHAP analysis confirmed that cycle had the largest average contribution to model predictions, followed by sensor_11, sensor_4, and sensor_9.

This analysis provides insight into which measurements contribute most strongly to the model's RUL predictions.

Individual prediction explanations were also generated for selected test-engine predictions.

#### Maintenance Decision Support

Predicted RUL values were converted into prototype maintenance-risk categories.

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

For the 100 FD001 test engines, the prototype classification produced:

Maintenance Risk	Engines

NORMAL	38
MONITOR	32
HIGH MAINTENANCE PRIORITY	30

These thresholds are experimental prototype decision rules and are not real-world defence maintenance standards.

## Streamlit Prototype

AYUSH includes an interactive Streamlit dashboard for exploring the official FD001 test predictions.

The dashboard provides:

Test-engine selection

Current operating cycle

Predicted RUL

Maintenance-risk classification

Actual RUL

Prediction error

Absolute error

Fleet-level maintenance-risk overview

The application is intended as a demonstration of how RUL predictions can be presented as maintenance-oriented decision-support information.

## Project Structure
```
AYUSH/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── FD001/
│   │
│   └── processed/
│       └── FD001/
│
├── models/
│   └── fd001_final_model.pkl
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_rul_target.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_final_model.ipynb
│   ├── 08_test_evaluation.ipynb
│   ├── 09_explainability.ipynb
│   └── 10_maintenance_decision_support.ipynb
│
├── reports/
│   └── FD001/
│
├── src/
│   ├── init.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── explainability.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```
## Technology Stack

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

SHAP

Plotly

Streamlit

Jupyter Notebook

Limitations

The current implementation has several limitations:

The benchmark dataset is simulated rather than real defence-equipment data.

Results may not directly generalize to real-world defence equipment.

Different equipment types may exhibit different degradation patterns.

The FD001 dataset contains one operating condition and one fault mode.

Prototype maintenance thresholds are not operational maintenance standards.

Real deployment would require equipment-specific sensor data and domain-expert validation.

The model is intended for research and demonstration rather than operational maintenance decisions.

## Future Scope

Potential extensions include:

Evaluation across additional C-MAPSS subsets

Multiple operating conditions

Multiple fault modes

Evaluation on real-world equipment sensor data

Real-time health monitoring

Edge or embedded deployment

Equipment-specific model adaptation

Integration with maintenance-management systems

Digital-twin integration

Continuous health-state monitoring

## Competition

ML Bubble 2026 - Machine Learning Awareness & Skill Building Challenge

Organized by: Army Institute of Technology (AIT), Pune

Domain: Defence & National Security

> Disclaimer

AYUSH is an academic/research prototype developed using publicly available simulated equipment-degradation data.

It is intended to demonstrate machine-learning techniques for predictive maintenance, Remaining Useful Life estimation, explainability, and decision support.

The project does not represent a deployed military system, does not use classified defence data, and should not be interpreted as an operational defence maintenance system.

## Author

Tisya Ahuja

B.Tech - Computer Science

B.S. - Data Science

GitHub: https://github.com/tisya-ahuja

Repository: https://github.com/tisya-ahuja/AYUSH

Dataset

NASA C-MAPSS FD001

The project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset, specifically the FD001 subset.

FD001 contains simulated degradation trajectories for turbofan engines under a single operating condition and a single fault mode.

The FD001 dataset contains:

* 100 training engines
* 100 test engines
* 21 sensor measurements
* 3 operational-setting columns in the original raw representation
* Sequential operating-cycle observations
* Complete failure trajectories for the training data
* Truncated trajectories for the test data

The project uses the dataset as a reproducible benchmark for Remaining Useful Life prediction rather than as real defence-equipment data.

Dataset Source

NASA Prognostics Center of Excellence Data Repository:

"https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/" (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

«Note: C-MAPSS is a simulated aircraft-engine degradation dataset. It is not classified or proprietary defence-equipment data. AYUSH uses it as a reproducible benchmark to investigate a predictive-maintenance methodology that may have potential applications to defence equipment.»

Methodology

The project follows a staged machine-learning workflow.

NASA C-MAPSS FD001
        |
        v
Data Audit & Cleaning
        |
        v
RUL Target Generation
        |
        v
Exploratory Data Analysis
        |
        v
Engine-Level Train/Validation Split
        |
        v
Baseline Model Comparison
        |
        +----------------------+
        |                      |
        v                      v
Linear Regression       Random Forest
        |                      |
        +----------+-----------+
                   |
                   v
                XGBoost
                   |
                   v
          Feature Engineering
                   |
                   v
       Feature-Engineered XGBoost
                   |
                   v
            Model Comparison
                   |
                   v
          Final XGBoost Selection
                   |
                   v
       Official FD001 Test Evaluation

1. Data Preparation

The FD001 training data was processed into a clean tabular dataset.

The cleaned training representation contains:

unit  - engine identifier
cycle - operating cycle
18 selected input features
RUL   - Remaining Useful Life target

The resulting training dataset contains:

20,631 observations
100 engines
18 model features

The final baseline feature set contains:

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

"unit" is retained for trajectory-level processing but is not used as a model feature.

2. RUL Target Generation

For each training engine, RUL was calculated from its final observed failure cycle.

Conceptually:

RUL = Final Cycle of Engine - Current Cycle

For example:

Final cycle = 200
Current cycle = 150

RUL = 50 cycles

This produces a decreasing RUL trajectory for each engine as it approaches failure.

The resulting training dataset contains an RUL value for every training observation.

3. Exploratory Data Analysis

Exploratory analysis was performed to understand:

* Engine lifetime distributions
* Sensor variability
* Sensor behavior across operating cycles
* Relationships between sensors
* Correlations between sensors and operating cycle
* Sensor behavior across individual engines

The EDA artifacts are stored under:

reports/FD001/eda/

The analysis helped identify sensors with stronger degradation-related behavior and informed the later temporal-feature experiment.

4. Engine-Level Validation Strategy

A major consideration in this project was avoiding leakage between observations belonging to the same engine.

Instead of randomly splitting individual rows, the training engines were divided at the engine level.

The split used:

Training engines:    80
Validation engines:  20
Overlapping engines: 0

This produced:

Training observations:   16,561
Validation observations:  4,070
Total observations:      20,631

The same engine-level methodology was used when evaluating the feature-engineered experiment.

This prevents observations from the same degradation trajectory from appearing in both training and validation sets.

5. Baseline Models

Three regression models were evaluated.

Linear Regression

Used as a simple interpretable baseline.

Random Forest Regressor

Used as a nonlinear ensemble baseline capable of modelling nonlinear relationships between sensor measurements and RUL.

XGBoost Regressor

Used as the primary gradient-boosting model and ultimately selected as the final model based on validation performance.

6. Baseline Model Results

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Validation Results

Model| MAE| RMSE| R²
Linear Regression| 25.155| 31.656| 0.768
Random Forest| 23.939| 31.606| 0.768
XGBoost| 23.678| 31.189| 0.774

Lower MAE and RMSE are better, while higher R² is better.

Result

XGBoost achieved the strongest validation performance among the three baseline models:

MAE  = 23.678
RMSE = 31.189
R²   = 0.774

Therefore, XGBoost was selected for the subsequent feature-engineering experiment and final-model development.

7. Model Diagnostics

Prediction diagnostics were generated for the evaluated models, including:

* Actual vs. predicted RUL plots
* Residual plots
* Feature-importance plots for tree-based models

The visualizations are stored under:

reports/FD001/figures/

Model-specific figures are organized into:

reports/FD001/figures/
├── linear_regression/
├── random_forest/
├── xgboost/
├── fe_xgboost/
├── comparison/
└── final/

8. Feature Engineering Experiment

A temporal feature-engineering experiment was performed after the baseline comparison.

The goal was to investigate whether recent sensor behavior could improve RUL prediction beyond the original sensor values.

Seven sensors were selected for temporal feature construction:

sensor_11
sensor_4
sensor_9
sensor_12
sensor_7
sensor_14
sensor_15

For each selected sensor, the following features were created.

One-Cycle Difference

sensor_diff_1

This captures the change in sensor value between consecutive cycles.

Five-Cycle Rolling Mean

sensor_rolling_mean_5

This captures recent local sensor behavior.

Five-Cycle Rolling Standard Deviation

sensor_rolling_std_5

This captures recent sensor variability.

In total:

7 sensors × 3 temporal features = 21 new features

The feature-engineered dataset therefore contains:

* 41 columns
* 39 model features
* 1 unit identifier
* 1 RUL target

The engineered dataset is stored at:

data/processed/FD001/train_FD001_feature_engineered.csv

9. Feature Engineering Results

The feature-engineered XGBoost model was evaluated using the same engine-level validation methodology.

Comparison

Model| MAE| RMSE| R²
XGBoost Baseline| 23.678| 31.189| 0.774
XGBoost + Temporal Features| 23.778| 32.045| 0.762

The temporal feature-engineering experiment did not improve the baseline XGBoost model.

Compared with the baseline:

* MAE increased slightly
* RMSE increased
* R² decreased

Therefore, the final model uses the simpler 18-feature baseline representation rather than the 39-feature engineered representation.

This is an important experimental result: feature engineering was evaluated empirically rather than assumed to improve performance.

10. Final Model

Based on the validation experiments, the final model selected for FD001 is:

XGBoost Regressor

The final model uses the original 18-feature representation.

The final feature list is stored in:

reports/FD001/final_features.csv

The trained model is saved as:

models/fd001_final_model.pkl

The final model was trained using the selected baseline feature representation after model and feature-engineering comparison.

11. Official FD001 Test Evaluation

After model selection, the saved final XGBoost model was evaluated on the official FD001 test set.

The official test set contains:

13,096 observations
100 test engines

The test trajectories are truncated before failure, so their true RUL values cannot be derived simply from the maximum observed test cycle.

Instead, the official:

RUL_FD001.txt

values supplied with the benchmark were used as the ground-truth RUL values.

For each test engine, the final observed cycle was selected:

100 test engines
        |
        v
100 final observations
        |
        v
100 predictions

These predictions were then compared against the 100 official RUL values.

12. Official Test Results

The final XGBoost model achieved the following performance on the official FD001 test set:

Metric| Test Performance
MAE| 19.627
RMSE| 26.767
R²| 0.585

Interpretation

The model's official test-set Mean Absolute Error is approximately:

19.63 operating cycles

This means that, on average, the predicted RUL differs from the provided test-set RUL by approximately 19.6 cycles in absolute terms.

The test-set RMSE of approximately 26.77 cycles indicates that larger prediction errors remain for some engines.

The R² value of 0.585 indicates that the model explains a substantial portion, but not all, of the variation in the official test-set RUL values.

The official test results should be interpreted separately from the validation results because they are evaluated on a different set of engines.

13. Final Test Artifacts

The official test predictions are stored in:

reports/FD001/fd001_test_predictions.csv

The final test metrics are stored in:

reports/FD001/fd001_test_results.csv

The final test visualizations are stored in:

reports/FD001/figures/final/

Including:

FD001_Test_Actual_vs_Predicted_RUL.png
FD001_Test_Error_Distribution.png

14. Explainability

Model explainability is an intended part of the AYUSH framework.

The current implementation includes tree-model feature-importance analysis for the Random Forest and XGBoost models.

The XGBoost feature-importance analysis indicates that several features contribute strongly to the model, with "cycle" and "sensor_11" among the most influential features in the trained baseline model.

Feature importance should be interpreted as model-specific importance rather than as a direct causal explanation of equipment failure.

Future work may extend this analysis using SHAP or other local explanation methods to explain individual predictions.

15. Maintenance Decision Support

The predicted RUL can potentially be translated into prototype maintenance-risk categories.

For example:

Higher RUL
    |
    v
NORMAL
    |
    v
MONITOR
    |
    v
HIGH MAINTENANCE PRIORITY
    |
    v
Lower RUL

These categories are conceptual only.

Any thresholds used in a future prototype would be experimental decision rules and must not be interpreted as real-world defence maintenance standards.

The current project focuses on RUL prediction rather than operational maintenance authorization.

16. Project Structure

The repository currently contains the following major components:

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

«The repository structure may continue to evolve as additional project components are developed.»

17. Notebook Workflow

The current experimental workflow is organized across the project notebooks.

Data and EDA

01_data_audit.ipynb
02_eda.ipynb
03_rul_target.ipynb

These notebooks cover data inspection, exploratory analysis, and RUL preparation.

Model Development

05_baseline_models.ipynb

Contains the baseline model experiments:

* Linear Regression
* Random Forest
* XGBoost
* Model diagnostics
* Feature importance

Feature Engineering

06_feature_engineering.ipynb

Contains:

* Temporal feature construction
* Rolling statistics
* Feature-engineered XGBoost
* Baseline vs. feature-engineered comparison

Final Model

07_final_model.ipynb

Contains:

* Final feature selection
* Final XGBoost training
* Model serialization
* Final feature list

Official Test Evaluation

08_test_evaluation.ipynb

Contains:

* Official FD001 test-set preparation
* Final-cycle extraction
* Final-model prediction
* Official RUL comparison
* Test metrics
* Final prediction visualizations

18. Technology Stack

The project is implemented primarily in Python using:

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Jupyter Notebook
Joblib

Additional explainability and application technologies such as SHAP and Streamlit may be incorporated in future development.

19. Current Project Status

Completed

* Project concept and problem definition
* NASA C-MAPSS FD001 dataset preparation
* Data audit
* Exploratory data analysis
* RUL target generation
* Engine-level train/validation splitting
* Linear Regression baseline
* Random Forest baseline
* XGBoost baseline
* Baseline model comparison
* XGBoost feature-importance analysis
* Temporal feature-engineering experiment
* Feature-engineered XGBoost comparison
* Final model selection
* Final XGBoost training
* Official FD001 test evaluation
* Test prediction generation
* Test performance reporting
* Prediction visualization
* Error-distribution visualization

Current Best Model

Model: XGBoost
Features: 18 baseline features

Validation Performance

MAE  = 23.678
RMSE = 31.189
R²   = 0.774

Official FD001 Test Performance

MAE  = 19.627
RMSE = 26.767
R²   = 0.585

20. Limitations

The current implementation has several limitations.

Simulated Benchmark

The model is trained and evaluated on the NASA C-MAPSS simulated turbofan-engine dataset rather than real defence-equipment data.

Domain Transfer

Performance on C-MAPSS does not guarantee equivalent performance on real defence equipment.

Different equipment types may exhibit substantially different degradation mechanisms, sensor characteristics, operating conditions, and failure modes.

Limited FD001 Scope

The current experiment focuses on FD001:

* One operating condition
* One fault mode
* 100 training engines
* 100 test engines

The results should therefore not be generalized to all equipment types or operating environments.

Model Limitations

The current XGBoost model provides a point estimate of RUL. It does not currently provide a calibrated uncertainty interval for every prediction.

Explainability Limitations

Feature importance describes how the trained model uses input variables. It should not be interpreted as proof that a particular sensor is causally responsible for equipment degradation.

Operational Limitations

The project does not provide real-world maintenance authorization or operational recommendations.

Any maintenance-risk thresholds developed in the future would require domain-expert validation and equipment-specific calibration.

21. Future Scope

Potential future extensions include:

* Evaluation across C-MAPSS FD002, FD003, and FD004
* Multiple operating conditions
* Multiple fault modes
* Additional temporal-feature experiments
* Hyperparameter optimization
* Cross-validation strategies suitable for equipment trajectories
* Uncertainty estimation
* SHAP-based local explanations
* Real-time health monitoring
* Prototype maintenance-risk dashboards
* Edge or embedded deployment
* Integration with maintenance-management systems
* Equipment-specific model adaptation
* Real-world equipment sensor data
* Digital-twin integration
* Continuous equipment health-state monitoring

22. Competition

ML Bubble 2026 - Machine Learning Awareness & Skill Building Challenge

Organized by:

Army Institute of Technology (AIT), Pune

Domain:

Defense & National Security

23. Disclaimer

AYUSH is an academic/research prototype developed using publicly available simulated equipment-degradation data.

It is intended to demonstrate machine-learning techniques for predictive maintenance and decision-support research.

The project:

* Does not use classified defence data
* Does not use proprietary military equipment data
* Does not represent a deployed military system
* Does not provide operational maintenance instructions
* Does not establish real-world defence maintenance standards

Any future application to defence equipment would require equipment-specific data, domain-expert validation, safety evaluation, and appropriate operational authorization.

24. Author

Tisya Ahuja

B.Tech - Computer Science
B.S. - Data Science

GitHub:

"https://github.com/tisya-ahuja" (https://github.com/tisya-ahuja)

Project Repository:

"https://github.com/tisya-ahuja/AYUSH" (https://github.com/tisya-ahuja/AYUSH)

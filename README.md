# AYUSH: ML-Based Remaining Useful Life Prediction for Defence Equipment

> An explainable machine-learning framework for predicting the Remaining Useful Life (RUL) of equipment from historical sensor data to support predictive maintenance and decision-making.

## Project Overview

AYUSH is a machine-learning-based predictive maintenance framework designed to estimate the Remaining Useful Life (RUL) of equipment using historical sensor measurements.

The project explores how machine learning can identify degradation patterns in equipment and provide an early indication of when maintenance may be required.

The initial implementation will use the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001) as a publicly available and reproducible benchmark for predictive-maintenance research.

The project is being developed as part of ML Bubble 2026 - Machine Learning Awareness & Skill Building Challenge, organized by Army Institute of Technology (AIT), Pune.

## Problem Statement

Defence equipment can undergo gradual degradation before eventual component failure. Unexpected failures can lead to:

- Equipment downtime
- Increased maintenance requirements
- Reduced equipment availability
- Unplanned maintenance interventions
- Increased operational costs

Traditional maintenance approaches may rely heavily on predefined maintenance schedules rather than continuously estimated equipment health.

AYUSH aims to investigate whether historical equipment sensor measurements can be used by machine-learning models to estimate Remaining Useful Life and support proactive maintenance decisions.

## Research Question

> Can machine-learning models accurately estimate the Remaining Useful Life of equipment from historical sensor measurements while providing interpretable information to support maintenance decision-making?

## Proposed Solution

AYUSH follows a predictive-maintenance pipeline:

```text
Equipment Sensor Data
                    |
                    v
     Data Preprocessing
                    |
                    v
     Feature Engineering
                    |
                    v
         ML Models
                    |
                    v
        RUL Prediction
                    |
                    v
         Model Analysis
                    |
                    v
        Explainability
                    |
                    v
 Maintenance Risk
 & Decision Support
```

The system will estimate the number of remaining operating cycles before failure and use the prediction to provide maintenance-oriented risk information.

## Dataset

### NASA C-MAPSS - FD001

The initial implementation will use the NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset.

The FD001 subset contains multivariate time-series observations representing equipment degradation trajectories.

Key characteristics include:

- 100 training trajectories
- 100 test trajectories
- One operating condition
- One fault mode
- Multiple sensor measurements
- Sequential operating-cycle observations

The dataset is intended for prognostics and Remaining Useful Life prediction research.

### Dataset Source

NASA Prognostics Center of Excellence Data Repository

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

> Note: The dataset represents simulated aircraft-engine degradation and is not classified or proprietary defence-equipment data. AYUSH uses it as a reproducible benchmark to investigate a predictive-maintenance methodology with potential application to defence equipment.

## Machine Learning Methodology

### 1. Data Preprocessing

The dataset will be analyzed and prepared through:

- Data quality checks
- Missing-value analysis
- Sensor analysis
- Removal of non-informative features where appropriate
- Temporal data preparation
- Engine-level train/test separation

Special attention will be given to avoiding data leakage between equipment trajectories.

### 2. RUL Target Generation

For each training trajectory, Remaining Useful Life will be derived from the operating cycle and the final cycle of the corresponding equipment trajectory.

Conceptually:

RUL = Final Failure Cycle - Current Cycle

For example:

Final cycle = 200

Current cycle = 150

RUL = 50 cycles

### 3. Feature Engineering

Where appropriate, temporal features will be investigated, including:

- Current sensor values
- Sensor differences between consecutive cycles
- Rolling statistics
- Recent sensor trends
- Sensor variability
- Operating-cycle information

### 4. Model Development

Multiple regression models will be compared:

Baseline

- Linear Regression

Ensemble ML

- Random Forest Regressor

Gradient Boosting

- XGBoost Regressor

The final model will be selected based on experimentally observed performance rather than assuming a particular algorithm will perform best.

## Evaluation

The models will be evaluated using regression metrics appropriate for Remaining Useful Life prediction.

Mean Absolute Error - MAE

Measures the average absolute difference between predicted and actual RUL.

Root Mean Squared Error - RMSE

Penalizes larger prediction errors more strongly.

R² Score

Measures the proportion of variation in the target explained by the model.

### Planned Comparison

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>R²</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Linear Regression</td>
            <td>TBD</td>
            <td>TBD</td>
            <td>TBD</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>TBD</td>
            <td>TBD</td>
            <td>TBD</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>TBD</td>
            <td>TBD</td>
            <td>TBD</td>
        </tr>
    </tbody>
</table>

Model results will be added after training and validation.

No performance values are being reported before the experiments are conducted.

## Explainability

Predictive maintenance requires more than a numerical prediction.

AYUSH will investigate model explainability using techniques such as:

- Feature importance
- SHAP-based analysis

The objective is to understand which sensor characteristics contribute most strongly to an individual RUL prediction.

The eventual system will aim to provide information such as:

- Predicted RUL: [Model Output]
- Maintenance Risk: [Prototype Risk Level]
- Important Contributing Factors:
    1. Sensor-related feature
    2. Sensor trend feature
    3. Operating-condition feature

This will help make the ML predictions more interpretable for decision-support purposes.

## Maintenance Decision Support

The predicted RUL can subsequently be translated into prototype maintenance-risk categories.

For example:

```text
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
```

Any risk thresholds used in the prototype will be explicitly treated as experimental decision rules, not as real-world defence maintenance standards.

## Expected Outcome

The project aims to develop a reproducible ML prototype that can:

- Estimate Remaining Useful Life from historical sensor data
- Compare multiple regression approaches
- Identify degradation-related patterns
- Evaluate model performance using appropriate regression metrics
- Provide interpretable information about model predictions
- Translate predicted RUL into maintenance-oriented risk information

## Project Status

Current Status: Proposal / Development in Progress

### Completed

- Project concept finalized
- Problem domain identified
- ML problem defined
- Initial dataset selected
- Proposed methodology established
- GitHub repository created

### Upcoming

- Dataset acquisition and exploration
- Exploratory data analysis
- RUL target generation
- Feature engineering
- Model training
- Model comparison
- Explainability analysis
- Prototype dashboard
- Final documentation

## Project Structure

<table>
    <thead>
        <tr>
            <th>Path</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>data/raw/</td>
            <td>Raw dataset files</td>
        </tr>
        <tr>
            <td>data/processed/</td>
            <td>Cleaned and prepared data</td>
        </tr>
        <tr>
            <td>notebooks/</td>
            <td>Exploration, analysis, and experimentation notebooks</td>
        </tr>
        <tr>
            <td>src/</td>
            <td>Reusable source code for preprocessing, training, and evaluation</td>
        </tr>
        <tr>
            <td>models/</td>
            <td>Saved model artifacts</td>
        </tr>
        <tr>
            <td>app/</td>
            <td>Prototype application or dashboard</td>
        </tr>
        <tr>
            <td>reports/</td>
            <td>Documentation and generated reports</td>
        </tr>
        <tr>
            <td>README.md</td>
            <td>Project overview and setup information</td>
        </tr>
        <tr>
            <td>requirements.txt</td>
            <td>Python dependency list</td>
        </tr>
        <tr>
            <td>.gitignore</td>
            <td>Ignored files and folders</td>
        </tr>
    </tbody>
</table>

The repository structure will evolve as development progresses.

## Limitations

The initial implementation has several limitations:

- The benchmark dataset is simulated rather than real defence-equipment data.
- Results may not directly generalize to real-world defence equipment.
- Different equipment types may exhibit different degradation patterns.
- Prototype maintenance thresholds are not operational maintenance standards.
- Real deployment would require equipment-specific sensor data and domain-expert validation.

These limitations will be considered when interpreting the model results.

## Future Scope

Potential future extensions include:

- Evaluation across additional C-MAPSS subsets
- Multiple operating conditions
- Multiple fault modes
- Real-world equipment sensor data
- Real-time health monitoring
- Edge/embedded model deployment
- Integration with maintenance-management systems
- Equipment-specific model adaptation
- Digital-twin integration
- Continuous health-state monitoring

## Technology Stack

The planned implementation will primarily use:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- SHAP
- Streamlit
- Jupyter Notebook

The final dependency list will be updated as the implementation develops.

## Competition

ML Bubble 2026 - Machine Learning Awareness & Skill Building Challenge

Organized by: Army Institute of Technology (AIT), Pune

Domain: Defense & National Security

## Disclaimer

AYUSH is an academic/research prototype developed using publicly available simulated equipment-degradation data.

It is intended to demonstrate machine-learning techniques for predictive maintenance and decision support. It does not represent a deployed military system, does not use classified defence data, and should not be interpreted as an operational maintenance system.

## Author

Tisya Ahuja

B.Tech - Computer Science

B.Sc. - Data Science

GitHub: https://github.com/tisya-ahuja
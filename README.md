# ali-mortality-prediction
Machine learning analysis for prediction using NHANES data
# ALI Mortality Prediction using Machine Learning

A machine learning analysis to predict mortality in patients with Acute Lung Injury (ALI) using multiple algorithms and comprehensive evaluation metrics.

## 🎯 Overview

This project compares 6 machine learning models (Logistic Regression, Random Forest, Decision Tree, XGBoost, SVM, KNN) for ALI mortality prediction using clinical and demographic variables.

**Key Features:**

- Multiple model comparison with ROC, PR curves, and calibration analysis
- Bootstrap confidence intervals and cross-validation
- SHAP interpretability analysis and feature importance
- Comprehensive visualization and statistical evaluation

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/ali-mortality-prediction.git
cd ali-mortality-prediction
pip install -r requirements.txt
```

### Usage

```bash
# Run individual analysis scripts
python 01_mortality_roc.py              # ROC curve analysis
python 02_mortality_roc_95ci.py         # ROC with confidence intervals  
python 03_mortality_pr.py               # Precision-Recall curves
python 04_mortality_calibration.py      # Calibration curves
python 05_mortality_shap.py             # SHAP analysis
python 06_mortality_rf_importance.py    # Feature importance
python 07_mortality_confusion_matrix.py # Confusion matrix
```

## 📊 Data

**Dataset:** NHANES-based ALI patient data **Variables:** 19 clinical and demographic features

- Demographics: Gender, Age, Ethnicity, Education, Marital status
- Clinical: BMI, CVD, Diabetes, Hypertension, Smoking status
- Laboratory: HbA1c, ALT, AST, BUN, Total cholesterol, HDL
- Lifestyle: Drinking status, PIR (Poverty Income Ratio)
- Outcome: ALI severity score, Mortality (target variable)

## 📈 Results

| Model               | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
| ------------------- | ------- | -------- | --------- | ------ | -------- |
| XGBoost             | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Random Forest       | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Logistic Regression | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| SVM                 | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Decision Tree       | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| KNN                 | 0.XXX   | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |

*Results will be automatically generated after running the analysis*

## 📁 Files

```
├── 01_mortality_roc.py                  # ROC curve analysis
├── 02_mortality_roc_95ci.py             # ROC with confidence intervals
├── 03_mortality_pr.py                   # Precision-Recall curves
├── 04_mortality_calibration.py          # Model calibration analysis
├── 05_mortality_shap.py                 # SHAP interpretability
├── 06_mortality_rf_importance.py        # Random Forest feature importance
├── 07_mortality_confusion_matrix.py     # Confusion matrix visualization
├── requirements.txt                     # Python dependencies
├── ALI-mortality-train-data.xlsx        # Training dataset
├── ALI-mortality-test-data.xlsx         # Testing dataset
└── README.md                           # This file
```

## 🔧 Requirements

- Python 3.7+
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- seaborn
- shap

See `requirements.txt` for complete list.

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{ali_mortality_prediction,
  author = {[Your Name]},
  title = {ALI Mortality Prediction using Machine Learning},
  year = {2024},
  url = {https://github.com/your-username/ali-mortality-prediction}
}
```
# Stroke Risk Prediction using Machine Learning

A comprehensive machine learning analysis to predict stroke risk using NHANES data with multiple algorithms and advanced evaluation techniques.

## 🎯 Overview

This project compares 6 machine learning models (Logistic Regression, Random Forest, Decision Tree, XGBoost, SVM, KNN) for stroke risk prediction using clinical and demographic variables from NHANES database.

**Key Features:**

- SMOTE oversampling for class imbalance handling
- Multiple model comparison with ROC, PR curves, and calibration analysis
- Bootstrap confidence intervals and statistical evaluation
- SHAP interpretability analysis and feature importance ranking
- Data preprocessing with normalization and one-hot encoding

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/stroke-risk-prediction.git
cd stroke-risk-prediction
pip install -r requirements.txt
```

### Usage

```bash
# Data preprocessing
python normalization.py                     # Feature normalization and encoding
python smote.py                             # SMOTE oversampling for class balance

# Model analysis
python 02_stroke_roc_95ci.py                # ROC with confidence intervals  
python 03_stroke_pr.py                      # Precision-Recall curves
python 04_stroke_calibration.py             # Calibration curve analysis
python 05_stroke_shap.py                    # SHAP interpretability analysis
python 06_stroke_rf_importance.py           # Random Forest feature importance
python 07_stroke_confusion_matrix.py        # Confusion matrix visualization
```

## 📊 Data

**Dataset:** NHANES-based stroke risk assessment data **Features:** 19 clinical and demographic variables

- Demographics: Gender, Age, Ethnicity, Education level, Marital status
- Clinical: BMI, CVD, Diabetes, Hypertension, Smoking status
- Laboratory: HbA1c, ALT, AST, BUN, Total cholesterol, HDL
- Lifestyle: Drinking status, PIR (Poverty Income Ratio)
- Severity: ALI score
- **Target:** Stroke occurrence (binary outcome)

**Data Processing:**

- MinMax normalization for continuous variables
- One-hot encoding for categorical variables
- SMOTE oversampling to handle class imbalance
- Train-test split with stratification

## 📈 Results

| Model               | AUC-ROC | 95% CI       | Accuracy | Precision | Recall | F1-Score |
| ------------------- | ------- | ------------ | -------- | --------- | ------ | -------- |
| XGBoost             | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Random Forest       | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Logistic Regression | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| SVM                 | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| Decision Tree       | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |
| KNN                 | 0.XXX   | [X.XX, X.XX] | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    |

*Results automatically generated after running analysis*

## 📁 Project Structure

```
├── normalization.py                        # Data preprocessing and normalization
├── smote.py                                # SMOTE oversampling implementation
├── 02_stroke_roc_95ci.py                   # ROC analysis with confidence intervals
├── 03_stroke_pr.py                         # Precision-Recall curve analysis
├── 04_stroke_calibration.py                # Model calibration assessment
├── 05_stroke_shap.py                       # SHAP interpretability analysis
├── 06_stroke_rf_importance.py              # Random Forest feature importance
├── 07_stroke_confusion_matrix.py           # Confusion matrix visualization
├── requirements.txt                        # Python dependencies
├── stroke-train-data.xlsx                  # Training dataset
├── stroke-test-data.xlsx                   # Testing dataset
├── stroke-shapdata.csv                     # SHAP analysis dataset
└── README.md                              # This file
```

## 🔧 Requirements

**Core Libraries:**

- scikit-learn (machine learning algorithms)
- xgboost (gradient boosting)
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- shap (model interpretability)
- imblearn (SMOTE oversampling)

See `requirements.txt` for complete dependencies.

## 📊 Key Findings

**Class Imbalance Handling:**

- Original dataset shows significant class imbalance
- SMOTE successfully balances training data
- Improved model performance on minority class

**Model Performance:**

- [Best performing model] achieves highest AUC-ROC
- Feature importance reveals [top predictive factors]
- Calibration analysis shows [model reliability assessment]

**Clinical Insights:**

- [Key risk factors identified through SHAP analysis]
- [Important demographic and clinical predictors]
- [Model interpretability highlights]

## 📝 Citation

```bibtex
@software{stroke_risk_prediction,
  author = {[Your Name]},
  title = {Stroke Risk Prediction using Machine Learning},
  year = {2024},
  url = {https://github.com/your-username/stroke-risk-prediction}
}
```

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

- Author: [jiaxin_fan@163.com]
- Email: [your.email@example.com]
- Institution: [Department of Neurology, The Second Affiliated Hospital of Xi'an Jiaotong University, Xi'an, China.]

------

**Note:** This is a research tool and should not be used for clinical decision-making without proper validation.

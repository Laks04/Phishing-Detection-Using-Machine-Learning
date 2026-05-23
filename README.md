# Phishing Detection Using Machine Learning
 
## Overview
 
This project builds and evaluates multiple machine learning classifiers to detect phishing URLs using URL-based features. The dataset contains ~96,000 labeled samples (50% phishing, 50% benign) across 13 numerical features extracted from domain characteristics.
 
---
 
## Dataset
 
**File:** `data_Features.csv`
 
**Size:** 96,005 rows × 14 columns (after dropping the `domain` column: 13 features + 1 label)
 
| Feature | Description |
|---|---|
| `ranking` | Domain ranking score |
| `mld_res` | Main landing domain resolution result |
| `mld.ps_res` | Public suffix resolution result |
| `card_rem` | Cardinality remainder |
| `ratio_Rrem` | Ratio of remaining redirects |
| `ratio_Arem` | Ratio of remaining anchors |
| `jaccard_RR` | Jaccard similarity (redirect-redirect) |
| `jaccard_RA` | Jaccard similarity (redirect-anchor) |
| `jaccard_AR` | Jaccard similarity (anchor-redirect) |
| `jaccard_AA` | Jaccard similarity (anchor-anchor) |
| `jaccard_ARrd` | Jaccard similarity (anchor-redirect, rounded) |
| `jaccard_ARrem` | Jaccard similarity (anchor-redirect, remainder) |
| `label` | Target variable: `1` = Phishing, `0` = Benign |
 
**Class Balance:** Nearly equal — 48,009 benign (50.06%) vs. 47,903 phishing (49.94%).
 
---
 
## Project Structure
 
```
├── data_Features.csv           # Main dataset
├── Phishing_Detection.ipynb    # Main notebook
└── README.md
```
 
---
 
## Setup & Requirements
 
### Install Dependencies
 
```bash
pip install ydata-profiling catboost xgboost imbalanced-learn scikit-learn lightgbm keras tensorflow lazypredict
```
 
### Key Libraries
 
| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn`, `plotly` | Visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost`, `lightgbm`, `catboost` | Gradient boosting models |
| `imbalanced-learn` | SMOTE oversampling |
| `lazypredict` | Quick baseline model comparison |
| `keras`, `tensorflow` | Deep learning (imported, available for extension) |
 
---
 
## Pipeline
 
### 1. Data Loading
Load `data_Features.csv` using pandas with `on_bad_lines='skip'` and `encoding='latin1'`.
 
### 2. Data Cleaning
- Drop the `domain` column (non-numeric identifier)
- Convert all columns to numeric, coercing errors to `NaN`
- Drop rows with any `NaN` values (~92 rows dropped)
- Remove 1 duplicate row
- **Final clean dataset:** 95,912 rows × 13 columns
### 3. EDA
- Histograms and boxplots for all numerical features
- Correlation matrix heatmap
- Class balance bar chart and pie chart
Notable correlations:
- `jaccard_RR`, `jaccard_RA`, `jaccard_AR`, `jaccard_AA` are highly correlated (~0.98)
- `ranking` negatively correlates with `mld_res` (-0.67) and `jaccard_ARrd` (-0.62)
- `label` has the strongest positive correlation with `ranking` (0.50)
### 4. Feature Engineering
- Checked for features with >90% repeated max values → none found, no columns dropped
- Train/test split: **80/20**, stratified by label
- **SMOTE oversampling** applied to training set only → balanced to 38,407 samples per class (76,814 total)
### 5. Baseline Model Comparison (LazyPredict)
Run on a subsample (20,000 train / 5,000 test) for speed:
 
| Model | Accuracy | F1 Score |
|---|---|---|
| ExtraTreesClassifier | 0.94 | 0.94 |
| RandomForestClassifier | 0.94 | 0.93 |
| BaggingClassifier | 0.93 | 0.93 |
| XGBClassifier | 0.92 | 0.92 |
| LGBMClassifier | 0.91 | 0.91 |
 
### 6. Full Model Training & Evaluation
Each classifier is trained with 10-fold cross-validation, then evaluated on the held-out test set. Metrics reported: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Report.
 
**Models evaluated:**
 
| Model | Test F1 (macro) | Test Accuracy |
|---|---|---|
| **Random Forest** ⭐ | **0.9555** | **0.96** |
| Extra Trees Classifier | 0.9543 | 0.95 |
| Bagging Classifier | 0.9505 | 0.95 |
| Decision Tree | 0.9400 | 0.94 |
| XGBoost | 0.9294 | 0.93 |
| Extra Tree (single) | 0.9286 | 0.93 |
| K Nearest Neighbors | 0.9196 | 0.92 |
| LGBM | 0.9140 | 0.91 |
| CatBoost | 0.9041 | 0.90 |
| Gradient Boosting | 0.8790 | 0.88 |
| AdaBoost | 0.8230 | 0.82 |
| Bernoulli Naive Bayes | 0.7536 | 0.76 |
| Gaussian Naive Bayes | 0.7447 | 0.75 |
| Logistic Regression | 0.6409 | 0.66 |
| SGD Classifier | 0.6048 | 0.64 |
 
> **Champion Model: Random Forest** with F1 macro = 0.9555, Accuracy = 0.96
 
### 7. Fusion Classifier (Stacking + Soft Voting)
 
A novel ensemble approach was applied by:
 
1. **Stacking Classifier (TP-optimized):** XGBoost + LGBM → meta-learner: Logistic Regression
2. **Stacking Classifier (TN-optimized):** Gradient Boosting + Gaussian Naive Bayes → meta-learner: Logistic Regression
3. **Soft Voting Ensemble:** Combines both stacking classifiers
**Ensemble F1 (macro): 0.9191** — slightly below the Random Forest champion but demonstrates the fusion approach's viability as a more balanced TP/TN detector.
 
---
 
## Key Findings
 
- **Random Forest** is the best single model for this task (F1 = 0.9555)
- Tree-based ensemble methods (Random Forest, Extra Trees, Bagging) consistently outperform boosting and linear models
- The dataset is well-balanced, making SMOTE have minimal impact on overall accuracy but helping ensure the model doesn't develop class bias
- Jaccard-based features and domain ranking are the most informative features for phishing detection
---
 
## Usage
 
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
# Load and prepare data
df = pd.read_csv("data_Features.csv", on_bad_lines='skip', encoding='latin1')
df = df.drop("domain", axis=1)
df = df.apply(pd.to_numeric, errors="coerce").dropna()
 
X = df.drop("label", axis=1)
y = df["label"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Train champion model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict
y_pred = model.predict(X_test)
```
 
---
 
## Notes
 
- The attempt to install `scikit-learn==0.24` failed (build error) but did not affect the project since `scikit-learn 1.7.2` was available and used throughout
- The `domain` column is dropped before modeling as it is a unique string identifier, not a usable feature
- The notebook was originally run on Google Colab (evidenced by `from google.colab import files`)

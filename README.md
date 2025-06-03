# Credit Card Fraud Detection

This project applies machine learning to detect fraudulent credit card transactions using the 2013 anonymized Kaggle dataset. It addresses severe class imbalance using SMOTE and evaluates performance with metrics focused on the minority (fraud) class.

## Objective

- Detect fraudulent transactions while minimizing false negatives.
- Address extreme class imbalance using oversampling and appropriate metrics.
- Compare and evaluate multiple classifiers with a focus on fraud recall and F1-score.

## Tools & Techniques

- **Preprocessing:** StandardScaler, PCA (exploratory)
- **Balancing:** SMOTE (Synthetic Minority Oversampling Technique)
- **Models Used:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- **Evaluation Metrics:**
  - Precision, Recall, F1-Score
  - ROC AUC and PR AUC
  - Confusion Matrix

## Model Comparison (Average Cross-Validation Results)

| Model               | ROC AUC | PR AUC | Accuracy | Precision | Recall | F1-Score |
|--------------------|---------|--------|----------|-----------|--------|----------|
| **Random Forest**   | 0.9743  | 0.8347 | 0.9993   | 0.8164    | 0.8020 | **0.8069** |
| XGBoost             | 0.9747  | 0.8433 | 0.9989   | 0.6517    | 0.8426 | 0.7336 |
| CatBoost            | 0.9724  | 0.8178 | 0.9983   | 0.5150    | 0.8274 | 0.6329 |
| LightGBM            | 0.9758  | 0.7508 | 0.9942   | 0.2064    | 0.8351 | 0.3307 |
| K-Nearest Neighbors | 0.9335  | 0.5875 | 0.9977   | 0.4145    | 0.8554 | 0.5581 |
| Logistic Regression | 0.9816  | 0.7454 | 0.9726   | 0.0543    | 0.9011 | 0.1023 |

**Conclusion:**  
- **Random Forest** performed best overall, achieving the highest F1-score and strong PR AUC.
- **XGBoost** was also highly competitive and stable across metrics.

## Final Test Set Results (Random Forest)

- **Accuracy:** 0.9993  
- **ROC AUC:** 0.968  
- **PR AUC:** 0.862  
- **F1-Score (Fraud Class):** 0.80

# Key Takeaways

- SMOTE + ensemble learning significantly improved fraud detection performance.
- ROC AUC alone can be misleading — PR AUC and F1-score for the minority class are more reliable in imbalanced scenarios.
- Random Forest was the most consistent and interpretable model overall.

## Files

- `Credit Card Fraud Analysis.ipynb`: Full notebook with data prep, modeling, and evaluation

#### Dataset Used:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

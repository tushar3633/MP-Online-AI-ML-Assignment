# Employee Attrition Prediction using Decision Tree and Random Forest

Name: Tushar Verma   

Registration Number: 23BCE10097

Application Number: IN26011832

Batch Number: 1A

Email: tushar366.verma@gmail.com  

## Objective

The objective of this project is to develop machine learning models to predict employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset. The project compares the performance of Decision Tree and Random Forest classifiers using various evaluation metrics to determine the better-performing model.

---

## Dataset Link

IBM HR Analytics Employee Attrition & Performance Dataset

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Methodology

1. Loaded the IBM HR Analytics Employee Attrition dataset using Pandas.
2. Performed data exploration and identified numerical, categorical, and target variables.
3. Checked for missing values and removed unnecessary columns.
4. Encoded categorical features using Label Encoding.
5. Split the dataset into 80% training and 20% testing sets.
6. Built a Decision Tree Classifier.
7. Built a Random Forest Classifier with 100 estimators.
8. Evaluated both models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
9. Visualized feature importance for the Random Forest model.
10. Compared the performance of both models.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------:|----------:|--------:|----------:|
| Decision Tree | 75.85% | 29.31% | 36.17% | 32.38% |
| Random Forest | 83.33% | 42.86% | 12.77% | 19.67% |

---

## Model Comparison

- Random Forest achieved the highest overall accuracy (83.33%).
- Decision Tree achieved better Recall and F1-Score, identifying more employees who were likely to leave.
- Random Forest generated fewer false positive predictions than Decision Tree.
- Decision Tree is easier to interpret, while Random Forest provides better generalization by combining multiple decision trees.

---

## Conclusion

Both Decision Tree and Random Forest models were successfully implemented for employee attrition prediction. Random Forest achieved the highest overall accuracy and precision, making it the better-performing model for this dataset. The ensemble learning approach of Random Forest reduces overfitting and improves prediction stability. However, the Decision Tree demonstrated higher recall and F1-score, making it more effective at identifying employees who were actually likely to leave. A limitation of the Decision Tree is its tendency to overfit the training data, whereas Random Forest requires greater computational resources and is less interpretable due to its ensemble structure.

---

# Customer Churn Prediction using Logistic Regression

## Project Information

**Author:** Tushar Verma
**Registration Number:** 23BCE10097
**Application Number:** IN26011831
**Batch Number:** 1A
**Email ID:** [tushar366.verma@gmail.com](tushar366.verma@gmail.com)

---

# Objective

The objective of this project is to develop a **Logistic Regression** model to predict customer churn for a telecommunications company using customer demographic information and service usage patterns. The project demonstrates the end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, and performance evaluation.

---

# Dataset

**Telco Customer Churn Dataset**

**Source:** Kaggle – Telco Customer Churn Dataset

**Dataset Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer information such as:

* Customer demographics
* Account information
* Subscription services
* Billing details
* Contract type
* Tenure
* Monthly and total charges
* Churn status (target variable)

---

# Technologies and Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Kaggle API

---

# Methodology

### 1. Data Collection and Understanding

* Loaded the dataset using Pandas.
* Explored the dataset structure, feature types, and target variable.
* Identified numerical and categorical features for preprocessing.

### 2. Data Preprocessing

* Converted the `TotalCharges` column to numeric format.
* Handled missing values by replacing them with the median value.
* Encoded the target variable (`Churn`) into binary values:

  * **Yes → 1**
  * **No → 0**
* Applied **One-Hot Encoding** (`drop_first=True`) to categorical variables.
* Standardized numerical features using **StandardScaler** to improve model convergence and performance.

### 3. Data Splitting

* Divided the dataset into:

  * **80% Training Data**
  * **20% Testing Data**
* Used `train_test_split` from Scikit-learn with a fixed random state to ensure reproducibility.

### 4. Model Development

* Implemented the **Logistic Regression** algorithm using Scikit-learn's `LogisticRegression`.
* Trained the model on the standardized training data.
* Set `max_iter=1000` to ensure convergence during optimization.

### 5. Model Evaluation

The trained model was evaluated using multiple classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score

A **Confusion Matrix** heatmap was also generated to visualize the classification performance and identify correctly and incorrectly classified instances.

---

# Results

The Logistic Regression model achieved the following performance on the test dataset:

* **Accuracy:** **81.97%**
* **Precision:** **68.31%**
* **Recall:** **59.52%**
* **F1-Score:** **63.61%**

### Key Findings

The most influential factors affecting customer churn included:

* Contract type
* Customer tenure
* Monthly charges

The model successfully identified the majority of churn and non-churn cases while maintaining a good balance between precision and recall.

---

# Conclusion

This project demonstrates that **Logistic Regression** is an effective baseline algorithm for predicting customer churn in the telecommunications industry. With an overall **accuracy of 81.97%**, the model provides reliable predictions while remaining computationally efficient and easily interpretable.

Despite its strong performance, Logistic Regression is limited by its linear decision boundary and may not fully capture complex, non-linear relationships between customer attributes. For example, interactions between short customer tenure, high monthly charges, and specific contract types may not be adequately modeled.

Future improvements could involve implementing more advanced machine learning algorithms such as:

* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier

These ensemble methods can model non-linear relationships more effectively and may improve the identification of customers at risk of churning.

---

# Future Scope

* Perform feature engineering to enhance predictive performance.
* Optimize the model through hyperparameter tuning.
* Compare Logistic Regression with advanced ensemble learning algorithms.
* Deploy the trained model as a web application using Flask or Streamlit for real-time customer churn prediction.

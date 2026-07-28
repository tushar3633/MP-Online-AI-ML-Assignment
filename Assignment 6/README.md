# Weather Condition Classification using Support Vector Machine (SVM)

**Name:** Tushar Verma 

**Registration Number:** 23BCE10097

**Application Number:** IN26011831

**Batch Number:** 1A

**Email:** tushar366.verma@gmail.com

---

## Objective

The objective of this assignment is to develop a Support Vector Machine (SVM) classifier that predicts whether the weather is **Warm** or **Cool** using meteorological data obtained from the Open-Meteo API. The model uses temperature, relative humidity, surface pressure, and wind speed as input features and classifies weather conditions using the RBF kernel.

---

## API Documentation Link

- Open-Meteo API: https://open-meteo.com/
- API Endpoint Used:
  https://api.open-meteo.com/v1/forecast

---

## Libraries Used

- Python
- Pandas
- NumPy
- Requests
- Matplotlib
- Scikit-learn

---

## Methodology

1. Fetched hourly weather data from the Open-Meteo API.
2. Converted the JSON response into a Pandas DataFrame.
3. Selected the following input features:
   - Temperature
   - Relative Humidity
   - Surface Pressure
   - Wind Speed
4. Created the target variable **Weather_Class**:
   - Warm → Temperature ≥ 25°C
   - Cool → Temperature < 25°C
5. Checked for missing values and removed unnecessary columns.
6. Encoded the target variable using LabelEncoder.
7. Split the dataset into 80% training and 20% testing sets.
8. Standardized the feature values using StandardScaler.
9. Trained an SVM classifier with the RBF kernel.
10. Evaluated the model using Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.

---

## Results

The SVM classifier produced excellent performance on the test dataset.

| Metric | Value |
|---------|-------|
| Accuracy | **97.06%** |
| Precision | **100.00%** |
| Recall | **91.67%** |
| F1-Score | **95.65%** |

### Confusion Matrix Summary

- Correctly classified **Cool** weather: **22**
- Correctly classified **Warm** weather: **11**
- Incorrectly classified **Warm as Cool**: **1**
- Incorrectly classified **Cool as Warm**: **0**

The model correctly classified **33 out of 34** test samples, demonstrating excellent predictive performance.

---

## Conclusion

This assignment successfully implemented a Support Vector Machine (SVM) classifier to classify weather conditions as **Warm** or **Cool** using temperature, relative humidity, surface pressure, and wind speed obtained from the Open-Meteo API. The trained model achieved an **accuracy of 97.06%**, demonstrating excellent classification performance with only one misclassified test sample. Feature scaling using **StandardScaler** was essential because SVM relies on distance-based calculations, ensuring that all features contribute equally to the decision boundary. A major advantage of the SVM algorithm is its ability to achieve high accuracy and effectively handle nonlinear classification problems using the **RBF kernel**. However, SVM can become computationally expensive and slower to train when working with very large datasets.

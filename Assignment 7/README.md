# Customer Segmentation using K-Means Clustering and PCA

**Name:** Tushar Verma   

**Registration Number:** 23BCE10097   

**Application Number:** IN26011831 

**Batch Number:** 1A

**Email ID:** tushar366.verma@gmail.com  

## Objective
The objective of this project is to segment mall customers into distinct behavioral groups based on demographic details, annual income, and spending scores using K-Means clustering, and visualize the segments in 2D using Principal Component Analysis (PCA)[cite: 2].

## Dataset Link
- [Kaggle: Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)[cite: 2]

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kaggle`

## Methodology
1. **Data Understanding**: Identified numerical (`Age`, `Annual Income`, `Spending Score`) and categorical (`Gender`) attributes[cite: 2].
2. **Data Preprocessing**:
   - Confirmed zero missing values[cite: 2].
   - Dropped non-informative identifier `CustomerID`[cite: 2].
   - Binary encoded `Gender` (`Male`: 1, `Female`: 0)[cite: 2].
   - Standardized features using `StandardScaler`[cite: 2].
3. **Model Development**:
   - Used the Elbow Method (WCSS) to select optimal clusters ($K = 5$)[cite: 2].
   - Trained `KMeans` with $K = 5$ to assign cluster labels[cite: 2].
   - Applied `PCA(n_components=2)` for dimensionality reduction[cite: 2].
4. **Visualization**: Plotted the Elbow Curve, Customer Segment distributions, and 2D PCA Cluster projections[cite: 2].

## Results
- **Optimal Clusters ($K$):** 5 distinct customer groups[cite: 2].
- **PCA Variance Explained:** Component 1 (33.69%) + Component 2 (26.23%) = **59.92% Total Variance Explained**[cite: 2].
- **Segments Identified**:
  1. *High Income, High Spending*: Target VIP Customers.
  2. *High Income, Low Spending*: Potential growth segment.
  3. *Low Income, High Spending*: Impulsive shoppers.
  4. *Low Income, Low Spending*: Frugal shoppers.
  5. *Average Income, Moderate Spending*: Mainstream baseline customers.

## Conclusion
Combining K-Means clustering with PCA enables effective multidimensional customer segmentation for targeted marketing campaigns[cite: 2]. While K-Means is restricted by spherical cluster assumptions, PCA effectively simplifies high-dimensional feature spaces for clear business insight[cite: 2].

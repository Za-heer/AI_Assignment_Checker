# Assignment 5: Feature Scaling for KNN
# Apply feature scaling before training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Synthetic dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Predict
test_data = np.array([2, 3])  # Error: Shape mismatch, should be [[2, 3]]
print(f"Prediction: {knn.predict(test_data)}")
#KNN

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

error_rate = []

# Train and calculate error rate for k = 1 to 40
for k in range(1, 41):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = 1 - accuracy_score(y_test, pred)
    error_rate.append(error)

# Find best k
best_k = error_rate.index(min(error_rate)) + 1
print(f"Best K value: {best_k}")
print(f"Lowest Error Rate: {round(min(error_rate), 4)}")

# Plot Error Rate vs K
plt.figure(figsize=(10, 5))
plt.plot(range(1, 41), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

# Train final model with best K
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f'Confusion Matrix (K = {best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

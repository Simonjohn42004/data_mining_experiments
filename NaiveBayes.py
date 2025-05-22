#NAIVE BAYES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Create and save a simple dataset to CSV
df = pd.DataFrame({
    'feature1': [5.1, 4.9, 6.2, 5.9, 6.9, 5.5, 6.7, 6.0],
    'feature2': [3.5, 3.0, 3.4, 3.0, 3.1, 2.3, 3.3, 2.7],
    'target':   [0, 0, 1, 1, 1, 0, 1, 0]
})
df.to_csv('data.csv', index=False)

# Load CSV
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2']]
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 2))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

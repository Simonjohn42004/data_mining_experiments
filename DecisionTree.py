#Decision tree - iris

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, clf.predict(X_te)) * 100:.2f}%")

plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.title("Iris Decision Tree")
plt.show()

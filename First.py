# 1. Imports & Load Dataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset as DataFrame
data = load_breast_cancer(as_frame=True)
df = pd.concat([data.frame], axis=1)
df.head()

# 2. Inspect & Preprocess
df.info()
df.isnull().sum()  # should be zero; dataset is clean :contentReference[oaicite:1]{index=1}

X = data.data
y = data.target  # 0 = malignant, 1 = benign :contentReference[oaicite:2]{index=2}


# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Baseline Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5. Visualize the Tree
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree – Baseline")
plt.show()

# 6. Hyperparameter Tuning
param_grid = {
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_leaf": [1, 3, 5, 10]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
best_clf = grid.best_estimator_

# 7. Evaluation of Tuned Model
y_pred2 = best_clf.predict(X_test)
print("Tuned Accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt="d")
plt.title("Confusion Matrix – Tuned")
plt.show()

# 8. Feature Importances
imp = pd.Series(best_clf.feature_importances_, index=data.feature_names)
imp = imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(x=imp.values, y=imp.index)
plt.title("Top 15 Feature Importances")
plt.show()

# 9. Final Tree Visualization
plt.figure(figsize=(20,10))
plot_tree(
    best_clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree – Tuned Model")
plt.show()

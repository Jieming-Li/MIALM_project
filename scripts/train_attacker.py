import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

data_path = r"C:\Users\jli93\Desktop\MIALM_project\attack_data\attack_train_data.pkl"

with open(data_path, "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])
y = np.array(data["y"])

#split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#logistic regression model
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=500,
    class_weight="balanced"
)

#train
clf.fit(X_train, y_train)

#evaluate
preds = clf.predict(X_val)
probs = clf.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, preds))
print("ROC-AUC:", roc_auc_score(y_val, probs))

# Save the trained attack model
save_path = r"C:\Users\jli93\Desktop\MIALM_project\attack_model\attack_model_logreg.pkl"
with open(save_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Attack model saved to {save_path}")

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = pd.read_csv("crop_recommendation.csv")
print("Rows and columns:", data.shape)
print("Missing values:", data.isnull().sum().sum())

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

rows = []
best_name = ""
best_acc = 0
best_model = None

for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: test accuracy = {acc*100:.2f}%, CV accuracy = {cv*100:.2f}%")
    rows.append({
        "Model": name,
        "Test Accuracy (%)": round(acc * 100, 2),
        "CV Accuracy (%)": round(cv * 100, 2),
    })

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

results = pd.DataFrame(rows).sort_values("Test Accuracy (%)", ascending=False)
print("\nBest model:", best_name)

best_pred = best_model.predict(X_test)
print(classification_report(y_test, best_pred))

os.makedirs("charts", exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, xticks_rotation=90, cmap="Greens", colorbar=False, ax=ax
)
plt.title("Confusion Matrix - " + best_name)
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=110)
plt.close()

rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", color="green", figsize=(7, 4))
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=110)
plt.close()

saved = {
    "model": best_model,
    "name": best_name,
    "accuracy": round(best_acc * 100, 2),
    "results": results,
}
joblib.dump(saved, "model.pkl")
print("Model saved as model.pkl")

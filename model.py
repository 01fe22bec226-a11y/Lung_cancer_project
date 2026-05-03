import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("Lung_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Inputs
X = df[['AGE', 'SMOKING', 'COUGHING']]

# Output
y = df['LUNG_CANCER'].astype(str).str.strip().str.upper()
y = y.map({'YES': 1, 'NO': 0})

print("Class counts:")
print(y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Better model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Test
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "model.pkl")
print("New model saved")
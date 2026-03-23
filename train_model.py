import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path
import pickle

# Get project root dynamically
base_dir = Path(__file__).parent

# Load dataset
data_path = base_dir / "data" / "ops_data.csv"
df = pd.read_csv(data_path)

# Features & target
X = df[["handle_time", "priority", "agent_experience", "queue_load"]]
y = df["SLA_breached"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
model_path = base_dir / "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
from sklearn.metrics import accuracy_score, roc_auc_score

# After prediction
accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC AUC:", roc)
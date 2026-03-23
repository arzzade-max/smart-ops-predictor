import pickle
import pandas as pd
import os

# Get project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(base_dir, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

def recommend_action(data):
    if data["handle_time"] > 40:
        return "Escalate to senior agent"
    elif data["queue_load"] > 200:
        return "Redistribute workload"
    else:
        return "Normal processing"

def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    action = recommend_action(data)
    return prediction, action


# Test
if __name__ == "__main__":
    sample = {
        "handle_time": 45,
        "priority": 1,
        "agent_experience": 2,
        "queue_load": 220
    }

    pred, action = predict(sample)

    print("Prediction:", "SLA Breach" if pred == 1 else "No Risk")
    print("Action:", action)
import streamlit as st
from pathlib import Path
import pickle
import pandas as pd

# Load model
base_dir = Path(__file__).parent.parent
model_path = base_dir / "model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🚀 Smart Ops Predictor")

handle_time = st.slider("Handle Time", 5, 60)
priority = st.selectbox("Priority", [1, 2, 3])
agent_experience = st.slider("Agent Experience", 1, 10)
queue_load = st.slider("Queue Load", 50, 300)

data = pd.DataFrame([{
    "handle_time": handle_time,
    "priority": priority,
    "agent_experience": agent_experience,
    "queue_load": queue_load
}])

if st.button("Predict"):
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of SLA Breach")
    else:
        st.success("✅ Low Risk")

    st.write(f"Risk Score: {prob:.2%}")
    st.progress(float(prob))

    if handle_time > 40:
        st.write("👉 Action: Escalate to senior agent")
    elif queue_load > 200:
        st.write("👉 Action: Redistribute workload")
    else:
        st.write("👉 Action: Normal processing")
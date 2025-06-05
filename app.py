import streamlit as st
import pandas as pd
import joblib

# Load files
similarity = joblib.load("model.pkl")
df = joblib.load("med_mapping.pkl")

st.title("💊 Disease-based Medication Recommendation System")

# Dropdown to select disease
diseases = df["Disease"].unique()
selected_disease = st.selectbox("Select Disease", diseases)

# Find all rows with selected disease
indices = df[df["Disease"] == selected_disease].index.tolist()

# Average similarity with other entries
avg_scores = similarity[indices].mean(axis=0)
top_indices = avg_scores.argsort()[::-1][1:6]  # top 5 similar, excluding itself

# Get medication recommendations
recommended = df.iloc[top_indices][["Disease", "Medication"]].drop_duplicates()

st.subheader("Recommended Medications:")
for _, row in recommended.iterrows():
    st.write(f"✅ {row['Medication']} (for {row['Disease']})")

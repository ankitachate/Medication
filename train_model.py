import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("medications.csv")

# Combine disease-medication into a string (for text vectorization)
df["combined"] = df["Disease"].astype(str) + " " + df["Medication"].astype(str)

# Create Count Vectors
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df["combined"])

# Compute cosine similarity
similarity = cosine_similarity(count_matrix)

# Save model and mapping
joblib.dump(similarity, "model.pkl")
joblib.dump(df, "med_mapping.pkl")  # Save full dataframe for lookup

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the travel data
df = pd.read_csv('sample_data2.csv')

# Map accessibility and safety ratings to numeric values for fuzzy logic
accessibility_mapping = {'Easy': 3, 'Moderate': 2, 'Hard': 1}
df['accessibility_numeric'] = df['accessibility'].map(accessibility_mapping)

# Vectorize activities, destination type, and climate
df['combined_features'] = df['activities'] + " " + df['destination_type'] + " " + df['climate']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the fuzzy membership functions for budget
budget_range = np.arange(0, 50001, 1)
low_budget = fuzz.trapmf(budget_range, [0, 0, 15000, 25000])
medium_budget = fuzz.trimf(budget_range, [15000, 30000, 45000])
high_budget = fuzz.trapmf(budget_range, [30000, 40000, 50000, 50000])

# Function to classify budget category using fuzzy logic
def classify_budget(budget):
    low_score = fuzz.interp_membership(budget_range, low_budget, budget)
    medium_score = fuzz.interp_membership(budget_range, medium_budget, budget)
    high_score = fuzz.interp_membership(budget_range, high_budget, budget)

    if low_score >= medium_score and low_score >= high_score:
        return "Low"
    elif medium_score >= low_score and medium_score >= high_score:
        return "Medium"
    else:
        return "High"

# Apply fuzzy classification to the average cost per day
df['fuzzy_budget_category'] = df['avg_cost_per_day'].apply(classify_budget)

# Combine both content-based and fuzzy logic for final score
def combine_scores(index, cosine_sim, df):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_similar_destinations = sim_scores[1:4]
    final_recommendations = []

    for idx, _ in top_similar_destinations:
        safety = df['safety_rating'].iloc[idx]
        accessibility_numeric = df['accessibility_numeric'].iloc[idx]

        # Fuzzy logic scores
        safety_score = fuzz.interp_membership(np.arange(0, 11, 1), fuzz.trimf(np.arange(0, 11, 1), [0, 0, 5]), safety)
        accessibility_score = accessibility_numeric / 3.0  # Normalize accessibility score

        final_score = 0.7 * cosine_sim[index][idx] + 0.3 * (safety_score + accessibility_score)
        final_recommendations.append((df['destination'].iloc[idx], final_score))

    return final_recommendations

# Define a Pydantic model for request validation
class RecommendationRequest(BaseModel):
    budget: int
    safety_rating: float
    accessibility: str

# Define the /recommendations route
@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    # Classify budget using fuzzy logic
    budget_category = classify_budget(request.budget)

    # # Find destinations based on content-based filtering and fuzzy logic
    # index = 0  # Example index; you can change this based on user preferences
    # recommendations = combine_scores(index, cosine_sim, df)

    safety_score = fuzz.interp_membership(np.arange(0, 11, 1), fuzz.trimf(np.arange(0, 11, 1), [0, 0, 5]), request.safety_rating)
    accessibility_score = accessibility_mapping.get(request.accessibility, 1) / 3.0  # Normalize accessibility score
    
    # Create a score for each destination based on user input
    df['input_match_score'] = df.apply(
        lambda row: abs(row['safety_rating'] - request.safety_rating) * 0.3 + abs(row['accessibility_numeric'] - accessibility_score) * 0.3, axis=1
    )
    
    # Get the index of the destination with the best match to the user input
    closest_match_idx = df['input_match_score'].idxmin()
    
    # Get recommendations based on the closest match
    recommendations = combine_scores(closest_match_idx, cosine_sim, df)

    return {"budget_category": budget_category, "recommendations": recommendations}

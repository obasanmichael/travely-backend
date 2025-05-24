import numpy as np
import skfuzzy as fuzz
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Improved budget classification using fuzzy logic
def classify_budget(budget):
    """
    Classify budget using improved fuzzy logic with less overlap
    """
    budget_range = np.arange(0, 50001, 1)
    
    # Better defined membership functions
    low_budget = fuzz.trapmf(budget_range, [0, 0, 12000, 18000])
    medium_budget = fuzz.trimf(budget_range, [15000, 25000, 35000])  
    high_budget = fuzz.trapmf(budget_range, [32000, 40000, 50000, 50000])

    low_score = fuzz.interp_membership(budget_range, low_budget, budget)
    medium_score = fuzz.interp_membership(budget_range, medium_budget, budget)
    high_score = fuzz.interp_membership(budget_range, high_budget, budget)

    if low_score >= medium_score and low_score >= high_score:
        return "Low"
    elif medium_score >= low_score and medium_score >= high_score:
        return "Medium"
    else:
        return "High"
    
# Get budget match score - improved to be more selective
def get_budget_match_score(destination_cost, user_budget):
    """
    Calculate how well a destination's cost matches user's budget
    Uses fuzzy logic concepts to calculate a gradual match score rather than binary
    """
    # Calculate budget difference ratio
    if destination_cost > user_budget:
        # Penalize more heavily if destination cost is above user budget
        ratio = user_budget / destination_cost
        # Apply exponential penalty for destinations that are significantly above budget
        return ratio ** 1.5
    else:
        # Less penalty for destinations under budget
        ratio = 0.8 + 0.2 * (destination_cost / user_budget)
        return min(1.0, ratio)  # Cap at 1.0

def load_data():
    """
    Load and prepare the destination data from CSV
    """
    # Check if the file exists
    if not os.path.exists('result2.csv'):
        raise FileNotFoundError("result2.csv file not found. Please make sure it exists in the current directory.")
    
    try:
        # Load the updated travel data
        df = pd.read_csv('result2.csv')  
        
        # Clean column names (strip quotes if they exist)
        df.columns = [col.strip('"') for col in df.columns]
        
        # Create activities from Main Activities Available - handle if column doesn't exist
        if 'Main Activities Available' in df.columns:
            df['activities'] = df['Main Activities Available']
        else:
            # If column doesn't exist, create empty activities column
            df['activities'] = ""
        
        # Map column names to match our logic - with safety checks
        column_mapping = {
            'State': 'state',
            'City': 'city',
            'Destination Name': 'destination',
            'Destination Type': 'destination_type',
            'Climate': 'climate',
            'Best Season to Visit': 'best_season',
            'Least cost per day in Naira': 'avg_cost_per_day',
            'Safety Rating': 'safety_rating',
            'Accessibility': 'accessibility',
            'Accommodation Type': 'accommodation_type',
            'Nearby Hotel': 'nearby_hotel',
            'Hotel Price Range (Naira)': 'hotel_price_range',
            'Feeding Cost Range (Naira)': 'feeding_cost_range',
            'Other Necessities Range (Naira)': 'necessities_range',
        }
        
        # Only rename columns that actually exist
        rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Ensure all required columns exist, create them if they don't
        required_columns = ['state', 'city', 'destination', 'destination_type', 'climate', 
                           'best_season', 'avg_cost_per_day', 'accommodation_type', 
                           'nearby_hotel', 'hotel_price_range', 'feeding_cost_range', 
                           'necessities_range', 'activities']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""  # Add empty column if missing
        
        # Convert avg_cost_per_day to numeric values with error handling
        df['avg_cost_per_day'] = pd.to_numeric(df['avg_cost_per_day'], errors='coerce').fillna(20000)  # Default to medium cost
                
        # Standardize text fields for better matching
        for col in ['destination_type', 'activities']:
            df[col] = df[col].fillna('').str.lower().str.strip()
            
        # Create searchable fields - separately maintained for better filtering
        df['activities_keywords'] = df['activities'].fillna('').str.lower()
        df['destination_type_keywords'] = df['destination_type'].fillna('').str.lower()
        
        return df
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error loading data: {str(e)}")
        # Return empty DataFrame with required columns as fallback
        empty_df = pd.DataFrame(columns=required_columns)
        return empty_df

def get_recommendations(budget, destination_type=None, activity_type=None):
    """
    Generate destination recommendations based on user preferences
    
    Parameters:
    - budget: user's budget in Naira
    - destination_type: type of destination (e.g., Nature, Cultural, Urban Leisure)
    - activity_type: preferred activities (e.g., Hiking, Swimming)
    
    Returns:
    - Dictionary with user budget category and recommendations
    """
    try:
        df = load_data()
        
        # Validate inputs
        try:
            budget = float(budget)
        except (ValueError, TypeError):
            budget = 25000  # Default to medium budget
            
        # Handle empty DataFrame
        if df.empty:
            return {
                "user_budget_category": classify_budget(budget),
                "recommendations": []
            }
            
        # Apply pre-filtering to get a more diverse set of results
        candidates_df = df.copy()
        
        # Apply strong filters first (must-have preferences)
        temp_filtered_df = candidates_df.copy()
        
        if destination_type:
            destination_type = str(destination_type).lower().strip()
            # More flexible matching for destination type
            mask = temp_filtered_df['destination_type_keywords'].str.contains(destination_type, case=False, na=False)
            filtered_df = temp_filtered_df[mask]
            if not filtered_df.empty:
                temp_filtered_df = filtered_df
        
        if activity_type and not temp_filtered_df.empty:
            activity_type = str(activity_type).lower().strip()
            mask = temp_filtered_df['activities_keywords'].str.contains(activity_type, case=False, na=False)
            filtered_df = temp_filtered_df[mask]
            if not filtered_df.empty:
                temp_filtered_df = filtered_df
        
        # Use pre-filtered results if we have enough, otherwise use original dataset
        if len(temp_filtered_df) >= 5:
            candidates_df = temp_filtered_df
        
        # Classify budget using fuzzy logic
        budget_category = classify_budget(budget)
        
        # Create feature vectors for content-based filtering
        # This is where we implement content-based filtering for personalized recommendations
        
        # Create feature vectors for destinations
        destination_features = {
            'destination_type': candidates_df['destination_type'].fillna(''),
            'activities': candidates_df['activities'].fillna('')
        }
        
        # Create user preference vectors
        user_preferences = {
            'destination_type': str(destination_type).lower() if destination_type else '',
            'activities': str(activity_type).lower() if activity_type else ''
        }
        
        # Default values if user didn't specify preferences
        if not user_preferences['destination_type'] and not user_preferences['activities']:
            user_preferences['activities'] = 'nature adventure leisure'
            
        # Create TF-IDF matrices for each feature
        tfidf_matrices = {}
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Calculate similarity scores for each feature using content-based filtering
        feature_similarities = {}
        
        for feature in ['destination_type', 'activities']:
            # Only process if we have user preferences for this feature
            if user_preferences[feature]:
                # Combine destination features with user preference
                combined_texts = list(destination_features[feature]) + [user_preferences[feature]]
                
                # Create TF-IDF matrix
                try:
                    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)
                    
                    # Get similarity between user preferences and destinations
                    user_vector = tfidf_matrix[-1]  # Last row is user preferences
                    destination_vectors = tfidf_matrix[:-1]  # All other rows are destinations
                    
                    # Calculate cosine similarity
                    similarities = cosine_similarity(user_vector, destination_vectors).flatten()
                    feature_similarities[feature] = similarities
                except:
                    # Handle case where vectorizer fails (e.g., empty strings)
                    feature_similarities[feature] = np.ones(len(candidates_df))
            else:
                # If no user preference for this feature, assign neutral scores
                feature_similarities[feature] = np.ones(len(candidates_df))
        
        # Initialize a scaler to normalize scores
        scaler = MinMaxScaler()
        
        # Calculate scores for each destination
        recommendations = []
        
        for index in range(len(candidates_df)):
            try:
                destination_data = candidates_df.iloc[index]
                
                # Basic destination info
                destination = str(destination_data['destination'])
                state = str(destination_data['state'])
                city = str(destination_data['city'])
                dest_type = str(destination_data['destination_type'])
                avg_cost = float(destination_data['avg_cost_per_day'])
                best_season = str(destination_data['best_season'])
                accommodation = str(destination_data['accommodation_type'])
                activities = str(destination_data['activities'])
                dest_climate = str(destination_data['climate'])
                nearby_hotel = str(destination_data['nearby_hotel'])
                hotel_price_range = str(destination_data['hotel_price_range'])
                feeding_cost_range = str(destination_data['feeding_cost_range'])
                necessities_range = str(destination_data['necessities_range'])
                
                # Calculate individual scores
                
                # 1. Budget Score - using fuzzy logic for gradual matching
                budget_score = get_budget_match_score(avg_cost, budget)
                
                # 2. Content Similarity Score (from TF-IDF) - content-based filtering
                # Weighted combination of feature similarities
                content_weights = {
                    'destination_type': 0.5,
                    'activities': 0.5
                }
                
                # Calculate weighted content similarity
                content_score = sum(
                    content_weights[feature] * feature_similarities[feature][index]
                    for feature in feature_similarities
                )
                
                # Calculate final weighted score - combining fuzzy logic and content-based filtering
                final_score = (
                    0.4 * budget_score + 
                    0.6 * content_score
                )
                
                # Create recommendation object
                recommendations.append({
                    "destination": destination,
                    "state": state,
                    "city": city,
                    "destination_type": dest_type,
                    "activities": activities,
                    "climate": dest_climate,
                    "avg_cost_per_day": avg_cost,
                    "best_season": best_season,
                    "accommodation_type": accommodation,
                    "nearby_hotel": nearby_hotel,
                    "hotel_price_range": hotel_price_range,
                    "feeding_cost_range": feeding_cost_range,
                    "necessities_range": necessities_range,
                    "budget_category": classify_budget(avg_cost),
                    "score": float(final_score)
                })
            except Exception as e:
                # Skip this destination if there's an error processing it
                print(f"Error processing destination at index {index}: {str(e)}")
                continue

        # Sorting by final score
        recommendations_sorted = sorted(recommendations, key=lambda x: x["score"], reverse=True)
        
        # Return top recommendations, but ensure diversity
        # We ensure we don't return too many of the same state
        state_counts = {}
        diversified_recommendations = []
        
        for rec in recommendations_sorted:
            state = rec["state"]
            # Limit max 2 recommendations from same state
            if state_counts.get(state, 0) < 2:
                state_counts[state] = state_counts.get(state, 0) + 1
                diversified_recommendations.append(rec)
            
            if len(diversified_recommendations) >= 5:
                break
        
        # Backfill if we don't have enough diversified recommendations
        if len(diversified_recommendations) < 5 and len(recommendations_sorted) > len(diversified_recommendations):
            for rec in recommendations_sorted:
                if rec not in diversified_recommendations:
                    diversified_recommendations.append(rec)
                if len(diversified_recommendations) >= 5:
                    break

        return {
            "user_budget_category": budget_category,
            "recommendations": diversified_recommendations
        }
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in get_recommendations: {str(e)}")
        # Return empty recommendations as fallback
        return {
            "user_budget_category": classify_budget(budget) if budget else "Medium",
            "recommendations": []
        }
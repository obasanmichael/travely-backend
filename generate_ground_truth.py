import pandas as pd

# Load your destinations dataset
df = pd.read_csv("result2.csv")
print(df.columns)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# dest_type = dest_type.lower()
# filtered = df[df["destination_type"] == dest_type]


# Example test cases (inputs you'd want to evaluate)
test_cases = [
    {"user_id": "U1", "budget": 25000, "destination_type": "nature", "activity_type": "hiking"},
    {"user_id": "U2", "budget": 15000, "destination_type": "cultural", "activity_type": "museum"},
    {"user_id": "U3", "budget": 20000, "destination_type": "urban leisure", "activity_type": "shopping"}
]

# Helper function to build ground truth
def build_ground_truth(df, case):
    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["destination_type"] = df["destination_type"].str.strip().str.lower()
    df["main_activities_available"] = df["main_activities_available"].str.strip().str.lower()
    
    dest_type = case.get("destination_type", "").lower()
    activity_type = case.get("activity_type", "").lower()
    
    if dest_type:
        df = df[df["destination_type"].str.contains(dest_type, na=False)]
    if activity_type:
        df = df[df["main_activities_available"].str.contains(activity_type, na=False)]
    
    destinations = df["destination_name"].tolist()
    print("Filtered destinations:", destinations)
    return destinations

# Generate ground truth lists for each case
for case in test_cases:
    preferred_destinations = build_ground_truth(df, case)
    case["preferred_destinations"] = preferred_destinations

# Display ground truth-enhanced test cases
for case in test_cases:
    print(f"\nUser {case['user_id']} Ground Truth Destinations:")
    print(case["preferred_destinations"])

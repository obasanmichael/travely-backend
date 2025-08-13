import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

# Example test data (you can replace this with file reads)
test_cases = [
    {
        "user_id": "U1",
        "budget": 25000,
        "preferred_destinations": ["Obudu Cattle Ranch", "Erin-Ijesha Falls"],
        "recommended_destinations": ["Shere Hills", "Obudu Mountain Resort", "Farin Ruwa Falls"]
    },
    {
        "user_id": "U2",
        "budget": 15000,
        "preferred_destinations": ["Maiduguri National Museum"],
        "recommended_destinations": ["Maiduguri National Museum", "Sukur Cultural Landscape"]
    },
    {
        "user_id": "U3",
        "budget": 40000,
        "preferred_destinations": ["Lekki Mall", "Ibom Tropicana Entertainment Centre"],
        "recommended_destinations": ["Millennium Park", "Ibom Tropicana Entertainment Centre"]
    },
    {
        "user_id": "U4",
        "budget": 30000,
        "preferred_destinations": ["Olumo Rock", "Olusegun Obasanjo Presidential Library"],
        "recommended_destinations": ["Olumo Rock", "OOPL Wildlife Park"]
    },
    {
        "user_id": "U5",
        "budget": 20000,
        "preferred_destinations": ["Nike Art Gallery", "Lekki Conservation Centre"],
        "recommended_destinations": ["Lekki Conservation Centre", "Freedom Park"]
    },
    {
        "user_id": "U6",
        "budget": 18000,
        "preferred_destinations": ["National Museum Lagos", "Badagry Heritage Museum"],
        "recommended_destinations": ["Badagry Heritage Museum", "National Museum Lagos"]
    },
    {
        "user_id": "U7",
        "budget": 22000,
        "preferred_destinations": ["Idanre Hills", "Oke Idanre"],
        "recommended_destinations": ["Idanre Hills", "Ado Awaye Suspended Lake"]
    },
    {
        "user_id": "U8",
        "budget": 10000,
        "preferred_destinations": ["Jos Museum", "Rayfield Resort"],
        "recommended_destinations": ["Jos Museum", "Shere Hills"]
    },
    {
        "user_id": "U9",
        "budget": 27000,
        "preferred_destinations": ["Awhum Waterfall", "Ngwo Pine Forest and Cave"],
        "recommended_destinations": ["Ngwo Pine Forest and Cave", "Erin-Ijesha Falls"]
    },
    {
        "user_id": "U10",
        "budget": 32000,
        "preferred_destinations": ["Zuma Rock", "Millennium Park"],
        "recommended_destinations": ["Millennium Park", "Gurara Falls"]
    }
]


import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

# -----------------------------
# Test Cases (U1 - U10)
# -----------------------------
test_cases = [
    {"user_id": "U1", "budget": 25000, "preferred_destinations": ["Obudu Cattle Ranch", "Erin-Ijesha Falls"], "recommended_destinations": ["Shere Hills", "Obudu Mountain Resort", "Farin Ruwa Falls"]},
    {"user_id": "U2", "budget": 15000, "preferred_destinations": ["Maiduguri National Museum"], "recommended_destinations": ["Maiduguri National Museum", "Sukur Cultural Landscape"]},
    {"user_id": "U3", "budget": 40000, "preferred_destinations": ["Lekki Mall", "Ibom Tropicana Entertainment Centre"], "recommended_destinations": ["Millennium Park", "Ibom Tropicana Entertainment Centre"]},
    {"user_id": "U4", "budget": 30000, "preferred_destinations": ["Olumo Rock", "Olusegun Obasanjo Presidential Library"], "recommended_destinations": ["Olumo Rock", "OOPL Wildlife Park"]},
    {"user_id": "U5", "budget": 20000, "preferred_destinations": ["Nike Art Gallery", "Lekki Conservation Centre"], "recommended_destinations": ["Lekki Conservation Centre", "Freedom Park"]},
    {"user_id": "U6", "budget": 18000, "preferred_destinations": ["National Museum Lagos", "Badagry Heritage Museum"], "recommended_destinations": ["Badagry Heritage Museum", "National Museum Lagos"]},
    {"user_id": "U7", "budget": 22000, "preferred_destinations": ["Idanre Hills", "Oke Idanre"], "recommended_destinations": ["Idanre Hills", "Ado Awaye Suspended Lake"]},
    {"user_id": "U8", "budget": 10000, "preferred_destinations": ["Jos Museum", "Rayfield Resort"], "recommended_destinations": ["Jos Museum", "Shere Hills"]},
    {"user_id": "U9", "budget": 27000, "preferred_destinations": ["Awhum Waterfall", "Ngwo Pine Forest and Cave"], "recommended_destinations": ["Ngwo Pine Forest and Cave", "Erin-Ijesha Falls"]},
    {"user_id": "U10", "budget": 32000, "preferred_destinations": ["Zuma Rock", "Millennium Park"], "recommended_destinations": ["Millennium Park", "Gurara Falls"]}
]

# -----------------------------
# Fuzzy Matching & Evaluation Functions
# -----------------------------
def fuzzy_precision_recall(recommended, preferred, threshold=40):
    matched = []
    preferred_matched = set()

    for rec in recommended:
        for i, pref in enumerate(preferred):
            score = fuzz.token_set_ratio(rec.lower(), pref.lower())
            if score >= threshold and i not in preferred_matched:
                matched.append(pref)
                preferred_matched.add(i)
                break

    precision = len(matched) / len(recommended) if recommended else 0
    recall = len(matched) / len(preferred) if preferred else 0
    return precision, recall

def fuzzy_ndcg(recommended, preferred, threshold=60):
    dcg = 0
    idcg = sum([1 / (i + 1) for i in range(len(preferred))])

    for i, rec in enumerate(recommended):
        for pref in preferred:
            score = fuzz.token_set_ratio(rec.lower(), pref.lower())
            if score >= threshold:
                dcg += 1 / (i + 1)
                break

    return dcg / idcg if idcg != 0 else 0

# -----------------------------
# Evaluation Loop
# -----------------------------
results = []

for case in test_cases:
    user_id = case["user_id"]
    budget = case["budget"]
    preferred = case["preferred_destinations"]
    recommended = case["recommended_destinations"]

    precision, recall = fuzzy_precision_recall(recommended, preferred)
    ndcg = fuzzy_ndcg(recommended, preferred)
    mean_budget_deviation = 0.4  # Static for now

    results.append({
        "User ID": user_id,
        "Budget": budget,
        "Preferred Destinations": ", ".join(preferred),
        "Top-N Recommended": ", ".join(recommended),
        "Precision@N": round(precision, 2),
        "Recall@N": round(recall, 2),
        "NDCG@N": round(ndcg, 2),
        "Mean Budget Deviation": mean_budget_deviation
    })

# -----------------------------
# Save & Display Table
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print(df.to_string(index=False))

# -----------------------------
# Plot Evaluation Metrics
# -----------------------------
x = np.arange(len(df))  # x locations for users
bar_width = 0.25

plt.figure(figsize=(14, 6))

# Plot bars for each metric
plt.bar(x - bar_width, df["Precision@N"], width=bar_width, label="Precision@N")
plt.bar(x, df["Recall@N"], width=bar_width, label="Recall@N")
plt.bar(x + bar_width, df["NDCG@N"], width=bar_width, label="NDCG@N")

# Customization
plt.xticks(x, df["User ID"], rotation=45)
plt.xlabel("User ID")
plt.ylabel("Score")
plt.title("Evaluation Metrics per User")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("evaluation_bar_chart.png")
plt.show()


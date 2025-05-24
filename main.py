from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from recommendation_logic import get_recommendations
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI(
    title="Travely API",
    description="API for personalized travel recommendations in Nigeria",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    budget: float
    destination_type: Optional[str] = None
    activity_type: Optional[str] = None

class Recommendation(BaseModel):
    destination: str
    state: str
    city: str
    destination_type: str
    activities: str
    climate: str
    avg_cost_per_day: float
    best_season: str
    accommodation_type: str
    nearby_hotel: str
    hotel_price_range: str
    feeding_cost_range: str
    necessities_range: str
    budget_category: str
    score: float

class RecommendationResponse(BaseModel):
    user_budget_category: str
    recommendations: List[Recommendation]

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to provide better error details"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"An error occurred: {str(exc)}",
            "traceback": traceback.format_exc()
        },
    )

@app.get("/", tags=["Home"])
def home():
    """
    Welcome endpoint for the Travely Recommendation API
    """
    return {
        "message": "Welcome to the Travely Recommendation API",
        "documentation": "/docs",
        "version": "2.0.0"
    }

@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend(request: RecommendationRequest):
    """
    Get personalized travel recommendations based on user preferences
    """
    try:
        # Input validation
        if request.budget <= 0:
            raise HTTPException(status_code=400, detail="Budget must be greater than 0")
        
        # Call recommendation logic with simplified inputs
        recommendations = get_recommendations(
            budget=request.budget,
            destination_type=request.destination_type,
            activity_type=request.activity_type
        )
        
        # Check if recommendations were returned
        if not recommendations or "recommendations" not in recommendations:
            return {
                "user_budget_category": "Medium",
                "recommendations": []
            }
            
        return recommendations
        
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in /recommendations endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Health check endpoint for monitoring
@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API is running properly
    """
    return {"status": "healthy"}
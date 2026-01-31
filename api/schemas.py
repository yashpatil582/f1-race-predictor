"""
Pydantic schemas for the F1 Race Predictor API.

Defines request and response models for all API endpoints.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DriverPrediction(BaseModel):
    """Prediction result for a single driver."""
    driver_code: str = Field(..., description="Driver abbreviation (e.g., 'VER', 'HAM')")
    team: str = Field(..., description="Constructor/team name")
    grid_position: int = Field(..., description="Starting grid position")
    predicted_position: float = Field(..., description="Predicted finishing position")
    confidence_lower: float = Field(..., description="Lower bound of 95% confidence interval")
    confidence_upper: float = Field(..., description="Upper bound of 95% confidence interval")
    position_probabilities: Dict[int, float] = Field(
        default_factory=dict,
        description="Probability distribution over finishing positions"
    )


class RacePredictionResponse(BaseModel):
    """Response model for race prediction endpoint."""
    year: int = Field(..., description="Season year")
    round_number: int = Field(..., description="Round number in the season")
    event_name: str = Field(..., description="Name of the Grand Prix")
    predictions: List[DriverPrediction] = Field(..., description="Predictions for all drivers")
    model_confidence: float = Field(..., description="Overall model confidence (0-100)")


class FeatureImportance(BaseModel):
    """Feature importance score."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score (0-1)")


class ModelInfoResponse(BaseModel):
    """Response model for model information endpoint."""
    is_fitted: bool = Field(..., description="Whether the model is trained")
    training_races: Optional[int] = Field(None, description="Number of races used for training")
    training_samples: Optional[int] = Field(None, description="Number of training samples")
    validation_mae: Optional[float] = Field(None, description="Mean Absolute Error on validation set")
    num_features: Optional[int] = Field(None, description="Number of features used")
    feature_names: Optional[List[str]] = Field(None, description="List of feature names")
    model_type: Optional[str] = Field(None, description="Type of ML model used")
    feature_importance: Optional[List[FeatureImportance]] = Field(
        None, description="Top feature importance scores"
    )


class HistoricalPrediction(BaseModel):
    """Single historical prediction result."""
    driver_code: str
    team: str
    grid_position: int
    predicted_position: float
    actual_position: Optional[int]
    error: Optional[float]


class RaceHistoricalResult(BaseModel):
    """Historical prediction results for a single race."""
    round_number: int
    event_name: str
    mae: float
    predictions: List[HistoricalPrediction]


class HistoricalPredictionResponse(BaseModel):
    """Response model for historical predictions endpoint."""
    year: int = Field(..., description="Season year")
    races: List[RaceHistoricalResult] = Field(..., description="Results for each race")
    season_mae: float = Field(..., description="Average MAE across the season")
    season_accuracy_within_3: float = Field(..., description="Percentage of predictions within 3 positions")


class RaceEvent(BaseModel):
    """Information about a race event."""
    round_number: int
    event_name: str
    date: str
    country: str
    circuit: str


class ScheduleResponse(BaseModel):
    """Response model for schedule endpoint."""
    year: int
    races: List[RaceEvent]


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

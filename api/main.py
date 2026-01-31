"""
F1 Race Outcome Predictor - FastAPI Backend

Provides REST API endpoints for:
- Race outcome predictions
- Historical prediction analysis
- Model information and metrics
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    DriverPrediction,
    RacePredictionResponse,
    ModelInfoResponse,
    HistoricalPredictionResponse,
    RaceHistoricalResult,
    HistoricalPrediction,
    ScheduleResponse,
    RaceEvent,
    HealthResponse,
    ErrorResponse,
    FeatureImportance,
)

from src.ml.data_collector import F1DataCollector
from src.ml.race_predictor import RacePredictor
from src.ml.model_evaluator import ModelEvaluator


# Initialize FastAPI app
app = FastAPI(
    title="F1 Race Outcome Predictor API",
    description="""
    ML-powered API for predicting F1 race finishing positions.

    Features:
    - Predict race outcomes based on qualifying, weather, and historical data
    - Analyze historical prediction accuracy
    - View model metrics and feature importance
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_collector = F1DataCollector()
predictor = RacePredictor(data_collector=data_collector)
historical_data_cache = {}


def load_model_if_exists():
    """Try to load a pre-trained model."""
    try:
        predictor.load_model()
        return True
    except FileNotFoundError:
        return False


# Try to load model on startup
model_loaded = load_model_if_exists()


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_fitted,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_fitted,
        version="1.0.0"
    )


@app.get(
    "/predict/{year}/{round_number}",
    response_model=RacePredictionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict_race(
    year: int,
    round_number: int,
    use_cached_model: bool = Query(True, description="Use pre-trained model if available")
):
    """
    Predict race finishing positions for a specific race.

    - **year**: Season year (e.g., 2024)
    - **round_number**: Round number in the season (1-24)
    """
    if not predictor.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first using /model/train endpoint."
        )

    try:
        # Get race data
        race_data = data_collector.collect_race_data(year, round_number)
        if race_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Race data not found for {year} Round {round_number}"
            )

        # Get historical data for features
        cache_key = f"{year}_{round_number}"
        if cache_key not in historical_data_cache:
            # Collect historical data
            historical = []
            for y in range(year - 2, year + 1):
                try:
                    season_data = data_collector.collect_season_data(y)
                    historical.extend(season_data)
                except Exception:
                    continue

            # Filter to races before this one
            historical = [
                r for r in historical
                if r.year < year or (r.year == year and r.round_number < round_number)
            ]
            historical_data_cache[cache_key] = historical

        historical = historical_data_cache[cache_key]

        if len(historical) < 5:
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical data for prediction"
            )

        # Make prediction
        prediction = predictor.predict_race(race_data, historical)

        # Convert to response model
        driver_predictions = [
            DriverPrediction(
                driver_code=p.driver_code,
                team=p.team,
                grid_position=p.grid_position,
                predicted_position=round(p.predicted_position, 2),
                confidence_lower=round(p.confidence_lower, 2),
                confidence_upper=round(p.confidence_upper, 2),
                position_probabilities={
                    k: round(v, 4) for k, v in p.position_probabilities.items()
                }
            )
            for p in prediction.predictions
        ]

        return RacePredictionResponse(
            year=prediction.year,
            round_number=prediction.round_number,
            event_name=prediction.event_name,
            predictions=driver_predictions,
            model_confidence=round(prediction.model_confidence, 1)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/historical/{year}",
    response_model=HistoricalPredictionResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_historical_predictions(year: int):
    """
    Get historical predictions vs actual results for a season.

    - **year**: Season year to analyze
    """
    if not predictor.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Model not trained"
        )

    try:
        # Collect all data
        all_races = []
        for y in range(year - 2, year + 1):
            try:
                season_data = data_collector.collect_season_data(y)
                all_races.extend(season_data)
            except Exception:
                continue

        # Get races for the specified year
        year_races = [r for r in all_races if r.year == year]

        if not year_races:
            raise HTTPException(
                status_code=404,
                detail=f"No races found for {year}"
            )

        # Evaluate each race
        evaluator = ModelEvaluator(predictor)
        race_results = []
        all_errors = []

        for race in year_races:
            historical = [
                r for r in all_races
                if r.year < race.year or (r.year == race.year and r.round_number < race.round_number)
            ]

            if len(historical) < 5:
                continue

            try:
                eval_result = evaluator.evaluate_race(race, historical)

                predictions = [
                    HistoricalPrediction(
                        driver_code=p['driver_code'],
                        team=p['team'],
                        grid_position=p['grid_position'],
                        predicted_position=p['predicted_position'],
                        actual_position=p['actual_position'],
                        error=p['error']
                    )
                    for p in eval_result.predictions
                ]

                race_results.append(RaceHistoricalResult(
                    round_number=eval_result.round_number,
                    event_name=eval_result.event_name,
                    mae=round(eval_result.mae, 2),
                    predictions=predictions
                ))

                all_errors.extend([p['error'] for p in eval_result.predictions])

            except Exception as e:
                print(f"Error evaluating {race.event_name}: {e}")
                continue

        if not race_results:
            raise HTTPException(
                status_code=404,
                detail=f"Could not evaluate any races for {year}"
            )

        import numpy as np
        season_mae = float(np.mean(all_errors))
        within_3 = sum(1 for e in all_errors if e <= 3) / len(all_errors) * 100

        return HistoricalPredictionResponse(
            year=year,
            races=race_results,
            season_mae=round(season_mae, 2),
            season_accuracy_within_3=round(within_3, 1)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the trained model."""
    info = predictor.get_model_info()

    if not info.get('is_fitted', False):
        return ModelInfoResponse(
            is_fitted=False,
            training_races=None,
            training_samples=None,
            validation_mae=None,
            num_features=None,
            feature_names=None,
            model_type=None,
            feature_importance=None
        )

    # Get feature importance
    importance = predictor.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    return ModelInfoResponse(
        is_fitted=True,
        training_races=info.get('training_races'),
        training_samples=info.get('training_samples'),
        validation_mae=info.get('validation_mae'),
        num_features=info.get('num_features'),
        feature_names=info.get('feature_names'),
        model_type=info.get('model_type'),
        feature_importance=[
            FeatureImportance(feature=f, importance=round(i, 4))
            for f, i in sorted_importance
        ]
    )


@app.post("/model/train", response_model=ModelInfoResponse)
async def train_model(
    training_years: str = Query("2022,2023", description="Comma-separated years for training"),
    save: bool = Query(True, description="Save the trained model")
):
    """
    Train the prediction model on historical data.

    - **training_years**: Comma-separated list of years to use for training
    - **save**: Whether to save the trained model to disk
    """
    try:
        years = [int(y.strip()) for y in training_years.split(",")]

        # Collect training data
        training_races = data_collector.collect_multi_season_data(years)

        if len(training_races) < 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient training data. Need at least 10 races."
            )

        # Train model
        metrics = predictor.train(training_races)

        if save:
            predictor.save_model()

        return await get_model_info()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schedule/{year}", response_model=ScheduleResponse)
async def get_schedule(year: int):
    """
    Get the race schedule for a season.

    - **year**: Season year
    """
    try:
        schedule = data_collector.get_schedule(year)

        races = [
            RaceEvent(
                round_number=int(row['RoundNumber']),
                event_name=row['EventName'],
                date=str(row['EventDate'].date()),
                country=row['Country'],
                circuit=row['Location']
            )
            for _, row in schedule.iterrows()
        ]

        return ScheduleResponse(year=year, races=races)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

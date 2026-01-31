"""
F1 Race Predictor - XGBoost model for predicting race finishing positions.

This module implements an XGBoost-based model that predicts race finishing
positions based on qualifying results, historical data, weather, and circuit
characteristics.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using fallback model.")

from .data_collector import RaceData, F1DataCollector
from .feature_engineer import FeatureEngineer


@dataclass
class PredictionResult:
    """Container for a single driver's prediction."""
    driver_code: str
    team: str
    grid_position: int
    predicted_position: float
    confidence_lower: float
    confidence_upper: float
    position_probabilities: Dict[int, float]


@dataclass
class RacePrediction:
    """Container for full race prediction results."""
    year: int
    round_number: int
    event_name: str
    predictions: List[PredictionResult]
    model_confidence: float


class RacePredictor:
    """
    XGBoost-based F1 race outcome predictor.

    Predicts finishing positions based on:
    - Qualifying performance
    - Historical driver/constructor statistics
    - Weather conditions
    - Circuit characteristics
    - Recent form
    """

    def __init__(
        self,
        model_dir: str = "models",
        data_collector: Optional[F1DataCollector] = None
    ):
        """
        Initialize the race predictor.

        Args:
            model_dir: Directory to save/load models
            data_collector: F1DataCollector instance
        """
        self.model_dir = model_dir
        self.data_collector = data_collector or F1DataCollector()
        self.feature_engineer = FeatureEngineer(self.data_collector)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = self.feature_engineer.get_feature_names()
        self.is_fitted = False

        # Training metadata
        self.training_races = 0
        self.training_samples = 0
        self.val_mae = None

    def train(
        self,
        races: List[RaceData],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the prediction model.

        Args:
            races: List of historical race data
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of training metrics
        """
        print(f"Preparing training data from {len(races)} races...")

        # Extract features
        X, y = self.feature_engineer.prepare_training_data(races, min_historical_races=5)

        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train XGBoost model
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            # Fallback to simple gradient boosting
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state
            )

        print("Training model...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)] if HAS_XGBOOST else None,
            verbose=False
        )

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )

        self.is_fitted = True
        self.training_races = len(races)
        self.training_samples = len(X)
        self.val_mae = val_mae

        metrics = {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'training_samples': len(X),
            'training_races': len(races),
        }

        print(f"\nTraining Results:")
        print(f"  Train MAE: {train_mae:.2f} positions")
        print(f"  Val MAE: {val_mae:.2f} positions")
        print(f"  Val RMSE: {val_rmse:.2f} positions")
        print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

        return metrics

    def predict_race(
        self,
        race: RaceData,
        historical_races: List[RaceData]
    ) -> RacePrediction:
        """
        Predict finishing positions for a race.

        Args:
            race: The race to predict
            historical_races: Historical races for feature calculation

        Returns:
            RacePrediction with all driver predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction. Call train() first.")

        # Extract features for this race
        features_df = self.feature_engineer.extract_features(race, historical_races)

        # Prepare prediction data
        X_pred = features_df[self.feature_names].copy()
        X_pred = X_pred.fillna(X_pred.median())
        X_pred_scaled = self.scaler.transform(X_pred)

        # Get predictions
        predictions = self.model.predict(X_pred_scaled)

        # Calculate prediction uncertainty using tree variance
        if HAS_XGBOOST:
            # Get predictions from each tree for uncertainty
            booster = self.model.get_booster()
            dmatrix = xgb.DMatrix(X_pred_scaled)
            tree_preds = []
            for i in range(self.model.n_estimators):
                tree_pred = booster.predict(dmatrix, iteration_range=(0, i+1))
                tree_preds.append(tree_pred)
            tree_preds = np.array(tree_preds[-20:])  # Last 20 trees
            pred_std = np.std(tree_preds, axis=0)
        else:
            # Estimate uncertainty from validation MAE
            pred_std = np.full(len(predictions), self.val_mae or 3.0)

        # Build prediction results
        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            pred_pos = predictions[i]
            std = pred_std[i] if isinstance(pred_std, np.ndarray) else pred_std

            # Calculate position probabilities (simplified)
            probs = self._calculate_position_probabilities(pred_pos, std)

            result = PredictionResult(
                driver_code=row['driver_code'],
                team=row['team'],
                grid_position=int(row['grid_position']),
                predicted_position=float(pred_pos),
                confidence_lower=float(max(1, pred_pos - 1.96 * std)),
                confidence_upper=float(min(20, pred_pos + 1.96 * std)),
                position_probabilities=probs
            )
            results.append(result)

        # Sort by predicted position
        results.sort(key=lambda x: x.predicted_position)

        # Calculate overall confidence
        avg_uncertainty = np.mean(pred_std) if isinstance(pred_std, np.ndarray) else pred_std
        confidence = max(0, min(100, 100 - avg_uncertainty * 10))

        return RacePrediction(
            year=race.year,
            round_number=race.round_number,
            event_name=race.event_name,
            predictions=results,
            model_confidence=confidence
        )

    def _calculate_position_probabilities(
        self,
        predicted_pos: float,
        std: float
    ) -> Dict[int, float]:
        """
        Calculate probability distribution over finishing positions.

        Args:
            predicted_pos: Mean predicted position
            std: Standard deviation of prediction

        Returns:
            Dictionary mapping position to probability
        """
        probs = {}
        total = 0

        for pos in range(1, 21):
            # Use normal distribution centered at predicted position
            z = (pos - predicted_pos) / max(std, 0.5)
            prob = np.exp(-0.5 * z * z)
            probs[pos] = prob
            total += prob

        # Normalize
        for pos in probs:
            probs[pos] = probs[pos] / total

        return probs

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first.")

        if HAS_XGBOOST:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_

        return dict(zip(self.feature_names, importance))

    def save_model(self, filename: str = "race_predictor.pkl"):
        """
        Save the trained model to disk.

        Args:
            filename: Name of the file to save
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving.")

        os.makedirs(self.model_dir, exist_ok=True)
        filepath = os.path.join(self.model_dir, filename)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_races': self.training_races,
            'training_samples': self.training_samples,
            'val_mae': self.val_mae,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model saved to {filepath}")

    def load_model(self, filename: str = "race_predictor.pkl"):
        """
        Load a trained model from disk.

        Args:
            filename: Name of the file to load
        """
        filepath = os.path.join(self.model_dir, filename)

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_races = model_data['training_races']
        self.training_samples = model_data['training_samples']
        self.val_mae = model_data['val_mae']
        self.is_fitted = True

        print(f"Model loaded from {filepath}")
        print(f"  Training races: {self.training_races}")
        print(f"  Training samples: {self.training_samples}")
        print(f"  Validation MAE: {self.val_mae:.2f}")

    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.

        Returns:
            Dictionary with model metadata
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'message': 'Model not trained yet'
            }

        return {
            'is_fitted': True,
            'training_races': self.training_races,
            'training_samples': self.training_samples,
            'validation_mae': self.val_mae,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_type': 'XGBoost' if HAS_XGBOOST else 'GradientBoostingRegressor',
        }


def train_and_evaluate(
    training_years: List[int] = [2022, 2023],
    validation_year: int = 2024,
    save_model: bool = True
) -> Tuple[RacePredictor, Dict]:
    """
    Convenience function to train and evaluate the model.

    Args:
        training_years: Years to use for training
        validation_year: Year to use for validation
        save_model: Whether to save the trained model

    Returns:
        Tuple of (trained predictor, evaluation metrics)
    """
    # Collect data
    collector = F1DataCollector()

    print("Collecting training data...")
    training_races = collector.collect_multi_season_data(training_years)

    print(f"\nCollecting {validation_year} validation data...")
    val_races = collector.collect_season_data(validation_year)

    # Combine for feature calculation
    all_races = training_races + val_races

    # Train model
    predictor = RacePredictor(data_collector=collector)
    metrics = predictor.train(training_races)

    # Evaluate on validation year
    print(f"\n=== Evaluating on {validation_year} season ===")
    val_predictions = []
    val_actuals = []

    for race in val_races:
        historical = [r for r in all_races if r.year < race.year or
                     (r.year == race.year and r.round_number < race.round_number)]

        if len(historical) < 5:
            continue

        try:
            prediction = predictor.predict_race(race, historical)

            for pred in prediction.predictions:
                actual_result = race.race_results[
                    race.race_results['driver_code'] == pred.driver_code
                ]
                if not actual_result.empty and pd.notna(actual_result.iloc[0]['finish_position']):
                    val_predictions.append(pred.predicted_position)
                    val_actuals.append(actual_result.iloc[0]['finish_position'])
        except Exception as e:
            print(f"Error predicting {race.event_name}: {e}")

    if val_predictions:
        val_mae = mean_absolute_error(val_actuals, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))

        # Calculate top-N accuracy
        correct_top3 = sum(
            1 for pred, actual in zip(val_predictions, val_actuals)
            if round(pred) <= 3 and actual <= 3
        )
        total_top3 = sum(1 for actual in val_actuals if actual <= 3)
        top3_accuracy = correct_top3 / total_top3 if total_top3 > 0 else 0

        print(f"\n{validation_year} Validation Results:")
        print(f"  MAE: {val_mae:.2f} positions")
        print(f"  RMSE: {val_rmse:.2f} positions")
        print(f"  Top-3 Prediction Accuracy: {top3_accuracy:.1%}")

        metrics['validation_year_mae'] = val_mae
        metrics['validation_year_rmse'] = val_rmse
        metrics['top3_accuracy'] = top3_accuracy

    if save_model:
        predictor.save_model()

    return predictor, metrics

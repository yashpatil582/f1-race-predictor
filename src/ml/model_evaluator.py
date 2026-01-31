"""
F1 Model Evaluator - Accuracy metrics and visualization for the race predictor.

This module provides tools for evaluating model performance, generating
comparison visualizations, and analyzing prediction accuracy.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_collector import RaceData, F1DataCollector
from .race_predictor import RacePredictor, RacePrediction


@dataclass
class RaceEvaluation:
    """Evaluation results for a single race."""
    year: int
    round_number: int
    event_name: str
    mae: float
    rmse: float
    exact_matches: int
    within_1: int
    within_3: int
    total_drivers: int
    predictions: List[Dict]


class ModelEvaluator:
    """
    Evaluates the F1 race prediction model performance.

    Provides metrics including:
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - Top-N prediction accuracy
    - Position distribution analysis
    """

    def __init__(self, predictor: RacePredictor):
        """
        Initialize the evaluator.

        Args:
            predictor: Trained RacePredictor instance
        """
        self.predictor = predictor
        self.evaluations: List[RaceEvaluation] = []

    def evaluate_race(
        self,
        race: RaceData,
        historical_races: List[RaceData]
    ) -> RaceEvaluation:
        """
        Evaluate model predictions for a single race.

        Args:
            race: The race to evaluate
            historical_races: Historical races for feature calculation

        Returns:
            RaceEvaluation with metrics and predictions
        """
        # Get predictions
        prediction = self.predictor.predict_race(race, historical_races)

        # Compare to actual results
        predictions_data = []
        errors = []

        for pred in prediction.predictions:
            actual_result = race.race_results[
                race.race_results['driver_code'] == pred.driver_code
            ]

            if actual_result.empty:
                continue

            actual_pos = actual_result.iloc[0]['finish_position']
            if pd.isna(actual_pos):
                continue

            error = abs(pred.predicted_position - actual_pos)
            errors.append(error)

            predictions_data.append({
                'driver_code': pred.driver_code,
                'team': pred.team,
                'grid_position': pred.grid_position,
                'predicted_position': round(pred.predicted_position, 1),
                'actual_position': int(actual_pos),
                'error': round(error, 1),
                'confidence_lower': round(pred.confidence_lower, 1),
                'confidence_upper': round(pred.confidence_upper, 1),
            })

        if not errors:
            raise ValueError(f"No valid comparisons for {race.event_name}")

        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        exact_matches = sum(1 for e in errors if e < 0.5)
        within_1 = sum(1 for e in errors if e <= 1)
        within_3 = sum(1 for e in errors if e <= 3)

        evaluation = RaceEvaluation(
            year=race.year,
            round_number=race.round_number,
            event_name=race.event_name,
            mae=mae,
            rmse=rmse,
            exact_matches=exact_matches,
            within_1=within_1,
            within_3=within_3,
            total_drivers=len(predictions_data),
            predictions=predictions_data
        )

        self.evaluations.append(evaluation)
        return evaluation

    def evaluate_season(
        self,
        races: List[RaceData],
        all_historical: List[RaceData]
    ) -> Dict:
        """
        Evaluate model on an entire season.

        Args:
            races: Races to evaluate
            all_historical: All historical data for feature calculation

        Returns:
            Dictionary of aggregate metrics
        """
        season_evaluations = []

        for race in races:
            historical = [
                r for r in all_historical
                if r.year < race.year or
                   (r.year == race.year and r.round_number < race.round_number)
            ]

            if len(historical) < 5:
                continue

            try:
                eval_result = self.evaluate_race(race, historical)
                season_evaluations.append(eval_result)
                print(f"  {race.event_name}: MAE={eval_result.mae:.2f}")
            except Exception as e:
                print(f"  {race.event_name}: Error - {e}")

        if not season_evaluations:
            return {'error': 'No valid evaluations'}

        # Aggregate metrics
        all_mae = [e.mae for e in season_evaluations]
        all_rmse = [e.rmse for e in season_evaluations]
        total_exact = sum(e.exact_matches for e in season_evaluations)
        total_within_1 = sum(e.within_1 for e in season_evaluations)
        total_within_3 = sum(e.within_3 for e in season_evaluations)
        total_drivers = sum(e.total_drivers for e in season_evaluations)

        return {
            'races_evaluated': len(season_evaluations),
            'avg_mae': np.mean(all_mae),
            'std_mae': np.std(all_mae),
            'avg_rmse': np.mean(all_rmse),
            'best_race_mae': min(all_mae),
            'worst_race_mae': max(all_mae),
            'exact_accuracy': total_exact / total_drivers if total_drivers > 0 else 0,
            'within_1_accuracy': total_within_1 / total_drivers if total_drivers > 0 else 0,
            'within_3_accuracy': total_within_3 / total_drivers if total_drivers > 0 else 0,
            'total_predictions': total_drivers,
        }

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics from all evaluations.

        Returns:
            Dictionary of overall statistics
        """
        if not self.evaluations:
            return {'error': 'No evaluations performed'}

        all_mae = [e.mae for e in self.evaluations]
        all_errors = []

        for eval_result in self.evaluations:
            for pred in eval_result.predictions:
                all_errors.append(pred['error'])

        return {
            'total_races': len(self.evaluations),
            'total_predictions': len(all_errors),
            'overall_mae': np.mean(all_errors),
            'overall_rmse': np.sqrt(np.mean(np.array(all_errors) ** 2)),
            'race_avg_mae': np.mean(all_mae),
            'race_std_mae': np.std(all_mae),
            'best_race': min(self.evaluations, key=lambda x: x.mae).event_name,
            'worst_race': max(self.evaluations, key=lambda x: x.mae).event_name,
        }

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """
        Get all predictions as a DataFrame.

        Returns:
            DataFrame with all prediction data
        """
        rows = []
        for eval_result in self.evaluations:
            for pred in eval_result.predictions:
                rows.append({
                    'year': eval_result.year,
                    'round': eval_result.round_number,
                    'event': eval_result.event_name,
                    **pred
                })
        return pd.DataFrame(rows)

    def get_position_accuracy(self) -> Dict[int, Dict]:
        """
        Get accuracy broken down by finishing position.

        Returns:
            Dictionary mapping position to accuracy metrics
        """
        df = self.get_predictions_dataframe()
        if df.empty:
            return {}

        position_stats = {}
        for pos in range(1, 21):
            pos_data = df[df['actual_position'] == pos]
            if not pos_data.empty:
                position_stats[pos] = {
                    'count': len(pos_data),
                    'avg_error': pos_data['error'].mean(),
                    'avg_predicted': pos_data['predicted_position'].mean(),
                }

        return position_stats

    def get_grid_vs_finish_analysis(self) -> Dict:
        """
        Analyze prediction accuracy vs grid position changes.

        Returns:
            Dictionary with analysis results
        """
        df = self.get_predictions_dataframe()
        if df.empty:
            return {}

        df['grid_change'] = df['actual_position'] - df['grid_position']
        df['predicted_change'] = df['predicted_position'] - df['grid_position']

        # Categorize predictions
        df['prediction_type'] = 'correct_direction'
        df.loc[
            (df['grid_change'] > 0) & (df['predicted_change'] <= 0),
            'prediction_type'
        ] = 'missed_drop'
        df.loc[
            (df['grid_change'] < 0) & (df['predicted_change'] >= 0),
            'prediction_type'
        ] = 'missed_gain'
        df.loc[
            abs(df['error']) < 1,
            'prediction_type'
        ] = 'accurate'

        return {
            'avg_grid_change': df['grid_change'].mean(),
            'predicted_avg_change': df['predicted_change'].mean(),
            'prediction_types': df['prediction_type'].value_counts().to_dict(),
            'overtake_correlation': df['grid_change'].corr(df['predicted_change']),
        }

    def generate_comparison_data(
        self,
        race_evaluation: RaceEvaluation
    ) -> List[Dict]:
        """
        Generate data for predicted vs actual comparison charts.

        Args:
            race_evaluation: Evaluation to visualize

        Returns:
            List of data points for visualization
        """
        comparison_data = []

        # Sort by predicted position
        sorted_preds = sorted(
            race_evaluation.predictions,
            key=lambda x: x['predicted_position']
        )

        for i, pred in enumerate(sorted_preds, 1):
            comparison_data.append({
                'predicted_rank': i,
                'driver': pred['driver_code'],
                'team': pred['team'],
                'grid': pred['grid_position'],
                'predicted': pred['predicted_position'],
                'actual': pred['actual_position'],
                'error': pred['error'],
                'confidence_range': f"{pred['confidence_lower']:.0f}-{pred['confidence_upper']:.0f}",
            })

        return comparison_data

    def get_feature_importance_analysis(self) -> Dict:
        """
        Get feature importance with context.

        Returns:
            Dictionary with feature importance analysis
        """
        importance = self.predictor.get_feature_importance()

        # Categorize features
        categories = {
            'Qualifying': ['grid_position', 'gap_to_pole', 'q3_reached', 'q2_reached',
                          'quali_pace_percentile', 'front_row', 'top_5_start', 'top_10_start'],
            'Driver History': ['driver_avg_finish', 'driver_best_finish', 'driver_dnf_rate',
                              'driver_races_completed', 'driver_total_points'],
            'Constructor': ['constructor_avg_finish', 'constructor_best_finish',
                           'constructor_total_points'],
            'Weather': ['track_temp', 'air_temp', 'humidity', 'wind_speed',
                       'rain_probability', 'is_wet_race'],
            'Circuit': ['circuit_length', 'num_corners', 'overtake_difficulty',
                       'is_street_circuit', 'is_high_downforce'],
            'Form': ['last_5_avg', 'position_trend', 'consecutive_points_finishes',
                    'momentum_score'],
        }

        category_importance = {}
        for cat, features in categories.items():
            cat_imp = sum(importance.get(f, 0) for f in features)
            category_importance[cat] = cat_imp

        return {
            'feature_importance': importance,
            'category_importance': category_importance,
            'top_5_features': sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

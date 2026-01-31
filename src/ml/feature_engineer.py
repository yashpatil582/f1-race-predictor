"""
F1 Feature Engineering - Extract ML features from raw F1 data.

This module transforms raw F1 data into ML-ready features including
qualifying performance, historical statistics, weather conditions,
and circuit characteristics.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .data_collector import RaceData, F1DataCollector


# Circuit characteristics (manually curated based on F1 knowledge)
CIRCUIT_CHARACTERISTICS = {
    # Circuit: (overtake_difficulty, street_circuit, high_downforce)
    # overtake_difficulty: 1 (easy) to 10 (hard)
    'Monaco': (10, True, True),
    'Singapore': (8, True, True),
    'Melbourne': (7, True, False),
    'Baku': (5, True, False),
    'Las Vegas': (6, True, False),
    'Jeddah': (4, True, False),
    'Miami': (6, True, False),
    'Monza': (3, False, False),
    'Spa-Francorchamps': (4, False, False),
    'Silverstone': (5, False, False),
    'Suzuka': (7, False, True),
    'Bahrain': (4, False, False),
    'Spielberg': (5, False, False),
    'Budapest': (9, False, True),
    'Zandvoort': (8, False, True),
    'Imola': (7, False, True),
    'Shanghai': (5, False, False),
    'Sakhir': (4, False, False),
    'Austin': (5, False, False),
    'Mexico City': (6, False, False),
    'Interlagos': (5, False, False),
    'Lusail': (5, False, False),
    'Yas Marina': (6, False, False),
}


class FeatureEngineer:
    """
    Transforms raw F1 data into ML features.

    Extracts 20+ features across categories:
    - Qualifying performance
    - Historical driver/constructor stats
    - Weather conditions
    - Circuit characteristics
    - Form and momentum
    """

    def __init__(self, data_collector: Optional[F1DataCollector] = None):
        """
        Initialize the feature engineer.

        Args:
            data_collector: F1DataCollector instance for historical data
        """
        self.data_collector = data_collector or F1DataCollector()

    def extract_features(
        self,
        race: RaceData,
        historical_races: List[RaceData]
    ) -> pd.DataFrame:
        """
        Extract all features for a race.

        Args:
            race: The race to extract features for
            historical_races: Historical races for calculating stats

        Returns:
            DataFrame with features for each driver
        """
        features_list = []

        # Get all drivers from qualifying
        for _, quali_row in race.qualifying_results.iterrows():
            driver_code = quali_row['driver_code']
            team = quali_row['team']

            # Extract all feature categories
            quali_features = self._extract_qualifying_features(
                quali_row, race.qualifying_results
            )

            historical_features = self._extract_historical_features(
                historical_races, driver_code, team,
                race.year, race.round_number
            )

            weather_features = self._extract_weather_features(race.weather_data)

            circuit_features = self._extract_circuit_features(
                race.circuit_name, race.circuit_info
            )

            form_features = self._extract_form_features(
                historical_races, driver_code,
                race.year, race.round_number
            )

            # Combine all features
            driver_features = {
                'driver_code': driver_code,
                'team': team,
                'year': race.year,
                'round': race.round_number,
                **quali_features,
                **historical_features,
                **weather_features,
                **circuit_features,
                **form_features,
            }

            # Get actual finish position (target variable)
            race_result = race.race_results[
                race.race_results['driver_code'] == driver_code
            ]
            if not race_result.empty:
                driver_features['finish_position'] = race_result.iloc[0]['finish_position']
                driver_features['dnf'] = 1 if race_result.iloc[0]['status'] not in [
                    'Finished', '+1 Lap', '+2 Laps', '+3 Laps'
                ] else 0
            else:
                driver_features['finish_position'] = None
                driver_features['dnf'] = None

            features_list.append(driver_features)

        return pd.DataFrame(features_list)

    def _extract_qualifying_features(
        self,
        quali_row: pd.Series,
        quali_results: pd.DataFrame
    ) -> Dict:
        """
        Extract qualifying-related features.

        Features:
        - grid_position: Starting grid position
        - gap_to_pole: Time gap to pole position (seconds)
        - q3_reached: Whether driver reached Q3
        - q2_reached: Whether driver reached Q2
        - quali_pace_percentile: Relative pace vs field
        """
        grid_pos = quali_row['grid_position']
        q3_time = quali_row['q3_time']
        q2_time = quali_row['q2_time']
        q1_time = quali_row['q1_time']

        # Calculate gap to pole
        pole_time = quali_results['q3_time'].min()
        if pd.notna(q3_time) and pd.notna(pole_time):
            gap_to_pole = q3_time - pole_time
        elif pd.notna(q2_time):
            best_q2 = quali_results['q2_time'].min()
            gap_to_pole = (q2_time - best_q2) + 0.5  # Penalty for not reaching Q3
        elif pd.notna(q1_time):
            best_q1 = quali_results['q1_time'].min()
            gap_to_pole = (q1_time - best_q1) + 1.0  # Larger penalty
        else:
            gap_to_pole = 3.0  # Default large gap

        # Calculate pace percentile
        best_times = []
        for _, row in quali_results.iterrows():
            if pd.notna(row['q3_time']):
                best_times.append(row['q3_time'])
            elif pd.notna(row['q2_time']):
                best_times.append(row['q2_time'])
            elif pd.notna(row['q1_time']):
                best_times.append(row['q1_time'])

        driver_best = q3_time or q2_time or q1_time
        if driver_best and best_times:
            percentile = (sum(1 for t in best_times if t > driver_best) / len(best_times)) * 100
        else:
            percentile = 50.0

        return {
            'grid_position': grid_pos,
            'gap_to_pole': gap_to_pole,
            'q3_reached': 1 if pd.notna(q3_time) else 0,
            'q2_reached': 1 if pd.notna(q2_time) else 0,
            'quali_pace_percentile': percentile,
            'front_row': 1 if grid_pos <= 2 else 0,
            'top_5_start': 1 if grid_pos <= 5 else 0,
            'top_10_start': 1 if grid_pos <= 10 else 0,
        }

    def _extract_historical_features(
        self,
        historical_races: List[RaceData],
        driver_code: str,
        team: str,
        year: int,
        round_number: int
    ) -> Dict:
        """
        Extract historical performance features.

        Features:
        - driver_avg_finish: Driver's average finish position
        - driver_best_finish: Driver's best finish
        - driver_dnf_rate: Driver's DNF rate
        - driver_races: Number of races driver has completed
        - constructor_avg_finish: Constructor's average finish
        - constructor_points: Constructor's total points
        """
        # Get driver history
        driver_history = self.data_collector.get_driver_history(
            historical_races, driver_code, year, round_number
        )

        # Get constructor history
        constructor_history = self.data_collector.get_constructor_history(
            historical_races, team, year, round_number
        )

        return {
            'driver_avg_finish': driver_history['avg_finish'],
            'driver_best_finish': driver_history['best_finish'],
            'driver_dnf_rate': driver_history['dnf_rate'],
            'driver_races_completed': driver_history['races_completed'],
            'driver_total_points': driver_history['total_points'],
            'constructor_avg_finish': constructor_history['avg_finish'],
            'constructor_best_finish': constructor_history['best_finish'],
            'constructor_total_points': constructor_history['total_points'],
        }

    def _extract_weather_features(
        self,
        weather_data: Optional[pd.DataFrame]
    ) -> Dict:
        """
        Extract weather-related features.

        Features:
        - track_temp: Average track temperature
        - air_temp: Average air temperature
        - humidity: Average humidity
        - rain_probability: Estimated rain probability
        - is_wet_race: Whether it's a wet race
        """
        if weather_data is None or weather_data.empty:
            return {
                'track_temp': 35.0,  # Default values
                'air_temp': 25.0,
                'humidity': 50.0,
                'wind_speed': 10.0,
                'rain_probability': 0.0,
                'is_wet_race': 0,
            }

        track_temp = weather_data['TrackTemp'].mean() if 'TrackTemp' in weather_data else 35.0
        air_temp = weather_data['AirTemp'].mean() if 'AirTemp' in weather_data else 25.0
        humidity = weather_data['Humidity'].mean() if 'Humidity' in weather_data else 50.0
        wind_speed = weather_data['WindSpeed'].mean() if 'WindSpeed' in weather_data else 10.0

        # Check for rain
        is_wet = 0
        rain_prob = 0.0
        if 'Rainfall' in weather_data:
            rainfall = weather_data['Rainfall']
            if rainfall.any():
                is_wet = 1
                rain_prob = rainfall.mean()

        return {
            'track_temp': float(track_temp) if pd.notna(track_temp) else 35.0,
            'air_temp': float(air_temp) if pd.notna(air_temp) else 25.0,
            'humidity': float(humidity) if pd.notna(humidity) else 50.0,
            'wind_speed': float(wind_speed) if pd.notna(wind_speed) else 10.0,
            'rain_probability': float(rain_prob),
            'is_wet_race': is_wet,
        }

    def _extract_circuit_features(
        self,
        circuit_name: str,
        circuit_info: dict
    ) -> Dict:
        """
        Extract circuit-related features.

        Features:
        - circuit_length: Track length in km
        - overtake_difficulty: 1-10 scale
        - is_street_circuit: Boolean
        - is_high_downforce: Boolean
        - num_corners: Number of corners
        """
        # Get circuit characteristics
        circuit_chars = CIRCUIT_CHARACTERISTICS.get(
            circuit_name,
            (5, False, False)  # Default values
        )

        return {
            'circuit_length': circuit_info.get('length') or 5.0,
            'num_corners': circuit_info.get('corners') or 15,
            'overtake_difficulty': circuit_chars[0],
            'is_street_circuit': 1 if circuit_chars[1] else 0,
            'is_high_downforce': 1 if circuit_chars[2] else 0,
        }

    def _extract_form_features(
        self,
        historical_races: List[RaceData],
        driver_code: str,
        year: int,
        round_number: int
    ) -> Dict:
        """
        Extract recent form features.

        Features:
        - last_5_avg: Average finish in last 5 races
        - position_trend: Improving/declining trend
        - consecutive_points: Races with points in a row
        - momentum_score: Composite form score
        """
        # Get recent races
        past_races = [
            r for r in historical_races
            if (r.year < year) or
               (r.year == year and r.round_number < round_number)
        ]

        # Sort by date (most recent first)
        past_races = sorted(past_races, key=lambda x: (x.year, x.round_number), reverse=True)

        # Get last 5 results
        last_5_positions = []
        consecutive_points = 0
        counting_consecutive = True

        for race in past_races[:10]:  # Look at last 10 races
            result = race.race_results[
                race.race_results['driver_code'] == driver_code
            ]
            if not result.empty:
                pos = result.iloc[0]['finish_position']
                points = result.iloc[0]['points']

                if pd.notna(pos) and len(last_5_positions) < 5:
                    last_5_positions.append(pos)

                if counting_consecutive:
                    if points > 0:
                        consecutive_points += 1
                    else:
                        counting_consecutive = False

        # Calculate trend
        if len(last_5_positions) >= 3:
            recent = np.mean(last_5_positions[:2])
            older = np.mean(last_5_positions[2:])
            trend = older - recent  # Positive = improving
        else:
            trend = 0.0

        # Calculate momentum score
        last_5_avg = np.mean(last_5_positions) if last_5_positions else 15.0
        momentum = (20 - last_5_avg) / 20 * 50 + consecutive_points * 5 + trend * 2

        return {
            'last_5_avg': last_5_avg,
            'position_trend': trend,
            'consecutive_points_finishes': consecutive_points,
            'momentum_score': momentum,
        }

    def prepare_training_data(
        self,
        races: List[RaceData],
        min_historical_races: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target vector for training.

        Args:
            races: List of all races to use
            min_historical_races: Minimum races before using for training

        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        all_features = []

        for i, race in enumerate(races):
            # Need at least some historical data
            historical = races[:i]
            if len(historical) < min_historical_races:
                continue

            try:
                race_features = self.extract_features(race, historical)
                all_features.append(race_features)
            except Exception as e:
                print(f"Error extracting features for {race.year} R{race.round_number}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data could be extracted")

        # Combine all race features
        df = pd.concat(all_features, ignore_index=True)

        # Remove rows with missing target
        df = df.dropna(subset=['finish_position'])

        # Define feature columns (exclude identifiers and target)
        feature_cols = [
            'grid_position', 'gap_to_pole', 'q3_reached', 'q2_reached',
            'quali_pace_percentile', 'front_row', 'top_5_start', 'top_10_start',
            'driver_avg_finish', 'driver_best_finish', 'driver_dnf_rate',
            'driver_races_completed', 'driver_total_points',
            'constructor_avg_finish', 'constructor_best_finish', 'constructor_total_points',
            'track_temp', 'air_temp', 'humidity', 'wind_speed',
            'rain_probability', 'is_wet_race',
            'circuit_length', 'num_corners', 'overtake_difficulty',
            'is_street_circuit', 'is_high_downforce',
            'last_5_avg', 'position_trend', 'consecutive_points_finishes', 'momentum_score',
        ]

        X = df[feature_cols].copy()
        y = df['finish_position'].copy()

        # Handle any remaining NaN values
        X = X.fillna(X.median())

        return X, y

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names used by the model."""
        return [
            'grid_position', 'gap_to_pole', 'q3_reached', 'q2_reached',
            'quali_pace_percentile', 'front_row', 'top_5_start', 'top_10_start',
            'driver_avg_finish', 'driver_best_finish', 'driver_dnf_rate',
            'driver_races_completed', 'driver_total_points',
            'constructor_avg_finish', 'constructor_best_finish', 'constructor_total_points',
            'track_temp', 'air_temp', 'humidity', 'wind_speed',
            'rain_probability', 'is_wet_race',
            'circuit_length', 'num_corners', 'overtake_difficulty',
            'is_street_circuit', 'is_high_downforce',
            'last_5_avg', 'position_trend', 'consecutive_points_finishes', 'momentum_score',
        ]

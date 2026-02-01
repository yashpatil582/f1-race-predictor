"""
F1 Data Collector - Historical data aggregation for ML training.

This module fetches and aggregates historical F1 race data from FastF1
for training the race outcome prediction model.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import fastf1
import pandas as pd
import numpy as np

from ..lib.settings import get_settings


@dataclass
class RaceData:
    """Container for race data used in ML training."""
    year: int
    round_number: int
    event_name: str
    circuit_name: str
    qualifying_results: pd.DataFrame
    race_results: pd.DataFrame
    weather_data: Optional[pd.DataFrame]
    lap_data: pd.DataFrame
    circuit_info: dict


class F1DataCollector:
    """
    Collects and aggregates historical F1 data for ML model training.

    Fetches qualifying results, race results, weather data, and lap times
    from FastF1 for specified seasons.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data collector.

        Args:
            cache_dir: Directory for FastF1 cache. Uses settings if not provided.
        """
        self.cache_dir = cache_dir
        self._enable_cache()

    def _enable_cache(self):
        """Enable FastF1 cache for faster data retrieval."""
        if self.cache_dir:
            cache_path = self.cache_dir
        else:
            settings = get_settings()
            cache_path = settings.cache_location

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        fastf1.Cache.enable_cache(cache_path)

    def get_schedule(self, year: int) -> pd.DataFrame:
        """Get the race schedule for a given year."""
        schedule = fastf1.get_event_schedule(year)
        # Filter out testing events
        return schedule[~schedule.apply(lambda x: x.is_testing(), axis=1)]

    def collect_race_data(
        self,
        year: int,
        round_number: int,
        include_weather: bool = True,
        include_laps: bool = True
    ) -> Optional[RaceData]:
        """
        Collect all relevant data for a single race.

        Args:
            year: Season year
            round_number: Round number in the season
            include_weather: Whether to fetch weather data
            include_laps: Whether to fetch lap-by-lap data

        Returns:
            RaceData object or None if data unavailable
        """
        try:
            # Get event info
            schedule = self.get_schedule(year)
            event = schedule[schedule['RoundNumber'] == round_number].iloc[0]

            # Load qualifying session
            quali_session = fastf1.get_session(year, round_number, 'Q')
            quali_session.load(telemetry=False, weather=include_weather)

            # Load race session
            race_session = fastf1.get_session(year, round_number, 'R')
            race_session.load(telemetry=False, weather=include_weather, laps=include_laps)

            # Extract qualifying results
            quali_results = self._extract_qualifying_results(quali_session)

            # Extract race results
            race_results = self._extract_race_results(race_session)

            # Extract weather data
            weather_data = None
            if include_weather and hasattr(race_session, 'weather_data'):
                weather_data = race_session.weather_data

            # Extract lap data
            lap_data = pd.DataFrame()
            if include_laps:
                lap_data = self._extract_lap_data(race_session)

            # Get circuit info
            circuit_info = self._extract_circuit_info(race_session)

            return RaceData(
                year=year,
                round_number=round_number,
                event_name=event['EventName'],
                circuit_name=event['Location'],
                qualifying_results=quali_results,
                race_results=race_results,
                weather_data=weather_data,
                lap_data=lap_data,
                circuit_info=circuit_info
            )

        except Exception as e:
            print(f"Error collecting data for {year} round {round_number}: {e}")
            return None

    def _extract_qualifying_results(self, session) -> pd.DataFrame:
        """Extract qualifying results from a session."""
        results = session.results

        quali_data = []
        for _, row in results.iterrows():
            if pd.isna(row.get('Position')):
                continue

            driver_data = {
                'driver_code': row['Abbreviation'],
                'driver_number': row['DriverNumber'],
                'team': row['TeamName'],
                'grid_position': int(row['Position']),
                'q1_time': self._timedelta_to_seconds(row.get('Q1')),
                'q2_time': self._timedelta_to_seconds(row.get('Q2')),
                'q3_time': self._timedelta_to_seconds(row.get('Q3')),
            }
            quali_data.append(driver_data)

        return pd.DataFrame(quali_data)

    def _extract_race_results(self, session) -> pd.DataFrame:
        """Extract race results from a session."""
        results = session.results

        race_data = []
        for _, row in results.iterrows():
            driver_data = {
                'driver_code': row['Abbreviation'],
                'driver_number': row['DriverNumber'],
                'team': row['TeamName'],
                'finish_position': int(row['Position']) if pd.notna(row.get('Position')) else None,
                'grid_position': int(row['GridPosition']) if pd.notna(row.get('GridPosition')) else None,
                'status': row.get('Status', 'Unknown'),
                'points': float(row['Points']) if pd.notna(row.get('Points')) else 0.0,
                'fastest_lap_time': self._timedelta_to_seconds(row.get('FastestLapTime')),
                'total_time': self._timedelta_to_seconds(row.get('Time')),
            }
            race_data.append(driver_data)

        return pd.DataFrame(race_data)

    def _extract_lap_data(self, session) -> pd.DataFrame:
        """Extract lap-by-lap data from a session."""
        laps = session.laps

        lap_data = []
        for _, lap in laps.iterrows():
            lap_info = {
                'driver_code': lap['Driver'],
                'lap_number': int(lap['LapNumber']),
                'lap_time': self._timedelta_to_seconds(lap.get('LapTime')),
                'sector1_time': self._timedelta_to_seconds(lap.get('Sector1Time')),
                'sector2_time': self._timedelta_to_seconds(lap.get('Sector2Time')),
                'sector3_time': self._timedelta_to_seconds(lap.get('Sector3Time')),
                'compound': lap.get('Compound', 'UNKNOWN'),
                'tyre_life': int(lap['TyreLife']) if pd.notna(lap.get('TyreLife')) else 0,
                'stint': int(lap['Stint']) if pd.notna(lap.get('Stint')) else 1,
                'is_pit_out_lap': bool(lap.get('PitOutTime')) if pd.notna(lap.get('PitOutTime')) else False,
                'is_pit_in_lap': bool(lap.get('PitInTime')) if pd.notna(lap.get('PitInTime')) else False,
            }
            lap_data.append(lap_info)

        return pd.DataFrame(lap_data)

    def _extract_circuit_info(self, session) -> dict:
        """Extract circuit information from a session."""
        try:
            circuit = session.get_circuit_info()
            return {
                'length': circuit.length if hasattr(circuit, 'length') else None,
                'rotation': circuit.rotation if hasattr(circuit, 'rotation') else 0,
                'corners': len(circuit.corners) if hasattr(circuit, 'corners') else None,
            }
        except Exception:
            return {'length': None, 'rotation': 0, 'corners': None}

    def _timedelta_to_seconds(self, td) -> Optional[float]:
        """Convert pandas Timedelta to seconds."""
        if pd.isna(td):
            return None
        return td.total_seconds()

    def collect_season_data(
        self,
        year: int,
        max_rounds: Optional[int] = None
    ) -> List[RaceData]:
        """
        Collect data for all races in a season.

        Args:
            year: Season year
            max_rounds: Maximum number of rounds to collect (for testing)

        Returns:
            List of RaceData objects
        """
        schedule = self.get_schedule(year)
        races = []

        rounds = schedule['RoundNumber'].unique()
        if max_rounds:
            rounds = rounds[:max_rounds]

        for round_num in rounds:
            print(f"Collecting data for {year} Round {round_num}...")
            race_data = self.collect_race_data(year, int(round_num))
            if race_data:
                races.append(race_data)

        return races

    def collect_multi_season_data(
        self,
        years: List[int],
        max_rounds_per_year: Optional[int] = None
    ) -> List[RaceData]:
        """
        Collect data for multiple seasons.

        Args:
            years: List of season years
            max_rounds_per_year: Maximum rounds per season (for testing)

        Returns:
            List of RaceData objects from all seasons
        """
        all_races = []
        for year in years:
            print(f"\n=== Collecting {year} season data ===")
            season_data = self.collect_season_data(year, max_rounds_per_year)
            all_races.extend(season_data)

        return all_races

    def save_collected_data(self, races: List[RaceData], filepath: str):
        """Save collected race data to a pickle file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(races, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(races)} races to {filepath}")

    def load_collected_data(self, filepath: str) -> List[RaceData]:
        """Load previously collected race data from a pickle file."""
        with open(filepath, 'rb') as f:
            races = pickle.load(f)
        print(f"Loaded {len(races)} races from {filepath}")
        return races

    def get_driver_history(
        self,
        races: List[RaceData],
        driver_code: str,
        before_year: int,
        before_round: int
    ) -> Dict:
        """
        Get historical statistics for a driver before a specific race.

        Args:
            races: List of all race data
            driver_code: Driver abbreviation (e.g., 'VER', 'HAM')
            before_year: Year of the race to predict
            before_round: Round number of the race to predict

        Returns:
            Dictionary of historical statistics
        """
        # Filter races before the specified race
        past_races = [
            r for r in races
            if (r.year < before_year) or
               (r.year == before_year and r.round_number < before_round)
        ]

        # Collect finish positions
        positions = []
        dnf_count = 0
        points_total = 0

        for race in past_races:
            driver_result = race.race_results[
                race.race_results['driver_code'] == driver_code
            ]
            if not driver_result.empty:
                result = driver_result.iloc[0]
                if pd.notna(result['finish_position']):
                    positions.append(result['finish_position'])
                    if result['status'] not in ['Finished', '+1 Lap', '+2 Laps']:
                        dnf_count += 1
                points_total += result['points']

        # Calculate statistics
        if positions:
            return {
                'avg_finish': np.mean(positions),
                'best_finish': min(positions),
                'worst_finish': max(positions),
                'races_completed': len(positions),
                'dnf_rate': dnf_count / len(positions) if positions else 0,
                'total_points': points_total,
                'last_5_avg': np.mean(positions[-5:]) if len(positions) >= 5 else np.mean(positions),
            }
        else:
            return {
                'avg_finish': 15.0,  # Default for unknown drivers
                'best_finish': 20,
                'worst_finish': 20,
                'races_completed': 0,
                'dnf_rate': 0.1,
                'total_points': 0,
                'last_5_avg': 15.0,
            }

    def get_constructor_history(
        self,
        races: List[RaceData],
        team: str,
        before_year: int,
        before_round: int
    ) -> Dict:
        """
        Get historical statistics for a constructor before a specific race.

        Args:
            races: List of all race data
            team: Team name
            before_year: Year of the race to predict
            before_round: Round number of the race to predict

        Returns:
            Dictionary of historical statistics
        """
        past_races = [
            r for r in races
            if (r.year < before_year) or
               (r.year == before_year and r.round_number < before_round)
        ]

        positions = []
        points_total = 0

        for race in past_races:
            team_results = race.race_results[
                race.race_results['team'] == team
            ]
            for _, result in team_results.iterrows():
                if pd.notna(result['finish_position']):
                    positions.append(result['finish_position'])
                points_total += result['points']

        if positions:
            return {
                'avg_finish': np.mean(positions),
                'best_finish': min(positions),
                'total_points': points_total,
                'races_completed': len(positions) // 2,  # Divide by 2 for 2 drivers
            }
        else:
            return {
                'avg_finish': 12.0,
                'best_finish': 20,
                'total_points': 0,
                'races_completed': 0,
            }

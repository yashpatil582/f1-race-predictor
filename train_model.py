#!/usr/bin/env python3
"""
Quick script to train the F1 race predictor model.
"""
import sys
sys.path.insert(0, '.')

from src.ml.data_collector import F1DataCollector
from src.ml.race_predictor import RacePredictor

def main():
    print("=" * 60)
    print("F1 Race Outcome Predictor - Model Training")
    print("=" * 60)

    # Initialize
    collector = F1DataCollector()
    predictor = RacePredictor(data_collector=collector)

    # Collect training data (just 2023 for faster training)
    print("\nCollecting 2023 season data...")
    training_races = collector.collect_season_data(2023)

    print(f"\nCollected {len(training_races)} races")

    if len(training_races) < 5:
        print("Not enough races collected. Check FastF1 cache/connection.")
        return

    # Train
    print("\nTraining model...")
    metrics = predictor.train(training_races)

    # Save
    print("\nSaving model...")
    predictor.save_model()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Validation MAE: {metrics['val_mae']:.2f} positions")
    print(f"Training samples: {metrics['training_samples']}")

    # Quick test prediction
    print("\nTesting prediction on 2023 Bahrain GP...")
    if len(training_races) > 1:
        test_race = training_races[-1]  # Last race
        historical = training_races[:-1]

        prediction = predictor.predict_race(test_race, historical)

        print(f"\nPredictions for {prediction.event_name}:")
        print("-" * 50)
        print(f"{'Driver':<8} {'Grid':>5} {'Pred':>6} {'Actual':>7}")
        print("-" * 50)

        for pred in prediction.predictions[:10]:
            actual = test_race.race_results[
                test_race.race_results['driver_code'] == pred.driver_code
            ]
            actual_pos = int(actual.iloc[0]['finish_position']) if not actual.empty and actual.iloc[0]['finish_position'] else '-'
            print(f"{pred.driver_code:<8} {pred.grid_position:>5} {pred.predicted_position:>6.1f} {actual_pos:>7}")

if __name__ == "__main__":
    main()

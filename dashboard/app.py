"""
F1 Race Outcome Predictor - Streamlit Dashboard

Interactive dashboard for visualizing race predictions,
model performance, and feature importance.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ml.data_collector import F1DataCollector
from src.ml.race_predictor import RacePredictor
from src.ml.model_evaluator import ModelEvaluator


# Page config
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E10600;
    }
    .prediction-correct {
        background-color: rgba(0, 200, 0, 0.2);
    }
    .prediction-close {
        background-color: rgba(255, 200, 0, 0.2);
    }
    .prediction-off {
        background-color: rgba(255, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load or initialize the predictor."""
    collector = F1DataCollector()
    predictor = RacePredictor(data_collector=collector)
    try:
        predictor.load_model()
        st.success("Loaded pre-trained model")
    except FileNotFoundError:
        st.info("No pre-trained model found. Please train a model first.")
    return predictor, collector


@st.cache_data(ttl=3600)
def get_schedule(year: int):
    """Get race schedule for a year."""
    collector = F1DataCollector()
    return collector.get_schedule(year)


@st.cache_data(ttl=3600)
def collect_race_data(year: int, round_number: int):
    """Collect race data."""
    collector = F1DataCollector()
    return collector.collect_race_data(year, round_number)


def main():
    st.markdown('<h1 class="main-header">🏎️ F1 Race Outcome Predictor</h1>', unsafe_allow_html=True)

    # Load predictor
    predictor, collector = load_predictor()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Race Prediction", "Historical Analysis", "Model Information", "Train Model"]
    )

    if page == "Race Prediction":
        show_prediction_page(predictor, collector)
    elif page == "Historical Analysis":
        show_historical_page(predictor, collector)
    elif page == "Model Information":
        show_model_info_page(predictor)
    else:
        show_training_page(predictor, collector)


def show_prediction_page(predictor: RacePredictor, collector: F1DataCollector):
    """Display race prediction interface."""
    st.header("Race Outcome Prediction")

    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox("Select Year", [2025, 2024, 2023, 2022], index=0)

    with col2:
        schedule = get_schedule(year)
        race_options = {
            f"Round {row['RoundNumber']}: {row['EventName']}": row['RoundNumber']
            for _, row in schedule.iterrows()
        }
        selected_race = st.selectbox("Select Race", list(race_options.keys()))
        round_number = race_options[selected_race]

    if st.button("Predict Race Outcome", type="primary"):
        if not predictor.is_fitted:
            st.error("Model not trained. Please train the model first.")
            return

        with st.spinner("Generating predictions..."):
            try:
                # Collect race data
                race_data = collect_race_data(year, round_number)

                if race_data is None:
                    st.error("Could not load race data")
                    return

                # Get historical data
                historical = []
                for y in range(year - 2, year + 1):
                    try:
                        season_data = collector.collect_season_data(y)
                        historical.extend(season_data)
                    except Exception:
                        continue

                historical = [
                    r for r in historical
                    if r.year < year or (r.year == year and r.round_number < round_number)
                ]

                if len(historical) < 5:
                    st.error("Insufficient historical data for prediction")
                    return

                # Make prediction
                prediction = predictor.predict_race(race_data, historical)

                # Display results
                st.subheader(f"Predictions for {prediction.event_name}")

                # Confidence metric
                st.metric("Model Confidence", f"{prediction.model_confidence:.1f}%")

                # Create comparison dataframe
                results_data = []
                for pred in prediction.predictions:
                    actual_result = race_data.race_results[
                        race_data.race_results['driver_code'] == pred.driver_code
                    ]
                    actual_pos = None
                    if not actual_result.empty and pd.notna(actual_result.iloc[0]['finish_position']):
                        actual_pos = int(actual_result.iloc[0]['finish_position'])

                    results_data.append({
                        'Driver': pred.driver_code,
                        'Team': pred.team,
                        'Grid': pred.grid_position,
                        'Predicted': round(pred.predicted_position, 1),
                        'Actual': actual_pos,
                        'Error': abs(pred.predicted_position - actual_pos) if actual_pos else None,
                        'CI': f"{pred.confidence_lower:.0f}-{pred.confidence_upper:.0f}"
                    })

                df = pd.DataFrame(results_data)

                # Show predictions table
                st.dataframe(
                    df.style.background_gradient(subset=['Predicted'], cmap='RdYlGn_r'),
                    use_container_width=True
                )

                # Visualization
                col1, col2 = st.columns(2)

                with col1:
                    # Predicted vs Actual chart
                    if df['Actual'].notna().any():
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df['Driver'],
                            y=df['Predicted'],
                            mode='markers+lines',
                            name='Predicted',
                            marker=dict(size=12, color='#E10600'),
                            line=dict(color='#E10600', dash='dash')
                        ))

                        fig.add_trace(go.Scatter(
                            x=df['Driver'],
                            y=df['Actual'],
                            mode='markers+lines',
                            name='Actual',
                            marker=dict(size=12, color='#00D2BE'),
                            line=dict(color='#00D2BE')
                        ))

                        fig.update_layout(
                            title="Predicted vs Actual Positions",
                            xaxis_title="Driver",
                            yaxis_title="Position",
                            yaxis=dict(autorange="reversed"),
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Position probability heatmap for top 5
                    top_drivers = prediction.predictions[:5]
                    prob_data = []

                    for driver in top_drivers:
                        for pos in range(1, 11):
                            prob_data.append({
                                'Driver': driver.driver_code,
                                'Position': pos,
                                'Probability': driver.position_probabilities.get(pos, 0) * 100
                            })

                    prob_df = pd.DataFrame(prob_data)
                    prob_pivot = prob_df.pivot(index='Driver', columns='Position', values='Probability')

                    fig = px.imshow(
                        prob_pivot,
                        labels=dict(x="Position", y="Driver", color="Probability %"),
                        title="Position Probability Distribution (Top 5)",
                        color_continuous_scale="RdYlGn_r",
                        aspect="auto"
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                # Calculate metrics if actual results available
                if df['Actual'].notna().any():
                    st.subheader("Prediction Accuracy")

                    mae = df['Error'].dropna().mean()
                    exact = (df['Error'].dropna() < 0.5).sum()
                    within_3 = (df['Error'].dropna() <= 3).sum()
                    total = df['Error'].notna().sum()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error", f"{mae:.2f} positions")
                    col2.metric("Exact Predictions", f"{exact}/{total}")
                    col3.metric("Within 3 Positions", f"{within_3}/{total} ({within_3/total*100:.0f}%)")

            except Exception as e:
                st.error(f"Error generating predictions: {e}")


def show_historical_page(predictor: RacePredictor, collector: F1DataCollector):
    """Display historical analysis."""
    st.header("Historical Prediction Analysis")

    if not predictor.is_fitted:
        st.warning("Model not trained. Please train the model first.")
        return

    year = st.selectbox("Select Year to Analyze", [2025, 2024, 2023], index=0)

    if st.button("Analyze Season", type="primary"):
        with st.spinner(f"Analyzing {year} season..."):
            try:
                # Collect all data
                all_races = []
                for y in range(year - 2, year + 1):
                    try:
                        season_data = collector.collect_season_data(y)
                        all_races.extend(season_data)
                    except Exception:
                        continue

                year_races = [r for r in all_races if r.year == year]

                if not year_races:
                    st.error(f"No races found for {year}")
                    return

                # Evaluate
                evaluator = ModelEvaluator(predictor)
                metrics = evaluator.evaluate_season(year_races, all_races)

                # Display metrics
                st.subheader("Season Summary")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Races Evaluated", metrics['races_evaluated'])
                col2.metric("Average MAE", f"{metrics['avg_mae']:.2f}")
                col3.metric("Within 3 Accuracy", f"{metrics['within_3_accuracy']*100:.1f}%")
                col4.metric("Total Predictions", metrics['total_predictions'])

                # MAE by race chart
                race_maes = [e.mae for e in evaluator.evaluations]
                race_names = [e.event_name for e in evaluator.evaluations]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=race_names,
                    y=race_maes,
                    marker_color=['#E10600' if m > metrics['avg_mae'] else '#00D2BE' for m in race_maes]
                ))

                fig.add_hline(y=metrics['avg_mae'], line_dash="dash", line_color="yellow",
                             annotation_text=f"Avg: {metrics['avg_mae']:.2f}")

                fig.update_layout(
                    title="Mean Absolute Error by Race",
                    xaxis_title="Race",
                    yaxis_title="MAE (positions)",
                    template="plotly_dark",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)

                # Position accuracy breakdown
                position_stats = evaluator.get_position_accuracy()

                if position_stats:
                    positions = sorted(position_stats.keys())
                    errors = [position_stats[p]['avg_error'] for p in positions]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=positions,
                        y=errors,
                        mode='lines+markers',
                        marker=dict(size=10, color='#E10600'),
                        line=dict(color='#E10600')
                    ))

                    fig.update_layout(
                        title="Prediction Error by Actual Finishing Position",
                        xaxis_title="Actual Position",
                        yaxis_title="Average Error",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing season: {e}")


def show_model_info_page(predictor: RacePredictor):
    """Display model information."""
    st.header("Model Information")

    info = predictor.get_model_info()

    if not info.get('is_fitted', False):
        st.warning("No model trained yet")
        return

    # Model overview
    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", info.get('model_type', 'Unknown'))
    col2.metric("Training Races", info.get('training_races', 0))
    col3.metric("Validation MAE", f"{info.get('validation_mae', 0):.2f}")

    st.subheader("Features")
    st.write(f"Number of features: {info.get('num_features', 0)}")

    # Feature importance
    st.subheader("Feature Importance")

    try:
        importance = predictor.get_feature_importance()

        # Sort and display
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f[0] for f in sorted_importance[:15]],
            x=[f[1] for f in sorted_importance[:15]],
            orientation='h',
            marker_color='#E10600'
        ))

        fig.update_layout(
            title="Top 15 Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_dark",
            height=500,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature categories
        st.subheader("Feature Categories")

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

        cat_importance = {}
        for cat, features in categories.items():
            cat_importance[cat] = sum(importance.get(f, 0) for f in features)

        fig = px.pie(
            values=list(cat_importance.values()),
            names=list(cat_importance.keys()),
            title="Importance by Feature Category",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading feature importance: {e}")


def show_training_page(predictor: RacePredictor, collector: F1DataCollector):
    """Display model training interface."""
    st.header("Train Model")

    st.info("""
    Train the prediction model on historical F1 data.
    Recommended: Use 2022-2023 for training to predict 2024 races.
    """)

    training_years = st.multiselect(
        "Select Training Years",
        [2022, 2023, 2024, 2025],
        default=[2022, 2023]
    )

    if st.button("Train Model", type="primary"):
        if not training_years:
            st.error("Please select at least one training year")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Collect training data
            status_text.text("Collecting training data...")
            training_races = []

            for i, year in enumerate(training_years):
                status_text.text(f"Collecting {year} data...")
                season_data = collector.collect_season_data(year)
                training_races.extend(season_data)
                progress_bar.progress((i + 1) / (len(training_years) + 1))

            if len(training_races) < 10:
                st.error("Insufficient training data")
                return

            # Train model
            status_text.text("Training model...")
            metrics = predictor.train(training_races)
            progress_bar.progress(0.9)

            # Save model
            status_text.text("Saving model...")
            predictor.save_model()
            progress_bar.progress(1.0)

            status_text.text("Training complete!")

            # Show results
            st.success("Model trained successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Training MAE", f"{metrics['train_mae']:.2f}")
            col2.metric("Validation MAE", f"{metrics['val_mae']:.2f}")
            col3.metric("Training Samples", metrics['training_samples'])

        except Exception as e:
            st.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()

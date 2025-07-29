import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import requests
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HungarianGPPredictor:
    """
    Comprehensive F1 Hungarian Grand Prix Predictor
    Combines historical data, weather, and driver performance metrics
    """
    
    def __init__(self, cache_dir="f1_cache_hungary"):
        self.cache_dir = cache_dir
        self.setup_cache()
        self.historical_data = {}
        self.weather_data = None
        self.models = {}
        self.scalers = {}
        
        # Hungarian GP specific characteristics
        self.track_characteristics = {
            "overtaking_difficulty": 0.9,  # Very hard to overtake (0-1 scale)
            "qualifying_importance": 0.95,  # Grid position very important
            "weather_sensitivity": 0.7,  # Moderate weather impact
            "tire_degradation": 0.6,  # Medium tire wear
            "downforce_importance": 0.8  # High downforce track
        }
    
    def setup_cache(self):
        """Setup FastF1 caching"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        fastf1.Cache.enable_cache(self.cache_dir)
        print(f"✓ Cache setup complete: {self.cache_dir}")
    
    def load_historical_data(self, years=[2022, 2023, 2024]):
        """Load historical Hungarian GP data"""
        print("Loading historical Hungarian GP data...")
        
        for year in years:
            try:
                print(f"  Loading {year} Hungarian GP...")
                
                # Hungarian GP is typically round 11-13, let's try multiple rounds
                session = None
                for round_num in [11, 12, 13, 10, 14]:  # Try common Hungarian GP round numbers
                    try:
                        session = fastf1.get_session(year, round_num, "R")
                        session.load()
                        
                        # Check if this is actually Hungarian GP
                        if "hun" in session.session_info['Meeting']['Name'].lower() or \
                           "budapest" in session.session_info['Meeting']['Name'].lower():
                            break
                    except:
                        continue
                
                if session is None:
                    print(f"    ⚠️ Could not find {year} Hungarian GP")
                    continue
                
                # Extract race data
                laps = session.laps.copy()
                results = session.results.copy()
                
                # Process lap data
                race_laps = laps[['Driver', 'LapTime', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus']].copy()
                race_laps = race_laps.dropna(subset=['LapTime'])
                race_laps['LapTime_seconds'] = race_laps['LapTime'].dt.total_seconds()
                
                # Get qualifying data
                try:
                    quali = fastf1.get_session(year, round_num, "Q")
                    quali.load()
                    quali_results = quali.results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].copy()
                    quali_results['BestQualiTime'] = quali_results[['Q1', 'Q2', 'Q3']].min(axis=1)
                    quali_results['BestQualiTime_seconds'] = quali_results['BestQualiTime'].dt.total_seconds()
                except:
                    print(f"    ⚠️ Could not load {year} qualifying data")
                    quali_results = None
                
                self.historical_data[year] = {
                    'race_laps': race_laps,
                    'results': results,
                    'quali': quali_results,
                    'session_info': session.session_info
                }
                
                print(f"    ✓ {year} data loaded successfully")
                
            except Exception as e:
                print(f"    ❌ Error loading {year}: {str(e)}")
        
        print(f"✓ Historical data loaded for {len(self.historical_data)} years")
    
    def get_weather_forecast(self, api_key=None):
        """Get weather forecast for Hungarian GP (August 3, 2025)"""
        print("Getting weather forecast for Budapest...")
        
        if not api_key:
            print("  Using fallback weather data (no API key provided)")
            self.weather_data = {
                "temperature": 28.0,  # Typical August temperature in Budapest
                "humidity": 60,
                "wind_speed": 12.0,
                "description": "partly cloudy",
                "rain_probability": 15,
                "track_temp": 45.0  # Estimated track temperature
            }
            return
        
        try:
            # OpenWeather API for Budapest
            url = "http://api.openweathermap.org/data/2.5/forecast"
            params = {
                "q": "Budapest,HU",
                "appid": api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for forecast around race time (August 3, 2025, 3:00 PM local)
                target_date = "2025-08-03 13:00:00"  # UTC time for 3 PM local
                
                forecast = None
                if "list" in data:
                    # Find closest forecast to race time
                    for item in data["list"]:
                        if target_date in item["dt_txt"]:
                            forecast = item
                            break
                    
                    if not forecast and data["list"]:
                        forecast = data["list"][0]  # Use first available
                
                if forecast:
                    self.weather_data = {
                        "temperature": forecast["main"]["temp"],
                        "humidity": forecast["main"]["humidity"],
                        "wind_speed": forecast["wind"]["speed"],
                        "description": forecast["weather"][0]["description"],
                        "rain_probability": forecast.get("pop", 0) * 100,
                        "track_temp": forecast["main"]["temp"] + 15  # Track usually 15°C warmer
                    }
                    print(f"  ✓ Weather forecast retrieved: {forecast['dt_txt']}")
                else:
                    raise Exception("No forecast data found")
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            print(f"  ⚠️ Weather API failed: {str(e)}")
            print("  Using fallback weather data")
            self.weather_data = {
                "temperature": 28.0,
                "humidity": 60,
                "wind_speed": 12.0,
                "description": "partly cloudy",
                "rain_probability": 15,
                "track_temp": 45.0
            }
    
    def create_driver_features(self):
        """Create comprehensive driver performance features"""
        print("Creating driver performance features...")
        
        driver_stats = {}
        
        for year, data in self.historical_data.items():
            race_laps = data['race_laps']
            results = data['results']
            
            # Calculate driver statistics
            for driver in race_laps['Driver'].unique():
                if driver not in driver_stats:
                    driver_stats[driver] = {
                        'avg_lap_time': [],
                        'consistency': [],
                        'tire_management': [],
                        'qualifying_performance': [],
                        'race_positions': [],
                        'years_experience': 0
                    }
                
                driver_laps = race_laps[race_laps['Driver'] == driver]
                
                # Average lap time (excluding outliers)
                clean_laps = driver_laps[
                    (driver_laps['LapTime_seconds'] > 60) & 
                    (driver_laps['LapTime_seconds'] < 120)
                ]
                
                if len(clean_laps) > 5:
                    avg_time = clean_laps['LapTime_seconds'].mean()
                    consistency = clean_laps['LapTime_seconds'].std()
                    
                    driver_stats[driver]['avg_lap_time'].append(avg_time)
                    driver_stats[driver]['consistency'].append(consistency)
                    driver_stats[driver]['years_experience'] += 1
                
                # Tire management (performance over stint)
                if len(clean_laps) > 10:
                    tire_deg = self.calculate_tire_degradation(clean_laps)
                    driver_stats[driver]['tire_management'].append(tire_deg)
                
                # Race result
                driver_result = results[results['Abbreviation'] == driver]
                if not driver_result.empty:
                    position = driver_result['Position'].iloc[0]
                    driver_stats[driver]['race_positions'].append(position)
        
        # Convert to final features
        self.driver_features = {}
        for driver, stats in driver_stats.items():
            if stats['avg_lap_time']:  # Only include drivers with data
                self.driver_features[driver] = {
                    'avg_performance': np.mean(stats['avg_lap_time']),
                    'consistency': np.mean(stats['consistency']),
                    'tire_management': np.mean(stats['tire_management']) if stats['tire_management'] else 1.0,
                    'avg_position': np.mean(stats['race_positions']) if stats['race_positions'] else 10.0,
                    'experience': stats['years_experience'],
                    'hungary_specialist': 1 if len(stats['avg_lap_time']) >= 2 else 0  # Raced Hungary 2+ times
                }
        
        print(f"  ✓ Features created for {len(self.driver_features)} drivers")
    
    def calculate_tire_degradation(self, driver_laps):
        """Calculate tire degradation rate for a driver"""
        if len(driver_laps) < 10:
            return 1.0
        
        # Group by stint (tire compound changes)
        stints = []
        current_compound = None
        current_stint = []
        
        for _, lap in driver_laps.iterrows():
            if lap['Compound'] != current_compound:
                if current_stint:
                    stints.append(current_stint)
                current_stint = [lap['LapTime_seconds']]
                current_compound = lap['Compound']
            else:
                current_stint.append(lap['LapTime_seconds'])
        
        if current_stint:
            stints.append(current_stint)
        
        # Calculate degradation for longest stint
        if stints:
            longest_stint = max(stints, key=len)
            if len(longest_stint) >= 8:
                # Compare first 3 laps vs last 3 laps of stint
                early_pace = np.mean(longest_stint[2:5])
                late_pace = np.mean(longest_stint[-3:])
                degradation = (late_pace - early_pace) / early_pace
                return max(0, degradation)  # Positive degradation only
        
        return 1.0  # Default degradation
    
    def prepare_training_data(self):
        """Prepare training dataset from historical data"""
        print("Preparing training data...")
        
        training_data = []
        
        for year, data in self.historical_data.items():
            if data['quali'] is None:
                continue
            
            results = data['results']
            quali = data['quali']
            
            for _, result in results.iterrows():
                driver = result['Abbreviation']
                
                if driver not in self.driver_features:
                    continue
                
                # Get qualifying position
                quali_data = quali[quali['Abbreviation'] == driver]
                if quali_data.empty:
                    continue
                
                quali_pos = quali_data['Position'].iloc[0]
                quali_time = quali_data['BestQualiTime_seconds'].iloc[0]
                
                if pd.isna(quali_time) or pd.isna(quali_pos):
                    continue
                
                # Create feature vector
                features = {
                    'quali_position': quali_pos,
                    'quali_time': quali_time,
                    'driver_avg_performance': self.driver_features[driver]['avg_performance'],
                    'driver_consistency': self.driver_features[driver]['consistency'],
                    'tire_management': self.driver_features[driver]['tire_management'],
                    'driver_experience': self.driver_features[driver]['experience'],
                    'hungary_specialist': self.driver_features[driver]['hungary_specialist'],
                    'weather_factor': 1.0,  # Placeholder - would use historical weather
                    'year': year
                }
                
                # Target: final race position
                race_position = result['Position']
                
                if not pd.isna(race_position) and race_position <= 20:  # Valid finishing position
                    features['target'] = race_position
                    training_data.append(features)
        
        self.training_df = pd.DataFrame(training_data)
        print(f"  ✓ Training data prepared: {len(self.training_df)} samples")
        
        if len(self.training_df) < 10:
            print("  ⚠️ Limited training data - predictions may be less accurate")
    
    def train_models(self):
        """Train multiple ML models"""
        print("Training prediction models...")
        
        if len(self.training_df) < 5:
            print("  ❌ Insufficient training data")
            return
        
        # Prepare features and target
        feature_cols = [col for col in self.training_df.columns if col not in ['target', 'year']]
        X = self.training_df[feature_cols]
        y = self.training_df['target']
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models_to_train.items():
            try:
                model.fit(X_scaled, y)
                
                # Evaluate on training data
                y_pred = model.predict(X_scaled)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                self.models[name] = {
                    'model': model,
                    'mae': mae,
                    'r2': r2,
                    'feature_cols': feature_cols
                }
                
                print(f"  ✓ {name}: MAE={mae:.2f}, R²={r2:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {str(e)}")
        
        # Select best model
        if self.models:
            best_model = min(self.models.keys(), key=lambda k: self.models[k]['mae'])
            self.best_model = best_model
            print(f"  ✓ Best model: {best_model}")
        else:
            print("  ❌ No models trained successfully")
    
    def predict_2025_race(self, quali_results_2025):
        """Predict 2025 Hungarian GP results"""
        print("Predicting 2025 Hungarian GP results...")
        
        if not self.models:
            print("  ❌ No trained models available")
            return None
        
        predictions = []
        
        for driver_data in quali_results_2025:
            driver = driver_data['driver']
            
            # Get driver features (use average if driver not in historical data)
            if driver in self.driver_features:
                driver_feats = self.driver_features[driver]
            else:
                # Use average features for new drivers
                avg_performance = np.mean([f['avg_performance'] for f in self.driver_features.values()])
                avg_consistency = np.mean([f['consistency'] for f in self.driver_features.values()])
                
                driver_feats = {
                    'avg_performance': avg_performance,
                    'consistency': avg_consistency,
                    'tire_management': 1.0,
                    'experience': 1,
                    'hungary_specialist': 0
                }
            
            # Create feature vector
            features = {
                'quali_position': driver_data['quali_position'],
                'quali_time': driver_data['quali_time'],
                'driver_avg_performance': driver_feats['avg_performance'],
                'driver_consistency': driver_feats['consistency'],
                'tire_management': driver_feats['tire_management'],
                'driver_experience': driver_feats['experience'],
                'hungary_specialist': driver_feats['hungary_specialist'],
                'weather_factor': self.calculate_weather_factor()
            }
            
            # Make prediction using best model
            model_info = self.models[self.best_model]
            feature_vector = [[features[col] for col in model_info['feature_cols']]]
            feature_vector_scaled = self.scalers['main'].transform(feature_vector)
            
            predicted_position = model_info['model'].predict(feature_vector_scaled)[0]
            
            predictions.append({
                'driver': driver,
                'quali_position': driver_data['quali_position'],
                'predicted_position': max(1, min(20, round(predicted_position))),
                'confidence': self.calculate_prediction_confidence(features)
            })
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        return predictions
    
    def calculate_weather_factor(self):
        """Calculate weather impact factor"""
        if not self.weather_data:
            return 1.0
        
        factor = 1.0
        
        # Rain impact
        if self.weather_data['rain_probability'] > 30:
            factor *= 1.2  # Rain increases unpredictability
        
        # Temperature impact
        if self.weather_data['track_temp'] > 50:
            factor *= 1.1  # Hot track affects tire performance
        
        # Wind impact
        if self.weather_data['wind_speed'] > 20:
            factor *= 1.05  # Strong wind affects handling
        
        return factor
    
    def calculate_prediction_confidence(self, features):
        """Calculate confidence level for prediction (0-100%)"""
        confidence = 100
        
        # Reduce confidence for drivers without historical data
        if features['hungary_specialist'] == 0:
            confidence -= 15
        
        # Reduce confidence in bad weather
        if self.weather_data and self.weather_data['rain_probability'] > 50:
            confidence -= 20
        
        # Reduce confidence for inconsistent drivers
        if features['driver_consistency'] > 2.0:  # High standard deviation
            confidence -= 10
        
        return max(50, confidence)  # Minimum 50% confidence

# Example usage and testing
def main():
    """Main execution function"""
    print("🏎️  HUNGARIAN GRAND PRIX 2025 PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HungarianGPPredictor()
    
    # Load historical data
    predictor.load_historical_data([2022, 2023, 2024])
    # predictor.get_weather_forecast("your_openweather_api_key")
    predictor.get_weather_forecast("api-key-here")  # Uses fallback data
    
    # Create driver features
    predictor.create_driver_features()
    
    # Prepare training data
    predictor.prepare_training_data()
    
    # Train models
    predictor.train_models()
    
    # Example 2025 qualifying results (you would replace this with actual data)
    example_quali_2025 = [
        {"driver": "VER", "quali_position": 1, "quali_time": 76.5},
        {"driver": "LEC", "quali_position": 2, "quali_time": 76.7},
        {"driver": "NOR", "quali_position": 3, "quali_time": 76.8},
        {"driver": "PIA", "quali_position": 4, "quali_time": 76.9},
        {"driver": "SAI", "quali_position": 5, "quali_time": 77.0},
        {"driver": "RUS", "quali_position": 6, "quali_time": 77.1},
        {"driver": "HAM", "quali_position": 7, "quali_time": 77.2},
        {"driver": "ALO", "quali_position": 8, "quali_time": 77.3},
        {"driver": "STR", "quali_position": 9, "quali_time": 77.4},
        {"driver": "TSU", "quali_position": 10, "quali_time": 77.5},
    ]
    
    # Make predictions
    if predictor.models:
        predictions = predictor.predict_2025_race(example_quali_2025)
        
        if predictions:
            print("\n🏁 2025 HUNGARIAN GP RACE PREDICTIONS")
            print("=" * 60)
            print(f"{'Pos':<4} {'Driver':<8} {'Quali':<6} {'Confidence':<10} {'Notes'}")
            print("-" * 40)
            
            for i, pred in enumerate(predictions):
                notes = ""
                if pred['predicted_position'] != pred['quali_position']:
                    change = pred['quali_position'] - pred['predicted_position']
                    notes = f"({'+' if change > 0 else ''}{change})"
                
                print(f"{pred['predicted_position']:<4} {pred['driver']:<8} "
                      f"P{pred['quali_position']:<5} {pred['confidence']:<10.0f}% {notes}")
            
            # Weather summary
            if predictor.weather_data:
                print(f"\n🌤️  WEATHER CONDITIONS")
                print("-" * 30)
                print(f"Temperature: {predictor.weather_data['temperature']:.1f}°C")
                print(f"Track Temp: {predictor.weather_data['track_temp']:.1f}°C")
                print(f"Rain Chance: {predictor.weather_data['rain_probability']:.0f}%")
                print(f"Wind Speed: {predictor.weather_data['wind_speed']:.1f} km/h")
        
        else:
            print("❌ Prediction failed")
    else:
        print("❌ No models available for prediction")

if __name__ == "__main__":
    main()

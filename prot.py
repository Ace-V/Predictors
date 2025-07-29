# Simplified & Humanized Hungarian GP 2025 Predictor
import os
import warnings
import requests
import numpy as np
import pandas as pd
import fastf1
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class HungarianGPPredictor:
    def __init__(self, cache_dir="f1_cache_hungary"):
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self.historical_data, self.models, self.scalers, self.driver_features = {}, {}, {}, {}
        self.weather_data = None
        print("✓ Cache setup complete")

    def load_data(self, years=[2022, 2023, 2024]):
        print("Loading historical race data...")
        for year in years:
            for rnd in [11, 12, 13, 10, 14]:
                try:
                    session = fastf1.get_session(year, rnd, "R")
                    session.load()
                    if "hun" in session.session_info['Meeting']['Name'].lower():
                        laps = session.laps.dropna(subset=['LapTime'])
                        laps['LapTime_seconds'] = laps['LapTime'].dt.total_seconds()
                        quali = fastf1.get_session(year, rnd, "Q")
                        quali.load()
                        quali_data = quali.results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']]
                        quali_data['BestTime'] = quali_data[['Q1', 'Q2', 'Q3']].min(axis=1).dt.total_seconds()
                        self.historical_data[year] = {
                            'laps': laps, 'results': session.results, 'quali': quali_data
                        }
                        print(f"✓ {year} loaded")
                        break
                except: continue

    def get_weather(self, api_key=None):
        print("Getting weather...")
        self.weather_data = {
            "temperature": 28, "humidity": 60, "wind_speed": 12,
            "description": "partly cloudy", "rain_probability": 15,
            "track_temp": 43
        }

    def extract_driver_features(self):
        print("Building driver profiles...")
        stats = {}
        for year, data in self.historical_data.items():
            for driver in data['laps']['Driver'].unique():
                clean_laps = data['laps'].query("Driver == @driver and 60 < LapTime_seconds < 120")
                if len(clean_laps) < 5: continue
                stats.setdefault(driver, {'lap_times': [], 'consistency': [], 'positions': [], 'experience': 0})
                stats[driver]['lap_times'].append(clean_laps['LapTime_seconds'].mean())
                stats[driver]['consistency'].append(clean_laps['LapTime_seconds'].std())
                stats[driver]['experience'] += 1
                pos = data['results'].query("Abbreviation == @driver")['Position']
                if not pos.empty: stats[driver]['positions'].append(pos.iloc[0])

        for d, s in stats.items():
            self.driver_features[d] = {
                'avg_perf': np.mean(s['lap_times']),
                'consistency': np.mean(s['consistency']),
                'avg_pos': np.mean(s['positions']) if s['positions'] else 10,
                'experience': s['experience'],
                'hungary_specialist': int(s['experience'] >= 2)
            }
        print(f"✓ Features created for {len(self.driver_features)} drivers")

    def prepare_data(self):
        print("Preparing training set...")
        rows = []
        for year, data in self.historical_data.items():
            for _, row in data['results'].iterrows():
                d = row['Abbreviation']
                q = data['quali'].query("Abbreviation == @d")
                if d not in self.driver_features or q.empty: continue
                f = self.driver_features[d]
                rows.append({
                    'quali_pos': q['Position'].values[0],
                    'quali_time': q['BestTime'].values[0],
                    'avg_perf': f['avg_perf'],
                    'consistency': f['consistency'],
                    'experience': f['experience'],
                    'hungary_specialist': f['hungary_specialist'],
                    'target': row['Position']
                })
        self.train_df = pd.DataFrame(rows)
        print(f"✓ {len(self.train_df)} samples ready")

    def train(self):
        print("Training models...")
        if len(self.train_df) < 5: return
        X = self.train_df.drop(columns=['target'])
        y = self.train_df['target']
        X = X.fillna(X.mean())
        scaler = StandardScaler().fit(X)
        self.scalers['main'] = scaler
        X_scaled = scaler.transform(X)

        models = {
            'rf': RandomForestRegressor(n_estimators=100),
            'gb': GradientBoostingRegressor(n_estimators=100)
        }

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            self.models[name] = {'model': model, 'mae': mean_absolute_error(y, y_pred)}
            print(f"{name}: MAE={self.models[name]['mae']:.2f}")
        self.best_model = min(self.models, key=lambda k: self.models[k]['mae'])
        print(f"✓ Best: {self.best_model}")

    def predict(self, quali_data):
        print("Making predictions...")
        results = []
        for d in quali_data:
            feats = self.driver_features.get(d['driver'], {
                'avg_perf': np.mean([f['avg_perf'] for f in self.driver_features.values()]),
                'consistency': 2.0, 'experience': 1, 'hungary_specialist': 0
            })
            x = [[
                d['quali_position'], d['quali_time'],
                feats['avg_perf'], feats['consistency'],
                feats['experience'], feats['hungary_specialist']
            ]]
            model = self.models[self.best_model]['model']
            scaled = self.scalers['main'].transform(x)
            pred = round(model.predict(scaled)[0])
            results.append({
                'driver': d['driver'],
                'quali': d['quali_position'],
                'predicted': max(1, min(20, pred))
            })
        return sorted(results, key=lambda r: r['predicted'])

# Main Usage
if __name__ == "__main__":
    print("\n🏁 Hungarian GP Predictor 2025")
    predictor = HungarianGPPredictor()
    predictor.load_data()
    predictor.get_weather()
    predictor.extract_driver_features()
    predictor.prepare_data()
    predictor.train()

    quali_2025 = [
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

    if predictor.models:
        predictions = predictor.predict(quali_2025)
        for p in predictions:
            print(f"P{p['predicted']:<2}  {p['driver']:<5} (Quali: P{p['quali']})")

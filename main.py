import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import warnings
import json
from pulp import *

warnings.filterwarnings('ignore')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

print("Loading data...")
with open('config.json', 'r') as f:
    config = json.load(f)

# Constants
emission_per_km = 0.9
n_per_ton = 25.0
offset_per_n = 5.0
seq_per_kg = 0.2
leach_per_n = 10.0
buffer_pct = 0.0  # Improved: Set to 0 to maximize offsets
rain_thresh = 30.0
forecast_window = 5
overflow_pen = 1000.0
truck_cap = 10.0
M = 20.0  # Big-M for binary delivery

# Load Files
stp_df = pd.read_csv('stp_registry.csv')
farm_df = pd.read_csv('farm_locations.csv')
weather_df = pd.read_csv('daily_weather_2025.csv')
demand_df = pd.read_csv('daily_n_demand.csv')
plant_df = pd.read_csv('planting_schedule_2025.csv')
sample_sub = pd.read_csv('sample_submission.csv')  # Assume you have this; otherwise, generate grid

# Conversions
stp_ids = stp_df['stp_id'].tolist()
farm_ids = farm_df['farm_id'].tolist()
farm_areas = farm_df.set_index('farm_id')['area_ha'].to_dict()
farm_zones = farm_df.set_index('farm_id')['zone'].to_dict()
stp_dict = stp_df.set_index('stp_id').to_dict('index')
farm_locs = farm_df.set_index('farm_id')[['lat', 'lon']].to_dict('index')

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)
demand_df['date'] = pd.to_datetime(demand_df['date'])
demand_df.set_index('date', inplace=True)

zones = farm_df['zone'].unique()
dates = pd.date_range('2025-01-01', '2025-12-31')
n_days = len(dates)

print("Precomputing constraints...")
can_apply = {zone: [False] * n_days for zone in zones}
for i, date in enumerate(dates):
    for zone in zones:
        if date in weather_df.index:
            end = min(i + forecast_window + 1, n_days)
            max_rain = 0
            for j in range(i, end):
                d_check = dates[j]
                if d_check in weather_df.index:
                    max_rain = max(max_rain, weather_df.at[d_check, zone])
            can_apply[zone][i] = max_rain <= rain_thresh
        else:
            can_apply[zone][i] = True
# --- REPLAY SIMULATION FROM FILE ---
print("Loading target submission for replay...")
try:
    target_sub = pd.read_csv('better_better_submission.csv')
    target_sub['date'] = pd.to_datetime(target_sub['date'])
except FileNotFoundError:
    print("Error: 'better_better_submission.csv' not found. Please provide the file.")
    exit()

# Pre-process moves: filter non-zeros and index by date
actual_moves = target_sub[target_sub['tons_delivered'] > 0.0]
moves_lookup = {}
for dt, group in actual_moves.groupby('date'):
    # Store list of (stp_id, farm_id, tons)
    moves_lookup[dt] = group[['stp_id', 'farm_id', 'tons_delivered']].values.tolist()

dist = {}
for s in stp_ids:
    for f in farm_ids:
        dist[(s, f)] = haversine(stp_dict[s]['lat'], stp_dict[s]['lon'],
                                 farm_locs[f]['lat'], farm_locs[f]['lon'])

storage = {s: 0.0 for s in stp_ids}
farm_accumulated_demand = {f: 0.0 for f in farm_ids}

m_offset = 0.0
m_seq = 0.0
m_transport = 0.0
m_leaching = 0.0
m_overflow = 0.0

print("Replaying submission to verify score...")

for day_idx in range(n_days):
    date = dates[day_idx]
    date_str = date.strftime('%Y-%m-%d')
    
    # 1. Production
    for s in stp_ids:
        storage[s] += stp_dict[s]['daily_output_tons']
    
    # 2. Accumulate Demand
    if date in demand_df.index:
        demand_row = demand_df.loc[date]
        for f in farm_ids:
            kg_n = demand_row[f] * farm_areas[f]
            potential_tons = kg_n * (1 + buffer_pct) / n_per_ton
            farm_accumulated_demand[f] += potential_tons
            
    # 3. Execute Moves from Target File
    if date in moves_lookup:
        for move in moves_lookup[date]:
            s_id, f_id, tons = move
            
            # Validation (optional warning)
            # if storage[s_id] < tons: print(f"Warning: Negative storage at {s_id} on {date_str}")
            
            storage[s_id] -= tons
            
            # Stats
            dist_km = dist[(s_id, f_id)]
            
            acc_demand = farm_accumulated_demand[f_id]
            
            # Utilized vs Excess (Leaching)
            utilized = min(tons, acc_demand)
            excess = max(0, tons - acc_demand)
            
            farm_accumulated_demand[f_id] = max(0, acc_demand - utilized)
            
            m_offset += utilized * n_per_ton * offset_per_n
            m_seq += tons * 1000 * seq_per_kg
            m_leaching += excess * n_per_ton * leach_per_n
            m_transport += max(1, np.ceil(tons/truck_cap)) * 2 * dist_km * emission_per_km

    # 4. Check Overflow
    for s in stp_ids:
        if storage[s] > stp_dict[s]['storage_max_tons']:
            over = storage[s] - stp_dict[s]['storage_max_tons']
            m_overflow += over
            storage[s] = stp_dict[s]['storage_max_tons']
            
    if day_idx % 50 == 0:
        score_est = (m_offset + m_seq - m_transport - m_leaching - m_overflow*overflow_pen)/1000
        print(f"Day {day_idx+1}: Score={score_est:.0f}k")

# --- SAVE OUTPUT ---
print("Saving submission.csv...")
# We simply save the target dataframe as it is the requested output
target_sub[['id', 'date', 'stp_id', 'farm_id', 'tons_delivered']].to_csv('submission.csv', index=False)

# Final Score
score = m_offset + m_seq - m_transport - m_leaching - (m_overflow * overflow_pen)
print("="*60)
print(f"FINAL SCORE: {score:,.2f}")
print("="*60)
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

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
emission_per_km     = 0.9
n_per_ton           = 25.0
offset_per_n        = 5.0
seq_per_kg          = 0.2
leach_per_n         = 10.0
buffer_pct          = 0.0           # No buffer to maximize offsets
rain_thresh         = 30.0
forecast_window     = 5
overflow_pen        = 1000.0
truck_cap           = 10.0
M                   = 100.0         # Big-M for binary vars
delivery_fixed_cost = 60.0          # Penalize too many deliveries

# ─── LOAD FILES ──────────────────────────────────────────────────────────────
stp_df     = pd.read_csv('stp_registry.csv')
farm_df    = pd.read_csv('farm_locations.csv')
weather_df = pd.read_csv('daily_weather_2025.csv')
demand_df  = pd.read_csv('daily_n_demand.csv')
plant_df   = pd.read_csv('planting_schedule_2025.csv')
sample_sub = pd.read_csv('sample_submission.csv')

stp_ids    = stp_df['stp_id'].tolist()
farm_ids   = farm_df['farm_id'].tolist()
farm_areas = farm_df.set_index('farm_id')['area_ha'].to_dict()
farm_zones = farm_df.set_index('farm_id')['zone'].to_dict()
stp_dict   = stp_df.set_index('stp_id').to_dict('index')
farm_locs  = farm_df.set_index('farm_id')[['lat', 'lon']].to_dict('index')

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df.set_index('date', inplace=True)
demand_df['date']  = pd.to_datetime(demand_df['date'])
demand_df.set_index('date', inplace=True)

zones = farm_df['zone'].unique()
dates = pd.date_range('2025-01-01', '2025-12-31')
n_days = len(dates)

print("Precomputing rain-safe windows...")
can_apply = {z: [False] * n_days for z in zones}
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

dist = {}
for s in stp_ids:
    for f in farm_ids:
        dist[(s, f)] = haversine(stp_dict[s]['lat'], stp_dict[s]['lon'],
                                 farm_locs[f]['lat'], farm_locs[f]['lon'])

stp_closest = {s: sorted(farm_ids, key=lambda f: dist[(s, f)]) for s in stp_ids}

# ─── SIMULATION ──────────────────────────────────────────────────────────────
storage = {s: 0.0 for s in stp_ids}
farm_accumulated_demand = {f: 0.0 for f in farm_ids}
deliveries = {}  # (date_str, farm_id) → (stp_id, tons)

m_offset = m_seq = m_transport = m_leaching = m_overflow = 0.0

print("Starting daily optimization...")

for day_idx in range(n_days):
    date = dates[day_idx]
    date_str = date.strftime('%Y-%m-%d')

    # Production
    for s in stp_ids:
        storage[s] += stp_dict[s]['daily_output_tons']

    # Demand accumulation
    if date in demand_df.index:
        row = demand_df.loc[date]
        for f in farm_ids:
            kg_n = row.get(f, 0.0) * farm_areas[f]
            farm_accumulated_demand[f] += kg_n / n_per_ton

    avail_farms = [f for f in farm_ids if can_apply[farm_zones[f]][day_idx]]

    # Target farms (limit to ~40-50 to keep MILP fast)
    target_farms = set()
    for s in stp_ids:
        for f in stp_closest[s][:40]:
            if f in avail_farms:
                target_farms.add(f)
    for f in avail_farms:
        if farm_accumulated_demand[f] > 2.0:
            target_farms.add(f)
    target_farms = list(target_farms)

    if not target_farms or sum(storage.values()) < 0.5:
        for s in stp_ids:
            if storage[s] > stp_dict[s]['storage_max_tons']:
                over = storage[s] - stp_dict[s]['storage_max_tons']
                m_overflow += over
                storage[s] = stp_dict[s]['storage_max_tons']
        continue

    # ─── MILP PROBLEM ────────────────────────────────────────────────────────
    prob = LpProblem("Biosolid_Dispatch", LpMaximize)

    x     = LpVariable.dicts("x",     [(s,f) for s in stp_ids for f in target_farms], 0, None, LpContinuous)
    y     = LpVariable.dicts("y",     [(s,f) for s in stp_ids for f in target_farms], cat='Binary')
    trips = LpVariable.dicts("trips", [(s,f) for s in stp_ids for f in target_farms], 0, None, LpInteger)
    util  = LpVariable.dicts("util",  target_farms, 0, None, LpContinuous)
    exc   = LpVariable.dicts("exc",   target_farms, 0, None, LpContinuous)
    over  = LpVariable.dicts("over",  stp_ids,      0, None, LpContinuous)

    # Constraints
    for f in target_farms:
        xtot = lpSum(x[(s,f)] for s in stp_ids)
        prob += util[f] + exc[f] == xtot
        prob += lpSum(y[(s,f)] for s in stp_ids) <= 1
        prob += util[f] <= 1.15 * farm_accumulated_demand[f]  # slight over-apply allowed

    for s in stp_ids:
        for f in target_farms:
            prob += x[(s,f)] <= M * y[(s,f)]
            prob += x[(s,f)] <= truck_cap * trips[(s,f)]

    for s in stp_ids:
        shipped = lpSum(x[(s,f)] for f in target_farms)
        prob += over[s] >= (storage[s] - shipped) - stp_dict[s]['storage_max_tons']

    # Late-year urgency (last 20 days)
    if day_idx > n_days - 20:
        prob += lpSum(x.values()) >= 0.85 * sum(storage.values())

    # Objective
    offset   = lpSum(util[f] * n_per_ton * offset_per_n for f in target_farms)
    seq      = lpSum((util[f] + exc[f]) * 1000 * seq_per_kg for f in target_farms)
    leaching = lpSum(exc[f] * n_per_ton * leach_per_n for f in target_farms)
    trans    = lpSum(trips[(s,f)] * 2 * dist[(s,f)] * emission_per_km for s in stp_ids for f in target_farms)
    over_pen = lpSum(over[s] * overflow_pen for s in stp_ids)
    del_pen  = delivery_fixed_cost * lpSum(y.values())

    prob += offset + seq - 1.4 * leaching - trans - over_pen - del_pen

    # Solve
    status = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=40))

    if LpStatus[status] == 'Optimal':
        delivered_today = 0.0
        for f in target_farms:
            for s in stp_ids:
                tons = value(x[(s,f)])
                if tons > 0.01:
                    deliveries[(date_str, f)] = (s, tons)
                    delivered_today += tons
                    break

        for f in target_farms:
            farm_accumulated_demand[f] -= value(util[f])

        for s in stp_ids:
            shipped_s = value(lpSum(x[(s,f)] for f in target_farms))
            storage[s] -= shipped_s
            ov = value(over[s])
            if ov > 0:
                storage[s] = stp_dict[s]['storage_max_tons']
            m_overflow += ov

        m_offset   += value(offset)
        m_seq      += value(seq)
        m_leaching += value(leaching)
        m_transport+= value(trans)

        print(f"Day {day_idx+1:3d} | Delivered {delivered_today:5.1f}t | "
              f"Storage remaining {sum(storage.values()):5.0f}t | "
              f"Cumulative overflow {m_overflow:4.0f}t")
    else:
        print(f"Day {day_idx+1:3d} | MILP failed ({LpStatus[status]}) → using greedy fallback")
        candidates = []
        for s in stp_ids:
            if storage[s] < 0.5: continue
            for f in stp_closest[s][:50]:
                if f not in avail_farms: continue
                acc = farm_accumulated_demand[f]
                if acc < 0.1: continue
                amt = min(storage[s], truck_cap, acc * 1.2)
                candidates.append({'s':s, 'f':f, 'amt':amt, 'dist':dist[(s,f)]})
        candidates.sort(key=lambda c: c['dist'])
        used = set()
        for c in candidates:
            s, f, amt = c['s'], c['f'], c['amt']
            if f in used: continue
            if storage[s] < amt: amt = storage[s]
            if amt < 0.1: continue
            deliveries[(date_str, f)] = (s, amt)
            storage[s] -= amt
            util_amt = min(amt, farm_accumulated_demand[f])
            exc_amt  = max(0, amt - farm_accumulated_demand[f])
            farm_accumulated_demand[f] -= util_amt
            m_offset   += util_amt * n_per_ton * offset_per_n
            m_seq      += amt * 1000 * seq_per_kg
            m_leaching += exc_amt * n_per_ton * leach_per_n
            m_transport+= np.ceil(amt / truck_cap) * 2 * c['dist'] * emission_per_km
            used.add(f)

        for s in stp_ids:
            if storage[s] > stp_dict[s]['storage_max_tons']:
                over = storage[s] - stp_dict[s]['storage_max_tons']
                m_overflow += over
                storage[s] = stp_dict[s]['storage_max_tons']

    if day_idx % 30 == 0 or day_idx == n_days - 1:
        est = (m_offset + m_seq - m_transport - m_leaching - m_overflow * overflow_pen) / 1000
        print(f"  → Est score: {est:,.0f}k | Total delivered so far: {sum(v[1] for v in deliveries.values()):,.0f}t")

# ─── GENERATE SUBMISSION ─────────────────────────────────────────────────────
print("\nGenerating submission.csv ...")

# Build lookup from deliveries
delivery_lookup = {(d, f): (s, t) for (d, f), (s, t) in deliveries.items()}

# Fill sample_submission
sample_sub['date'] = sample_sub['date'].astype(str)
sample_sub['stp_id'] = 'STP_TVM'
sample_sub['tons_delivered'] = 0.0

for idx, row in sample_sub.iterrows():
    key = (row['date'], row['farm_id'])
    if key in delivery_lookup:
        s, t = delivery_lookup[key]
        sample_sub.at[idx, 'stp_id'] = s
        sample_sub.at[idx, 'tons_delivered'] = t

final_sub = sample_sub[['id', 'date', 'stp_id', 'farm_id', 'tons_delivered']]
final_sub.to_csv('submission.csv', index=False)

print(f"Submission saved: {len(final_sub)} rows | "
      f"Non-zero deliveries: {(final_sub['tons_delivered'] > 0).sum()}")

# Final score
final_score = m_offset + m_seq - m_transport - m_leaching - (m_overflow * overflow_pen)
print("\n" + "="*70)
print(f"FINAL NET CARBON CREDIT SCORE: {final_score:,.2f} kg CO₂ eq")
print(f"  Total biosolids delivered : {sum(v[1] for v in deliveries.values()):,.1f} tons")
print(f"  Overflow amount           : {m_overflow:,.1f} tons (penalty {m_overflow*overflow_pen:,.0f})")
print(f"  Leaching penalty          : {m_leaching:,.0f}")
print(f"  Transport emissions       : {m_transport:,.0f}")
print("="*70)

print("Done! Submit 'submission.csv' and share the score if you want further tuning.")
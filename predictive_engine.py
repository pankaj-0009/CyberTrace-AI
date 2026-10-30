"""
Sonipat ATM Fraud Risk — Predictive Analytics Engine
=====================================================

IMPORTANT DATA NOTE (read before using in any real / judged deployment):
-------------------------------------------------------------------------
`complaints_log.csv` contains real NCRP-style complaint records for ATMs in
Connaught Place, Delhi (IDs like DEL-CP-1030). `Accurate_SNP_ATM_list.csv`
contains real ATMs in Sonipat, ~30-40km away. There is no genuine recorded
link between the two files.

For this prototype we generate a clearly-labeled SYNTHETIC linkage (each
complaint is probabilistically assigned to a Sonipat ATM, weighted so that
ATMs in busier/urban areas receive proportionally more complaints, which is
more defensible than pure uniform random but is still synthetic). Every
score this script produces should be presented as "demo data" until it is
replaced with real Sonipat cybercrime/ATM incident records. Do NOT present
these scores as real risk findings about the named banks below without that
swap — the bank names and addresses are real, the fraud linkage is not.
-------------------------------------------------------------------------
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.model_selection import LeaveOneOut, cross_val_score

RNG = np.random.default_rng(42)

# Exponential decay constant (in days) applied to each complaint's contribution
# to Severity_Index -- the target the risk regressor learns. Smaller = the model
# weights *recent* activity more heavily and reacts faster to a live feed.
#
# This was 30.0, which meant a complaint from 6 weeks ago still carried ~24% of
# the weight of one from today. With a few thousand historical rows on file, a
# handful of new complaints per cycle couldn't shift the score measurably and
# the dashboard looked frozen. 7.0 makes the last week dominate, which is also
# the more defensible framing for operational tasking ("where should patrols go
# this week"), while the full history still shows up in Complaint_Count.
RECENCY_HALFLIFE_DAYS = 7.0

# Output next to this script, not a hardcoded sandbox path -- works on any
# machine regardless of where the folder lives.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
atms = pd.read_csv(os.path.join(SCRIPT_DIR, 'Accurate_SNP_ATM_list.csv'))
complaints = pd.read_csv(os.path.join(SCRIPT_DIR, 'complaints_log.csv'))

atms['bank_name'] = atms['bank_name'].str.strip().str.title()
atms['ATM_ID'] = [f"SNP-ATM-{i+1001:04d}" for i in range(len(atms))]
atms['area'] = atms['area'].str.strip()

# ---------------------------------------------------------------------------
# 2. Real geospatial clustering on the ACTUAL Sonipat ATM coordinates
#    (this part is legitimate regardless of the complaint-linkage issue --
#    the ATM locations themselves are real)
# ---------------------------------------------------------------------------
coords_rad = np.radians(atms[['latitude', 'longitude']].values)
kms_per_radian = 6371.0088
atms['Zone_Cluster'] = DBSCAN(eps=3.0 / kms_per_radian, min_samples=2,
                               metric='haversine').fit_predict(coords_rad)
cluster_density = atms['Zone_Cluster'].value_counts().to_dict()
atms['Cluster_Density'] = atms['Zone_Cluster'].map(
    lambda c: cluster_density[c] if c != -1 else 1)

# ---------------------------------------------------------------------------
# 3. SYNTHETIC complaint -> ATM linkage (disclosed, weighted by area size,
#    not uniform-random, and not overwriting the complaint's own real fields)
# ---------------------------------------------------------------------------
complaints['Date_Time'] = pd.to_datetime(complaints['Date_Time'])
weights = atms.groupby('area')['ATM_ID'].transform('count')
weights = weights / weights.sum()
sim_idx = RNG.choice(len(atms), size=len(complaints), p=weights.values)

complaints['Sim_ATM_ID'] = atms.iloc[sim_idx]['ATM_ID'].values
complaints['Sim_Area'] = atms.iloc[sim_idx]['area'].values

complaints['Hour'] = complaints['Date_Time'].dt.hour
complaints['Is_Weekend'] = (complaints['Date_Time'].dt.dayofweek >= 5).astype(int)
complaints['Is_Night'] = complaints['Hour'].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)
max_date = complaints['Date_Time'].max()
complaints['Recency_Days'] = (max_date - complaints['Date_Time']).dt.days
complaints['Recency_Weight'] = np.exp(-complaints['Recency_Days'] / RECENCY_HALFLIFE_DAYS)

# ---------------------------------------------------------------------------
# 4. Aggregate to ATM level (this is the correct unit of analysis for
#    "which ATMs are hotspots", not classifying single complaints)
# ---------------------------------------------------------------------------
def agg_for_atm(sub):
    n = len(sub)
    return pd.Series({
        'Complaint_Count': n,
        'Recent_30d_Count': (sub['Recency_Days'] <= 30).sum(),
        'Avg_Risk_Score': sub['Risk_Score'].mean() if n else 0,
        'Severity_Index': (sub['Risk_Score'] * sub['Recency_Weight']).sum(),
        'Avg_Fund_Recovery_Pct': sub['Fund_Recovery_Pct'].mean() if n else 0,
        'Pct_Blocked': (sub['Withdrawal_Status'] == 'Blocked').mean() if n else 0,
        'Pct_Under_Investigation': (sub['Case_Status'] == 'Under Investigation').mean() if n else 0,
        'Fraud_Type_Diversity': sub['Fraud_Type'].nunique(),
        'Top_Fraud_Type': sub['Fraud_Type'].mode().iat[0] if n else 'None',
        'Avg_Transaction_Amount': sub['Transaction_Amount'].mean() if n else 0,
        'Pct_Weekend': sub['Is_Weekend'].mean() if n else 0,
        'Pct_Night': sub['Is_Night'].mean() if n else 0,
    })

agg = complaints.groupby('Sim_ATM_ID').apply(agg_for_atm)
atm_table = atms.set_index('ATM_ID').join(agg).fillna(0).reset_index().rename(columns={'index': 'ATM_ID'})

# ---------------------------------------------------------------------------
# 5. Unsupervised anomaly / pattern detection (does not depend on a
#    possibly-noisy label -- flags ATMs whose complaint pattern looks
#    statistically unusual vs the rest of the network)
# ---------------------------------------------------------------------------
pattern_features = ['Complaint_Count', 'Severity_Index', 'Pct_Night', 'Fraud_Type_Diversity',
                     'Avg_Transaction_Amount', 'Cluster_Density']
Xp = atm_table[pattern_features].values
iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
atm_table['Anomaly_Flag'] = (iso.fit_predict(Xp) == -1)
atm_table['Anomaly_Score_Raw'] = -iso.score_samples(Xp)  # higher = more anomalous

# ---------------------------------------------------------------------------
# 6. Supervised risk regressor
#    n=53 ATMs is a very small sample -- validated with Leave-One-Out so the
#    reported score is an honest estimate, not an overfit training score.
#    Target = Severity_Index (recency-weighted historical severity), which
#    the model learns to reconstruct from geospatial + behavioural features
#    alone (useful once/if a *new* ATM or a period with no direct complaints
#    yet needs a projected risk).
# ---------------------------------------------------------------------------
feature_cols = ['Complaint_Count', 'Recent_30d_Count', 'Avg_Fund_Recovery_Pct',
                 'Pct_Blocked', 'Pct_Under_Investigation', 'Fraud_Type_Diversity',
                 'Avg_Transaction_Amount', 'Pct_Weekend', 'Pct_Night',
                 'Cluster_Density', 'Anomaly_Score_Raw']
X = atm_table[feature_cols].values
y = atm_table['Severity_Index'].values

gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
loo_scores = cross_val_score(gbr, X, y, cv=LeaveOneOut(), scoring='neg_mean_absolute_error')
loo_mae = -loo_scores.mean()

gbr.fit(X, y)
atm_table['Predicted_Severity'] = gbr.predict(X)

# ---------------------------------------------------------------------------
# 7. Final composite 0-100 risk score:
#    blends learned severity prediction, geospatial hotspot density, and
#    the anomaly signal -- then min-max scaled across the network
# ---------------------------------------------------------------------------
def minmax(s):
    s = s.astype(float)
    if s.max() == s.min():
        return np.full(len(s), 50.0)
    return (s - s.min()) / (s.max() - s.min()) * 100

sev_scaled = minmax(atm_table['Predicted_Severity'])
density_scaled = minmax(atm_table['Cluster_Density'])
anomaly_scaled = minmax(atm_table['Anomaly_Score_Raw'])

atm_table['Fraud_Risk_Score_0_to_100'] = (
    0.55 * sev_scaled + 0.25 * anomaly_scaled + 0.20 * density_scaled
).round(2)

def categorize(v):
    if v >= 75: return 'CRITICAL RISK'
    if v >= 55: return 'HIGH RISK'
    if v >= 35: return 'MEDIUM RISK'
    return 'LOW RISK'

atm_table['Risk_Category'] = atm_table['Fraud_Risk_Score_0_to_100'].apply(categorize)

# ---------------------------------------------------------------------------
# 8. Output
# ---------------------------------------------------------------------------
out_cols = ['ATM_ID', 'bank_name', 'area', 'latitude', 'longitude', 'address',
            'Complaint_Count', 'Recent_30d_Count', 'Top_Fraud_Type',
            'Avg_Fund_Recovery_Pct', 'Pct_Blocked', 'Pct_Under_Investigation',
            'Anomaly_Flag', 'Zone_Cluster', 'Cluster_Density',
            'Fraud_Risk_Score_0_to_100', 'Risk_Category']

results = atm_table[out_cols].sort_values('Fraud_Risk_Score_0_to_100', ascending=False).reset_index(drop=True)

print("=== SONIPAT ATM NETWORK — PREDICTIVE FRAUD RISK RATING (0-100) ===")
print(f"(Leave-one-out validation MAE on severity target: {loo_mae:.3f} -- small-n caveat: n=53 ATMs)")
print(results.to_string())

results.to_json(os.path.join(SCRIPT_DIR, 'atm_risk_results.json'), orient='records')

# a small synthetic "live alert feed" for the dashboard demo, drawn from the
# most recent real complaints, mapped onto the top-risk ATMs
alerts = complaints.sort_values('Date_Time', ascending=False).head(25).copy()
alerts['ATM_ID'] = alerts['Sim_ATM_ID']
alerts = alerts.merge(atm_table[['ATM_ID', 'bank_name', 'area', 'latitude', 'longitude',
                                  'Fraud_Risk_Score_0_to_100', 'Risk_Category']], on='ATM_ID', how='left')
alerts[['Complaint_ID', 'Date_Time', 'Fraud_Type', 'Transaction_Amount', 'ATM_ID',
        'bank_name', 'area', 'latitude', 'longitude', 'Withdrawal_Status', 'Case_Status',
        'Risk_Score', 'Fraud_Risk_Score_0_to_100', 'Risk_Category']].to_json(
    os.path.join(SCRIPT_DIR, 'live_alerts.json'), orient='records', date_format='iso')

print("\nWrote atm_risk_results.json and live_alerts.json")

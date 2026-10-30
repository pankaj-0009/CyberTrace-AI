import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN

# 1. Load Dataset & Preprocessing
sonipat_atms = pd.read_csv('Accurate_SNP_ATM_list.csv')
complaints = pd.read_csv('complaints_log.csv')

sonipat_atms['ATM_ID'] = [f"SNP-ATM-{i+1001:04d}" for i in range(len(sonipat_atms))]
sonipat_atms['bank_name'] = sonipat_atms['bank_name'].replace({'SBI': 'State Bank of India'})

np.random.seed(42)
mapped_indices = np.random.choice(len(sonipat_atms), size=len(complaints))

complaints['ATM_ID'] = sonipat_atms.iloc[mapped_indices]['ATM_ID'].values
complaints['Bank_Operator'] = sonipat_atms.iloc[mapped_indices]['bank_name'].values
complaints['Latitude'] = sonipat_atms.iloc[mapped_indices]['latitude'].values
complaints['Longitude'] = sonipat_atms.iloc[mapped_indices]['longitude'].values
complaints['Area'] = sonipat_atms.iloc[mapped_indices]['area'].values

# Feature Engineering
complaints['Date_Time'] = pd.to_datetime(complaints['Date_Time'])
complaints['Hour'] = complaints['Date_Time'].dt.hour
complaints['Day_of_Week'] = complaints['Date_Time'].dt.dayofweek
complaints['Month'] = complaints['Date_Time'].dt.month
complaints['Is_Weekend'] = complaints['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
complaints['Hour_Sin'] = np.sin(2 * np.pi * complaints['Hour'] / 24.0)
complaints['Hour_Cos'] = np.cos(2 * np.pi * complaints['Hour'] / 24.0)
complaints['Log_Amount'] = np.log1p(complaints['Transaction_Amount'])

coords_rad = np.radians(complaints[['Latitude', 'Longitude']].values)
kms_per_radian = 6371.0088
complaints['Sonipat_Zone_Cluster_5km'] = DBSCAN(eps=5.0 / kms_per_radian, min_samples=3, metric='haversine').fit_predict(coords_rad)
complaints['Sonipat_Hotspot_Cluster_2km'] = DBSCAN(eps=2.0 / kms_per_radian, min_samples=2, metric='haversine').fit_predict(coords_rad)

feature_cols = [
    'Hour', 'Hour_Sin', 'Hour_Cos', 'Day_of_Week', 'Month', 'Is_Weekend', 
    'Log_Amount', 'Risk_Score', 'Fund_Recovery_Pct',
    'Fraud_Type', 'Bank_Operator', 'Complaint_Source', 'Case_Status', 'Withdrawal_Status', 'Area',
    'Sonipat_Zone_Cluster_5km', 'Sonipat_Hotspot_Cluster_2km'
]

X = pd.get_dummies(complaints[feature_cols], columns=[
    'Fraud_Type', 'Bank_Operator', 'Complaint_Source', 
    'Case_Status', 'Withdrawal_Status', 'Area',
    'Sonipat_Zone_Cluster_5km', 'Sonipat_Hotspot_Cluster_2km'
])
y = complaints['ATM_ID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

model = GradientBoostingClassifier(n_estimators=180, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 2. Continuous 0-100 Risk Rating Function
def calculate_all_atm_risk_scores(incoming_complaint_features, model, sonipat_atms_df):
    raw_probs = model.predict_proba(incoming_complaint_features)[0]
    
    # Smooth log-transformation and Min-Max scaling to ensure relative risk across 0 to 100
    log_probs = np.log(raw_probs + 1e-12)
    min_lp, max_lp = log_probs.min(), log_probs.max()
    scores_0_100 = ((log_probs - min_lp) / (max_lp - min_lp)) * 100
    
    scored_atms = []
    for idx, atm_id in enumerate(model.classes_):
        score_val = round(scores_0_100[idx], 2)
        atm_info = sonipat_atms_df[sonipat_atms_df['ATM_ID'] == atm_id].iloc[0]
        
        if score_val >= 80:
            risk_category = "CRITICAL RISK"
        elif score_val >= 60:
            risk_category = "HIGH RISK"
        elif score_val >= 40:
            risk_category = "MEDIUM RISK"
        else:
            risk_category = "LOW RISK"
            
        scored_atms.append({
            'ATM_ID': atm_id,
            'Fraud_Risk_Score_0_to_100': score_val,
            'Risk_Category': risk_category,
            'Bank_Operator': atm_info['bank_name'],
            'Area': atm_info['area'],
            'Latitude': atm_info['latitude'],
            'Longitude': atm_info['longitude']
        })
        
    df_results = pd.DataFrame(scored_atms).sort_values(by='Fraud_Risk_Score_0_to_100', ascending=False).reset_index(drop=True)
    return df_results

# Execute for a complaint
sample_complaint = X_test.iloc[[0]]
full_risk_table = calculate_all_atm_risk_scores(sample_complaint, model, sonipat_atms)

# PRINT ALL 93 SCORES IN OUTPUT TERMINAL
print("=== COMPLETE ATM NETWORK FRAUD RISK RATING (0 TO 100) ===")
print(full_risk_table.to_string())
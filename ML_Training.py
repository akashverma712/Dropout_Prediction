import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load merged data
data = pd.read_csv("historical_data_large.csv")

# Feature Engineering
data['score_drop'] = data['last_score'] - data['current_score']
data['fee_pending'] = data['fee_status'].apply(lambda x: 1 if str(x).lower()=='pending' else 0)
data['num_failed'] = data['num_failed'].fillna(0).astype(int)
data['attendance'] = data['attendance'].fillna(0)

# Features and target
X = data[['attendance', 'score_drop', 'num_failed', 'fee_pending']]
y = data['dropout']

# Train model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X, y)

# Save model
with open("ml_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)


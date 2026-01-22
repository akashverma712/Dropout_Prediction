# file 1: flask_1.py
from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
import os
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DB_FILE = "student_db.db"

# ---------- Initialize Database ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            name TEXT,
            attendance REAL,
            last_score REAL,
            current_score REAL,
            num_failed INTEGER,
            fee_status TEXT,
            fee_pending INTEGER,
            score_drop REAL,
            performance_decline INTEGER,
            attendance_low INTEGER,
            high_attempts INTEGER,
            risk_probability REAL,
            risk_level TEXT,
            risk_color TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------- Load Pre-trained ML Model ----------
ML_MODEL_FILE = "ml_model.pkl"
if os.path.exists(ML_MODEL_FILE):
    with open(ML_MODEL_FILE, "rb") as f:
        ml_model = pickle.load(f)
else:
    ml_model = None
    print("⚠️ ML model not found. Using placeholder thresholds instead.")

# ---------- Process Uploaded Files ----------
def process_uploaded_files(attendance_file, scores_file, fees_file):
    # Read CSV files
    attendance_df = pd.read_csv(attendance_file)
    scores_df = pd.read_csv(scores_file)
    fees_df = pd.read_csv(fees_file)

    # Merge on student_id
    df = attendance_df.merge(scores_df, on="student_id", how="outer")
    df = df.merge(fees_df, on="student_id", how="outer")

    # Cleaning & Features (aligned with ML training for SIH transparency)
    df['attendance'] = df['attendance'].fillna(0)
    df['last_score'] = df['last_score'].fillna(0)
    df['current_score'] = df['current_score'].fillna(0)
    df['num_failed'] = df['num_failed'].fillna(0).astype(int)
    df['fee_status'] = df['fee_status'].fillna('Unknown')
    df['fee_pending'] = df['fee_status'].apply(lambda x: 1 if str(x).lower()=='pending' else 0)
    df['score_drop'] = df['last_score'] - df['current_score']
    df['performance_decline'] = df['score_drop'].apply(lambda x: 1 if x>0 else 0)
    df['attendance_low'] = df['attendance'].apply(lambda x: 1 if x < 70 else 0)  # Configurable threshold
    df['high_attempts'] = df['num_failed'].apply(lambda x: 1 if x > 2 else 0)  # Aligns with SIH "high number of attempts"

    # ---------- ML Risk Evaluation ----------
    risk_probs, risk_levels, risk_colors = [], [], []

    for _, row in df.iterrows():
        features = [[row['attendance'], row['score_drop'], row['num_failed'], row['fee_pending'],
                     row['attendance_low'], row['high_attempts']]]

        # Predict probability
        if ml_model:
            prob = ml_model.predict_proba(features)[0][1]  # probability of dropout
        else:
            # Placeholder logic (clear, rule-based for SIH transparency)
            risk_score = 0
            if row['attendance_low'] == 1:
                risk_score += 0.4
            if row['high_attempts'] == 1:
                risk_score += 0.3
            if row['fee_pending'] == 1:
                risk_score += 0.2
            if row['score_drop'] > 10:
                risk_score += 0.1
            prob = min(risk_score, 1.0)  # Normalized to prob

        # Map probability to 3-level risk (aligned with SIH color-coding)
        if prob >= 0.7:
            risk_level = "High"
            risk_color = "Red"
        elif prob >= 0.4:
            risk_level = "Medium"
            risk_color = "Yellow"
        else:
            risk_level = "Low"
            risk_color = "Green"

        risk_probs.append(prob)
        risk_levels.append(risk_level)
        risk_colors.append(risk_color)

    df['risk_probability'] = risk_probs
    df['risk_level'] = risk_levels
    df['risk_color'] = risk_colors

    return df

# ---------- Update Database ----------
def update_database(df):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Clear old data for fresh upload (SIH monthly reset)
    c.execute('DELETE FROM students')
    conn.commit()
    
    # Insert new data
    for _, row in df.iterrows():
        c.execute('''
            INSERT INTO students (student_id, name, attendance, last_score, current_score, num_failed, fee_status, fee_pending, score_drop, performance_decline, attendance_low, high_attempts, risk_probability, risk_level, risk_color)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(row['student_id']), row.get('name', ''), row['attendance'], row['last_score'], row['current_score'], row['num_failed'],
            row['fee_status'], row['fee_pending'], row['score_drop'], row['performance_decline'],
            row['attendance_low'], row['high_attempts'],
            row['risk_probability'], row['risk_level'], row['risk_color']
        ))
    conn.commit()
    conn.close()
# ---------- API Endpoint ----------
@app.route('/monthly_upload', methods=['POST'])
def monthly_upload():
    try:
        attendance_file = request.files['attendance']
        scores_file = request.files['scores']
        fees_file = request.files['fees']

        # Save uploaded files
        attendance_path = os.path.join(UPLOAD_FOLDER, attendance_file.filename)
        scores_path = os.path.join(UPLOAD_FOLDER, scores_file.filename)
        fees_path = os.path.join(UPLOAD_FOLDER, fees_file.filename)
        attendance_file.save(attendance_path)
        scores_file.save(scores_path)
        fees_file.save(fees_path)

        # Process files & evaluate risk
        df = process_uploaded_files(attendance_path, scores_path, fees_path)

        # Update database
        update_database(df)

        return jsonify({"message": "Monthly data uploaded, ML risk evaluated (Low/Medium/High), database updated!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run Flask ----------
if __name__ == '__main__':
    app.run(debug=True)
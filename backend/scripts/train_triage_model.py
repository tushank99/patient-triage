import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


np.random.seed(42)
N_SAMPLES = 5000

def generate_synthetic_clinical_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic clinical dataset reflecting ESI Level 1-5 triage cases:
    - ESI 1: Immediate resuscitation (Severe shock, arrest, SpO2 < 80, SBP < 70)
    - ESI 2: Emergent (Chest pain, severe hypoxia <= 92, tachycardia > 130, pediatric fever)
    - ESI 3: Urgent (Needs 2+ resources, moderate vitals, severe pain)
    - ESI 4: Semi-urgent (1 resource: simple laceration, mild pain, stable vitals)
    - ESI 5: Non-urgent (0 resources: prescription refill, suture removal, minor rash)
    """
    age = np.random.uniform(0.1, 90.0, n_samples)
    hr = np.random.normal(82, 22, n_samples).clip(30, 220)
    sbp = np.random.normal(122, 24, n_samples).clip(50, 240)
    dbp = np.random.normal(78, 14, n_samples).clip(30, 140)
    spo2 = np.random.normal(97, 4, n_samples).clip(65, 100)
    temp = np.random.normal(37.0, 0.9, n_samples).clip(34.0, 41.5)
    pain = np.random.choice(range(0, 11), size=n_samples, p=[0.2] + [0.08]*10)
    has_history = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    
    esi = []
    for a, h, s, d, sp, t, p, hx in zip(age, hr, sbp, dbp, spo2, temp, pain, has_history):
        # ESI 1: Resuscitation / Severe instability
        if sp < 82 or s <= 65 or h < 35:
            esi.append(1)
        # ESI 2: Emergent / High Risk
        elif sp <= 92 or s < 88 or h > 135 or (a <= 0.25 and t >= 38.0) or (a >= 65 and s < 95 and h > 110):
            esi.append(2)
        # ESI 3: Urgent (Multi-resource / Moderate distress)
        elif h > 105 or s > 165 or t >= 38.5 or (p >= 7 and (hx == 1 or a > 50)):
            esi.append(3)
        # ESI 4: Semi-Urgent (Single resource)
        elif p >= 4 or t > 37.8 or hx == 1:
            esi.append(4)
        # ESI 5: Non-Urgent (Zero resource)
        else:
            esi.append(5)
            
    df = pd.DataFrame({
        "age": age,
        "hr": hr,
        "bp_sys": sbp,
        "bp_dia": dbp,
        "spo2": spo2,
        "temp": temp,
        "pain": pain,
        "has_history": has_history,
        "esi": [e - 1 for e in esi] # XGBoost classes must be 0-indexed (0 to 4)
    })
    return df

def train_and_save():
    print("Generating synthetic clinical training dataset...")
    df = generate_synthetic_clinical_dataset(N_SAMPLES)
    
    X = df.drop(columns=["esi"])
    y = df["esi"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Multi-Class XGBoost Triage Model...")
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=5,
        random_state=42
    )
    
   
    model.fit(X_train.values, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test.values)
    print("\nModel Evaluation Report (ESI 1-5):")
    print(classification_report(y_test, y_pred, target_names=["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]))
    
    # Save model artifact
    output_dir = os.path.join(os.path.dirname(__file__), "..", "app", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_triage_model.joblib")
    
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
import sys
import os
import sys
from pathlib import Path

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import Base, engine, SessionLocal
from backend.app.models import Patient, Visit, AuditLog

def seed():
    print("Creating tables in database...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    # Clear existing data to allow clean re-runs
    db.query(AuditLog).delete()
    db.query(Visit).delete()
    db.query(Patient).delete()
    db.commit()

    print("Seeding deterministic patient scenarios...")

    scenarios = [
        {
            "mrn": "MRN-PED-001",
            "age": 0.25, # 3 months old
            "gender": "Female",
            "history": False,
            "summary": None,
            "complaint": "Mother reports lethargy, irritability, and warm to touch for past 6 hours.",
            "vitals": {"hr": 165, "sbp": 80, "dbp": 50, "rr": 48, "spo2": 97.0, "temp": 38.8, "pain": 4},
            "expected_esi": 2, # Pediatric fever under 3 months is high acuity
            "notes": "Pediatric Trap Scenario"
        },
        {
            "mrn": "MRN-SIL-002",
            "age": 54.0,
            "gender": "Male",
            "history": True,
            "summary": "History of mild asthma.",
            "complaint": "Feels slightly winded after walking up stairs. Denies chest pain.",
            "vitals": {"hr": 78, "sbp": 122, "dbp": 78, "rr": 26, "spo2": 89.0, "temp": 36.8, "pain": 0},
            "expected_esi": 2, # Silent hypoxia
            "notes": "Silent Critical Scenario"
        },
        {
            "mrn": "MRN-CNF-003",
            "age": 61.0,
            "gender": "Male",
            "history": True,
            "summary": "Hypertension, poorly controlled.",
            "complaint": "Sudden onset sharp tearing sensation between shoulder blades radiating to lower back.",
            "vitals": {"hr": 82, "sbp": 138, "dbp": 86, "rr": 18, "spo2": 98.0, "temp": 37.0, "pain": 8},
            "expected_esi": 1, # Suspected acute aortic syndrome / dissection
            "notes": "Conflicting Signals (Normal Vitals vs High-Risk Text)"
        },
        {
            "mrn": "MRN-ZER-004",
            "age": 29.0,
            "gender": "Female",
            "history": False,
            "summary": None,
            "complaint": "Unspecified right lower quadrant abdominal pain since early morning, nausea without vomiting.",
            "vitals": {"hr": 88, "sbp": 115, "dbp": 72, "rr": 16, "spo2": 99.0, "temp": 37.6, "pain": 6},
            "expected_esi": 3, # Needs imaging / workup (Rule out appendicitis)
            "notes": "Zero-History First-Time Walk-in"
        },
        {
            "mrn": "MRN-GER-005",
            "age": 82.0,
            "gender": "Female",
            "history": True,
            "summary": "Atrial fibrillation on anticoagulants, Osteoarthritis.",
            "complaint": "Mechanical fall at home after feeling slight dizziness. Mild swelling on right forehead.",
            "vitals": {"hr": 64, "sbp": 102, "dbp": 60, "rr": 14, "spo2": 96.0, "temp": 36.4, "pain": 2},
            "expected_esi": 2, # Geriatric fall on blood thinners is immediate CT head / high risk
            "notes": "Geriatric Anticoagulation Fall Risk"
        }
    ]

    # Add 10 additional standard ESI 3, 4, 5 routine cases to fill the 15-patient minimum
    standard_complaints = [
        ("MRN-STD-006", 35.0, "Male", "Minor wrist sprain after basketball game.", {"hr": 72, "sbp": 120, "dbp": 80, "rr": 14, "spo2": 100.0, "temp": 36.6, "pain": 3}, 4),
        ("MRN-STD-007", 22.0, "Female", "Superficial laceration on left forearm from broken glass, bleeding controlled.", {"hr": 76, "sbp": 118, "dbp": 76, "rr": 16, "spo2": 99.0, "temp": 36.7, "pain": 4}, 4),
        ("MRN-STD-008", 45.0, "Male", "Productive cough and mild congestion for 4 days, no shortness of breath.", {"hr": 70, "sbp": 124, "dbp": 82, "rr": 16, "spo2": 98.0, "temp": 37.4, "pain": 1}, 5),
        ("MRN-STD-009", 50.0, "Female", "Dysuria, increased urinary frequency for 2 days without flank pain.", {"hr": 80, "sbp": 130, "dbp": 84, "rr": 16, "spo2": 98.0, "temp": 37.1, "pain": 4}, 4),
        ("MRN-STD-010", 67.0, "Male", "Gradual onset bilateral lower extremity swelling over 2 weeks, stable.", {"hr": 84, "sbp": 142, "dbp": 88, "rr": 18, "spo2": 95.0, "temp": 36.8, "pain": 0}, 3),
        ("MRN-STD-011", 19.0, "Male", "Sore throat and difficulty swallowing solids, tonsils mildly erythematous.", {"hr": 78, "sbp": 116, "dbp": 74, "rr": 14, "spo2": 99.0, "temp": 38.0, "pain": 5}, 4),
        ("MRN-STD-012", 41.0, "Female", "Recurrent migraine headache similar to prior episodes, photophobia present.", {"hr": 74, "sbp": 126, "dbp": 80, "rr": 16, "spo2": 99.0, "temp": 36.9, "pain": 7}, 3),
        ("MRN-STD-013", 28.0, "Male", "Medication refill request, asymptomatic.", {"hr": 68, "sbp": 118, "dbp": 76, "rr": 12, "spo2": 100.0, "temp": 36.5, "pain": 0}, 5),
        ("MRN-STD-014", 73.0, "Male", "Chronic low back pain unchanged from baseline.", {"hr": 72, "sbp": 134, "dbp": 82, "rr": 14, "spo2": 97.0, "temp": 36.7, "pain": 5}, 4),
        ("MRN-STD-015", 31.0, "Female", "Small first-degree burn to left palm while cooking.", {"hr": 82, "sbp": 122, "dbp": 78, "rr": 16, "spo2": 99.0, "temp": 36.6, "pain": 4}, 4),
    ]

    for item in standard_complaints:
        scenarios.append({
            "mrn": item[0],
            "age": item[1],
            "gender": item[2],
            "history": False,
            "summary": None,
            "complaint": item[3],
            "vitals": item[4],
            "expected_esi": item[5],
            "notes": "Standard Routine Presentation"
        })

    for sc in scenarios:
        p = Patient(
            mrn=sc["mrn"],
            age_years=sc["age"],
            gender=sc["gender"],
            has_prior_history=sc["history"],
            prior_history_summary=sc["summary"]
        )
        db.add(p)
        db.flush()

        v = Visit(
            patient_id=p.id,
            chief_complaint=sc["complaint"],
            heart_rate=sc["vitals"]["hr"],
            systolic_bp=sc["vitals"]["sbp"],
            diastolic_bp=sc["vitals"]["dbp"],
            respiratory_rate=sc["vitals"]["rr"],
            oxygen_saturation=sc["vitals"]["spo2"],
            temperature_celsius=sc["vitals"]["temp"],
            pain_score=sc["vitals"]["pain"],
            final_esi_score=sc["expected_esi"],
            status="WAITING"
        )
        db.add(v)

    db.commit()
    db.close()
    print(f"Successfully seeded {len(scenarios)} realistic patient records into the database.")

if __name__ == "__main__":
    seed()

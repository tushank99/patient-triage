"""
Deterministic synthetic retrieval benchmark (labels by construction).

Each CONCEPT has chart wording (what documents say) and query wording (what a nurse would type: lay terms,
abbreviations, paraphrases). Documents are built only from chart wording plus FILLER sentences that share
vocabulary with queries (distractors). A chunk is relevant to a query iff it contains that concept's chart
sentence. Negative queries ask about concepts the patient does NOT have; they calibrate the low-confidence gate.

Limitation: templated text is cleaner than real notes, so absolute scores overstate real-world performance.
Use it to compare configurations, not to claim accuracy.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List

BENCHMARK_VERSION = "1.0"

CONCEPTS: List[Dict] = [
    {"id": "anticoag", "section": "Medications", "chart": ["Apixaban 5 mg twice daily for stroke prevention in atrial fibrillation.", "Warfarin 4 mg nightly with an INR target of 2 to 3 for atrial fibrillation."], "queries": ["on a blood thinner", "takes blood thinners for afib", "what does she take for the irregular heartbeat"]},
    {"id": "prior_mi", "section": "Past Medical History", "chart": ["NSTEMI in 2021 treated with a drug-eluting stent to the left anterior descending artery."], "queries": ["previous heart attack", "hx of MI", "has he had a stent"]},
    {"id": "copd", "section": "Past Medical History", "chart": ["Chronic obstructive pulmonary disease, GOLD stage 3, on home tiotropium."], "queries": ["COPD", "lung disease from smoking", "chronic breathing problems"]},
    {"id": "asthma_icu", "section": "Hospital Course", "chart": ["Near-fatal asthma exacerbation requiring intubation in the intensive care unit."], "queries": ["ever intubated for wheezing", "severe asthma attack in the past", "was the asthma bad enough for ICU"]},
    {"id": "pcn_allergy", "section": "Allergies", "chart": ["Penicillin: anaphylaxis with hypotension and throat swelling."], "queries": ["allergic to antibiotics", "penicillin allergy", "bad reaction to amoxicillin"]},
    {"id": "sulfa_allergy", "section": "Allergies", "chart": ["Sulfonamide antibiotics cause a diffuse maculopapular eruption."], "queries": ["sulfa allergy", "rash from an antibiotic"]},
    {"id": "insulin", "section": "Medications", "chart": ["Insulin glargine 24 units at bedtime and insulin lispro with meals."], "queries": ["sugar medication", "is she on insulin", "diabetes treatment"]},
    {"id": "dka", "section": "Hospital Course", "chart": ["Admitted to intensive care for diabetic ketoacidosis after an insulin pump failure."], "queries": ["DKA history", "high sugar emergency before"]},
    {"id": "ckd", "section": "Past Medical History", "chart": ["Chronic kidney disease stage 3b with an eGFR of 38."], "queries": ["kidney function", "CKD", "kidney disease"]},
    {"id": "hfref", "section": "Past Medical History", "chart": ["Heart failure with reduced ejection fraction of 30 percent."], "queries": ["weak heart", "CHF history", "how well does the heart pump"]},
    {"id": "diuretic", "section": "Medications", "chart": ["Furosemide 40 mg each morning."], "queries": ["water pill", "diuretic dose"]},
    {"id": "epilepsy", "section": "Past Medical History", "chart": ["Focal epilepsy controlled on levetiracetam 1000 mg twice daily."], "queries": ["fits", "seizure medicine", "epilepsy"]},
    {"id": "dvt", "section": "Hospital Course", "chart": ["Left femoral deep vein thrombosis treated with six months of anticoagulation."], "queries": ["blood clot in the leg before", "DVT history"]},
    {"id": "pregnancy", "section": "Obstetric History", "chart": ["Delivered a healthy infant by caesarean section in February 2021."], "queries": ["was she pregnant before", "previous delivery"]},
    {"id": "splenectomy", "section": "Surgical History", "chart": ["Splenectomy after a motorcycle collision in 2020, post-splenectomy vaccines given."], "queries": ["no spleen", "spleen removed", "asplenia"]},
    {"id": "valve", "section": "Surgical History", "chart": ["Mechanical aortic valve replacement in 2016."], "queries": ["heart valve surgery", "AVR"]},
    {"id": "tia", "section": "Past Medical History", "chart": ["Transient ischemic attack with right arm weakness that fully resolved."], "queries": ["mini stroke", "TIA before", "previous stroke symptoms"]},
    {"id": "htn", "section": "Past Medical History", "chart": ["Long-standing hypertension, poorly controlled despite two agents."], "queries": ["high blood pressure", "HTN"]},
    {"id": "stones", "section": "Past Medical History", "chart": ["Recurrent calcium oxalate nephrolithiasis, last ureteroscopy in 2024."], "queries": ["kidney stones", "stone history"]},
    {"id": "migraine", "section": "Past Medical History", "chart": ["Migraine with visual aura, three to four episodes per month."], "queries": ["headaches", "migraine history"]},
    {"id": "thyroid", "section": "Medications", "chart": ["Levothyroxine 75 micrograms daily for hypothyroidism."], "queries": ["thyroid medication", "thyroid problem"]},
    {"id": "smoker", "section": "Social History", "chart": ["Former smoker with a 35 pack-year history, quit in 2015."], "queries": ["does he smoke", "tobacco use"]},
    {"id": "alcohol", "section": "Social History", "chart": ["Drinks six beers daily with prior withdrawal seizures."], "queries": ["alcohol use", "how much does he drink"]},
    {"id": "codeine", "section": "Allergies", "chart": ["Codeine causes confusion and hallucinations."], "queries": ["reaction to painkillers", "codeine allergy"]},
    {"id": "latex", "section": "Allergies", "chart": ["Latex: contact urticaria."], "queries": ["latex allergy", "allergic to gloves"]},
    {"id": "falls", "section": "Assessment", "chart": ["Two mechanical falls in the past year; mobilises with a walker."], "queries": ["has she fallen before", "falls risk"]},
    {"id": "dementia", "section": "Past Medical History", "chart": ["Moderate Alzheimer dementia, lives in assisted living."], "queries": ["memory problems", "dementia"]},
    {"id": "inr", "section": "Labs", "chart": ["INR 3.4 on the most recent check, above target."], "queries": ["latest INR", "clotting level"]},
    {"id": "a1c", "section": "Labs", "chart": ["Hemoglobin A1c 9.1 percent."], "queries": ["sugar control", "A1c result"]},
    {"id": "potassium", "section": "Labs", "chart": ["Potassium 5.8 mmol/L, repeated at 5.6 on a non-hemolysed sample."], "queries": ["high potassium", "potassium level"]},
    {"id": "aorta", "section": "Imaging", "chart": ["Ascending aorta dilated to 4.6 centimetres on echocardiogram."], "queries": ["enlarged aorta", "aortic aneurysm"]},
    {"id": "gi_bleed", "section": "Hospital Course", "chart": ["Upper gastrointestinal bleed from a duodenal ulcer requiring transfusion."], "queries": ["stomach bleeding history", "GI bleed", "ulcer bleed"]},
]

FILLER = [
    "Vital signs were stable throughout the admission.", "Medications were reconciled with the patient at discharge.",
    "Follow up with primary care in two weeks.", "Patient was counselled on diet and exercise.", "No acute distress on examination.",
    "Laboratory results were otherwise unremarkable.", "Plan discussed with the patient and family, who agree.",
    "Ambulating independently at discharge.", "Pain controlled with acetaminophen.", "Chest X-ray showed no acute process.",
    "Allergies were reviewed and are unchanged.", "Social work assisted with discharge planning.",
    "Blood pressure was checked twice and recorded.", "Heart rhythm regular on auscultation.", "Breath sounds clear bilaterally.",
    "Kidney function and electrolytes to be repeated next week.", "Seen by the pharmacist for a medication review.",
    "Return precautions given for worsening symptoms.", "Sleeping well and eating normally.", "Lives with spouse, independent with activities.",
    "Skin intact without rashes.", "Abdomen soft and non-tender.", "Neurological examination non-focal.",
    "Influenza vaccine given this season.", "Weight stable compared with the last visit.", "Advised to keep a symptom diary.",
]

DOCS = {
    "discharge_summary": ["History of Present Illness", "Past Medical History", "Hospital Course", "Medications", "Allergies", "Plan"],
    "clinic_note": ["Past Medical History", "Surgical History", "Obstetric History", "Social History", "Medications", "Assessment", "Plan"],
    "er_note": ["History of Present Illness", "Allergies", "Assessment", "Plan"],
    "lab_summary": ["Labs", "Imaging", "Impression"],
}


@dataclass
class BenchDoc:
    id: str
    doc_type: str
    title: str
    text: str


@dataclass
class BenchPatient:
    mrn: str
    age: float
    sex: str
    concepts: List[str]
    docs: List[BenchDoc] = field(default_factory=list)
    chart_sentences: Dict[str, str] = field(default_factory=dict)  # concept -> exact sentence placed in the chart


@dataclass
class BenchQuery:
    id: str
    mrn: str
    text: str
    concept: str
    positive: bool
    relevant_text: str = ""  # chart sentence; a chunk containing it is relevant


def generate(n_patients: int = 12, concepts_per_patient: int = 7, seed: int = 7):
    rng = random.Random(seed)
    by_id = {c["id"]: c for c in CONCEPTS}
    patients: List[BenchPatient] = []
    queries: List[BenchQuery] = []
    for p in range(n_patients):
        mrn = f"BENCH-{p:03d}"
        chosen = rng.sample([c["id"] for c in CONCEPTS], concepts_per_patient)
        pat = BenchPatient(mrn, round(rng.uniform(20, 90), 1), rng.choice("MF"), chosen)
        # place each concept in a document that has its section
        placement: Dict[tuple, List[str]] = {}
        for cid in chosen:
            c = by_id[cid]
            options = [dt for dt, secs in DOCS.items() if c["section"] in secs]
            dt = rng.choice(options)
            sentence = rng.choice(c["chart"])
            pat.chart_sentences[cid] = sentence
            placement.setdefault((dt, c["section"]), []).append(sentence)
        for dt, sections in DOCS.items():
            parts = [dt.replace("_", " ").title()]
            for sec in sections:
                body = rng.sample(FILLER, rng.randint(2, 4)) + placement.get((dt, sec), [])
                rng.shuffle(body)
                parts.append(f"{sec}:\n" + " ".join(body))
            pat.docs.append(BenchDoc(f"{mrn}:{dt}", dt, dt.replace("_", " ").title(), "\n\n".join(parts) + "\n"))
        patients.append(pat)
        for i, cid in enumerate(chosen):
            qs = by_id[cid]["queries"]
            text = qs[(p + i) % len(qs)]
            if (p + i) % 3 == 0:  # some queries arrive embedded in a complaint, like triage text does
                text = f"{rng.choice(['Fell at home today.', 'Short of breath since morning.', 'Feels unwell.'])} {text}"
            queries.append(BenchQuery(f"{mrn}-q{i}", mrn, text, cid, True, pat.chart_sentences[cid]))
        absent = [c["id"] for c in CONCEPTS if c["id"] not in chosen]
        for j, cid in enumerate(rng.sample(absent, 3)):
            queries.append(BenchQuery(f"{mrn}-n{j}", mrn, by_id[cid]["queries"][0], cid, False))
    return patients, queries

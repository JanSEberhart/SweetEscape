import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# App Setup
# --------------------------------------------------
st.set_page_config(page_title="SweetEscape 🍬", layout="centered")

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_final_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_used.txt"

st.title("SweetEscape 🍬")
st.subheader("Diabetes-Risiko – Ersteinschätzung")
st.write(
    "Diese Anwendung liefert eine **erste Einschätzung** basierend auf einem Machine-Learning-Modell.\n\n"
    "**Hinweis:** Kein Ersatz für ärztliche Diagnostik."
)

# --------------------------------------------------
# Load model + expected features
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

if not FEATURES_PATH.exists():
    st.error(
        "features_used.txt fehlt. Bitte in Notebook 00 die Feature-Liste speichern "
        "oder aus `diabetes_fe.csv` erzeugen."
    )
    st.stop()

expected_features = [
    line.strip()
    for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def yes_no(label, default="Nein", help_text=None):
    """Ja/Nein Auswahl mit 'Ja' oben."""
    options = ["Ja", "Nein"]
    index = 0 if default == "Ja" else 1
    return st.selectbox(label, options, index=index, help=help_text) == "Ja"

def bmi_from(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)

def build_input_row(values: dict) -> pd.DataFrame:
    """
    Erstellt genau die Feature-Reihenfolge, die das Modell erwartet.
    Nicht abgefragte Features werden per Default auf 0 gesetzt.
    """
    row = {f: 0 for f in expected_features}
    row.update(values)
    return pd.DataFrame([row], columns=expected_features)

# --------------------------------------------------
# UI Inputs (Deutschland-tauglich)
# --------------------------------------------------
st.header("Körperliche Angaben")

height_cm = st.number_input("Größe (cm)", min_value=120, max_value=220, value=175)
weight_kg = st.number_input("Gewicht (kg)", min_value=40, max_value=220, value=75)

bmi = bmi_from(height_cm, weight_kg)
st.write(f"**BMI (berechnet):** {bmi:.1f}")

st.header("Soziodemografie")

sex = st.selectbox("Geschlecht", ["männlich", "weiblich"], index=0)
sex_val = 1 if sex == "männlich" else 0  # BRFSS-ähnliche Kodierung

age_choice = st.selectbox(
    "Altersgruppe",
    [
        "18–24", "25–29", "30–34", "35–39", "40–44",
        "45–49", "50–54", "55–59", "60–64", "65–69",
        "70–74", "75–79", "80+"
    ],
    index=6
)
age_val = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5,
    "45–49": 6, "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10,
    "70–74": 11, "75–79": 12, "80+": 13
}[age_choice]

EDU_MAP_DE = {
    1: "Keine Schule / Vorschule",
    2: "Grundschule",
    3: "Sekundarstufe I",
    4: "Sekundarstufe II (Abitur o. ä.)",
    5: "Hochschule (mind. 3 Jahre, kein Abschluss)",
    6: "Hochschulabschluss",
}
edu_label = st.selectbox("Bildung", list(EDU_MAP_DE.values()), index=3)
education_val = {v: k for k, v in EDU_MAP_DE.items()}[edu_label]

INCOME_MAP_DE = {
    1: "unter 10.000 €",
    2: "10.000–<15.000 €",
    3: "15.000–<20.000 €",
    4: "20.000–<25.000 €",
    5: "25.000–<35.000 €",
    6: "35.000–<50.000 €",
    7: "50.000–<75.000 €",
    8: "75.000 € oder mehr",
}
income_label = st.selectbox("Einkommen (grob)", list(INCOME_MAP_DE.values()), index=4)
income_val = {v: k for k, v in INCOME_MAP_DE.items()}[income_label]

st.header("Gesundheit & Lebensstil")

high_bp = yes_no("Bluthochdruck bekannt?")
high_chol = yes_no("Erhöhte Cholesterinwerte bekannt?")

smoke_status = st.selectbox("Rauchen", ["Nie geraucht", "Früher geraucht", "Aktuell Raucher"], index=0)
smoker_val = 0 if smoke_status == "Nie geraucht" else 1

stroke = yes_no("Schon einmal einen Schlaganfall gehabt?")
heart_disease = yes_no("Herzerkrankung oder Herzinfarkt bekannt?")

phys_activity = yes_no("In den letzten 30 Tagen körperlich aktiv gewesen?")
fruits = yes_no("Mindestens 1 Portion Obst pro Tag?")
veggies = yes_no("Mindestens 1 Portion Gemüse pro Tag?")
heavy_alcohol = yes_no("Regelmäßig hoher Alkoholkonsum?")

gen_health = st.slider(
    "Allgemeiner Gesundheitszustand",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = sehr gut, 5 = sehr schlecht"
)

# Optionaler Block (mehr Nähe zum Trainingssetup)
with st.expander("Optional: Weitere Angaben (für genauere Einschätzung)"):
    chol_check = yes_no("Cholesterin in den letzten 5 Jahren gemessen?")
    diff_walk = yes_no("Gehen/Treppensteigen fällt schwer?")
    no_doc_cost = yes_no(
        "Arztbesuch trotz Bedarf vermieden (z. B. aus finanziellen Gründen)?",
        help_text="Allgemeine Zugangshürde – nicht an ein bestimmtes Gesundheitssystem gebunden."
    )

    # AnyHealthcare ist im Original US-spezifisch; hier neutral als „Zugang“ formuliert
    any_healthcare = yes_no(
        "Zugang zu medizinischer Versorgung (neutral formuliert)",
        help_text="Im Datensatz ist dies US-spezifisch; hier neutral interpretiert."
    )

    ment_hlth = st.slider(
        "Mentale Gesundheit: Anzahl schlechter Tage im letzten Monat",
        0, 30, 0,
        help="Wie viele Tage in den letzten 30 Tagen war die mentale Gesundheit nicht gut?"
    )
    phys_hlth = st.slider(
        "Körperliche Gesundheit: Anzahl schlechter Tage im letzten Monat",
        0, 30, 0,
        help="Wie viele Tage in den letzten 30 Tagen war die körperliche Gesundheit nicht gut?"
    )

# --------------------------------------------------
# Build model input row (must match training feature names)
# --------------------------------------------------
values = {
    # Kernfeatures
    "HighBP": int(high_bp),
    "HighChol": int(high_chol),
    "BMI": float(bmi),
    "Smoker": int(smoker_val),
    "Stroke": int(stroke),
    "HeartDiseaseorAttack": int(heart_disease),
    "PhysActivity": int(phys_activity),
    "Fruits": int(fruits),
    "Veggies": int(veggies),
    "HvyAlcoholConsump": int(heavy_alcohol),
    "GenHlth": int(gen_health),

    # Soziodemografie
    "Sex": int(sex_val),
    "Age": int(age_val),
    "Education": int(education_val),
    "Income": int(income_val),

    # Optional / Defaults
    "CholCheck": int(chol_check),
    "DiffWalk": int(diff_walk),
    "NoDocbcCost": int(no_doc_cost),
    "AnyHealthcare": int(any_healthcare),
    "MentHlth": int(ment_hlth),
    "PhysHlth": int(phys_hlth),
}

# FE-Features, falls im Training enthalten
if "inactive" in expected_features:
    values["inactive"] = 1 - int(phys_activity)
if "low_fruits" in expected_features:
    values["low_fruits"] = 1 - int(fruits)
if "low_veggies" in expected_features:
    values["low_veggies"] = 1 - int(veggies)
if "poor_health" in expected_features:
    values["poor_health"] = 1 if gen_health >= 4 else 0
if "mental_physical_burden" in expected_features:
    values["mental_physical_burden"] = int(ment_hlth) + int(phys_hlth)
if "cardio_risk_sum" in expected_features:
    values["cardio_risk_sum"] = int(high_bp) + int(high_chol) + int(heart_disease) + int(stroke)
if "lifestyle_risk_sum" in expected_features:
    values["lifestyle_risk_sum"] = int(smoker_val) + int(heavy_alcohol) + (1 - int(phys_activity)) + (1 - int(fruits)) + (1 - int(veggies))

input_data = build_input_row(values)

# --------------------------------------------------
# Predict
# --------------------------------------------------
st.divider()
if st.button("Risiko einschätzen"):
    probs = model.predict_proba(input_data)[0]
    pred = int(probs.argmax())

    label_map = {
        0: "🟢 Kein Diabetes",
        1: "🟡 Prädiabetes",
        2: "🔴 Diabetes"
    }

    st.subheader("Ergebnis")
    st.success(label_map.get(pred, f"Unbekannte Klasse: {pred}"))

    st.write("**Modell-Wahrscheinlichkeiten:**")
    st.write({
        "Kein Diabetes": f"{probs[0]*100:.1f} %",
        "Prädiabetes": f"{probs[1]*100:.1f} %",
        "Diabetes": f"{probs[2]*100:.1f} %",
    })

st.caption(
    "Hinweis: Diese Anwendung ersetzt keine ärztliche Diagnose und basiert auf statistischen Zusammenhängen aus Umfragedaten."
)

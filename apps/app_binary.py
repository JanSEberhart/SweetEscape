import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import re

# --------------------------------------------------
# Setup
# --------------------------------------------------
st.set_page_config(page_title="SweetEscape 🍬 (Binary)", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_binary_model.joblib"

st.title("SweetEscape 🍬 – Diabetes-Risiko (Binary)")
st.caption(
    "Ersteinschätzung: **0 = kein Diabetes**, **1 = Risiko (Prädiabetes oder Diabetes)**. "
    "Kein Ersatz für ärztliche Diagnostik."
)

if not MODEL_PATH.exists():
    st.error(f"Modell nicht gefunden: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# Feature-Liste direkt aus dem Modell -> verhindert Feature-Mismatch
if not hasattr(model, "feature_names_in_"):
    st.error("Modell hat keine feature_names_in_. Bitte wie im Notebook als sklearn-Pipeline trainieren.")
    st.stop()

expected_features = list(model.feature_names_in_)


# --------------------------------------------------
# Helpers (Frage links, Antwort rechts) + unique keys
# --------------------------------------------------
def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9äöüß]+", "_", s)
    return s.strip("_")

def yn_row(question: str, key: str, default: str = "Nein", help_text: str | None = None) -> bool:
    q_col, a_col = st.columns([3, 2], vertical_alignment="center")
    with q_col:
        st.write(question)
        if help_text:
            st.caption(help_text)
    with a_col:
        options = ["Ja", "Nein"]
        index = 0 if default == "Ja" else 1
        return st.radio(
            " ",
            options,
            index=index,
            horizontal=True,
            label_visibility="collapsed",
            key=key,
        ) == "Ja"

def select_row(question: str, options: list[str], key: str, index: int = 0, help_text: str | None = None) -> str:
    q_col, a_col = st.columns([3, 2], vertical_alignment="center")
    with q_col:
        st.write(question)
        if help_text:
            st.caption(help_text)
    with a_col:
        return st.selectbox(" ", options, index=index, label_visibility="collapsed", key=key)

def number_row(question: str, key: str, min_v: int, max_v: int, value: int, step: int = 1, help_text: str | None = None) -> int:
    q_col, a_col = st.columns([3, 2], vertical_alignment="center")
    with q_col:
        st.write(question)
        if help_text:
            st.caption(help_text)
    with a_col:
        return st.number_input(
            " ",
            min_value=min_v,
            max_value=max_v,
            value=value,
            step=step,
            label_visibility="collapsed",
            key=key
        )

def slider_row(question: str, key: str, min_v: int, max_v: int, value: int, help_text: str | None = None) -> int:
    q_col, a_col = st.columns([3, 2], vertical_alignment="center")
    with q_col:
        st.write(question)
        if help_text:
            st.caption(help_text)
    with a_col:
        return st.slider(" ", min_v, max_v, value, label_visibility="collapsed", key=key)

def bmi_from(height_cm: float, weight_kg: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)

def build_row(values: dict) -> pd.DataFrame:
    row = {f: 0 for f in expected_features}
    row.update(values)
    return pd.DataFrame([row], columns=expected_features)

def safe_set(d: dict, key: str, value):
    if key in expected_features:
        d[key] = value


# --------------------------------------------------
# Inputs (kompakt, aber vollständig genug)
# --------------------------------------------------
st.subheader("Körpermaße")
height_cm = number_row("Körpergröße (cm)", key="h_cm", min_v=120, max_v=220, value=175)
weight_kg = number_row("Körpergewicht (kg)", key="w_kg", min_v=40, max_v=220, value=75)

bmi = bmi_from(height_cm, weight_kg)
st.write(f"**BMI (berechnet):** {bmi:.1f}")

st.subheader("Soziodemografie")
sex = select_row("Geschlecht", ["männlich", "weiblich"], key="sex", index=0)
sex_val = 1 if sex == "männlich" else 0

age_label = select_row(
    "Altersgruppe",
    ["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"],
    key="age_group",
    index=6
)
age_val = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5,
    "45–49": 6, "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10,
    "70–74": 11, "75–79": 12, "80+": 13
}[age_label]

education_label = select_row(
    "Höchster Bildungsabschluss",
    ["Keine Schule / Grundschule","Sekundarstufe I","Sekundarstufe II (Abitur o. ä.)","Hochschule (ohne Abschluss)","Hochschulabschluss"],
    key="education",
    index=2
)
education_val = {
    "Keine Schule / Grundschule": 2,
    "Sekundarstufe I": 3,
    "Sekundarstufe II (Abitur o. ä.)": 4,
    "Hochschule (ohne Abschluss)": 5,
    "Hochschulabschluss": 6
}[education_label]

income_label = select_row(
    "Einkommen (jährlich, grobe Einteilung)",
    ["unter 10.000 €","10.000–<15.000 €","15.000–<20.000 €","20.000–<25.000 €","25.000–<35.000 €","35.000–<50.000 €","50.000–<75.000 €","75.000 € oder mehr"],
    key="income",
    index=4
)
income_val = {
    "unter 10.000 €": 1, "10.000–<15.000 €": 2, "15.000–<20.000 €": 3,
    "20.000–<25.000 €": 4, "25.000–<35.000 €": 5, "35.000–<50.000 €": 6,
    "50.000–<75.000 €": 7, "75.000 € oder mehr": 8
}[income_label]


st.subheader("Gesundheit & Lebensstil")
high_bp = yn_row(
    "Bluthochdruck bekannt (Diagnose/Medikation)?",
    key="highbp",
    help_text="Richtwert: häufig ≥140 mmHg (oberer Wert) oder ≥90 mmHg (unterer Wert)."
)

high_chol = yn_row(
    "Erhöhte Cholesterinwerte bekannt?",
    key="highchol",
    help_text="Richtwert: Gesamtcholesterin häufig ≥240 mg/dL oder LDL ≥160 mg/dL."
)

stroke = yn_row("Vorgeschichte eines Schlaganfalls?", key="stroke")
heart = yn_row("Koronare Herzerkrankung oder Herzinfarkt bekannt?", key="heart")

smoke_status = select_row(
    "Rauchstatus",
    ["Nie geraucht", "Ehemaliger Raucher", "Aktiver Raucher"],
    key="smoke_status",
    index=0
)
smoker_val = 0 if smoke_status == "Nie geraucht" else 1

phys_activity = yn_row("Körperliche Aktivität in den letzten 30 Tagen?", key="phys_activity", default="Ja")
fruits = yn_row("Täglicher Obstkonsum?", key="fruits", default="Ja")
veggies = yn_row("Täglicher Gemüsekonsum?", key="veggies", default="Ja")
alcohol = yn_row("Regelmäßig erhöhter Alkoholkonsum?", key="alcohol", default="Nein")

gen_health = slider_row(
    "Allgemeiner Gesundheitszustand (1–5)",
    key="genhlth",
    min_v=1, max_v=5, value=3,
    help_text="1 = sehr gut, 5 = sehr schlecht"
)

# Zusätzliche Features (die du vorher „mal da hattest“) wieder rein – aber als Expander
with st.expander("Zusätzliche Angaben (optional)"):
    chol_check = yn_row("Cholesterin in den letzten 5 Jahren bestimmt?", key="cholcheck", default="Ja")
    diff_walk = yn_row("Einschränkung beim Gehen/Treppensteigen?", key="diffwalk", default="Nein")

    ment = slider_row(
        "Mentale Gesundheit: belastende Tage (0–30)",
        key="menthlth",
        min_v=0, max_v=30, value=0,
        help_text="Tage in den letzten 30 Tagen mit eingeschränkter mentaler Gesundheit."
    )
    phys = slider_row(
        "Körperliche Gesundheit: belastende Tage (0–30)",
        key="physhlth",
        min_v=0, max_v=30, value=0,
        help_text="Tage in den letzten 30 Tagen mit eingeschränkter körperlicher Gesundheit."
    )

    any_healthcare = yn_row("Regelmäßiger Zugang zu medizinischer Versorgung?", key="anyhealthcare", default="Ja")
    no_doc_cost = yn_row("Arztbesuch trotz Bedarf aus Kostengründen vermieden?", key="nodoccost", default="Nein")


# Wenn Expander nicht geöffnet wurde: definieren wir Defaults (damit unten kein NameError)
if "chol_check" not in locals():
    chol_check = True
if "diff_walk" not in locals():
    diff_walk = False
if "ment" not in locals():
    ment = 0
if "phys" not in locals():
    phys = 0
if "any_healthcare" not in locals():
    any_healthcare = True
if "no_doc_cost" not in locals():
    no_doc_cost = False


# --------------------------------------------------
# Build model input
# --------------------------------------------------
vals = {}

safe_set(vals, "HighBP", int(high_bp))
safe_set(vals, "HighChol", int(high_chol))
safe_set(vals, "CholCheck", int(chol_check))
safe_set(vals, "BMI", float(bmi))
safe_set(vals, "Smoker", int(smoker_val))
safe_set(vals, "Stroke", int(stroke))
safe_set(vals, "HeartDiseaseorAttack", int(heart))
safe_set(vals, "PhysActivity", int(phys_activity))
safe_set(vals, "Fruits", int(fruits))
safe_set(vals, "Veggies", int(veggies))
safe_set(vals, "HvyAlcoholConsump", int(alcohol))
safe_set(vals, "AnyHealthcare", int(any_healthcare))
safe_set(vals, "NoDocbcCost", int(no_doc_cost))
safe_set(vals, "GenHlth", int(gen_health))
safe_set(vals, "MentHlth", int(ment))
safe_set(vals, "PhysHlth", int(phys))
safe_set(vals, "DiffWalk", int(diff_walk))
safe_set(vals, "Sex", int(sex_val))
safe_set(vals, "Age", int(age_val))
safe_set(vals, "Education", int(education_val))
safe_set(vals, "Income", int(income_val))

# FE-Features (falls im Training enthalten)
if "inactive" in expected_features:
    vals["inactive"] = 1 - int(phys_activity)
if "low_fruits" in expected_features:
    vals["low_fruits"] = 1 - int(fruits)
if "low_veggies" in expected_features:
    vals["low_veggies"] = 1 - int(veggies)
if "poor_health" in expected_features:
    vals["poor_health"] = 1 if gen_health >= 4 else 0
if "mental_physical_burden" in expected_features:
    vals["mental_physical_burden"] = int(ment) + int(phys)
if "cardio_risk_sum" in expected_features:
    vals["cardio_risk_sum"] = int(high_bp) + int(high_chol) + int(heart) + int(stroke)
if "lifestyle_risk_sum" in expected_features:
    vals["lifestyle_risk_sum"] = int(smoker_val) + int(alcohol) + (1 - int(phys_activity)) + (1 - int(fruits)) + (1 - int(veggies))

input_df = build_row(vals)

# --------------------------------------------------
# Predict
# --------------------------------------------------
st.divider()
if st.button("Risiko berechnen", use_container_width=True):
    proba_risk = float(model.predict_proba(input_df)[0, 1])
    pred = int(proba_risk >= 0.5)

    if pred == 0:
        st.success("🟢 Kein Diabetes-Risiko (Modell)")
    else:
        st.warning("🔴 Diabetes-Risiko (Prädiabetes oder Diabetes) – Abklärung empfohlen")

    st.write(f"**Wahrscheinlichkeit für Risiko (Klasse 1):** {proba_risk * 100:.1f} %")
    st.caption("Für eine medizinische Abklärung sind Laborwerte (z. B. HbA1c, Nüchternglukose) entscheidend.")

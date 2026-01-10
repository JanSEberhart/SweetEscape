# SweetEscape 🍬

SweetEscape ist ein Machine-Learning-Projekt zur Klassifikation des Diabetes-Risikos
(**kein Diabetes**, **Prädiabetes**, **Diabetes**) auf Basis von Gesundheits- und Lebensstilmerkmalen
aus dem BRFSS-2015-Datensatz.

Ziel ist eine nachvollziehbare ML-Pipeline (Daten → Feature Engineering → Modellvergleich → finales Modell → Web-App).

---

## Projektstruktur

- `data/raw/` – originale Rohdaten (unverändert)
- `data/processed/` – verarbeitete Daten (Feature Engineering Output)
- `notebooks/` – nachvollziehbarer Workflow in zwei Schritten
  - `00_preprocessing_feature_engineering.ipynb` – Preprocessing & Feature Engineering, schreibt `diabetes_fe.csv`
  - `01_train_model.ipynb` – Training, Vergleich von Modellen, Auswahl & Speichern des finalen Modells
- `models/` – gespeichertes finales Modell (`diabetes_final_model.joblib`)
- `src/` – wiederverwendbarer Projektcode (z. B. Pfade/Konstanten, FE-Funktionen)
- `app.py` – Streamlit-Webanwendung (lokal ausführbar)

**Warum diese Struktur?**  
Sie trennt Rohdaten, Verarbeitung, Training und Deployment sauber und macht das Projekt reproduzierbar und prüfbar.

---

## Workflow (Reproduzierbarkeit)

1. **Rohdaten** liegen in `data/raw/diabetes_raw.csv`
2. Notebook **00** ausführen  
   → erzeugt `data/processed/diabetes_fe.csv` (moderates Feature Engineering)
3. Notebook **01** ausführen  
   → trainiert und vergleicht **3 Modelle** auf identischem Train/Test-Split  
   → speichert das **finale Modell** als `models/diabetes_final_model.joblib`
4. `app.py` lokal starten  
   → lädt das gespeicherte Modell und macht Vorhersagen für Nutzer-Eingaben

---

## Feature Engineering (Notebook 00)

Der Datensatz ist bereits stark vorverarbeitet (viele 0/1-Indikatoren und kategoriale Codes).
Daher wird bewusst **nur moderates Feature Engineering** durchgeführt, um Interpretierbarkeit zu erhalten.

Beispiele für abgeleitete Features:
- `inactive` (aus `PhysActivity`)
- `cardio_risk_sum` (Summenfeature aus kardiovaskulären Risikoindikatoren)
- `lifestyle_risk_sum` (Summenfeature aus Lifestyle-Risiken)
- `poor_health` (binär aus `GenHlth`)
- `mental_physical_burden` (Summenfeature aus `MentHlth` + `PhysHlth`)

Output: `data/processed/diabetes_fe.csv`

---

## Modelltraining & Vergleich (Notebook 01)

Es werden drei Modelle trainiert und fair verglichen (gleicher Split, gleiche Metriken):

1. **Logistic Regression** (`class_weight="balanced"`) – interpretierbare Baseline
2. **Random Forest** (`class_weight="balanced_subsample"`) – nicht-linearer Vergleich
3. **HistGradientBoosting** (mit `sample_weight`) – Boosting-Ansatz für tabellarische Daten

### Warum Macro-F1 als Hauptmetrik?

Der Datensatz ist stark unausgeglichen (viele Fälle „kein Diabetes“, sehr wenige „Prädiabetes“).
Accuracy wäre daher irreführend, weil ein Modell durch Vorhersage der Mehrheitsklasse bereits hoch abschneiden kann.

**Macro-F1** mittelt den F1-Score **über alle Klassen**, sodass Minderheitsklassen (Prädiabetes/Diabetes)
gleichwertig berücksichtigt werden. Zusätzlich werden **Recall der Klassen 1 und 2** sowie die **Confusion Matrix**
betrachtet, um kritische Fehlklassifikationen sichtbar zu machen.

### Ergebnis & finale Modellwahl

Basierend auf dem Vergleich wurde **Logistic Regression (balanced)** als finales Modell gewählt,
da es die beste Balance über alle Klassen (höchster Macro-F1) liefert und gut interpretierbar bleibt.

Gespeichert als:
- `models/diabetes_final_model.joblib`

---

## Web-App (lokal)

Die Web-Anwendung ist mit **Streamlit** umgesetzt und lädt das gespeicherte Modell.

Start (im Projekt-Root, venv aktiv):
```bash
streamlit run app.py

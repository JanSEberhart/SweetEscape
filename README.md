# SweetEscape 🍬

SweetEscape ist ein Machine-Learning-Projekt zur Klassifikation des Diabetes-Risikos  
(**kein Diabetes**, **Prädiabetes**, **Diabetes**) auf Basis von Gesundheits- und
Lebensstilmerkmalen aus dem BRFSS-2015-Datensatz.

Ziel ist der Aufbau einer **nachvollziehbaren und reproduzierbaren ML-Pipeline**  
(Daten → Feature Engineering → Modellvergleich → finales Modell → Web-App).

---

## Projektstruktur

- `data/raw/` – originale Rohdaten (unverändert)
- `data/processed/` – verarbeitete Daten (Output aus Feature Engineering)
- `notebooks/` – vollständiger, nachvollziehbarer Workflow
  - `00_preprocessing_feature_engineering.ipynb`  
    → Preprocessing & Feature Engineering, erzeugt `diabetes_fe.csv`
  - `01_train_model.ipynb`  
    → Modelltraining, Vergleich mehrerer Modelle, Auswahl & Speicherung des finalen Modells
- `models/` – gespeichertes finales Modell (`diabetes_final_model.joblib`)
- `src/` – wiederverwendbarer Projektcode (z. B. Hilfsfunktionen, Konstanten)
- `app.py` – Streamlit-Webanwendung (lokal ausführbar)
- `requirements.txt` – benötigte Python-Abhängigkeiten

**Warum diese Struktur?**  
Sie trennt Rohdaten, Verarbeitung, Training und Deployment sauber und macht das Projekt
reproduzierbar, wartbar und prüfbar.

---

## Workflow (Reproduzierbarkeit)

1. **Rohdaten** liegen in `data/raw/diabetes_raw.csv`
2. Notebook **00_preprocessing_feature_engineering.ipynb** ausführen  
   → erzeugt `data/processed/diabetes_fe.csv`
3. Notebook **01_train_model.ipynb** ausführen  
   → trainiert und vergleicht mehrere Modelle  
   → speichert das **finale Modell** als `models/diabetes_final_model.joblib`
4. `app.py` lokal starten  
   → lädt das gespeicherte Modell und führt Vorhersagen für Nutzereingaben aus

---

## Feature Engineering (Notebook 00)

Der Datensatz ist bereits stark vorverarbeitet (viele binäre Indikatoren und
kategoriale Codes).  
Daher wird bewusst **nur moderates Feature Engineering** durchgeführt, um die
Interpretierbarkeit der Merkmale zu erhalten.

Beispiele für abgeleitete Features:
- `inactive` – abgeleitet aus fehlender körperlicher Aktivität
- `cardio_risk_sum` – Summenfeature aus kardiovaskulären Risikofaktoren
- `lifestyle_risk_sum` – Summenfeature aus Lifestyle-Risiken
- `poor_health` – binär abgeleitet aus `GenHlth`
- `mental_physical_burden` – Kombination aus mentalen und körperlichen Belastungstagen

Output:
- `data/processed/diabetes_fe.csv`

---

## Modelltraining & Vergleich (Notebook 01)

Es werden **drei Modelle** unter identischen Bedingungen trainiert und verglichen
(gleicher Train/Test-Split, gleiche Metriken):

1. **Logistic Regression** (`class_weight="balanced"`)  
   → interpretierbare Baseline
2. **Random Forest** (`class_weight="balanced_subsample"`)  
   → nicht-lineares Vergleichsmodell
3. **HistGradientBoosting** (mit `sample_weight`)  
   → Boosting-Ansatz für tabellarische Daten

### Warum Macro-F1 als Hauptmetrik?

Der Datensatz ist stark unausgeglichen:
- viele Fälle **kein Diabetes**
- sehr wenige Fälle **Prädiabetes**

Eine hohe Accuracy wäre daher irreführend.

**Macro-F1** mittelt den F1-Score **über alle Klassen** und berücksichtigt damit
Minderheitsklassen gleichwertig. Zusätzlich werden:
- der **Recall** für Prädiabetes und Diabetes
- sowie die **Confusion Matrix**

zur Bewertung herangezogen.

### Finale Modellwahl

Basierend auf dem Vergleich wurde **Logistic Regression (balanced)** als finales Modell gewählt,
da es:
- den höchsten **Macro-F1-Score** erreicht
- stabile Ergebnisse für Minderheitsklassen liefert
- und gut interpretierbar bleibt

Gespeichert als:
- `models/diabetes_final_model.joblib`

---

## Web-App (lokal)

Die Web-Anwendung ist mit **Streamlit** umgesetzt und lädt das gespeicherte Modell.

Funktionen der App:
- Eingabe gesundheitlicher und lebensstilbezogener Merkmale
- automatische Berechnung des BMI aus Größe & Gewicht
- Ausgabe einer **probabilistischen Einschätzung** für alle drei Klassen

Nicht alle Angaben sind verpflichtend; **nicht ausgefüllte optionale Felder werden
neutral angenommen**.

**Hinweis:**  
Die Anwendung dient ausschließlich zur **ersten Risikoeinschätzung** und ersetzt
keine ärztliche Diagnose.

---

## Getting Started (lokal ausführen)

### Voraussetzungen
- Python **3.10 oder höher**
- Git

### Installation & Start

```bash
git clone https://github.com/JanSEberhart/SweetEscape.git
cd SweetEscape
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```


## Erweiterung: Binäre Klassifikation (App 2)

Zusätzlich zur Multiclass-Variante gibt es eine **binäre Klassifikation**:

- **0 = kein Diabetes**
- **1 = Diabetes-Risiko** (Prädiabetes **oder** Diabetes)

**Warum?**  
Die ursprüngliche Klasse „Prädiabetes“ ist im Datensatz sehr selten und schwer abzugrenzen. Durch das Zusammenfassen zu „Risiko“ wird das Lernproblem stabiler und die Metriken (z. B. F1/ROC-AUC) werden robuster.

### Dateien
- `data/processed/diabetes_fe_binary.csv` – Feature-engineerte Daten + Target `Diabetes_binary`
- `notebooks/02_train_binary_model.ipynb` – Training/Evaluation der binären Variante
- `models/diabetes_binary_model.joblib` – gespeichertes Binary-Modell
- `apps/app_binary.py` – Streamlit-App für die binäre Vorhersage

### App starten (Binary)
Im Projekt-Root (venv aktiv):

```bash
streamlit run apps/app_binary.py
```

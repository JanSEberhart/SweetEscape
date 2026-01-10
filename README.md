# SweetEscape 🍬

SweetEscape ist ein Machine-Learning-Projekt zur Klassifikation des Diabetes-Risikos
(kein Diabetes, Prädiabetes, Diabetes) auf Basis von Gesundheits- und Lebensstilmerkmalen
aus dem BRFSS-2015-Datensatz.

## Projektstruktur
- `data/raw` – originale Rohdaten
- `data/processed` – Feature-engineerte Daten
- `notebooks/00_*` – Datenvorverarbeitung & Feature Engineering
- `notebooks/01_*` – Modelltraining
- `models/` – gespeichertes finales Modell
- `app.py` – Streamlit-Webanwendung

## Workflow
1. Rohdaten laden (`data/raw`)
2. Feature Engineering → neue CSV (`data/processed`)
3. Modelltraining & Evaluation
4. Speicherung des finalen Modells
5. Deployment als Web-App

## Ziel
Unterstützung bei der Einschätzung des Diabetes-Risikos anhand bekannter medizinischer
Risikofaktoren.
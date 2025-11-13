# Risk Controlled Prediction Sets for Ontology Grounding (RCPS_OG)
Adapting methods for risk controlled prediction sets for grounding free text to biomedical ontologies.
# local installation 
- `pip install -e ./` 
# gilda resource file  can be gotten but running (note that this will take some time)
- `python -m gilda.generate_terms` 

# Steps to get calibration dataset
1. Pull the raw dataset from the NIH `bash scripts/pull_BioRed.sh`
2. Extract teh information into a tabular representation that is easier to work with, `python scripts/load_BioRed.py`

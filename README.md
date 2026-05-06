# Knapsack MIP Optimizer

A Streamlit app for the classic 0-1 knapsack problem as a mixed-integer program
(Pyomo + GLPK). Click items to pack them; compare your selection against the
optimum.

**Live demo:** https://knapsackmilp.streamlit.app/

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

GLPK must be on PATH. On Streamlit Cloud, packages.txt handles `glpk-utils`.

## Files

- `app.py` — Streamlit UI, Pyomo model, GLPK wrapper
- `Knapsack.ipynb` — formulation in a notebook
- `requirements.txt`, `packages.txt` — Python deps and system packages

# Knapsack MIP Optimizer

A Streamlit app for the classic 0-1 knapsack problem as a mixed-integer program
(Pyomo + GLPK). Click items to pack them; compare your selection against the
optimum.

**Live demo:** https://knapsack.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

GLPK must be on PATH:
- Debian/Ubuntu: `apt-get install glpk-utils`
- macOS: `brew install glpk`

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
with `glpk-utils` and the app dependencies; `fly.toml` configures auto-stop
machines (idle = $0/mo). Custom domain wired through Cloudflare DNS.

## Files

- `app.py` — Streamlit UI, Pyomo model, GLPK wrapper
- `Knapsack.ipynb` — formulation in a notebook
- `requirements.txt`, `packages.txt` — Python and system deps
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline

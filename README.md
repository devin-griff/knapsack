# Knapsack MIP Optimizer

A Streamlit app for the classic 0-1 knapsack problem as a mixed-integer program
(Pyomo + HiGHS). Click items to pack them; compare your selection against the
optimum. The in-app **📐 Formulation** tab summarizes the math, explains why the
greedy heuristic fails on the integer case, and links to the references: see
[References](#references) below.

**Live demo:** https://knapsack.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

HiGHS ships as a pip wheel (`highspy`), so `pip install` covers everything -
no separate solver install needed.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py`: Streamlit UI, Pyomo model, HiGHS wrapper
- `Knapsack.ipynb`: formulation in a notebook
- `requirements.txt`: Python deps
- `Dockerfile`, `fly.toml`, `.dockerignore`: Fly.io production image config
- `.github/workflows/deploy.yml`: auto-deploy pipeline

## References

[1] H. Kellerer, U. Pferschy, and D. Pisinger, *Knapsack Problems*. Springer,
Berlin, 2004.
[Springer](https://link.springer.com/book/10.1007/978-3-540-24777-7)

[2] Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex
method," *Mathematical Programming Computation*, vol. 10, no. 1, pp. 119–142,
2018.
[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)

[3] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo: Optimization
Modeling in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)

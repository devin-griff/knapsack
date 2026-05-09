# Minimal image for Streamlit + Pyomo + GLPK.
# Python 3.12 slim base; glpk-utils provides the `glpsol` solver Pyomo shells out to.
FROM python:3.12-slim

# System dependencies: GLPK for the MIP solver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends glpk-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source + favicon (referenced by st.set_page_config(page_icon=...)).
COPY app.py favicon.png ./

# Overwrite Streamlit's default static index.html title and favicon so the
# initial render — before the React app boots and applies set_page_config —
# already shows our app name and the blackletter-G favicon, instead of the
# default "Streamlit" title flashing for ~1s before being replaced.
RUN STATIC=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))") \
    && sed -i 's|<title>Streamlit</title>|<title>Knapsack MIP Optimizer</title>|' "$STATIC/index.html" \
    && cp /app/favicon.png "$STATIC/favicon.png"

# Streamlit listens on 8080 (Fly's expected internal port).
# --server.address=0.0.0.0 so it binds outside the container.
# --server.headless=true skips the email prompt on first run.
# --browser.gatherUsageStats=false opts out of telemetry.
# Run as a non-root user. If a future Streamlit (or transitive dep) RCE
# lands in the container, the attacker doesn't get root. Defense in depth.
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

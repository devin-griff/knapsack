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

# Streamlit listens on 8080 (Fly's expected internal port).
# --server.address=0.0.0.0 so it binds outside the container.
# --server.headless=true skips the email prompt on first run.
# --browser.gatherUsageStats=false opts out of telemetry.
EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

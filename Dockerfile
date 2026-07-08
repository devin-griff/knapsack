# Minimal image for Streamlit + Pyomo + HiGHS.
# Python 3.12 slim base; HiGHS ships as a pip wheel (`highspy`), so no system
# dependencies are needed.
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source + favicon (referenced by st.set_page_config(page_icon=...)).
COPY app.py favicon.png ./

# Overwrite Streamlit's default static index.html: title, favicon, and
# inject Open Graph + Twitter Card meta tags so links to this app on
# *.griffith-pse.com unfurl as a rich card on LinkedIn / Slack / iMessage.
RUN STATIC=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))") \
    && sed -i 's|<title>Streamlit</title>|<title>Knapsack</title>|' "$STATIC/index.html" \
    && sed -i 's|</head>|<link rel="icon" type="image/png" href="./favicon.png"/><meta property="og:type" content="website"/><meta property="og:title" content="Knapsack MIP Optimizer"/><meta property="og:description" content="A 0-1 mixed-integer program: pack the most-valuable items under a weight limit. Pyomo + HiGHS, runs in your browser."/><meta property="og:image" content="https://griffith-pse.com/images/knapsack.png"/><meta property="og:site_name" content="Griffith PSE"/><meta name="twitter:card" content="summary_large_image"/><meta name="twitter:title" content="Knapsack MIP Optimizer"/><meta name="twitter:description" content="A 0-1 mixed-integer program: pack the most-valuable items under a weight limit. Pyomo + HiGHS, runs in your browser."/><meta name="twitter:image" content="https://griffith-pse.com/images/knapsack.png"/></head>|' "$STATIC/index.html" \
    && cp /app/favicon.png "$STATIC/favicon.png" && cp /app/favicon.png "$STATIC/favicon.ico"

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

import copy
import html as html_module
import os
import tempfile

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.opt import TerminationCondition


MAX_ITEMS = 12

DEFAULT_DATA = {
    "items": [
        "laptop", "water_bottle", "tent", "sleeping_bag", "flashlight",
        "first_aid_kit", "stove", "jacket", "map", "camera",
    ],
    "value": {
        "laptop": 25, "water_bottle": 4, "tent": 18, "sleeping_bag": 14,
        "flashlight": 6, "first_aid_kit": 10, "stove": 12, "jacket": 11,
        "map": 5, "camera": 16,
    },
    "weight": {
        "laptop": 6, "water_bottle": 2, "tent": 9, "sleeping_bag": 5,
        "flashlight": 1, "first_aid_kit": 3, "stove": 4, "jacket": 4,
        "map": 1, "camera": 3,
    },
    "weight_limit": 14,
}


# ---------- Solver ----------

def build_model(data):
    m = pyo.ConcreteModel()
    m.ITEMS = pyo.Set(initialize=data["items"])
    m.value = pyo.Param(m.ITEMS, initialize=data["value"])
    m.weight = pyo.Param(m.ITEMS, initialize=data["weight"])
    m.y = pyo.Var(m.ITEMS, within=pyo.Binary)
    m.total_value = pyo.Objective(
        expr=sum(m.value[i] * m.y[i] for i in m.ITEMS),
        sense=pyo.maximize,
    )
    m.capacity = pyo.Constraint(
        expr=sum(m.weight[i] * m.y[i] for i in m.ITEMS) <= data["weight_limit"]
    )
    return m


def _solve_capturing(m):
    """Run the solver and return (results, log_text). Captures GLPK's
    subprocess stdout via two mechanisms (FD-level redirect + logfile=)
    so we get output reliably across platforms."""
    fd, log_path = tempfile.mkstemp(suffix=".glpk.log")
    os.close(fd)
    log_text = ""
    try:
        try:
            with capture_output(capture_fd=True) as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        except TypeError:
            with capture_output() as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        if not log_text.strip():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
            except OSError:
                pass
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass
    return results, log_text


def solve(data):
    if not data["items"]:
        return {"status": "no_items", "y": {}, "value": None, "log": ""}

    m = build_model(data)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "GLPK solver binary not found. On Streamlit Cloud add "
                "`glpk-utils` to packages.txt at the repo root. "
                f"({e})"
            ),
            "y": {},
            "value": None,
            "log": "",
        }

    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        y = {i: float(pyo.value(m.y[i])) for i in data["items"]}
        value = float(pyo.value(m.total_value))
        return {"status": "optimal", "y": y, "value": value, "log": log}
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "y": {}, "value": None, "log": log}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "y": {}, "value": None, "log": log}
    return {"status": str(tc), "y": {}, "value": None, "log": log}


# ---------- State ----------

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "selected" not in st.session_state:
        st.session_state.selected = set()
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.selected = set()
    st.session_state.optimal = None
    st.session_state["weight_limit_input"] = int(DEFAULT_DATA["weight_limit"])


def _toggle_item(item):
    sel = st.session_state.selected
    if item in sel:
        sel.discard(item)
    else:
        sel.add(item)


def _set_at_optimum():
    optimal = st.session_state.optimal
    if optimal and optimal["status"] == "optimal":
        st.session_state.selected = {
            i for i, v in optimal["y"].items() if v > 0.5
        }


# ---------- Utilities ----------

def data_to_df(data):
    return pd.DataFrame([
        {"Item": i, "Value": data["value"][i], "Weight": data["weight"][i]}
        for i in data["items"]
    ])


def df_to_data(df, weight_limit):
    df = df.copy()
    df["Item"] = df["Item"].astype("string").str.strip()
    df = df.dropna(subset=["Item"])
    df = df[df["Item"] != ""]
    df = df.drop_duplicates(subset=["Item"], keep="first")
    for col in ["Value", "Weight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    items = df["Item"].tolist()
    value = {row.Item: float(row.Value) for row in df.itertuples()}
    weight = {row.Item: float(row.Weight) for row in df.itertuples()}
    return {
        "items": items,
        "value": value,
        "weight": weight,
        "weight_limit": float(weight_limit),
    }


def colored_metric(label, value, color):
    style_color = f"color: {color};" if color else ""
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0;'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- LaTeX (instance formulation) ----------

_LATEX_ESCAPE = [
    ("\\", r"\textbackslash "),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
]


def _latex_text(s):
    for raw, esc in _LATEX_ESCAPE:
        s = s.replace(raw, esc)
    return f"\\text{{{s}}}"


def _build_lhs(coefs, items):
    parts = []
    first = True
    for i in items:
        c = float(coefs.get(i, 0.0))
        if c == 0:
            continue
        c_str = f"{c:g}"
        sep = "" if first else " + "
        parts.append(f"{sep}{c_str} \\, {_latex_text(i)}")
        first = False
    return "".join(parts) if parts else "0"


def build_instance_latex(data):
    items = data["items"]
    obj = _build_lhs(data["value"], items)
    cap_lhs = _build_lhs(data["weight"], items)
    rhs = f"{data['weight_limit']:g}"
    bounds_lhs = ", ".join(_latex_text(i) for i in items)
    rows = [
        r"\max \quad & " + obj + r" \\",
        r"\text{s.t.} \quad & " + cap_lhs + r" \le " + rhs + r" \quad \text{(capacity)} \\",
        f"& {bounds_lhs} \\in \\{{0,1\\}}",
    ]
    body = r"\begin{aligned}" + "\n".join(rows) + r"\end{aligned}"
    if len(items) > 7:
        body = r"\small " + body
    return body


# ---------- CSS ----------

CSS = """
<style>
/* Toggle/action buttons: allow multi-line labels and tighten spacing
   so 12 items fit in a 4x3 grid without scrolling. */
.stButton > button {
  white-space: pre-line;
  font-size: 0.85rem;
  padding: 0.4rem 0.5rem;
  line-height: 1.3;
  min-height: 60px;
}

/* Inert read-only cards for the Optimal column. */
.kp-card {
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: 0.5rem;
  padding: 0.4rem 0.5rem;
  text-align: center;
  background: transparent;
  color: rgba(49,51,63,1);
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  line-height: 1.3;
  min-height: 60px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.kp-card-on {
  background: var(--primary-color, #FF4B4B);
  color: white;
  border-color: var(--primary-color, #FF4B4B);
}
.kp-name { font-weight: 600; }
.kp-stats { font-size: 0.75rem; opacity: 0.85; }
</style>
"""


# ---------- Tabs ----------

def _grid_rows(items, cols=3):
    rows = []
    for r in range(0, len(items), cols):
        row = items[r:r + cols]
        while len(row) < cols:
            row.append(None)
        rows.append(row)
    return rows


def _render_optimal_card(item, data, on):
    cls = "kp-card kp-card-on" if on else "kp-card"
    safe = html_module.escape(item)
    v = data["value"][item]
    w = data["weight"][item]
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="kp-name">{safe}</div>'
        f'<div class="kp-stats">value {v:g} &middot; weight {w:g}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_optimizer_tab():
    data = st.session_state.data
    if not data["items"]:
        st.info("Add at least one item on the Data tab.")
        return

    # Action row
    act1, act2, _ = st.columns([1, 1, 6])
    with act1:
        if st.button("Run Optimizer", width="stretch", key="run_btn"):
            st.session_state.optimal = solve(data)
    optimal = st.session_state.optimal
    with act2:
        set_disabled = not (optimal and optimal["status"] == "optimal")
        st.button(
            "Set at Optimum",
            width="stretch",
            disabled=set_disabled,
            on_click=_set_at_optimum,
            key="set_opt_btn",
        )

    # Solver-status messages (only on non-optimal outcomes)
    if optimal:
        if optimal["status"] == "solver_missing":
            st.error(optimal.get("message", "Solver missing"))
        elif optimal["status"] == "infeasible":
            st.error("Infeasible — no selection satisfies the weight limit.")
        elif optimal["status"] == "unbounded":
            st.error("Unbounded problem.")
        elif optimal["status"] not in ("optimal", "no_items"):
            st.error(f"Solver returned: {optimal['status']}")

    # Three-column body: Your | Weight chart | Optimal
    your_col, chart_col, opt_col = st.columns([2, 1, 2])

    selected = st.session_state.selected
    your_value = sum(data["value"][i] for i in selected if i in data["value"])
    your_weight = sum(data["weight"][i] for i in selected if i in data["weight"])

    with your_col:
        st.markdown("**Your knapsack**")
        for row in _grid_rows(data["items"], cols=3):
            cs = st.columns(3)
            for c, item in zip(cs, row):
                with c:
                    if item is None:
                        st.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    v = data["value"][item]
                    w = data["weight"][item]
                    label = f"{item}\nvalue {v:g} · weight {w:g}"
                    btn_type = "primary" if item in selected else "secondary"
                    st.button(
                        label,
                        type=btn_type,
                        width="stretch",
                        on_click=_toggle_item,
                        args=(item,),
                        key=f"toggle_{item}",
                    )

        if optimal and optimal["status"] == "optimal":
            opt_value = float(optimal["value"])
            matches = abs(your_value - opt_value) < 1e-6
            your_color = "#16a34a" if matches else "#dc2626"
        else:
            your_color = None
        colored_metric("Your value", f"{your_value:g}", your_color)

    with chart_col:
        st.markdown("**Weight**")
        chart_rows = [{"source": "You", "value": float(your_weight)}]
        if optimal and optimal["status"] == "optimal":
            opt_items = {i for i, v in optimal["y"].items() if v > 0.5}
            opt_w = sum(data["weight"][i] for i in opt_items if i in data["weight"])
            chart_rows.append({"source": "Optimal", "value": float(opt_w)})

        chart_df = pd.DataFrame(chart_rows)
        bars = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("source:N", sort=["You", "Optimal"], title=None),
                y=alt.Y("value:Q", title="Weight"),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(
                        domain=["You", "Optimal"],
                        range=["#4C78A8", "#54A24B"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("source:N"),
                    alt.Tooltip("value:Q", format=".2f"),
                ],
            )
        )
        limit_df = pd.DataFrame([{"value": float(data["weight_limit"])}])
        rule = (
            alt.Chart(limit_df)
            .mark_rule(color="#dc2626", strokeWidth=2, strokeDash=[5, 5])
            .encode(
                y="value:Q",
                tooltip=[alt.Tooltip("value:Q", title="Weight limit")],
            )
        )
        chart = (bars + rule).properties(height=320)
        st.altair_chart(chart, width="stretch")

    with opt_col:
        st.markdown("**Optimal**")
        if optimal and optimal["status"] == "optimal":
            opt_set = {i for i, v in optimal["y"].items() if v > 0.5}
        else:
            st.caption("Click Run Optimizer to see the optimal selection.")
            opt_set = set()

        for row in _grid_rows(data["items"], cols=3):
            cs = st.columns(3)
            for c, item in zip(cs, row):
                with c:
                    if item is None:
                        st.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    _render_optimal_card(item, data, on=(item in opt_set))

        if optimal and optimal["status"] == "optimal":
            colored_metric("Optimal value", f"{float(optimal['value']):g}", "#16a34a")
        else:
            colored_metric("Optimal value", "—", None)


def render_data_tab():
    wl_col, _ = st.columns([1, 4])
    with wl_col:
        weight_limit = st.number_input(
            "Weight limit",
            min_value=0,
            value=int(st.session_state.data["weight_limit"]),
            step=1,
            format="%d",
            key="weight_limit_input",
        )

    st.subheader(f"Items (max {MAX_ITEMS})")
    df = data_to_df(st.session_state.data)
    table_col, _ = st.columns([2, 3])
    with table_col:
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
            height=(len(df) + 2) * 35 + 3,
            column_config={
                "Item": st.column_config.TextColumn("Item"),
                "Value": st.column_config.NumberColumn("Value", min_value=0, max_value=100),
                "Weight": st.column_config.NumberColumn("Weight", min_value=0, max_value=50),
            },
            key="data_editor",
        )

    warnings = []
    if len(edited) > MAX_ITEMS:
        warnings.append(f"Capped at {MAX_ITEMS} items; extra rows ignored.")
        edited = edited.head(MAX_ITEMS)
    names = edited["Item"].dropna().astype("string").str.strip()
    if names.duplicated().any():
        warnings.append("Duplicate item names were dropped (kept the first).")

    new_data = df_to_data(edited, weight_limit)

    if new_data != st.session_state.data:
        st.session_state.data = new_data
        st.session_state.optimal = None
        st.session_state.selected = (
            st.session_state.selected & set(new_data["items"])
        )
        st.rerun()

    for w in warnings:
        st.warning(w)

    if st.button("Reset to defaults"):
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    st.subheader("General Formulation")

    st.markdown("**Sets**")
    st.markdown(r"$\mathcal{I} = \{\text{items}\}$")

    st.markdown("**Parameters**")
    st.markdown(r"$v_i$ value of item $i \in \mathcal{I}$")
    st.markdown(r"$w_i$ weight of item $i \in \mathcal{I}$")
    st.markdown(r"$W$ knapsack weight capacity")

    st.markdown("**Variables**")
    st.markdown(r"$y_i \in \{0, 1\}$ whether item $i$ is packed")

    st.markdown("**Objective and Constraints**")
    st.latex(r"""
    \begin{gathered}
    \max_y \sum_{i \in \mathcal{I}} v_i y_i \quad \text{(value)} \\
    \text{s.t.} \quad \sum_{i \in \mathcal{I}} w_i y_i \le W \quad \text{(capacity)} \\
    y_i \in \{0, 1\} \quad \forall i \in \mathcal{I}
    \end{gathered}
    """)

    st.divider()
    st.subheader("Instance Formulation")
    data = st.session_state.data
    if not data["items"]:
        st.info("Add at least one item on the Data tab.")
        return
    st.latex(build_instance_latex(data))


def render_logs_tab():
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Run the optimizer to see solver logs.")
        return
    log = optimal.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


# ---------- Main ----------

st.set_page_config(page_title="Knapsack MIP Optimizer", layout="wide")
init_state()
st.markdown(CSS, unsafe_allow_html=True)
st.title("Knapsack MIP Optimizer")

optimizer_tab, data_tab, formulation_tab, logs_tab = st.tabs(
    ["🎯 Optimizer", "📋 Data", "📐 Formulation", "📜 Logs"]
)

with optimizer_tab:
    render_optimizer_tab()
with data_tab:
    render_data_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()

# =============================================================================
# Knapsack MIP Optimizer — a Streamlit tutorial app.
#
# This file builds an interactive web app around the classic 0-1 knapsack
# problem: given a set of items, each with a value and a weight, choose a
# subset that maximizes total value without exceeding a weight limit.
#
# It is solved here as a Mixed-Integer Program (MIP):
#   maximize   sum_i  v_i * y_i        (total value)
#   subject to sum_i  w_i * y_i <= W   (capacity)
#              y_i in {0, 1}           (each item is either packed or not)
#
# Library roadmap:
#   - streamlit  — the UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — the algebraic modeling layer. We declare sets, parameters,
#                  variables, an objective, and constraints, then hand the
#                  model to a solver.
#   - GLPK       — the actual MIP solver, called as a subprocess via Pyomo.
#   - pandas     — used as the data shape for the Streamlit data editor and
#                  for the bar chart in the optimizer tab.
#   - altair     — the bar chart comparing your weight to the optimum.
#
# File roadmap (matching the section banners below):
#   1. Solver       — model definition and a wrapper that captures GLPK logs.
#   2. State        — initializing and mutating st.session_state.
#   3. Utilities    — DataFrame <-> internal-dict conversion, colored metric.
#   4. LaTeX        — render the current instance as a formatted equation.
#   5. CSS          — styling for the item buttons and read-only cards.
#   6. Tabs         — one render_* function per tab.
#   7. Main         — page config and tab assembly at module bottom.
# =============================================================================

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


# Hard cap on item count. The data editor allows adding rows freely; anything
# beyond this gets truncated with a warning. Twelve fits the 4x3 button grid
# rendered in the Optimizer tab without scrolling.
MAX_ITEMS = 12

# Default instance shown on first load and after the "Reset to defaults"
# button. `items` is the order list; `value`/`weight` are dicts keyed by item
# name; `weight_limit` is the capacity W.

DEFAULT_DATA = {
    "items": [
        "laptop", "water bottle", "tent", "sleeping bag", "flashlight",
        "first aid kit", "stove", "jacket", "map", "camera",
        "knife", "compass",
    ],
    "value": {
        "laptop": 25, "water bottle": 4, "tent": 18, "sleeping bag": 14,
        "flashlight": 6, "first aid kit": 10, "stove": 12, "jacket": 11,
        "map": 5, "camera": 16, "knife": 8, "compass": 7,
    },
    "weight": {
        "laptop": 6, "water bottle": 2, "tent": 9, "sleeping bag": 5,
        "flashlight": 1, "first aid kit": 3, "stove": 4, "jacket": 4,
        "map": 1, "camera": 3, "knife": 1, "compass": 1,
    },
    "weight_limit": 14,
}


# ---------- Solver ----------
#
# Everything here is plain Pyomo: define a ConcreteModel, declare its sets,
# parameters, variables, objective, and constraints, then call a solver. The
# only twist is `_solve_capturing`, which works around the fact that GLPK's
# stdout comes from a subprocess and isn't picked up by ordinary Python
# stream redirection.

def build_model(data):
    # ConcreteModel is the eager flavor of Pyomo model: components are bound
    # to data at construction time (as opposed to AbstractModel + .create_instance()).
    m = pyo.ConcreteModel()

    # Set of item names. Indexes every per-item parameter and variable below.
    m.ITEMS = pyo.Set(initialize=data["items"])

    # Parameters: known data the solver does not change. v_i and w_i.
    m.value = pyo.Param(m.ITEMS, initialize=data["value"])
    m.weight = pyo.Param(m.ITEMS, initialize=data["weight"])

    # Decision variable y_i in {0, 1}. One binary per item: 1 = packed.
    m.y = pyo.Var(m.ITEMS, within=pyo.Binary)

    # Objective: maximize the value of packed items.
    m.total_value = pyo.Objective(
        expr=sum(m.value[i] * m.y[i] for i in m.ITEMS),
        sense=pyo.maximize,
    )

    # Single capacity constraint: total packed weight cannot exceed W.
    m.capacity = pyo.Constraint(
        expr=sum(m.weight[i] * m.y[i] for i in m.ITEMS) <= data["weight_limit"]
    )
    return m


def _solve_capturing(m):
    """Run the solver and return (results, log_text). Captures GLPK's
    subprocess stdout via two mechanisms (FD-level redirect + logfile=)
    so we get output reliably across platforms."""
    # Two capture paths run in parallel because each is unreliable on its own:
    #   1. capture_output(capture_fd=True) — redirects at the OS file
    #      descriptor level, which catches output from child processes like
    #      the GLPK binary. The capture_fd kwarg only exists in newer Pyomo,
    #      hence the TypeError fallback.
    #   2. logfile=log_path — asks GLPK itself to write its log to a file.
    #      Used as a backup if the FD capture comes back empty.
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
            # Older Pyomo without capture_fd — fall back to the plain form.
            with capture_output() as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        # If the in-memory capture missed the output, read the on-disk log.
        if not log_text.strip():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
            except OSError:
                pass
    finally:
        # Always clean up the temp log file.
        try:
            os.remove(log_path)
        except OSError:
            pass
    return results, log_text


def solve(data):
    # Top-level entrypoint used by the UI. Always returns a plain dict so the
    # caller can stash the result in session_state without holding on to a
    # live Pyomo model.

    # Empty problem — bail before constructing a model with no variables.
    if not data["items"]:
        return {"status": "no_items", "y": {}, "value": None, "log": ""}

    m = build_model(data)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        # Pyomo raises ApplicationError when the solver binary is missing
        # from PATH. Surfaces a friendly message in the UI rather than a
        # Python traceback.
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

    # Translate Pyomo's TerminationCondition enum into a small set of stable
    # status strings the UI knows how to render.
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        # Extract numeric values out of the model before returning. After
        # this point the Pyomo model is no longer needed.
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
    # Catch-all for anything unexpected (e.g. solver error, time limit).
    return {"status": str(tc), "y": {}, "value": None, "log": log}


# ---------- State ----------
#
# Streamlit re-executes the whole script on every interaction. Anything that
# must persist between runs lives in `st.session_state`, a dict-like store
# scoped to the user's browser session. The keys we use:
#   - data:                the current problem instance (items/values/weights/W)
#   - selected:            the user's current selection of items (a set)
#   - optimal:             the most recent solver result, or None
#   - _pending_reset:      a one-shot flag to reset on the next run
#   - weight_limit_input:  the value backing the number_input widget
#   - data_editor:         backing key for the data editor widget
#   - toggle_<item>:       backing key for each item toggle button

def init_state():
    # Idempotent initialization: only seed defaults the first time, otherwise
    # the user's edits would be wiped on every rerun.
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "selected" not in st.session_state:
        st.session_state.selected = set()
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    # The reset button can't directly mutate widget-backed keys without
    # raising a Streamlit error, so it sets a flag and reruns. We then apply
    # the reset *before* widgets are instantiated this run.
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    # Restore the default instance and clear any user-driven state.
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.selected = set()
    st.session_state.optimal = None
    # Reseed the number_input's backing key so the widget shows the default.
    st.session_state["weight_limit_input"] = int(DEFAULT_DATA["weight_limit"])


# Streamlit `on_click` callbacks. These run between the user's click and the
# next script rerun, so any state they mutate is visible to the rerun.
def _toggle_item(item):
    sel = st.session_state.selected
    if item in sel:
        sel.discard(item)
    else:
        sel.add(item)


def _set_at_optimum():
    # Copy the optimal solution into the user's `selected` set so the user
    # column visually matches the optimum on the next render.
    optimal = st.session_state.optimal
    if optimal and optimal["status"] == "optimal":
        st.session_state.selected = {
            i for i, v in optimal["y"].items() if v > 0.5
        }


# ---------- Utilities ----------
#
# Adapters between two data shapes:
#   - The "internal" dict shape used by the solver and most of the app.
#   - The DataFrame shape used by Streamlit's data editor widget.
# Plus one small custom replacement for st.metric that lets us tint the value.

def data_to_df(data):
    # Internal -> DataFrame. Used to seed the data editor each render.
    return pd.DataFrame([
        {"Item": i, "Value": data["value"][i], "Weight": data["weight"][i]}
        for i in data["items"]
    ])


def df_to_data(df, weight_limit):
    # DataFrame -> internal. The data editor lets users add/remove/edit rows
    # freely, so this normalizes whatever they typed: strip whitespace, drop
    # blanks and duplicates, coerce numerics, clamp to non-negative.
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


def colored_metric(label, value, color, align="left"):
    # st.metric doesn't support arbitrary value coloring, so we render our
    # own metric-shaped block via raw HTML. Used to flag matching/mismatching
    # values (green if your value equals the optimum, red otherwise).
    style_color = f"color: {color};" if color else ""
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0; text-align: {align};'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- LaTeX (instance formulation) ----------
#
# The Formulation tab shows two things: a static general formulation in math
# notation, and a dynamic "instance" formulation that substitutes the user's
# current values into the equation. The helpers here build the LaTeX source
# for the instance view; Streamlit's `st.latex` renders it.

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
    # Wrap an item name in \text{...} so it renders upright (not italicized
    # like math variables), with special characters escaped.
    for raw, esc in _LATEX_ESCAPE:
        s = s.replace(raw, esc)
    return f"\\text{{{s}}}"


def _build_lhs(coefs, items):
    # Build a "c1 * item1 + c2 * item2 + ..." style sum, skipping zero
    # coefficients and avoiding a leading "+".
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
    # Assemble the full instance formulation as a LaTeX `aligned` block:
    # objective, capacity constraint, then 0/1 bounds.
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
    # With many items the line gets long; \small keeps it on screen.
    if len(items) > 7:
        body = r"\small " + body
    return body


# ---------- CSS ----------
#
# Streamlit's default styling makes the item buttons too tall and prevents
# multi-line labels. We override a few rules so the 4x3 button grid fits and
# the read-only "Optimal" cards visually mirror the user's button grid.

CSS = """
<style>
/* Tighten the top of the main block so the title sits closer to the page
   top and the tabs are visible without scrolling. The minimum here is
   determined by Streamlit's sticky header (~3.75rem); going smaller hides
   the title underneath it. */
.block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 4rem !important;
}

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
  /* Streamlit puts 1rem of gap below each button row; matching that here so
     card rows line up vertically with the button rows on the left. */
  margin-bottom: 1rem;
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
/* Match the toggle buttons: uniform 0.85rem regular-weight text on both
   lines, no opacity adjustment. */
.kp-name { font-weight: 400; }
.kp-stats { font-size: 0.85rem; opacity: 1; }
</style>
"""


# ---------- Tabs ----------
#
# One render_* function per tab. The Optimizer tab is the main UI; Data lets
# the user edit items; Formulation shows the math; Logs shows GLPK output.

def _grid_rows(items, cols=3):
    # Chunk a flat list into rows of `cols` items, padding with None so each
    # row is the same length (lets the caller pair rows with `st.columns`).
    rows = []
    for r in range(0, len(items), cols):
        row = items[r:r + cols]
        while len(row) < cols:
            row.append(None)
        rows.append(row)
    return rows


def _render_optimal_card(item, data, on):
    # An inert read-only card matching the visual size of the toggle button
    # used for the user's selection. `on` highlights items chosen by the
    # solver. Item names are HTML-escaped because they come from user input.
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
    # Layout (top to bottom):
    #   1. Two action buttons (Run Optimizer, Set at Optimum)
    #   2. A status banner if the last run wasn't optimal
    #   3. Three columns: user's selection | weight chart | optimal selection
    data = st.session_state.data
    if not data["items"]:
        st.info("Add at least one item on the Data tab.")
        return

    # Action row. The third column is empty filler so the buttons stay
    # narrow on wide screens.
    act1, act2, _ = st.columns([1, 1, 6])
    with act1:
        # `st.button` returns True the run after it's clicked, so we can
        # branch on it directly. The result is stored in session_state so
        # subsequent reruns (e.g. from clicking item buttons) keep showing
        # the most recent solver result.
        if st.button("Run Optimizer", width="stretch", key="run_btn"):
            st.session_state.optimal = solve(data)
    optimal = st.session_state.optimal
    with act2:
        # "Set at Optimum" is only meaningful after a successful solve.
        # Using `on_click` instead of branching on the return value means
        # the callback runs *before* the rerun, so the new selection is
        # visible the same render.
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

    # Aggregates of the user's current selection.
    selected = st.session_state.selected
    your_value = sum(data["value"][i] for i in selected if i in data["value"])
    your_weight = sum(data["weight"][i] for i in selected if i in data["weight"])

    with your_col:
        # Left column: a 4x3 grid of item toggle buttons. Clicking calls
        # `_toggle_item` to flip the item's membership in `selected`.
        st.markdown("**Your knapsack**")
        for row in _grid_rows(data["items"], cols=3):
            cs = st.columns(3)
            for c, item in zip(cs, row):
                with c:
                    if item is None:
                        # Padding cell from `_grid_rows` — keeps columns aligned.
                        st.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    v = data["value"][item]
                    w = data["weight"][item]
                    # \n in a button label renders as a line break thanks to
                    # `white-space: pre-line` set in CSS above.
                    label = f"{item}\nvalue {v:g} · weight {w:g}"
                    # `primary` (red) for selected, `secondary` (gray) for not.
                    btn_type = "primary" if item in selected else "secondary"
                    st.button(
                        label,
                        type=btn_type,
                        width="stretch",
                        on_click=_toggle_item,
                        args=(item,),
                        key=f"toggle_{item}",
                    )

        # "Your value" sits beneath the user grid, right-aligned so it lands
        # next to the chart's "You" axis. Green when it matches the optimum,
        # red when it doesn't, no color before the first solve.
        if optimal and optimal["status"] == "optimal":
            opt_value = float(optimal["value"])
            matches = abs(your_value - opt_value) < 1e-6
            your_color = "#16a34a" if matches else "#dc2626"
        else:
            your_color = None
        colored_metric("Your value", f"{your_value:g}", your_color, align="right")

    with chart_col:
        # Middle column: bar chart of total weight (user vs optimum) with a
        # red dashed rule at the capacity W and a small text label naming
        # that rule. Built as three layered Altair charts: bars + rule + label.
        st.markdown("**Weight**")
        # Always include both bars so the chart layout stays stable. Before
        # a successful solve, the Optimal bar shows zero.
        if optimal and optimal["status"] == "optimal":
            # Items the solver selected (y_i > 0.5 since y_i is binary).
            opt_items = {i for i, v in optimal["y"].items() if v > 0.5}
            opt_w = sum(data["weight"][i] for i in opt_items if i in data["weight"])
        else:
            opt_w = 0.0
        chart_rows = [
            {"source": "You", "value": float(your_weight)},
            {"source": "Optimal", "value": float(opt_w)},
        ]

        chart_df = pd.DataFrame(chart_rows)
        # Layer 1: the bars. Encoded fields use Altair's `:N`/`:Q` shorthand
        # for nominal (categorical) and quantitative axes.
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
        # Layer 2: a single horizontal rule at y = weight_limit, plus a
        # text label sitting just above the line at the left edge of the
        # chart so the rule is identified at a glance.
        limit_df = pd.DataFrame([{"value": float(data["weight_limit"])}])
        rule = (
            alt.Chart(limit_df)
            .mark_rule(color="#dc2626", strokeWidth=2, strokeDash=[5, 5])
            .encode(
                y="value:Q",
                tooltip=[alt.Tooltip("value:Q", title="Weight limit")],
            )
        )
        label = (
            alt.Chart(limit_df)
            .mark_text(
                align="left",
                baseline="bottom",
                dx=4,
                dy=-2,
                color="#dc2626",
                fontSize=11,
            )
            .encode(
                y="value:Q",
                x=alt.value(0),
                text=alt.value(f"Weight limit ({data['weight_limit']:g})"),
            )
        )
        # `+` overlays the layers in Altair.
        chart = (bars + rule + label).properties(height=320)
        st.altair_chart(chart, width="stretch")

    with opt_col:
        # Right column: same 4x3 grid as the user column but rendered as
        # inert cards instead of buttons. Highlights items the solver chose.
        # When no solve has happened we render the prompt inline next to the
        # title so the cards below stay vertically aligned with the user grid.
        if optimal and optimal["status"] == "optimal":
            st.markdown("**Optimal knapsack**")
            opt_set = {i for i, v in optimal["y"].items() if v > 0.5}
        else:
            st.markdown(
                "**Optimal knapsack**"
                " <span style='color: rgba(49,51,63,0.6); font-size: 0.85rem;"
                " margin-left: 0.5rem;'>"
                "Click Run Optimizer to see the optimal selection.</span>",
                unsafe_allow_html=True,
            )
            opt_set = set()

        for row in _grid_rows(data["items"], cols=3):
            cs = st.columns(3)
            for c, item in zip(cs, row):
                with c:
                    if item is None:
                        st.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    _render_optimal_card(item, data, on=(item in opt_set))

        # "Optimal value" sits beneath the optimal grid, left-aligned so it
        # lands next to the chart's "Optimal" axis. Always green once a solve
        # has happened; em-dash placeholder beforehand.
        if optimal and optimal["status"] == "optimal":
            colored_metric("Optimal value", f"{float(optimal['value']):g}", "#16a34a", align="left")
        else:
            colored_metric("Optimal value", "—", None, align="left")


def render_data_tab():
    # Capacity input. The widget is bound to `weight_limit_input` in
    # session_state so `apply_reset` can seed it.
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

    # Editable items table. `num_rows="dynamic"` lets the user add/delete rows.
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

    # Validate / report on the edited table.
    warnings = []
    if len(edited) > MAX_ITEMS:
        warnings.append(f"Capped at {MAX_ITEMS} items; extra rows ignored.")
        edited = edited.head(MAX_ITEMS)
    names = edited["Item"].dropna().astype("string").str.strip()
    if names.duplicated().any():
        warnings.append("Duplicate item names were dropped (kept the first).")

    new_data = df_to_data(edited, weight_limit)

    # If the cleaned data differs from what we had, commit it to state and
    # rerun so other tabs see the change. We also invalidate any prior
    # solver result and drop selections for items that no longer exist.
    if new_data != st.session_state.data:
        st.session_state.data = new_data
        st.session_state.optimal = None
        st.session_state.selected = (
            st.session_state.selected & set(new_data["items"])
        )
        st.rerun()

    for w in warnings:
        st.warning(w)

    # Reset uses the deferred-flag pattern documented in `init_state`.
    if st.button("Reset to defaults"):
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    # Static reference math at the top, dynamic instance math at the bottom.
    # `st.markdown` with `$...$` renders inline LaTeX; `st.latex` renders a
    # display-style block. The general section is split across a 3-column
    # grid: the left column stacks Sets/Parameters/Variables (each as a
    # single markdown block so items stack tightly with no inter-paragraph
    # margin), the middle column holds the centered objective + constraints,
    # and the right column is empty padding so the equation lands at the
    # page midline rather than the right half.
    st.subheader("General Formulation")
    left, right, _ = st.columns([1, 1, 1])
    with left:
        st.markdown(
            "**Sets**  \n"
            r"$\mathcal{I} = \{\text{items}\}$"
        )
        st.markdown(
            "**Parameters**  \n"
            r"$v_i$ value of item $i \in \mathcal{I}$" "  \n"
            r"$w_i$ weight of item $i \in \mathcal{I}$" "  \n"
            r"$W$ knapsack weight capacity"
        )
        st.markdown(
            "**Variables**  \n"
            r"$y_i \in \{0, 1\}$ whether item $i$ is packed"
        )
    with right:
        # Title + display math in one centered block. Using `$$...$$` inside
        # st.markdown (rather than st.latex in its own component) lets us wrap
        # both in a single text-align:center div so the equation lines up
        # under the centered title.
        st.markdown(
            r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{gathered}
\max_y \sum_{i \in \mathcal{I}} v_i y_i \quad \text{(value)} \\
\text{s.t.} \quad \sum_{i \in \mathcal{I}} w_i y_i \le W \quad \text{(capacity)} \\
y_i \in \{0, 1\} \quad \forall i \in \mathcal{I}
\end{gathered}
$$

</div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Instance Formulation")
    data = st.session_state.data
    if not data["items"]:
        st.info("Add at least one item on the Data tab.")
        return
    st.latex(build_instance_latex(data))


def render_logs_tab():
    # Shows whatever GLPK printed during the last solve. The capture itself
    # happens in `_solve_capturing`; this tab just displays the result.
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
#
# Module-level code runs on every Streamlit rerun, so this section needs to
# be cheap and idempotent: configure the page, ensure session_state is set
# up, inject CSS, then assemble the four tabs.

# `set_page_config` must be the first Streamlit call; "wide" gives the
# three-column optimizer layout enough horizontal room.
st.set_page_config(page_title="Knapsack MIP Optimizer", layout="wide")

# Initialize session_state defaults and apply any pending reset.
init_state()

# Inject custom CSS once per rerun. `unsafe_allow_html=True` is required for
# raw HTML/CSS to be rendered through `st.markdown`.
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Knapsack MIP Optimizer"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([1, 1])
with _caption_col:
    st.markdown(
        "Try to beat the optimizer: toggle items in the **Optimizer** tab to pack "
        "the most valuable knapsack you can under the weight limit, then click "
        "**Run Optimizer** to see how your selection stacks up against the solver's. "
        "Edit items and capacity in the **Data** tab; the **Formulation** and "
        "**Logs** tabs show the underlying MIP and solver output."
    )

# `st.tabs` returns one container per label, used as a context manager to
# scope subsequent `st.*` calls into that tab.
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

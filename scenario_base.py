import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from collections import defaultdict, deque
import re
import json

st.title("FyreDrill Scenario Generator")

# --- Inputs ---
st.sidebar.header("General Settings")
num_days = st.sidebar.number_input("Number of Days", min_value=1, value=10)
num_metrics = st.sidebar.number_input("Number of Base Metrics", min_value=1, value=1)
num_calc = st.sidebar.number_input("Number of Calculated Metrics", min_value=0, value=1, step=1)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

base_date = pd.to_datetime("2025-05-13")
dates = pd.date_range(base_date, periods=num_days)
np.random.seed(seed)

# --- Base Metrics ---
st.header("Base Metrics")

metrics_data = []
metrics_dict = {}

for i in range(num_metrics):
    with st.expander(f"Metric {i+1}", expanded=(i == 0)):
        name = st.text_input("Metric Name", value=f"Metric_{i+1}", key=f"name_{i}")
        cols = st.columns(3)
        with cols[0]:
            num_breaks = st.number_input("# Breakpoints", min_value=1, value=1, key=f"bp_{i}")
        with cols[1]:
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.1, key=f"rand_{i}")
        with cols[2]:
            integer = st.checkbox("Integer", key=f"int_{i}")

        breakpoints = []
        params = []

        for segment in range(num_breaks):
            st.markdown(f"**Segment {segment}**")

            cols = st.columns(3)

            with cols[0]:
                start = st.number_input(
                    "Start day", min_value=0, max_value=num_days,
                    value=segment * (num_days // num_breaks),
                    key=f"bp_{i}_{segment}"
                )

            with cols[1]:
                slope = st.number_input(
                    "Slope", value=0.0, key=f"slope_{i}_{segment}"
                )

            with cols[2]:
                intercept = st.number_input(
                    "Intercept", value=1000, key=f"intercept_{i}_{segment}"
                )
            
            breakpoints.append(start)
            params.append((slope, intercept))
        
        breakpoints.append(num_days)

        metrics_data.append({
            "name": name,
            "segments": breakpoints,
            "params": list(params),
            "randomness": randomness,
            "integer": integer,
        })

    base_values = np.zeros(num_days)
    noise = np.zeros(num_days)
    for i, (start, end) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        m, b = params[i]
        x = np.arange(end - start)
        base_values[start:end] = m * x + b
        noise[start:end] = np.random.randn(end - start) * randomness * b
    if integer:
        actual_values = np.round(base_values + noise)
    else:
        actual_values = base_values + noise

    metrics_dict[name] = actual_values.astype(float)

    # Plot
    st.altair_chart(
        alt.Chart(pd.DataFrame({"Date": dates, name: actual_values})).mark_line(point=True).encode(
            x="Date:T", y=alt.Y(f"{name}:Q", title=name)
        ).properties(title=name, width=700)
    )


# --- Calculated Metrics (Chained) ---
st.header("Calculated Metrics")

calc_metrics = []

for i in range(num_calc):
    with st.expander(f"Calculated Metric {i+1}", expanded=(i == 0)):
        name = st.text_input("Calc Metric Name", value=f"Calc_{i+1}", key=f"calc_name_{i}")
        cols = st.columns(3)
        with cols[0]:
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.05, key=f"calc_random_{i}")
        with cols[1]:
            integer = st.checkbox("Integer", key=f"calc_int_{i}")
        with cols[2]:
            num_breaks = st.number_input("# Breakpoints", min_value=1, value=1, key=f"calc_bp_{i}")

        segments = []
        breakpoints = []

        for j in range(num_breaks):
            st.markdown(f"**Segment {j}**")
            cols = st.columns([1, 4])
            start_day = cols[0].number_input("Start Day", min_value=0, max_value=num_days, value=j * (num_days // num_breaks), key=f"calc_start_{i}_{j}")
            formula = cols[1].text_input("Formula", value=f"{metrics_data[0]['name']} * 0.05", key=f"calc_formula_{i}_{j}")
            breakpoints.append(start_day)
            segments.append(formula)
        
        breakpoints.append(num_days)  # end after last segment

        calc_metrics.append({
            "name": name,
            "segments": breakpoints,
            "formulas": segments,
            "randomness": randomness,
            "integer": integer,
        })


# --- Dependency Graph and Topo Sort ---
def extract_deps(expr, known_names):
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
    return [t for t in tokens if t in known_names]

all_metrics = metrics_dict.copy()
dep_graph = defaultdict(list)
in_degree = defaultdict(int)
calc_by_name = {}

# Build graph
known_names = set(all_metrics.keys()) | {m["name"] for m in calc_metrics}
for metric in calc_metrics:
    name = metric["name"]
    all_deps = set()
    for formula in metric["formulas"]:
        deps = extract_deps(formula, known_names)
        all_deps.update(deps)

    calc_by_name[name] = metric
    for dep in all_deps:
        dep_graph[dep].append(name)
        in_degree[name] += 1

queue = deque([m["name"] for m in calc_metrics])
calc_order = []

while queue:
    current = queue.popleft()
    calc_order.append(current)
    for neighbor in dep_graph[current]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)


# Evaluate in order
days = np.arange(num_days)

for name in calc_order:
    metric = calc_by_name[name]
    values = np.zeros(num_days)
    try:
        for formula, start, end in zip(metric["formulas"], metric["segments"][:-1], metric["segments"][1:]):
            local_env = {"np": np, "days": days[start:end]}
            local_env.update({k: v[start:end] for k, v in all_metrics.items()})
            segment_values = eval(formula, {}, local_env)
            values[start:end] = segment_values

        std = np.std(values)
        noise = np.random.randn(len(values)) * metric["randomness"] * std
        final_result = np.round(values + noise) if metric["integer"] else values + noise

        all_metrics[name] = final_result

        st.altair_chart(
            alt.Chart(pd.DataFrame({"Date": dates, name: final_result})).mark_line(color="#9467bd").encode(
                x="Date:T", y=alt.Y(f"{name}:Q", title=name)
            ).properties(title=name, width=700)
        )
    except Exception as e:
        st.error(f"Error computing {name}: {e}")


# --- Show Final Combined Data ---
st.header("📋 Combined Data Output")
final_df = pd.DataFrame({"day": dates})
for key, values in all_metrics.items():
    final_df[key] = values

st.dataframe(final_df)

# Optional JSON download (convert Timestamp to str first)
json_ready_df = final_df.copy()
json_ready_df["day"] = json_ready_df["day"].astype(str)

st.download_button(
    "Download as JSON",
    json.dumps(json_ready_df.to_dict(orient="list"), indent=2),
    file_name="metrics.json",
    mime="application/json"
)

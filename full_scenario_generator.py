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

        breakdown_enabled = st.checkbox("Enable Breakdown", key=f"breakdown_enabled_{i}")
        breakdown_segments = []
        breakdown_weights = []
        breakdown_randomness = 0.0

        if breakdown_enabled:
            num_segments = st.number_input("Number of Segments", min_value=2, value=2, key=f"num_segments_{i}")
            for s in range(num_segments):
                seg_name = st.text_input(f"Segment {s+1} Name", key=f"segment_name_{i}_{s}")
                weight = st.number_input(f"Segment {s+1} % (total = 100)", min_value=0.0, max_value=100.0, value=100.0/num_segments, key=f"segment_weight_{i}_{s}")
                breakdown_segments.append(seg_name)
                breakdown_weights.append(weight)
            breakdown_randomness = st.slider("Breakdown Randomness", 0.0, 1.0, 0.1, key=f"breakdown_randomness_{i}")

            if abs(sum(breakdown_weights) - 100.0) > 1e-6:
                st.warning("⚠️ Segment weights do not sum to 100% exactly. This may skew results.")

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
            "breakdown": breakdown_enabled,
            "breakdown_segments": breakdown_segments,
        })

    base_values = np.zeros(num_days)
    noise = np.zeros(num_days)
    for j, (start, end) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        m, b = params[j]
        x = np.arange(end - start)
        base_values[start:end] = m * x + b
        noise[start:end] = np.random.randn(end - start) * randomness * b
    if integer:
        actual_values = np.round(base_values + noise)
    else:
        actual_values = base_values + noise

    metrics_dict[name] = actual_values.astype(float)

    # Plot total metric
    st.altair_chart(
        alt.Chart(pd.DataFrame({"Date": dates, name: actual_values})).mark_line(point=True).encode(
            x="Date:T", y=alt.Y(f"{name}:Q", title=name)
        ).properties(title=name, width=700)
    )

    # Generate breakdown segments
    if breakdown_enabled and breakdown_segments:
        weights = np.array(breakdown_weights)
        weights /= weights.sum()
        proportions = np.random.dirichlet(weights * (1.0 - breakdown_randomness) + 1.0, size=num_days)

        all_segment_data = []
        for j, seg_name in enumerate(breakdown_segments):
            seg_values = actual_values * proportions[:, j]
            seg_metric_name = f"{name}_{seg_name}"
            metrics_dict[seg_metric_name] = seg_values
            segment_df = pd.DataFrame({
                "Date": dates,
                "Segment": seg_name,
                "Value": seg_values
            })
            all_segment_data.append(segment_df)

        stacked_df = pd.concat(all_segment_data, ignore_index=True)

        # Plot breakdown
        st.altair_chart(
            alt.Chart(stacked_df).mark_area(opacity=0.7).encode(
                x="Date:T", y="Value:Q", color="Segment:N"
            ).properties(title=f"{name} Breakdown", width=700)
        )

# --- Calculated Metrics (Chained) ---
st.header("Calculated Metrics")

calc_metrics = []

for i in range(num_calc):
    with st.expander(f"Calculated Metric {i+1}", expanded=(i == 0)):
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Calc Metric Name", value=f"Calc_{i+1}", key=f"calc_name_{i}")
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.05, key=f"calc_random_{i}")
        with cols[1]:
            default_formula = next(iter(metrics_dict.keys()), "")
            formula = st.text_area("Formula (use base metric names)", value=default_formula, key=f"calc_formula_{i}")
            integer = st.checkbox("Integer", key=f"calc_int_{i}")

        num_formula_segments = st.number_input("# Formula Segments", min_value=1, value=1, key=f"calc_segments_{i}")

        formula_segments = []
        formula_breakpoints = []
        for s in range(num_formula_segments):
            st.markdown(f"**Segment {s+1}**")
            seg_cols = st.columns(2)
            with seg_cols[0]:
                start = st.number_input("Start Day", min_value=0, max_value=num_days, value=s * (num_days // num_formula_segments), key=f"calc_start_{i}_{s}")
            with seg_cols[1]:
                seg_formula = st.text_input("Formula", value=formula if s == 0 else "", key=f"calc_seg_formula_{i}_{s}")
            formula_breakpoints.append(start)
            formula_segments.append(seg_formula)

        formula_breakpoints.append(num_days)

        calc_metrics.append({
            "name": name,
            "formulas": formula_segments,
            "breakpoints": formula_breakpoints,
            "randomness": randomness,
            "integer": integer,
        })

# --- Process Calculated Metrics ---
if calc_metrics:
    # --- Dependency Graph and Topo Sort ---
    def extract_deps(expr, known_names):
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr)
        return [t for t in tokens if t in known_names]

    all_metrics = metrics_dict.copy()
    dep_graph = defaultdict(list)
    in_degree = defaultdict(int)
    calc_by_name = {}

    # Initialize all calculated metrics in in_degree with 0
    for metric in calc_metrics:
        in_degree[metric["name"]] = 0
        calc_by_name[metric["name"]] = metric

    # Build dependency graph
    known_names = set(all_metrics.keys()) | {m["name"] for m in calc_metrics}
    for metric in calc_metrics:
        name = metric["name"]
        all_deps = set()
        for formula in metric["formulas"]:
            if formula.strip():  # Only process non-empty formulas
                deps = extract_deps(formula, known_names)
                all_deps.update(deps)
        
        for dep in all_deps:
            if dep != name:  # Avoid self-dependency
                dep_graph[dep].append(name)
                in_degree[name] += 1

    # Topological sort
    queue = deque([name for name in calc_by_name.keys() if in_degree[name] == 0])
    calc_order = []

    while queue:
        current = queue.popleft()
        calc_order.append(current)
        for neighbor in dep_graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for circular dependencies
    if len(calc_order) != len(calc_metrics):
        st.error("Circular dependency detected in calculated metrics!")
        remaining = [name for name in calc_by_name.keys() if name not in calc_order]
        st.error(f"Could not process: {remaining}")
    else:
        # Evaluate in dependency order
        days = np.arange(num_days)
        for name in calc_order:
            metric = calc_by_name[name]
            result = np.zeros(num_days)
            try:
                for j, (start, end) in enumerate(zip(metric["breakpoints"][:-1], metric["breakpoints"][1:])):
                    formula = metric["formulas"][j]
                    if formula.strip():  # Only evaluate non-empty formulas
                        # Create a safe evaluation environment
                        safe_dict = {"np": np, "days": days}
                        safe_dict.update(all_metrics)
                        
                        local_result = eval(formula, {"__builtins__": {}}, safe_dict)
                        if np.isscalar(local_result):
                            result[start:end] = local_result
                        else:
                            result[start:end] = local_result[start:end]
                    else:
                        result[start:end] = 0  # Default to 0 for empty formulas

                # Add noise if specified
                if metric["randomness"] > 0:
                    std = np.std(result) if np.std(result) > 0 else 1.0
                    noise = np.random.randn(len(result)) * metric["randomness"] * std
                    result = result + noise

                # Convert to integer if specified
                final_result = np.round(result).astype(int) if metric["integer"] else result

                all_metrics[name] = final_result

                # Plot the calculated metric
                st.altair_chart(
                    alt.Chart(pd.DataFrame({"Date": dates, name: final_result})).mark_line(color="#9467bd").encode(
                        x="Date:T", y=alt.Y(f"{name}:Q", title=name)
                    ).properties(title=f"{name} (Calculated)", width=700)
                )
                
            except Exception as e:
                st.error(f"Error computing {name}: {e}")
                # Set to zeros if there's an error
                all_metrics[name] = np.zeros(num_days)
else:
    all_metrics = metrics_dict.copy()

# --- Show Final Combined Data ---
st.header("📋 Combined Data Output")
final_df = pd.DataFrame({"Date": dates})
for key, values in all_metrics.items():
    final_df[key] = values

st.dataframe(final_df)

# Show summary
st.subheader("Data Summary")
st.write(f"Total metrics generated: {len(all_metrics)}")
st.write(f"Base metrics: {len(metrics_dict)}")
st.write(f"Calculated metrics: {len(all_metrics) - len(metrics_dict)}")

# Copyable Python array
st.subheader("📎 Python-Friendly Output")
st.code(f"data = {final_df.to_dict(orient='list')}", language="python")

# Optional JSON download (convert Timestamp to str first)
json_ready_df = final_df.copy()
json_ready_df["Date"] = json_ready_df["Date"].astype(str)

st.download_button(
    "Download as JSON",
    json.dumps(json_ready_df.to_dict(orient="list"), indent=2),
    file_name="metrics.json",
    mime="application/json"
)
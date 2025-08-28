import streamlit as st
import pandas as pd
import altair as alt

def render_sidebar_settings():
    st.sidebar.header("General Settings")

    time_scale = st.sidebar.selectbox("Time Scale", ["Minutes", "Hours", "Days", "Weeks", "Months", "Years"], index=2,)
    base_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-04-12").date())
    start_date = pd.to_datetime(base_date)

    if time_scale in ["Hours", "Minutes"]:
        hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=0)
        start_date = start_date + pd.to_timedelta(hour, unit="h")

    if time_scale == "Minutes":
        minute = st.sidebar.number_input("Minute", min_value=0, max_value=59, value=0)
        start_date = start_date + pd.to_timedelta(minute, unit="m")

    # # Round to appropriate granularity
    # if time_scale == "Minutes":
    #     start_date = start_date.floor("T")
    # elif time_scale == "Hours":
    #     start_date = start_date.floor("H")
    # elif time_scale == "Days":
    #     start_date = start_date.floor("D")
    # elif time_scale == "Weeks":
    #     start_date = start_date.to_period("W").start_time
    # elif time_scale == "Months":
    #     start_date = start_date.to_period("M").start_time
    # elif time_scale == "Years":
    #     start_date = start_date.to_period("Y").start_time

    num_times = st.sidebar.number_input(
        f"Number of {time_scale.lower()}", min_value=1, value=10
    )
    num_metrics = st.sidebar.number_input("Number of Base Metrics", min_value=1, value=1)
    num_calc = st.sidebar.number_input("Number of Calculated Metrics", min_value=0, value=1, step=1)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    freq_map = {
        "Minutes": "T",
        "Hours": "H",
        "Days": "D",
        "Weeks": "W-MON",
        "Months": "MS",
        "Years": "YS",
    }
    frequency = freq_map[time_scale]

    return start_date, frequency, num_times, num_metrics, num_calc, seed

def render_base_metric_config(i, num_days):
    """Render configuration UI for a single base metric"""
    with st.expander(f"Metric {i+1}", expanded=(i == 0)):
        name = st.text_input("Metric Name", value=f"Metric_{i+1}", key=f"name_{i}")
        
        cols = st.columns(3)
        with cols[0]:
            num_breaks = st.number_input("# Breakpoints", min_value=1, value=1, key=f"bp_{i}")
        with cols[1]:
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.1, key=f"rand_{i}")
        with cols[2]:
            integer = st.checkbox("Integer", key=f"int_{i}")

        # Breakdown configuration
        breakdown_config = render_breakdown_config(i)
        
        # Breakpoint configuration
        breakpoints, params = render_breakpoint_config(i, num_breaks, num_days)
        
        return {
            "name": name,
            "segments": breakpoints,
            "params": params,
            "randomness": randomness,
            "integer": integer,
            "breakdown": breakdown_config
        }

def render_breakdown_config(i):
    """Render breakdown configuration for a metric"""
    breakdown_enabled = st.checkbox("Enable Breakdown", key=f"breakdown_enabled_{i}")
    
    if not breakdown_enabled:
        return {"enabled": False}
    
    num_segments = st.number_input("Number of Segments", min_value=2, value=2, key=f"num_segments_{i}")
    segments = []
    weights = []
    
    for s in range(num_segments):
        seg_name = st.text_input(f"Segment {s+1} Name", key=f"segment_name_{i}_{s}")
        weight = st.number_input(
            f"Segment {s+1} % (total = 100)", 
            min_value=0.0, max_value=100.0, 
            value=100.0/num_segments, 
            key=f"segment_weight_{i}_{s}"
        )
        segments.append(seg_name)
        weights.append(weight)
    
    breakdown_randomness = st.slider("Breakdown Randomness", 0.0, 1.0, 0.1, key=f"breakdown_randomness_{i}")
    
    if abs(sum(weights) - 100.0) > 1e-6:
        st.warning("⚠️ Segment weights do not sum to 100% exactly. This may skew results.")
    
    return {
        "enabled": True,
        "segments": segments,
        "weights": weights,
        "randomness": breakdown_randomness
    }

def render_breakpoint_config(i, num_breaks, num_days):
    """Render breakpoint configuration for a metric"""
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
            slope = st.number_input("Slope", value=0.0, key=f"slope_{i}_{segment}")
        
        with cols[2]:
            intercept = st.number_input("Intercept", value=1000, key=f"intercept_{i}_{segment}")
        
        breakpoints.append(start)
        params.append((slope, intercept))
    
    breakpoints.append(num_days)
    return breakpoints, params

def render_calculated_metric_config(i, num_days, available_metrics):
    """Render configuration UI for a calculated metric"""
    with st.expander(f"Calculated Metric {i+1}", expanded=(i == 0)):
        cols = st.columns(2)
        
        with cols[0]:
            name = st.text_input("Calc Metric Name", value=f"Calc_{i+1}", key=f"calc_name_{i}")
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.05, key=f"calc_random_{i}")
        
        with cols[1]:
            # Create a simple default formula that references an actual base metric
            default_formula = next(iter(available_metrics), "1000")
            formula = st.text_area(
                "Formula (use base metric names)", 
                value=default_formula, 
                key=f"calc_formula_{i}",
                help="Example: Metric_1 * 0.8 + 100"
            )
            integer = st.checkbox("Integer", key=f"calc_int_{i}")
        
        # Formula segments
        num_formula_segments = st.number_input("# Formula Segments", min_value=1, value=1, key=f"calc_segments_{i}")
        formula_segments, formula_breakpoints = render_formula_segments(i, num_formula_segments, num_days, formula)
        
        return {
            "name": name,
            "formulas": formula_segments,
            "breakpoints": formula_breakpoints,
            "randomness": randomness,
            "integer": integer,
        }

def render_formula_segments(i, num_segments, num_days, default_formula):
    """Render formula segment configuration"""
    formula_segments = []
    formula_breakpoints = []
    
    for s in range(num_segments):
        st.markdown(f"**Segment {s+1}**")
        seg_cols = st.columns(2)
        
        with seg_cols[0]:
            start = st.number_input(
                "Start Day", min_value=0, max_value=num_days, 
                value=s * (num_days // num_segments), 
                key=f"calc_start_{i}_{s}"
            )
        
        with seg_cols[1]:
            seg_formula = st.text_input(
                "Formula", 
                value=default_formula if s == 0 else "", 
                key=f"calc_seg_formula_{i}_{s}"
            )
        
        formula_breakpoints.append(start)
        formula_segments.append(seg_formula)
    
    formula_breakpoints.append(num_days)
    return formula_segments, formula_breakpoints

def plot_metric(dates, name, values, color="steelblue", is_calculated=False):
    """Plot a single metric"""
    title_suffix = " (Calculated)" if is_calculated else ""
    chart = alt.Chart(pd.DataFrame({"Date": dates, name: values})).mark_line(
        point=True, color=color
    ).encode(
        x="Date:T", 
        y=alt.Y(f"{name}:Q", title=name)
    ).properties(
        title=f"{name}{title_suffix}", 
        width=700
    )
    
    st.altair_chart(chart)

def plot_breakdown(dates, name, segment_data):
    """Plot breakdown segments as stacked area chart"""
    all_segment_data = []
    for seg_name, seg_values in segment_data.items():
        segment_df = pd.DataFrame({
            "Date": dates,
            "Segment": seg_name,
            "Value": seg_values
        })
        all_segment_data.append(segment_df)
    
    if all_segment_data:
        stacked_df = pd.concat(all_segment_data, ignore_index=True)
        chart = alt.Chart(stacked_df).mark_area(opacity=0.7).encode(
            x="Date:T", 
            y="Value:Q", 
            color="Segment:N"
        ).properties(
            title=f"{name} Breakdown", 
            width=700
        )
        
        st.altair_chart(chart)

def render_data_output(final_df, all_metrics, base_metrics_count):
    """Render the final data output section"""
    st.header("📋 Combined Data Output")
    st.dataframe(final_df)
    
    # Show summary
    st.subheader("Data Summary")
    st.write(f"Total metrics generated: {len(all_metrics)}")
    st.write(f"Base metrics: {base_metrics_count}")
    st.write(f"Calculated metrics: {len(all_metrics) - base_metrics_count}")
    
    # Copyable Python array
    st.subheader("📎 Python-Friendly Output")
    st.code(f"data = {final_df.to_dict(orient='list')}", language="python")
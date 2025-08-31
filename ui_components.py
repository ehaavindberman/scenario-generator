import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

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

def render_base_metric_config(i, num_times, breakdown_manager):
    """Render configuration UI for a single base metric"""
    with st.expander(f"Base Metric {i+1}", expanded=(i == 0)):
        # Basic configuration
        cols = st.columns([2, 1, 1])
        with cols[0]:
            name = st.text_input("Metric Name", value=f"Metric_{i+1}", key=f"name_{i}")
        with cols[1]:
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.1, key=f"rand_{i}")
        with cols[2]:
            integer = st.checkbox("Integer", key=f"int_{i}")
        
        # Breakdown configuration
        breakdown_config = breakdown_manager.render_metric_breakdown_selector(name, f"base_{i}")
        
        # Breakpoint configuration
        num_breaks = st.number_input("# Breakpoints", min_value=1, value=1, key=f"bp_{i}")
        breakpoints, params = render_breakpoint_config(i, num_breaks, num_times)
        
        return {
            "name": name,
            "segments": breakpoints,
            "params": params,
            "randomness": randomness,
            "integer": integer,
            "breakdown": breakdown_config
        }

def render_breakpoint_config(i, num_breaks, num_times):
    """Render breakpoint configuration for a metric"""
    breakpoints = []
    params = []
    
    for segment in range(num_breaks):
        st.markdown(f"**Segment {segment + 1}**")
        cols = st.columns(3)
        
        with cols[0]:
            start = st.number_input(
                "Start day", min_value=0, max_value=num_times,
                value=segment * (num_times // num_breaks),
                key=f"bp_{i}_{segment}"
            )
        
        with cols[1]:
            slope = st.number_input("Slope", value=0.0, key=f"slope_{i}_{segment}")
        
        with cols[2]:
            intercept = st.number_input("Intercept", value=1000, key=f"intercept_{i}_{segment}")
        
        breakpoints.append(start)
        params.append((slope, intercept))
    
    breakpoints.append(num_times)
    return breakpoints, params

def render_calculated_metric_config(i, num_times, available_metrics, breakdown_manager):
    """Render configuration UI for a calculated metric"""
    with st.expander(f"Calculated Metric {i+1}", expanded=(i == 0)):
        # Basic configuration
        cols = st.columns([2, 1, 1])
        with cols[0]:
            name = st.text_input("Calc Metric Name", value=f"Calc_{i+1}", key=f"calc_name_{i}")
        with cols[1]:
            randomness = st.slider("Randomness Scale", 0.0, 1.0, 0.05, key=f"calc_random_{i}")
        with cols[2]:
            integer = st.checkbox("Integer", key=f"calc_int_{i}")
        
        # Show available metrics for reference
        if available_metrics:
            with st.expander("Available Metrics for Formulas", expanded=False):
                st.write("You can use these metric names in your formulas:")
                metric_list = list(available_metrics)
                # Display in columns for better readability
                cols = st.columns(3)
                for idx, metric in enumerate(metric_list):
                    with cols[idx % 3]:
                        st.code(metric)
        
        # Breakdown configuration
        breakdown_config = breakdown_manager.render_metric_breakdown_selector(name, f"calc_{i}")
        
        # Formula segments
        num_formula_segments = st.number_input("# Formula Segments", min_value=1, value=1, key=f"calc_segments_{i}")
        
        # Create a better default formula that references an actual base metric
        default_formula = next(iter(available_metrics), "1000")
        if default_formula != "1000":
            default_formula = f"{default_formula} * 0.8"
        
        formula_segments, formula_breakpoints = render_formula_segments(
            i, num_formula_segments, num_times, default_formula
        )
        
        return {
            "name": name,
            "formulas": formula_segments,
            "breakpoints": formula_breakpoints,
            "randomness": randomness,
            "integer": integer,
            "breakdown": breakdown_config
        }

def render_formula_segments(i, num_segments, num_times, default_formula):
    """Render formula segment configuration"""
    formula_segments = []
    formula_breakpoints = []
    
    for s in range(num_segments):
        st.markdown(f"**Formula Segment {s+1}**")
        seg_cols = st.columns([1, 3])
        
        with seg_cols[0]:
            start = st.number_input(
                "Start Time", min_value=0, max_value=num_times, 
                value=s * (num_times // num_segments), 
                key=f"calc_start_{i}_{s}"
            )
        
        with seg_cols[1]:
            seg_formula = st.text_area(
                "Formula", 
                value=default_formula if s == 0 else "", 
                key=f"calc_seg_formula_{i}_{s}",
                height=60,
                help="Example: Metric_1 * 0.8 + np.sin(times/10) * 100"
            )
        
        formula_breakpoints.append(start)
        formula_segments.append(seg_formula)
    
    formula_breakpoints.append(num_times)
    return formula_segments, formula_breakpoints

def plot_metric(dates, name, values, color="steelblue", is_calculated=False):
    """Plot a single metric"""
    title_suffix = " (Calculated)" if is_calculated else ""
    chart_data = pd.DataFrame({"Date": dates, "Value": values})
    
    chart = alt.Chart(chart_data).mark_line(
        point=True, color=color, strokeWidth=2
    ).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title=name),
        tooltip=["Date:T", "Value:Q"]
    ).properties(
        title=f"{name}{title_suffix}", 
        width=700,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)

def plot_breakdown(dates, metric_name, segment_data):
    """Plot breakdown segments as stacked area chart and individual line charts"""
    if not segment_data:
        return
    
    # Stacked area chart
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
        
        # Stacked area chart
        stacked_chart = alt.Chart(stacked_df).mark_area(opacity=0.7).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Value", stack="zero"),
            color=alt.Color("Segment:N", scale=alt.Scale(scheme="category20")),
            tooltip=["Date:T", "Segment:N", "Value:Q"]
        ).properties(
            title=f"{metric_name} - Breakdown (Stacked)", 
            width=700,
            height=300
        )
        
        st.altair_chart(stacked_chart, use_container_width=True)
        
        # Individual line charts for each segment
        with st.expander(f"Individual Breakdown Charts for {metric_name}", expanded=False):
            for seg_name, seg_values in segment_data.items():
                segment_chart_data = pd.DataFrame({
                    "Date": dates,
                    "Value": seg_values
                })
                
                segment_chart = alt.Chart(segment_chart_data).mark_line(
                    point=True, strokeWidth=2
                ).encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Value:Q", title=seg_name),
                    tooltip=["Date:T", "Value:Q"]
                ).properties(
                    title=f"{metric_name} - {seg_name}",
                    width=350,
                    height=200
                )
                
                st.altair_chart(segment_chart, use_container_width=True)

def render_data_output(final_df, all_metrics, base_metrics_count, breakdown_metrics_count):
    """Render the final data output section"""
    st.header("📋 Combined Data Output")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Metrics", len(all_metrics))
    with col2:
        st.metric("Base Metrics", base_metrics_count)
    with col3:
        st.metric("Calculated Metrics", len(all_metrics) - base_metrics_count - breakdown_metrics_count)
    with col4:
        st.metric("Breakdown Segments", breakdown_metrics_count)
    
    # Data preview with filtering
    st.subheader("📊 Data Preview")
    
    # Column selector for large datasets
    all_columns = list(final_df.columns)
    if len(all_columns) > 10:
        with st.expander("Select Columns to Display", expanded=False):
            selected_columns = st.multiselect(
                "Choose columns to show in preview",
                all_columns,
                default=all_columns[:10]  # Show first 10 by default
            )
        
        if selected_columns:
            display_df = final_df[["Date"] + [col for col in selected_columns if col != "Date"]]
        else:
            display_df = final_df.iloc[:, :11]  # Show first 11 columns if none selected
    else:
        display_df = final_df
    
    st.dataframe(display_df, use_container_width=True)
    
    # Data info
    with st.expander("📈 Data Information", expanded=False):
        st.write("**Dataset Shape:**", final_df.shape)
        st.write("**Date Range:**", f"{final_df['Date'].min()} to {final_df['Date'].max()}")
        
        # Basic statistics for numeric columns
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Basic Statistics:**")
            st.dataframe(final_df[numeric_cols].describe())
    
    # Copyable Python dictionary
    with st.expander("🐍 Python-Friendly Output", expanded=False):
        st.code(f"data = {final_df.to_dict(orient='list')}", language="python")
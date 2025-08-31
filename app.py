import streamlit as st
import pandas as pd
import numpy as np
from metric_generator import MetricGenerator, CalculatedMetricProcessor, BreakdownProcessor
from breakdown_manager import BreakdownManager
from ui_components import (
    render_sidebar_settings,
    render_base_metric_config,
    render_calculated_metric_config,
    plot_metric,
    plot_breakdown,
    render_data_output
)
from data_exporter import create_export_section, create_data_preview

def main():
    st.set_page_config(
        page_title="FyreDrill Scenario Generator",
        page_icon="🔥",
        layout="wide"
    )
    
    st.title("🔥 FyreDrill Scenario Generator")
    st.markdown("Generate realistic datasets with customizable base and calculated metrics, plus flexible breakdowns.")
    
    # Initialize breakdown manager
    breakdown_manager = BreakdownManager()
    
    # Sidebar settings
    start_date, frequency, num_times, num_metrics, num_calc, seed = render_sidebar_settings()

    # Generate date range
    base_date = pd.Timestamp(start_date)
    dates = pd.date_range(base_date, periods=num_times, freq=frequency)
    
    # Initialize generators
    generator = MetricGenerator(num_times, seed)
    breakdown_processor = BreakdownProcessor(breakdown_manager, seed)
    
    # --- Breakdown Management Section ---
    breakdown_manager.render_breakdown_management_ui()
    
    # Show current breakdown summary
    if breakdown_manager.breakdown_definitions:
        with st.expander("📊 Current Breakdown Definitions Summary", expanded=False):
            st.markdown(breakdown_manager.get_breakdown_info())
    
    st.divider()
    
    # --- Base Metrics Section ---
    st.header("📈 Base Metrics")
    
    base_metrics = {}
    base_configs = []
    breakdown_configs = {}
    
    for i in range(num_metrics):
        config = render_base_metric_config(i, num_times, breakdown_manager)
        base_configs.append(config)
        
        # Store breakdown config for later processing
        breakdown_configs[config["name"]] = config["breakdown"]
        
        # Generate the base metric
        values = generator.generate_base_metric(
            config["name"],
            config["segments"],
            config["params"],
            config["randomness"],
            config["integer"]
        )
        
        base_metrics[config["name"]] = values
        
        # Plot the metric
        plot_metric(dates, config["name"], values)
    
    # Apply breakdowns to base metrics
    base_metrics_with_breakdowns = breakdown_processor.apply_breakdowns_to_metrics(
        base_metrics, breakdown_configs
    )
    
    # Plot breakdown visualizations for base metrics
    for config in base_configs:
        if config["breakdown"].get("enabled", False) and config["breakdown"].get("selected"):
            # Extract segments for this metric
            metric_segments = {}
            for key, values in base_metrics_with_breakdowns.items():
                if key.startswith(f"{config['name']}_") and key != config["name"]:
                    segment_name = key[len(config["name"])+1:]  # Remove "MetricName_" prefix
                    metric_segments[segment_name] = values
            
            if metric_segments:
                plot_breakdown(dates, config["name"], metric_segments)
    
    # --- Calculated Metrics Section ---
    st.header("🧮 Calculated Metrics")
    
    calculated_metrics = {}
    calculated_configs = []
    
    if num_calc > 0:
        # Get all available metrics for calculated metric formulas
        available_metrics = list(base_metrics_with_breakdowns.keys())
        
        for i in range(num_calc):
            config = render_calculated_metric_config(i, num_times, available_metrics, breakdown_manager)
            calculated_configs.append(config)
            
            # Store breakdown config for calculated metrics
            breakdown_configs[config["name"]] = config["breakdown"]
        
        # Process calculated metrics
        try:
            processor = CalculatedMetricProcessor(base_metrics_with_breakdowns)
            calculated_metrics = processor.process_calculated_metrics(calculated_configs, num_times)
            
            # Plot calculated metrics
            for name, values in calculated_metrics.items():
                plot_metric(dates, name, values, color="#9467bd", is_calculated=True)
                
        except Exception as e:
            st.error(f"Error processing calculated metrics: {e}")
            st.info("Check your formulas and dependencies. Make sure all referenced metrics exist.")
            # Show available metrics for debugging
            with st.expander("Available Metrics for Debugging"):
                st.write("Available metrics for formulas:")
                for metric in available_metrics:
                    st.code(metric)
    
    # Apply breakdowns to calculated metrics
    calculated_with_breakdowns = breakdown_processor.apply_breakdowns_to_metrics(
        calculated_metrics, 
        {config["name"]: config["breakdown"] for config in calculated_configs}
    )
    
    # Plot breakdown visualizations for calculated metrics
    for config in calculated_configs:
        if config["breakdown"].get("enabled", False) and config["breakdown"].get("selected"):
            # Extract segments for this calculated metric
            metric_segments = {}
            for key, values in calculated_with_breakdowns.items():
                if key.startswith(f"{config['name']}_") and key != config["name"]:
                    segment_name = key[len(config["name"])+1:]  # Remove "MetricName_" prefix
                    metric_segments[segment_name] = values
            
            if metric_segments:
                plot_breakdown(dates, config["name"], metric_segments)
    
    # Combine all metrics
    all_metrics = {**base_metrics_with_breakdowns, **calculated_with_breakdowns}
    
    # Create final DataFrame
    final_df = pd.DataFrame({"Date": dates})
    for key, values in all_metrics.items():
        final_df[key] = values
    
    # Count breakdown segments
    breakdown_segments_count = len(all_metrics) - len(base_metrics) - len(calculated_metrics)
    
    # --- Data Output Section ---
    render_data_output(final_df, all_metrics, len(base_metrics), breakdown_segments_count)
    
    # --- Export Section ---
    create_export_section(final_df)
    
    # --- Data Preview Section ---
    create_data_preview(final_df)
    
    # --- Configuration Summary ---
    with st.expander("📋 Configuration Summary"):
        st.write("**Base Metrics:**")
        for config in base_configs:
            breakdown_info = ""
            if config["breakdown"].get("enabled", False):
                selected_breakdowns = config["breakdown"].get("selected", [])
                if selected_breakdowns:
                    breakdown_info = f" | Breakdowns: {', '.join(selected_breakdowns)}"
            
            st.write(f"- {config['name']}: {len(config['segments'])-1} segments, "
                    f"randomness={config['randomness']}, integer={config['integer']}{breakdown_info}")
        
        if calculated_configs:
            st.write("**Calculated Metrics:**")
            for config in calculated_configs:
                breakdown_info = ""
                if config["breakdown"].get("enabled", False):
                    selected_breakdowns = config["breakdown"].get("selected", [])
                    if selected_breakdowns:
                        breakdown_info = f" | Breakdowns: {', '.join(selected_breakdowns)}"
                
                st.write(f"- {config['name']}: {len(config['formulas'])} formula segments, "
                        f"randomness={config['randomness']}, integer={config['integer']}{breakdown_info}")
        
        # Breakdown summary
        if breakdown_manager.breakdown_definitions:
            st.write("**Global Breakdown Definitions:**")
            for breakdown_name, segments in breakdown_manager.breakdown_definitions.items():
                st.write(f"- {breakdown_name}: {len(segments)} segments ({', '.join(segments)})")
        
        # Final counts
        st.write("**Final Dataset:**")
        st.write(f"- Total columns: {len(all_metrics) + 1}")  # +1 for Date column
        st.write(f"- Base metrics: {len(base_metrics)}")
        st.write(f"- Calculated metrics: {len(calculated_metrics)}")
        st.write(f"- Breakdown segments: {breakdown_segments_count}")
        st.write(f"- Time periods: {num_times}")

if __name__ == "__main__":
    main()
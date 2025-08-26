import streamlit as st
import pandas as pd
import numpy as np
from metric_generator import MetricGenerator, CalculatedMetricProcessor
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
    st.markdown("Generate realistic datasets with customizable base and calculated metrics.")
    
    # Sidebar settings
    num_days, num_metrics, num_calc, seed = render_sidebar_settings()
    
    # Generate date range
    base_date = pd.to_datetime("2025-05-13")
    dates = pd.date_range(base_date, periods=num_days)
    
    # Initialize metric generator
    generator = MetricGenerator(num_days, seed)
    
    # --- Base Metrics Section ---
    st.header("📈 Base Metrics")
    
    base_metrics = {}
    base_metrics_configs = []
    
    for i in range(num_metrics):
        config = render_base_metric_config(i, num_days)
        base_metrics_configs.append(config)
        
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
        
        # Handle breakdown segments
        if config["breakdown"]["enabled"]:
            segment_data = generator.generate_breakdown_segments(
                values,
                config["breakdown"]["segments"],
                config["breakdown"]["weights"],
                config["breakdown"]["randomness"]
            )
            
            # Add segments to base_metrics with naming convention
            for seg_name, seg_values in segment_data.items():
                full_seg_name = f"{config['name']}_{seg_name}"
                base_metrics[full_seg_name] = seg_values
            
            # Plot breakdown
            plot_breakdown(dates, config["name"], segment_data)
    
    # --- Calculated Metrics Section ---
    st.header("🧮 Calculated Metrics")
    
    calculated_metrics = {}
    calc_metrics_configs = []
    
    if num_calc > 0:
        for i in range(num_calc):
            config = render_calculated_metric_config(i, num_days, base_metrics.keys())
            calc_metrics_configs.append(config)
        
        # Process calculated metrics
        try:
            processor = CalculatedMetricProcessor(base_metrics)
            calculated_metrics = processor.process_calculated_metrics(calc_metrics_configs, num_days)
            
            # Plot calculated metrics
            for name, values in calculated_metrics.items():
                plot_metric(dates, name, values, color="#9467bd", is_calculated=True)
                
        except Exception as e:
            st.error(f"Error processing calculated metrics: {e}")
            st.info("Check your formulas and dependencies. Make sure all referenced metrics exist.")
    
    # Combine all metrics
    all_metrics = {**base_metrics, **calculated_metrics}
    
    # Create final DataFrame
    final_df = pd.DataFrame({"Date": dates})
    for key, values in all_metrics.items():
        final_df[key] = values
    
    # --- Data Output Section ---
    render_data_output(final_df, all_metrics, len(base_metrics))
    
    # --- Export Section ---
    create_export_section(final_df)
    
    # --- Data Preview Section ---
    create_data_preview(final_df)
    
    # --- Configuration Summary ---
    with st.expander("📋 Configuration Summary"):
        st.write("**Base Metrics:**")
        for config in base_metrics_configs:
            st.write(f"- {config['name']}: {len(config['segments'])-1} segments, "
                    f"randomness={config['randomness']}, integer={config['integer']}")
            if config["breakdown"]["enabled"]:
                st.write(f"  └─ Breakdown: {', '.join(config['breakdown']['segments'])}")
        
        if calc_metrics_configs:
            st.write("**Calculated Metrics:**")
            for config in calc_metrics_configs:
                st.write(f"- {config['name']}: {len(config['formulas'])} formula segments, "
                        f"randomness={config['randomness']}, integer={config['integer']}")

if __name__ == "__main__":
    main()
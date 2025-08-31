import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json

class BreakdownManager:
    def __init__(self):
        # Initialize session state for breakdown definitions if not exists
        if "global_breakdown_definitions" not in st.session_state:
            st.session_state["global_breakdown_definitions"] = {
                "Device": ["mobile", "desktop", "tablet"],
                "Channel": ["organic", "paid", "social"],
                "Geography": ["US", "Europe", "Asia"]
            }
    
    @property
    def breakdown_definitions(self) -> Dict[str, List[str]]:
        """Get current breakdown definitions from session state"""
        return st.session_state["global_breakdown_definitions"]
    
    def render_breakdown_management_ui(self):
        """Render the breakdown definition management interface"""
        st.subheader("🔧 Breakdown Management")
        
        with st.expander("Manage Breakdown Definitions", expanded=False):
            st.markdown("Define the available breakdown categories and their segments. Weights are set per metric.")
            
            # Add new breakdown
            col1, col2 = st.columns([3, 1])
            with col1:
                new_breakdown = st.text_input("New breakdown name", key="new_breakdown_input")
            with col2:
                if st.button("Add Breakdown") and new_breakdown:
                    if new_breakdown not in self.breakdown_definitions:
                        st.session_state["global_breakdown_definitions"][new_breakdown] = []
                        st.rerun()
            
            # Manage existing breakdowns
            breakdowns_to_delete = []
            for breakdown_name, segments in self.breakdown_definitions.items():
                with st.container():
                    st.markdown(f"**📊 {breakdown_name}**")
                    
                    # Delete breakdown button
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button(f"🗑️", key=f"delete_breakdown_{breakdown_name}", help=f"Delete {breakdown_name}"):
                            breakdowns_to_delete.append(breakdown_name)
                            continue
                    
                    # Add new segment
                    with col1:
                        new_segment = st.text_input(f"New segment for {breakdown_name}", 
                                                   key=f"new_segment_{breakdown_name}")
                        if st.button(f"Add Segment", key=f"add_segment_{breakdown_name}") and new_segment:
                            if new_segment not in segments:
                                st.session_state["global_breakdown_definitions"][breakdown_name].append(new_segment)
                                st.rerun()
                    
                    # Display and manage existing segments
                    if segments:
                        segments_to_delete = []
                        st.write(f"Current segments:")
                        
                        # Display segments in a more compact format
                        for i, segment in enumerate(segments):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(f"• {segment}")
                            with col2:
                                if st.button("🗑️", key=f"delete_segment_{breakdown_name}_{segment}", help=f"Delete {segment}"):
                                    segments_to_delete.append(segment)
                        
                        # Delete marked segments
                        for segment in segments_to_delete:
                            st.session_state["global_breakdown_definitions"][breakdown_name].remove(segment)
                            st.rerun()
                    else:
                        st.info(f"No segments defined for {breakdown_name}")
                    
                    st.divider()
            
            # Delete marked breakdowns
            for breakdown_name in breakdowns_to_delete:
                del st.session_state["global_breakdown_definitions"][breakdown_name]
                st.rerun()
            
            # Export/Import breakdown definitions
            st.markdown("**Export/Import Breakdown Definitions**")
            col1, col2 = st.columns(2)
            
            with col1:
                breakdown_json = json.dumps(self.breakdown_definitions, indent=2)
                st.download_button(
                    "📥 Export Definitions",
                    breakdown_json,
                    file_name="breakdown_definitions.json",
                    mime="application/json"
                )
            
            with col2:
                uploaded_file = st.file_uploader("📤 Import Definitions", type=['json'])
                if uploaded_file is not None:
                    try:
                        imported_definitions = json.load(uploaded_file)
                        st.session_state["global_breakdown_definitions"].update(imported_definitions)
                        st.success("Breakdown definitions imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing definitions: {e}")
    
    def render_metric_breakdown_selector(self, metric_name: str, key_prefix: str) -> Dict[str, Any]:
        """Render breakdown selector for a specific metric with per-metric weights"""
        if not self.breakdown_definitions:
            return {"enabled": False, "selected": {}}
        
        enabled = st.checkbox(f"Enable Breakdowns for {metric_name}", key=f"{key_prefix}_breakdown_enabled")
        
        if not enabled:
            return {"enabled": False, "selected": {}}
        
        # Select which breakdowns to apply
        available_breakdowns = list(self.breakdown_definitions.keys())
        selected_breakdown_names = st.multiselect(
            f"Select Breakdowns for {metric_name}",
            available_breakdowns,
            key=f"{key_prefix}_breakdown_select"
        )
        
        selected_breakdowns = {}
        
        # Configure weights for each selected breakdown
        for breakdown_name in selected_breakdown_names:
            segments = self.breakdown_definitions[breakdown_name]
            if not segments:
                continue
                
            st.markdown(f"**Weights for {breakdown_name}:**")
            
            segment_weights = {}
            total_weight = 0
            
            # Create weight inputs for each segment
            cols = st.columns(min(len(segments), 3))  # Max 3 columns for layout
            for i, segment in enumerate(segments):
                with cols[i % len(cols)]:
                    weight = st.number_input(
                        f"{segment}",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0 / len(segments),  # Equal distribution by default
                        step=1.0,
                        key=f"{key_prefix}_{breakdown_name}_{segment}_weight"
                    )
                    segment_weights[segment] = weight
                    total_weight += weight
            
            # Show total and normalization info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total: {total_weight:.1f}%")
            with col2:
                if abs(total_weight - 100.0) > 0.1:
                    st.warning("⚠️ Weights don't sum to 100%")
            
            # Normalize weights to sum to 1.0
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in segment_weights.items()}
            else:
                normalized_weights = {k: 1.0/len(segments) for k in segment_weights.keys()}
            
            selected_breakdowns[breakdown_name] = {
                "segments": segments,
                "weights": normalized_weights
            }
        
        # Global randomness for breakdown generation
        breakdown_randomness = st.slider(
            f"Breakdown Randomness for {metric_name}",
            0.0, 1.0, 0.1,
            key=f"{key_prefix}_breakdown_randomness",
            help="Higher values make breakdown distributions more variable over time"
        )
        
        return {
            "enabled": True,
            "selected": selected_breakdowns,
            "randomness": breakdown_randomness
        }
    
    def generate_breakdown_segments(self, base_values: np.ndarray, breakdown_config: Dict[str, Dict], 
                                  randomness: float, seed: int) -> Dict[str, np.ndarray]:
        """Generate breakdown segments for multiple breakdowns applied to a metric"""
        if not breakdown_config:
            return {}
        
        np.random.seed(seed)
        all_segments = {}
        
        # Generate segments for each breakdown independently
        for breakdown_name, config in breakdown_config.items():
            segments = config["segments"]
            weights = config["weights"]
            
            if not segments:
                continue
            
            # Convert weights to array
            segment_names = list(segments)
            weights_array = np.array([weights.get(seg, 0.0) for seg in segment_names])
            
            if weights_array.sum() == 0:
                weights_array = np.ones_like(weights_array) / len(weights_array)
            else:
                weights_array = weights_array / weights_array.sum()
            
            # Generate proportions using Dirichlet distribution for time variation
            alpha = weights_array * (1.0 - randomness) * 100 + 1.0
            proportions = np.random.dirichlet(alpha, size=len(base_values))
            
            # Generate segments
            for j, segment_name in enumerate(segment_names):
                segment_key = f"{breakdown_name}_{segment_name}"
                seg_values = base_values * proportions[:, j]
                all_segments[segment_key] = seg_values
        
        return all_segments
    
    def get_breakdown_info(self) -> str:
        """Get a summary of current breakdown definitions for display"""
        if not self.breakdown_definitions:
            return "No breakdown definitions"
        
        info_lines = []
        for breakdown_name, segments in self.breakdown_definitions.items():
            segment_count = len(segments)
            segment_preview = segments[:3] if segments else []
            preview_text = ', '.join(segment_preview)
            if segment_count > 3:
                preview_text += '...'
            info_lines.append(f"**{breakdown_name}**: {segment_count} segments ({preview_text})")
        
        return "\n".join(info_lines)
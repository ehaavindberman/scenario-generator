import json
import streamlit as st
import pandas as pd

def export_to_json(df, filename="metrics.json"):
    """Export dataframe to JSON with download button"""
    # Convert Timestamp to string for JSON serialization
    json_ready_df = df.copy()
    json_ready_df["Date"] = json_ready_df["Date"].astype(str)
    
    json_data = json.dumps(json_ready_df.to_dict(orient="list"), indent=2)
    
    st.download_button(
        "Download as JSON",
        json_data,
        file_name=filename,
        mime="application/json"
    )

def export_to_csv(df, filename="metrics.csv"):
    """Export dataframe to CSV with download button"""
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        "Download as CSV",
        csv_data,
        file_name=filename,
        mime="text/csv"
    )

def create_export_section(df):
    """Create export section with multiple format options"""
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_to_json(df)
    
    with col2:
        export_to_csv(df)
    
    # Optional: Excel export if openpyxl is available
    try:
        import io
        
        def export_to_excel(df, filename="metrics.xlsx"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Metrics', index=False)
            
            st.download_button(
                "Download as Excel",
                buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with st.container():
            export_to_excel(df)
            
    except ImportError:
        st.info("Install openpyxl for Excel export functionality")

def create_data_preview(df, num_rows=5):
    """Create a data preview section"""
    st.subheader("📊 Data Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First few rows:**")
        st.dataframe(df.head(num_rows))
    
    with col2:
        st.write("**Data types and info:**")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null': [df[col].count() for col in df.columns]
        })
        st.dataframe(info_df)
    
    # Basic statistics
    st.write("**Basic Statistics:**")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())
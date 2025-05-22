#!/usr/bin/env python3
"""
Token-Economics PoC - Streamlit Dashboard
Visualizes token usage and cost data from runs of the driver script.
"""

import os
import json
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure the Streamlit page
st.set_page_config(
    page_title="Token Economics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_run_files() -> List[Path]:
    """Get list of available run files"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return []
    
    excel_files = list(runs_dir.glob("*.xlsx"))
    # Sort by modification time (newest first)
    excel_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return excel_files

def load_run_data(file_path: Path) -> Tuple[pd.DataFrame, str, bool]:
    """Load data from an Excel file"""
    # Read the Excel file
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_name = xls.sheet_names[0]  # Get the first sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    
    # Parse JSON responses if they're in string format
    if "json_response" in df.columns:
        df["json_response"] = df["json_response"].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x
        )
    
    # Check if this is a combined file (has transcript column)
    is_combined = "transcript" in df.columns
    
    return df, sheet_name, is_combined

def format_cost(cost: float) -> str:
    """Format cost value for display"""
    return f"${cost:.6f}"

def create_horizontal_bar_chart(data, x_field, y_field, title, color_scheme="blues"):
    """Create a horizontal bar chart using Altair"""
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y(y_field, sort=None, title=None),
        x=alt.X(x_field, title=None),
        tooltip=[y_field, x_field]
    ).properties(
        title=title,
        height=len(data) * 40  # Dynamic height based on number of bars
    )
    
    if color_scheme:
        chart = chart.encode(color=alt.Color(y_field, legend=None, scale=alt.Scale(scheme=color_scheme)))
    
    return chart

def main():
    """Main Streamlit app function"""
    st.title("Token Economics Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # File selection
    run_files = get_run_files()
    
    if not run_files:
        st.warning("No run files found. Please run driver.py first to generate data.")
        return
    
    # Option to upload a file
    uploaded_file = st.sidebar.file_uploader("Or upload an Excel file", type=["xlsx"])
    
    if uploaded_file:
        # Use the uploaded file
        df, run_id, is_combined = load_run_data(uploaded_file)
        file_name = uploaded_file.name
    else:
        # Use the selected file from the list
        file_options = {f.name: f for f in run_files}
        selected_filename = st.sidebar.selectbox(
            "Select a run file", 
            options=list(file_options.keys()),
            index=0
        )
        
        selected_file = file_options[selected_filename]
        df, run_id, is_combined = load_run_data(selected_file)
        file_name = selected_file.name
    
    # Check if this is a combined file and handle accordingly
    if is_combined:
        st.sidebar.info("📊 Combined results file detected. Select a transcript to filter results.")
        
        # Get unique transcripts
        transcripts = df["transcript"].unique().tolist()
        
        # Option to filter by transcript or view all
        transcript_options = ["All Transcripts"] + transcripts
        selected_transcript = st.sidebar.selectbox(
            "Filter by transcript",
            options=transcript_options,
            index=0
        )
        
        # Filter by selected transcript if not "All Transcripts"
        if selected_transcript != "All Transcripts":
            df = df[df["transcript"] == selected_transcript]
            st.subheader(f"Results for transcript: {selected_transcript}")
    
    # Column visibility
    st.sidebar.subheader("Column Visibility")
    
    # Determine which columns to show
    all_columns = df.columns.tolist()
    essential_columns = ["model", "status", "total_tokens", "total_cost_usd"]
    
    # If combined file, add transcript to essential columns
    if is_combined and selected_transcript == "All Transcripts":
        essential_columns.insert(0, "transcript")
    
    optional_columns = [col for col in all_columns if col not in essential_columns and col != "json_response"]
    
    # Always show essential columns
    columns_to_show = essential_columns.copy()
    
    # Let user select which optional columns to show
    for col in optional_columns:
        if st.sidebar.checkbox(f"Show {col}", value=True):
            columns_to_show.append(col)
    
    # Option to show JSON responses
    show_json = st.sidebar.checkbox("Show JSON responses", value=False)
    if show_json:
        columns_to_show.append("json_response")
    
    # Filter the DataFrame
    display_df = df[columns_to_show].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if col.endswith("_cost_usd"):
            display_df[col] = display_df[col].apply(format_cost)
    
    # Main view
    st.subheader(f"Results from: {file_name}")
    
    # Display the DataFrame
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
    )
    
    # Charts section
    st.subheader("Cost Comparison")
    
    # Prepare data for charts - ensure numeric types
    chart_df = df.copy()
    
    # If combined file, add transcript information to model names for charts
    if is_combined and selected_transcript == "All Transcripts":
        # Create a combined model+transcript label for the charts
        chart_df["model_label"] = chart_df.apply(
            lambda row: f"{row['model']} ({row['transcript'].split('.')[0]})", 
            axis=1
        )
        model_column = "model_label"
    else:
        model_column = "model"
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Token Usage", "Response Stats"])
    
    with tab1:
        st.subheader("Cost by Model")
        
        # Prepare data for horizontal bar chart
        cost_data = pd.DataFrame({
            'model': chart_df[model_column],
            'Input Cost': pd.to_numeric(chart_df['input_cost_usd'], errors='coerce'),
            'Output Cost': pd.to_numeric(chart_df['output_cost_usd'], errors='coerce')
        })
        
        # Melt the dataframe for easier charting
        cost_data_melted = pd.melt(
            cost_data, 
            id_vars=['model'],
            value_vars=['Input Cost', 'Output Cost'],
            var_name='Cost Type',
            value_name='Cost (USD)'
        )
        
        # Create horizontal bar chart with better label handling
        cost_chart = alt.Chart(cost_data_melted).mark_bar().encode(
            y=alt.Y('model:N', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0)),
            x=alt.X('Cost (USD):Q', title='Cost (USD)'),
            color=alt.Color('Cost Type:N', scale=alt.Scale(scheme='blues')),
            tooltip=['model', 'Cost Type', 'Cost (USD)']
        ).properties(
            title='Cost Breakdown by Model',
            height=max(300, len(cost_data) * 50)  # Increased minimum height
        )
        
        st.altair_chart(cost_chart, use_container_width=True)
        
        # Total cost chart with better label handling
        total_cost_data = pd.DataFrame({
            'model': chart_df[model_column],
            'Total Cost': pd.to_numeric(chart_df['total_cost_usd'], errors='coerce')
        })
        
        total_cost_chart = alt.Chart(total_cost_data).mark_bar().encode(
            y=alt.Y('model:N', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0)),
            x=alt.X('Total Cost:Q', title='Total Cost (USD)'),
            color=alt.Color('model:N', legend=None),
            tooltip=['model', 'Total Cost']
        ).properties(
            title='Total Cost by Model',
            height=max(300, len(total_cost_data) * 50)  # Increased minimum height
        )
        
        st.altair_chart(total_cost_chart, use_container_width=True)
    
    with tab2:
        st.subheader("Token Usage by Model")
        
        # Prepare token data
        token_data = pd.DataFrame({
            'model': chart_df[model_column],
            'Input Tokens': pd.to_numeric(chart_df['input_tokens'], errors='coerce'),
            'Output Tokens': pd.to_numeric(chart_df['output_tokens'], errors='coerce')
        })
        
        # Melt for charting
        token_data_melted = pd.melt(
            token_data,
            id_vars=['model'],
            value_vars=['Input Tokens', 'Output Tokens'],
            var_name='Token Type',
            value_name='Tokens'
        )
        
        token_chart = alt.Chart(token_data_melted).mark_bar().encode(
            y=alt.Y('model:N', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0)),
            x=alt.X('Tokens:Q', title='Number of Tokens'),
            color=alt.Color('Token Type:N', scale=alt.Scale(scheme='category10')),
            tooltip=['model', 'Token Type', 'Tokens']
        ).properties(
            title='Token Usage by Model',
            height=max(300, len(token_data) * 50)  # Increased minimum height
        )
        
        st.altair_chart(token_chart, use_container_width=True)
        
        # Total tokens chart
        total_token_data = pd.DataFrame({
            'model': chart_df[model_column],
            'Total Tokens': pd.to_numeric(chart_df['total_tokens'], errors='coerce')
        })
        
        total_token_chart = alt.Chart(total_token_data).mark_bar().encode(
            y=alt.Y('model:N', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0)),
            x=alt.X('Total Tokens:Q', title='Total Tokens'),
            color=alt.Color('model:N', legend=None),
            tooltip=['model', 'Total Tokens']
        ).properties(
            title='Total Tokens by Model',
            height=max(300, len(total_token_data) * 50)  # Increased minimum height
        )
        
        st.altair_chart(total_token_chart, use_container_width=True)
    
    with tab3:
        if "elapsed_seconds" in chart_df.columns:
            st.subheader("Response Time by Model")
            
            # Prepare time data
            time_data = pd.DataFrame({
                'model': chart_df[model_column],
                'Response Time': pd.to_numeric(chart_df['elapsed_seconds'], errors='coerce')
            })
            
            time_chart = alt.Chart(time_data).mark_bar().encode(
                y=alt.Y('model:N', title=None, axis=alt.Axis(labelLimit=200, labelAngle=0)),
                x=alt.X('Response Time:Q', title='Seconds'),
                color=alt.Color('model:N', legend=None),
                tooltip=['model', 'Response Time']
            ).properties(
                title='Response Time by Model',
                height=max(300, len(time_data) * 50)  # Increased minimum height
            )
            
            st.altair_chart(time_chart, use_container_width=True)
    
    # JSON response inspector
    if "json_response" in df.columns:
        st.subheader("Response Inspector")
        
        # Add a comparison mode toggle
        comparison_mode = st.checkbox("Enable Comparison Mode", value=True)
        
        if comparison_mode:
            # Side-by-side model comparison
            st.write("### Model Response Comparison")
            
            # Let user select models to compare
            models = df["model"].tolist()
            
            # Allow selecting multiple models for comparison
            selected_models = st.multiselect(
                "Select models to compare (2-4 recommended)",
                options=models,
                default=models[:min(2, len(models))]  # Default to first 2 models
            )
            
            # Ensure at least one model is selected
            if not selected_models:
                st.warning("Please select at least one model to view.")
                return
            
            # Get the selected models' responses
            model_rows = {model: df[df["model"] == model].iloc[0] for model in selected_models}
            
            # If more than 4 models, warn about crowded display
            if len(selected_models) > 4:
                st.warning("Comparing many models may make the display crowded.")
            
            # Display model metrics side by side
            num_cols = min(4, len(selected_models))  # Limit to 4 columns max
            metrics_cols = st.columns(num_cols)
            
            # Create metrics for each model
            for i, model_name in enumerate(selected_models[:num_cols]):
                with metrics_cols[i % num_cols]:
                    row = model_rows[model_name]
                    # If combined file, show transcript in the header
                    if is_combined:
                        model_display = f"{model_name}\n({row['transcript']})" if 'transcript' in row else model_name
                    else:
                        model_display = model_name
                    st.subheader(model_display)
                    st.metric("Input Tokens", row["input_tokens"])
                    st.metric("Output Tokens", row["output_tokens"])
                    st.metric("Total Cost", format_cost(row["total_cost_usd"]))
                    if "elapsed_seconds" in row:
                        st.metric("Response Time", f"{row['elapsed_seconds']:.2f}s")
            
            # If there are more models than columns, create a second row
            if len(selected_models) > num_cols:
                st.write("") # Spacer
                metrics_cols2 = st.columns(min(4, len(selected_models) - num_cols))
                for i, model_name in enumerate(selected_models[num_cols:]):
                    with metrics_cols2[i % len(metrics_cols2)]:
                        row = model_rows[model_name]
                        # If combined file, show transcript in the header
                        if is_combined:
                            model_display = f"{model_name}\n({row['transcript']})" if 'transcript' in row else model_name
                        else:
                            model_display = model_name
                        st.subheader(model_display)
                        st.metric("Input Tokens", row["input_tokens"])
                        st.metric("Output Tokens", row["output_tokens"])
                        st.metric("Total Cost", format_cost(row["total_cost_usd"]))
                        if "elapsed_seconds" in row:
                            st.metric("Response Time", f"{row['elapsed_seconds']:.2f}s")
            
            # Function to safely extract nested JSON values
            def extract_json_value(json_obj, path, default="N/A"):
                """Extract a nested value from a JSON object using a dot-separated path"""
                if not isinstance(json_obj, dict):
                    return default
                
                parts = path.split('.')
                current = json_obj
                
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return default
                
                return current
            
            # Display JSON responses side by side in expandable sections
            response_tabs = st.tabs(["Structured Comparison", "Raw JSON"])
            
            with response_tabs[0]:
                # Get JSON responses
                valid_jsons = []
                for model_name in selected_models:
                    json_resp = model_rows[model_name]["json_response"]
                    if isinstance(json_resp, dict):
                        valid_jsons.append((model_name, json_resp))
                
                if valid_jsons:
                    # Extract key parameter evaluations
                    st.subheader("Parameter Evaluations")
                    
                    # Check if we have parameter_evaluations
                    has_params = all("parameter_evaluations" in json_obj for _, json_obj in valid_jsons)
                    
                    if has_params:
                        # Create a comparison table of parameter scores
                        param_data = []
                        
                        # Find all unique parameters across models
                        all_params = set()
                        for _, json_obj in valid_jsons:
                            if "parameter_evaluations" in json_obj:
                                all_params.update(json_obj["parameter_evaluations"].keys())
                        
                        # For each parameter, get scores from all models
                        for param in sorted(all_params):
                            row = {"Parameter": param}
                            
                            for model_name, json_obj in valid_jsons:
                                # Get the score for this parameter from this model
                                if "parameter_evaluations" in json_obj and param in json_obj["parameter_evaluations"]:
                                    param_obj = json_obj["parameter_evaluations"][param]
                                    if isinstance(param_obj, dict) and "score" in param_obj:
                                        row[model_name] = param_obj["score"]
                                    else:
                                        row[model_name] = "N/A"
                                else:
                                    row[model_name] = "N/A"
                            
                            param_data.append(row)
                        
                        # Create a DataFrame for the parameter scores
                        param_df = pd.DataFrame(param_data)
                        st.dataframe(param_df, use_container_width=True)
                        
                        # Create bar chart comparison of scores
                        param_melted = pd.melt(
                            param_df, 
                            id_vars=["Parameter"],
                            var_name="Model",
                            value_name="Score"
                        )
                        
                        # Convert scores to numeric, ignoring non-numeric values
                        param_melted["Score"] = pd.to_numeric(param_melted["Score"], errors="coerce")
                        
                        # Filter out rows with NaN scores
                        param_melted = param_melted.dropna(subset=["Score"])
                        
                        if not param_melted.empty:
                            # Create the comparison chart
                            param_chart = alt.Chart(param_melted).mark_bar().encode(
                                y=alt.Y("Parameter:N", title=None),
                                x=alt.X("Score:Q", title="Parameter Score (1-10)"),
                                color=alt.Color("Model:N"),
                                tooltip=["Parameter", "Model", "Score"]
                            ).properties(
                                title="Parameter Scores Comparison",
                                height=max(300, len(all_params) * 40)  # Increased minimum height
                            )
                            
                            st.altair_chart(param_chart, use_container_width=True)
                    
                    # Compare overall decisions
                    st.subheader("Overall Decisions")
                    decisions = []
                    
                    for model_name, json_obj in valid_jsons:
                        decision = extract_json_value(json_obj, "overall_decision", "N/A")
                        decisions.append({"Model": model_name, "Decision": decision})
                    
                    # Display decisions as a table
                    decisions_df = pd.DataFrame(decisions)
                    st.dataframe(decisions_df, use_container_width=True)
                    
                    # Show key reasons for decisions
                    st.subheader("Key Reasons")
                    for model_name, json_obj in valid_jsons:
                        reasons = extract_json_value(json_obj, "key_reasons", "N/A")
                        st.write(f"**{model_name}**: {reasons}")
                
                else:
                    st.warning("Invalid JSON responses for comparison. Please check the raw JSON view.")
            
            with response_tabs[1]:
                # Show raw JSON side by side instead of tabs
                st.write("### Raw JSON Comparison")
                
                # Add a slider to control how many models to show at once
                if len(selected_models) > 2:
                    max_models_per_row = st.slider(
                        "Models per row", 
                        min_value=1, 
                        max_value=min(4, len(selected_models)), 
                        value=min(3, len(selected_models))
                    )
                else:
                    max_models_per_row = len(selected_models)
                
                # Show models in rows with the selected number of columns
                for i in range(0, len(selected_models), max_models_per_row):
                    # Get the models for this row (up to max_models_per_row)
                    row_models = selected_models[i:i+max_models_per_row]
                    
                    # Create columns for this row
                    json_cols = st.columns(len(row_models))
                    
                    # Display each model's JSON in its column
                    for j, model_name in enumerate(row_models):
                        with json_cols[j]:
                            st.subheader(model_name)
                            st.json(model_rows[model_name]["json_response"])
                            
                            # Add download button for each JSON
                            json_data = model_rows[model_name]["json_response"]
                            if isinstance(json_data, dict):
                                try:
                                    json_str = json.dumps(json_data, indent=2)
                                    st.download_button(
                                        f"Download {model_name} JSON",
                                        json_str,
                                        file_name=f"{model_name}_response.json",
                                        mime="application/json"
                                    )
                                except (TypeError, OverflowError, ValueError) as e:
                                    st.warning(f"Could not create download for this response: {str(e)}")
        
        else:
            # Original single model response inspector
            # Model selector for response inspection
            models = df["model"].tolist()
            selected_model = st.selectbox("Select model to inspect", models)
            
            # Get the selected model's response
            selected_row = df[df["model"] == selected_model].iloc[0]
            
            # Display response metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input Tokens", selected_row["input_tokens"])
            with col2:
                st.metric("Output Tokens", selected_row["output_tokens"])
            with col3:
                st.metric("Total Cost", format_cost(selected_row["total_cost_usd"]))
            
            # Display JSON response
            json_resp = selected_row["json_response"]
            if isinstance(json_resp, dict):
                st.json(json_resp)
            else:
                st.text(str(json_resp))

if __name__ == "__main__":
    main() 
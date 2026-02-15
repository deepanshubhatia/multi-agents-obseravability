#!/usr/bin/env python3
"""
Local Observability Dashboard

A Streamlit-based dashboard for visualizing LLM, tool, and agent metrics locally.
Run with: streamlit run observability_dashboard.py
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.observability.local_observability import (
    get_metrics_store
)

# Page config
st.set_page_config(
    page_title="LLM Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_metrics():
    """Load all metrics from the database"""
    store = get_metrics_store()
    return {
        "llm": store.get_llm_metrics(limit=500),
        "tools": store.get_tool_metrics(limit=500),
        "agents": store.get_agent_metrics(limit=500),
        "summary": store.get_summary_stats()
    }


def render_overview_tab(metrics):
    """Render the overview tab"""
    st.header("📊 Overview")

    summary = metrics["summary"]

    # Create metric columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 LLM Calls")
        st.metric("Total Calls", summary["llm"]["total_calls"])
        st.metric("Avg Duration", f"{summary['llm']['avg_duration_seconds']:.2f}s")
        st.metric("Success Rate", f"{summary['llm']['success_rate_percent']:.1f}%")

    with col2:
        st.markdown("### 🔧 Tool Executions")
        st.metric("Total Calls", summary["tools"]["total_calls"])
        st.metric("Avg Duration", f"{summary['tools']['avg_duration_seconds']:.2f}s")
        st.metric("Success Rate", f"{summary['tools']['success_rate_percent']:.1f}%")

    with col3:
        st.markdown("### 🤠 Agent Actions")
        st.metric("Total Actions", summary["agents"]["total_actions"])
        st.metric("Avg Duration", f"{summary['agents']['avg_duration_seconds']:.2f}s")
        st.metric("Success Rate", f"{summary['agents']['success_rate_percent']:.1f}%")

    st.divider()

    # Timeline chart
    llm_df = pd.DataFrame(metrics["llm"])
    tools_df = pd.DataFrame(metrics["tools"])
    agents_df = pd.DataFrame(metrics["agents"])

    if not llm_df.empty or not tools_df.empty or not agents_df.empty:
        st.subheader("📈 Activity Timeline")

        fig = go.Figure()

        if not llm_df.empty:
            llm_df['timestamp'] = pd.to_datetime(llm_df['timestamp'])
            llm_counts = llm_df.groupby(llm_df['timestamp'].dt.floor('min')).size()
            fig.add_trace(go.Scatter(
                x=llm_counts.index,
                y=llm_counts.values,
                mode='lines+markers',
                name='LLM Calls',
                line=dict(color='#1f77b4')
            ))

        if not tools_df.empty:
            tools_df['timestamp'] = pd.to_datetime(tools_df['timestamp'])
            tools_counts = tools_df.groupby(tools_df['timestamp'].dt.floor('min')).size()
            fig.add_trace(go.Scatter(
                x=tools_counts.index,
                y=tools_counts.values,
                mode='lines+markers',
                name='Tool Calls',
                line=dict(color='#2ca02c')
            ))

        if not agents_df.empty:
            agents_df['timestamp'] = pd.to_datetime(agents_df['timestamp'])
            agents_counts = agents_df.groupby(agents_df['timestamp'].dt.floor('min')).size()
            fig.add_trace(go.Scatter(
                x=agents_counts.index,
                y=agents_counts.values,
                mode='lines+markers',
                name='Agent Actions',
                line=dict(color='#ff7f0e')
            ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Count",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


def render_llm_tab(metrics):
    """Render the LLM calls tab"""
    st.header("🤖 LLM Calls")

    llm_df = pd.DataFrame(metrics["llm"])

    if llm_df.empty:
        st.info("No LLM calls recorded yet. Run some tasks to see metrics here.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        agents = ["All"] + list(llm_df['agent_name'].unique())
        selected_agent = st.selectbox("Agent", agents)
    with col2:
        models = ["All"] + list(llm_df['model'].unique())
        selected_model = st.selectbox("Model", models)

    # Apply filters
    filtered_df = llm_df.copy()
    if selected_agent != "All":
        filtered_df = filtered_df[filtered_df['agent_name'] == selected_agent]
    if selected_model != "All":
        filtered_df = filtered_df[filtered_df['model'] == selected_model]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", len(filtered_df))
    with col2:
        st.metric("Success Rate", f"{filtered_df['success'].mean() * 100:.1f}%")
    with col3:
        st.metric("Avg Duration", f"{filtered_df['duration_seconds'].mean():.2f}s")
    with col4:
        st.metric("Total Tokens", filtered_df['prompt_tokens'].sum() + filtered_df['completion_tokens'].sum())

    # Duration distribution
    st.subheader("Duration Distribution")
    fig = px.histogram(filtered_df, x='duration_seconds', nbins=20,
                       title='LLM Call Duration Distribution',
                       labels={'duration_seconds': 'Duration (seconds)', 'count': 'Count'})
    st.plotly_chart(fig, use_container_width=True)

    # Duration by agent
    st.subheader("Duration by Agent")
    fig = px.box(filtered_df, x='agent_name', y='duration_seconds',
                 title='LLM Call Duration by Agent',
                 labels={'agent_name': 'Agent', 'duration_seconds': 'Duration (seconds)'})
    st.plotly_chart(fig, use_container_width=True)

    # Recent calls table
    st.subheader("Recent Calls")
    display_df = filtered_df[['timestamp', 'agent_name', 'model', 'duration_seconds',
                              'prompt_tokens', 'completion_tokens', 'success']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_df.head(20), use_container_width=True)

    # Prompt/Completion viewer
    st.subheader("📝 Prompt/Completion Viewer")
    selected_id = st.selectbox("Select a call to view details",
                               filtered_df['id'].tolist())

    if selected_id:
        call = filtered_df[filtered_df['id'] == selected_id].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Prompt:**")
            st.text_area("", call['prompt'], height=200, key="prompt_view")
        with col2:
            st.markdown("**Completion:**")
            st.text_area("", call['completion'], height=200, key="completion_view")


def render_tools_tab(metrics):
    """Render the tools tab"""
    st.header("🔧 Tool Executions")

    tools_df = pd.DataFrame(metrics["tools"])

    if tools_df.empty:
        st.info("No tool executions recorded yet. Run some tasks to see metrics here.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        tools = ["All"] + list(tools_df['tool_name'].unique())
        selected_tool = st.selectbox("Tool", tools)
    with col2:
        agents = ["All"] + list(tools_df['agent_name'].unique())
        selected_agent = st.selectbox("Agent", agents)

    # Apply filters
    filtered_df = tools_df.copy()
    if selected_tool != "All":
        filtered_df = filtered_df[filtered_df['tool_name'] == selected_tool]
    if selected_agent != "All":
        filtered_df = filtered_df[filtered_df['agent_name'] == selected_agent]

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Calls", len(filtered_df))
    with col2:
        st.metric("Success Rate", f"{filtered_df['success'].mean() * 100:.1f}%")
    with col3:
        st.metric("Avg Duration", f"{filtered_df['duration_seconds'].mean():.3f}s")

    # Tool usage pie chart
    st.subheader("Tool Usage Distribution")
    tool_counts = filtered_df['tool_name'].value_counts()
    fig = px.pie(values=tool_counts.values, names=tool_counts.index,
                 title='Tool Usage Distribution')
    st.plotly_chart(fig, use_container_width=True)

    # Duration by tool
    st.subheader("Duration by Tool")
    fig = px.box(filtered_df, x='tool_name', y='duration_seconds',
                 title='Tool Execution Duration',
                 labels={'tool_name': 'Tool', 'duration_seconds': 'Duration (seconds)'})
    st.plotly_chart(fig, use_container_width=True)

    # Recent executions table
    st.subheader("Recent Executions")
    display_df = filtered_df[['timestamp', 'agent_name', 'tool_name',
                              'duration_seconds', 'success']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_df.head(20), use_container_width=True)


def render_agents_tab(metrics):
    """Render the agents tab"""
    st.header("🤠 Agent Actions")

    agents_df = pd.DataFrame(metrics["agents"])

    if agents_df.empty:
        st.info("No agent actions recorded yet. Run some tasks to see metrics here.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        agents = ["All"] + list(agents_df['agent_name'].unique())
        selected_agent = st.selectbox("Agent", agents)
    with col2:
        actions = ["All"] + list(agents_df['action'].unique())
        selected_action = st.selectbox("Action", actions)

    # Apply filters
    filtered_df = agents_df.copy()
    if selected_agent != "All":
        filtered_df = filtered_df[filtered_df['agent_name'] == selected_agent]
    if selected_action != "All":
        filtered_df = filtered_df[filtered_df['action'] == selected_action]

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Actions", len(filtered_df))
    with col2:
        st.metric("Success Rate", f"{filtered_df['success'].mean() * 100:.1f}%")
    with col3:
        st.metric("Avg Duration", f"{filtered_df['duration_seconds'].mean():.2f}s")

    # Actions by agent
    st.subheader("Actions by Agent")
    agent_action_counts = filtered_df.groupby(['agent_name', 'action']).size().reset_index(name='count')
    fig = px.bar(agent_action_counts, x='agent_name', y='count', color='action',
                 title='Actions by Agent',
                 labels={'agent_name': 'Agent', 'count': 'Count', 'action': 'Action'})
    st.plotly_chart(fig, use_container_width=True)

    # Duration by agent
    st.subheader("Duration by Agent")
    fig = px.box(filtered_df, x='agent_name', y='duration_seconds',
                 title='Agent Action Duration',
                 labels={'agent_name': 'Agent', 'duration_seconds': 'Duration (seconds)'})
    st.plotly_chart(fig, use_container_width=True)

    # Success rate by agent
    st.subheader("Success Rate by Agent")
    success_by_agent = filtered_df.groupby('agent_name')['success'].mean().reset_index()
    success_by_agent['success'] = success_by_agent['success'] * 100
    fig = px.bar(success_by_agent, x='agent_name', y='success',
                 title='Success Rate by Agent (%)',
                 labels={'agent_name': 'Agent', 'success': 'Success Rate (%)'})
    st.plotly_chart(fig, use_container_width=True)

    # Recent actions table
    st.subheader("Recent Actions")
    display_df = filtered_df[['timestamp', 'agent_name', 'agent_type', 'action',
                              'task_id', 'duration_seconds', 'success']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(display_df.head(20), use_container_width=True)


def render_export_tab(metrics):
    """Render the export tab"""
    st.header("📤 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export to JSON")
        if st.button("Export All Metrics to JSON"):
            from src.observability.local_observability import export_local_metrics
            output_path = Path("observability_data/metrics_export.json")
            export_local_metrics(str(output_path))
            st.success(f"Exported to {output_path}")
            with open(output_path, 'r') as f:
                st.download_button("Download JSON", f, file_name="metrics_export.json")

    with col2:
        st.subheader("Export to CSV")
        if st.button("Export Metrics to CSV"):
            import tempfile
            import zipfile

            store = get_metrics_store()

            with tempfile.TemporaryDirectory() as tmpdir:
                llm_path = Path(tmpdir) / "llm_metrics.csv"
                tool_path = Path(tmpdir) / "tool_metrics.csv"
                agent_path = Path(tmpdir) / "agent_metrics.csv"

                pd.DataFrame(metrics["llm"]).to_csv(llm_path, index=False)
                pd.DataFrame(metrics["tools"]).to_csv(tool_path, index=False)
                pd.DataFrame(metrics["agents"]).to_csv(agent_path, index=False)

                zip_path = Path("observability_data/metrics_export.zip")
                zip_path.parent.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'w') as zf:
                    zf.write(llm_path, "llm_metrics.csv")
                    zf.write(tool_path, "tool_metrics.csv")
                    zf.write(agent_path, "agent_metrics.csv")

                st.success(f"Exported to {zip_path}")

    # Summary display
    st.subheader("Current Summary")
    st.json(metrics["summary"])


def main():
    """Main dashboard entry point"""
    st.title("📊 LLM Observability Dashboard")
    st.markdown("Local metrics and tracing for LLM applications")

    # Sidebar
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Overview", "LLM Calls", "Tools", "Agents", "Export"])

    # Load metrics
    try:
        metrics = load_metrics()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.info("Make sure you have run some tasks to generate metrics.")
        return

    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (30s)"):
        time.sleep(0.1)  # Small delay to prevent rapid refresh
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Render selected tab
    if tab == "Overview":
        render_overview_tab(metrics)
    elif tab == "LLM Calls":
        render_llm_tab(metrics)
    elif tab == "Tools":
        render_tools_tab(metrics)
    elif tab == "Agents":
        render_agents_tab(metrics)
    elif tab == "Export":
        render_export_tab(metrics)


if __name__ == "__main__":
    main()
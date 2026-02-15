"""
Local Observability Dashboard

A Streamlit-based dashboard for visualizing LLM, Tool, and Agent metrics locally.
Run with: streamlit run src/observability/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.observability.local_observability import (
    get_metrics_store,
    LocalMetricsStore,
)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds / 60:.1f}m"


def main():
    st.set_page_config(
        page_title="Local LLM Observability Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Local LLM Observability Dashboard")
    st.markdown("Real-time metrics for LLM calls, Tools, and Agents")

    # Get metrics store
    store = get_metrics_store()

    # Refresh button
    if st.button("🔄 Refresh", key="refresh"):
        st.rerun()

    # Summary stats
    st.header("📈 Summary Statistics")
    stats = store.get_summary_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "LLM Calls",
            stats["llm"]["total_calls"],
            f"Avg: {format_duration(stats['llm']['avg_duration_seconds'])}"
        )
        st.progress(stats["llm"]["success_rate_percent"] / 100)
        st.caption(f"Success Rate: {stats['llm']['success_rate_percent']:.1f}%")

    with col2:
        st.metric(
            "Tool Calls",
            stats["tools"]["total_calls"],
            f"Avg: {format_duration(stats['tools']['avg_duration_seconds'])}"
        )
        st.progress(stats["tools"]["success_rate_percent"] / 100)
        st.caption(f"Success Rate: {stats['tools']['success_rate_percent']:.1f}%")

    with col3:
        st.metric(
            "Agent Actions",
            stats["agents"]["total_actions"],
            f"Avg: {format_duration(stats['agents']['avg_duration_seconds'])}"
        )
        st.progress(stats["agents"]["success_rate_percent"] / 100)
        st.caption(f"Success Rate: {stats['agents']['success_rate_percent']:.1f}%")

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 LLM Calls", "🔧 Tools", "👤 Agents", "📉 Charts"])

    # LLM Calls Tab
    with tab1:
        st.header("LLM Calls")

        llm_metrics = store.get_llm_metrics(limit=100)

        if llm_metrics:
            # Convert to DataFrame
            df = pd.DataFrame(llm_metrics)

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                agents = ["All"] + list(df["agent_name"].unique())
                selected_agent = st.selectbox("Agent", agents, key="llm_agent_filter")
            with col2:
                models = ["All"] + list(df["model"].unique())
                selected_model = st.selectbox("Model", models, key="llm_model_filter")

            # Apply filters
            if selected_agent != "All":
                df = df[df["agent_name"] == selected_agent]
            if selected_model != "All":
                df = df[df["model"] == selected_model]

            # Display metrics
            for _, row in df.iterrows():
                with st.expander(
                    f"{'✅' if row['success'] else '❌'} {row['agent_name']} - {row['model']} ({row['timestamp'][:19]})",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Prompt:**")
                        st.code(row['prompt'][:500] + "..." if len(row['prompt']) > 500 else row['prompt'])
                    with col2:
                        st.markdown("**Completion:**")
                        st.code(row['completion'][:500] + "..." if len(row['completion']) > 500 else row['completion'])

                    st.markdown(f"**Duration:** {format_duration(row['duration_seconds'])} | "
                                f"**Tokens:** {row['prompt_tokens']} prompt / {row['completion_tokens']} completion")

                    if row['error']:
                        st.error(f"Error: {row['error']}")
        else:
            st.info("No LLM metrics recorded yet. Run some agents to see metrics here.")

    # Tools Tab
    with tab2:
        st.header("Tool Calls")

        tool_metrics = store.get_tool_metrics(limit=100)

        if tool_metrics:
            df = pd.DataFrame(tool_metrics)

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                tools = ["All"] + list(df["tool_name"].unique())
                selected_tool = st.selectbox("Tool", tools, key="tool_filter")
            with col2:
                agents = ["All"] + list(df["agent_name"].unique())
                selected_agent = st.selectbox("Agent", agents, key="tool_agent_filter")

            # Apply filters
            if selected_tool != "All":
                df = df[df["tool_name"] == selected_tool]
            if selected_agent != "All":
                df = df[df["agent_name"] == selected_agent]

            # Display metrics
            for _, row in df.iterrows():
                with st.expander(
                    f"{'✅' if row['success'] else '❌'} {row['tool_name']} by {row['agent_name']} ({row['timestamp'][:19]})",
                    expanded=False
                ):
                    st.markdown("**Parameters:**")
                    st.json(row['parameters'])

                    if row['result']:
                        st.markdown("**Result:**")
                        st.code(row['result'][:500] + "..." if len(row['result']) > 500 else row['result'])

                    st.markdown(f"**Duration:** {format_duration(row['duration_seconds'])}")

                    if row['error']:
                        st.error(f"Error: {row['error']}")
        else:
            st.info("No tool metrics recorded yet. Run some agents to see metrics here.")

    # Agents Tab
    with tab3:
        st.header("Agent Actions")

        agent_metrics = store.get_agent_metrics(limit=100)

        if agent_metrics:
            df = pd.DataFrame(agent_metrics)

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                agents = ["All"] + list(df["agent_name"].unique())
                selected_agent = st.selectbox("Agent", agents, key="agent_filter")
            with col2:
                actions = ["All"] + list(df["action"].unique())
                selected_action = st.selectbox("Action", actions, key="action_filter")

            # Apply filters
            if selected_agent != "All":
                df = df[df["agent_name"] == selected_agent]
            if selected_action != "All":
                df = df[df["action"] == selected_action]

            # Display metrics
            for _, row in df.iterrows():
                with st.expander(
                    f"{'✅' if row['success'] else '❌'} {row['agent_name']} - {row['action']} ({row['timestamp'][:19]})",
                    expanded=False
                ):
                    st.markdown(f"**Agent Type:** {row['agent_type']}")
                    st.markdown(f"**Task ID:** {row['task_id']}")
                    st.markdown(f"**Duration:** {format_duration(row['duration_seconds'])}")

                    if row.get('metadata'):
                        try:
                            import json
                            metadata = json.loads(row['metadata'])
                            st.markdown("**Metadata:**")
                            st.json(metadata)
                        except:
                            pass

                    if row['error']:
                        st.error(f"Error: {row['error']}")
        else:
            st.info("No agent metrics recorded yet. Run some agents to see metrics here.")

    # Charts Tab
    with tab4:
        st.header("Analytics Charts")

        llm_df = pd.DataFrame(store.get_llm_metrics(limit=1000))
        tool_df = pd.DataFrame(store.get_tool_metrics(limit=1000))
        agent_df = pd.DataFrame(store.get_agent_metrics(limit=1000))

        if not llm_df.empty or not tool_df.empty or not agent_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                # Duration over time
                st.subheader("Duration Over Time")

                fig = go.Figure()

                if not llm_df.empty:
                    llm_df['timestamp'] = pd.to_datetime(llm_df['timestamp'])
                    fig.add_trace(go.Scatter(
                        x=llm_df['timestamp'],
                        y=llm_df['duration_seconds'],
                        mode='lines+markers',
                        name='LLM Calls',
                        line=dict(color='blue')
                    ))

                if not tool_df.empty:
                    tool_df['timestamp'] = pd.to_datetime(tool_df['timestamp'])
                    fig.add_trace(go.Scatter(
                        x=tool_df['timestamp'],
                        y=tool_df['duration_seconds'],
                        mode='lines+markers',
                        name='Tool Calls',
                        line=dict(color='green')
                    ))

                if not agent_df.empty:
                    agent_df['timestamp'] = pd.to_datetime(agent_df['timestamp'])
                    fig.add_trace(go.Scatter(
                        x=agent_df['timestamp'],
                        y=agent_df['duration_seconds'],
                        mode='lines+markers',
                        name='Agent Actions',
                        line=dict(color='orange')
                    ))

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Duration (seconds)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Call distribution pie chart
                st.subheader("Call Distribution")

                labels = []
                values = []
                colors = []

                if not llm_df.empty:
                    labels.append('LLM Calls')
                    values.append(len(llm_df))
                    colors.append('blue')

                if not tool_df.empty:
                    labels.append('Tool Calls')
                    values.append(len(tool_df))
                    colors.append('green')

                if not agent_df.empty:
                    labels.append('Agent Actions')
                    values.append(len(agent_df))
                    colors.append('orange')

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors,
                    hole=.3
                )])
                st.plotly_chart(fig, use_container_width=True)

            # Agent breakdown
            if not llm_df.empty:
                st.subheader("LLM Calls by Agent")
                agent_counts = llm_df['agent_name'].value_counts()
                fig = px.bar(x=agent_counts.index, y=agent_counts.values, labels={'x': 'Agent', 'y': 'Calls'})
                st.plotly_chart(fig, use_container_width=True)

            if not tool_df.empty:
                st.subheader("Tool Usage")
                tool_counts = tool_df['tool_name'].value_counts()
                fig = px.bar(x=tool_counts.index, y=tool_counts.values, labels={'x': 'Tool', 'y': 'Calls'})
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No metrics recorded yet. Run some agents to see charts here.")

    # Export section
    st.divider()
    st.header("📦 Export Data")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export to JSON"):
            data = store.export_to_json()
            st.download_button(
                "Download JSON",
                data=json.dumps(data, indent=2, default=str),
                file_name=f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col2:
        if st.button("View Raw Data"):
            st.json(store.export_to_json())


if __name__ == "__main__":
    main()
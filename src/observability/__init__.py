"""
AgentOps Observability Integration

This module provides proper instrumentation for LLM calls, tools, and agents
using AgentOps SDK for comprehensive traceability. Also provides local
observability with SQLite storage and Streamlit dashboard.
"""

from .agentops_instrumentation import (
    InstrumentedOllamaClient,
    InstrumentedToolRegistry,
    InstrumentedBaseAgent,
    init_agentops,
    end_agentops_session,
    instrument_llm_call,
    instrument_tool_call,
    instrument_agent_action,
    create_llm_span,
    create_tool_span,
    create_agent_span,
    is_agentops_initialized,
    get_agentops_status,
)

# Local observability
from .local_observability import (
    LocalMetricsStore,
    LocalLLMSpan,
    LocalToolSpan,
    LocalAgentSpan,
    LLMMetric,
    ToolMetric,
    AgentMetric,
    create_local_llm_span,
    create_local_tool_span,
    create_local_agent_span,
    get_metrics_store,
    get_local_metrics_summary,
    export_local_metrics,
)

__all__ = [
    # AgentOps integration
    "InstrumentedOllamaClient",
    "InstrumentedToolRegistry",
    "InstrumentedBaseAgent",
    "init_agentops",
    "end_agentops_session",
    "instrument_llm_call",
    "instrument_tool_call",
    "instrument_agent_action",
    "create_llm_span",
    "create_tool_span",
    "create_agent_span",
    "is_agentops_initialized",
    "get_agentops_status",
    # Local observability
    "LocalMetricsStore",
    "LocalLLMSpan",
    "LocalToolSpan",
    "LocalAgentSpan",
    "LLMMetric",
    "ToolMetric",
    "AgentMetric",
    "create_local_llm_span",
    "create_local_tool_span",
    "create_local_agent_span",
    "get_metrics_store",
    "get_local_metrics_summary",
    "export_local_metrics",
]
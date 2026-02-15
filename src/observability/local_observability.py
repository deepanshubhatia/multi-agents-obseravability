"""
Local Observability Module

Provides local tracing, metrics collection, and dashboard for LLM calls, tools, and agents.
This is an alternative to cloud-based services like AgentOps.

Features:
- Local SQLite database for metrics storage
- Phoenix (Arize) integration for trace visualization
- Built-in dashboard with Streamlit
- Export to JSON/CSV for analysis
"""

import os
import json
import time
import sqlite3
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from contextlib import contextmanager
import uuid
from loguru import logger

# Try to import Phoenix for local tracing
try:
    import phoenix as px
    from phoenix.trace.langchain import OpenInferenceTracer
    from phoenix.trace import using_span
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Local database path
DEFAULT_DB_PATH = Path("observability_data/metrics.db")


@dataclass
class LLMMetric:
    """LLM call metric"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_name: str = "unknown"
    model: str = "unknown"
    prompt: str = ""
    completion: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class ToolMetric:
    """Tool execution metric"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_name: str = "unknown"
    tool_name: str = "unknown"
    parameters: Dict = field(default_factory=dict)
    result: str = ""
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class AgentMetric:
    """Agent action metric"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_name: str = "unknown"
    agent_type: str = "unknown"
    action: str = "unknown"
    task_id: str = ""
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""
    metadata: Dict = field(default_factory=dict)


class LocalMetricsStore:
    """SQLite-based local metrics storage"""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # LLM metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    agent_name TEXT,
                    model TEXT,
                    prompt TEXT,
                    completion TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    duration_seconds REAL,
                    success INTEGER,
                    error TEXT,
                    metadata TEXT
                )
            ''')

            # Tool metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    agent_name TEXT,
                    tool_name TEXT,
                    parameters TEXT,
                    result TEXT,
                    duration_seconds REAL,
                    success INTEGER,
                    error TEXT
                )
            ''')

            # Agent metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    agent_name TEXT,
                    agent_type TEXT,
                    action TEXT,
                    task_id TEXT,
                    duration_seconds REAL,
                    success INTEGER,
                    error TEXT,
                    metadata TEXT
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_timestamp ON tool_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_timestamp ON agent_metrics(timestamp)')

            conn.commit()

    def save_llm_metric(self, metric: LLMMetric):
        """Save an LLM metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO llm_metrics
                (id, timestamp, agent_name, model, prompt, completion, prompt_tokens,
                 completion_tokens, duration_seconds, success, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.id, metric.timestamp, metric.agent_name, metric.model,
                metric.prompt[:5000], metric.completion[:5000],  # Truncate long text
                metric.prompt_tokens, metric.completion_tokens,
                metric.duration_seconds, int(metric.success), metric.error,
                json.dumps(metric.metadata)
            ))
            conn.commit()

    def save_tool_metric(self, metric: ToolMetric):
        """Save a tool metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO tool_metrics
                (id, timestamp, agent_name, tool_name, parameters, result,
                 duration_seconds, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.id, metric.timestamp, metric.agent_name, metric.tool_name,
                json.dumps(metric.parameters), metric.result[:1000],
                metric.duration_seconds, int(metric.success), metric.error
            ))
            conn.commit()

    def save_agent_metric(self, metric: AgentMetric):
        """Save an agent metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO agent_metrics
                (id, timestamp, agent_name, agent_type, action, task_id,
                 duration_seconds, success, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.id, metric.timestamp, metric.agent_name, metric.agent_type,
                metric.action, metric.task_id, metric.duration_seconds,
                int(metric.success), metric.error, json.dumps(metric.metadata)
            ))
            conn.commit()

    def get_llm_metrics(self, limit: int = 100, agent_name: str = None) -> List[Dict]:
        """Get LLM metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if agent_name:
                cursor.execute('''
                    SELECT * FROM llm_metrics
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (agent_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM llm_metrics
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_tool_metrics(self, limit: int = 100, tool_name: str = None) -> List[Dict]:
        """Get tool metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if tool_name:
                cursor.execute('''
                    SELECT * FROM tool_metrics
                    WHERE tool_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (tool_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM tool_metrics
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_agent_metrics(self, limit: int = 100, agent_name: str = None) -> List[Dict]:
        """Get agent metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if agent_name:
                cursor.execute('''
                    SELECT * FROM agent_metrics
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (agent_name, limit))
            else:
                cursor.execute('''
                    SELECT * FROM agent_metrics
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # LLM stats
            cursor.execute('SELECT COUNT(*) FROM llm_metrics')
            llm_count = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(duration_seconds) FROM llm_metrics WHERE success = 1')
            llm_avg_duration = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(success) * 100.0 / COUNT(*) FROM llm_metrics')
            llm_success_rate = cursor.fetchone()[0] or 0

            # Tool stats
            cursor.execute('SELECT COUNT(*) FROM tool_metrics')
            tool_count = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(duration_seconds) FROM tool_metrics WHERE success = 1')
            tool_avg_duration = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(success) * 100.0 / COUNT(*) FROM tool_metrics')
            tool_success_rate = cursor.fetchone()[0] or 0

            # Agent stats
            cursor.execute('SELECT COUNT(*) FROM agent_metrics')
            agent_count = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(duration_seconds) FROM agent_metrics WHERE success = 1')
            agent_avg_duration = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(success) * 100.0 / COUNT(*) FROM agent_metrics')
            agent_success_rate = cursor.fetchone()[0] or 0

            return {
                "llm": {
                    "total_calls": llm_count,
                    "avg_duration_seconds": round(llm_avg_duration, 3),
                    "success_rate_percent": round(llm_success_rate, 1)
                },
                "tools": {
                    "total_calls": tool_count,
                    "avg_duration_seconds": round(tool_avg_duration, 3),
                    "success_rate_percent": round(tool_success_rate, 1)
                },
                "agents": {
                    "total_actions": agent_count,
                    "avg_duration_seconds": round(agent_avg_duration, 3),
                    "success_rate_percent": round(agent_success_rate, 1)
                }
            }

    def export_to_json(self, output_path: Path = None) -> Dict:
        """Export all metrics to JSON"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "llm_metrics": self.get_llm_metrics(limit=1000),
            "tool_metrics": self.get_tool_metrics(limit=1000),
            "agent_metrics": self.get_agent_metrics(limit=1000),
            "summary": self.get_summary_stats()
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        return data


# Global metrics store instance
_metrics_store: Optional[LocalMetricsStore] = None


def get_metrics_store(db_path: Path = DEFAULT_DB_PATH) -> LocalMetricsStore:
    """Get or create the global metrics store"""
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = LocalMetricsStore(db_path)
    return _metrics_store


class LocalLLMSpan:
    """Context manager for LLM call tracking with local storage"""

    def __init__(self, model: str, prompt: str, agent_name: str = "unknown", **kwargs):
        self.model = model
        self.prompt = prompt
        self.agent_name = agent_name
        self.kwargs = kwargs
        self.start_time = None
        self.completion = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.error = None
        self.metric = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        # Save to local metrics store
        metric = LLMMetric(
            agent_name=self.agent_name,
            model=self.model,
            prompt=self.prompt,
            completion=self.completion,
            prompt_tokens=self.prompt_tokens or len(self.prompt.split()),
            completion_tokens=self.completion_tokens or len(self.completion.split()),
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or "",
            metadata=self.kwargs
        )

        self.metric = metric
        get_metrics_store().save_llm_metric(metric)
        logger.debug(f"LLM metric saved: {metric.id[:8]}... ({duration:.2f}s)")

        return False

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        metric = LLMMetric(
            agent_name=self.agent_name,
            model=self.model,
            prompt=self.prompt,
            completion=self.completion,
            prompt_tokens=self.prompt_tokens or len(self.prompt.split()),
            completion_tokens=self.completion_tokens or len(self.completion.split()),
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or "",
            metadata=self.kwargs
        )

        self.metric = metric
        get_metrics_store().save_llm_metric(metric)
        logger.debug(f"LLM metric saved: {metric.id[:8]}... ({duration:.2f}s)")

        return False


class LocalToolSpan:
    """Context manager for tool execution tracking with local storage"""

    def __init__(self, tool_name: str, agent_name: str = "unknown", parameters: Dict = None, **kwargs):
        self.tool_name = tool_name
        self.agent_name = agent_name
        self.parameters = parameters or {}
        self.kwargs = kwargs
        self.start_time = None
        self.result = None
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        metric = ToolMetric(
            agent_name=self.agent_name,
            tool_name=self.tool_name,
            parameters=self.parameters,
            result=str(self.result)[:1000] if self.result else "",
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or ""
        )

        get_metrics_store().save_tool_metric(metric)
        logger.debug(f"Tool metric saved: {self.tool_name} ({duration:.2f}s)")

        return False

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        metric = ToolMetric(
            agent_name=self.agent_name,
            tool_name=self.tool_name,
            parameters=self.parameters,
            result=str(self.result)[:1000] if self.result else "",
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or ""
        )

        get_metrics_store().save_tool_metric(metric)
        logger.debug(f"Tool metric saved: {self.tool_name} ({duration:.2f}s)")

        return False


class LocalAgentSpan:
    """Context manager for agent action tracking with local storage"""

    def __init__(self, agent_name: str, agent_type: str, action: str = "execute", **kwargs):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.action = action
        self.kwargs = kwargs
        self.start_time = None
        self.result = None
        self.error = None
        self.task_id = kwargs.get('task_id', '')

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        metric = AgentMetric(
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            action=self.action,
            task_id=self.task_id,
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or "",
            metadata=self.kwargs
        )

        get_metrics_store().save_agent_metric(metric)
        logger.debug(f"Agent metric saved: {self.agent_name}/{self.action} ({duration:.2f}s)")

        return False

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        metric = AgentMetric(
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            action=self.action,
            task_id=self.task_id,
            duration_seconds=round(duration, 3),
            success=exc_type is None,
            error=self.error or "",
            metadata=self.kwargs
        )

        get_metrics_store().save_agent_metric(metric)
        logger.debug(f"Agent metric saved: {self.agent_name}/{self.action} ({duration:.2f}s)")

        return False


# Convenience functions
def create_local_llm_span(model: str, prompt: str, agent_name: str = "unknown", **kwargs):
    """Create a local LLM span for tracking"""
    return LocalLLMSpan(model=model, prompt=prompt, agent_name=agent_name, **kwargs)


def create_local_tool_span(tool_name: str, agent_name: str = "unknown", parameters: Dict = None, **kwargs):
    """Create a local tool span for tracking"""
    return LocalToolSpan(tool_name=tool_name, agent_name=agent_name, parameters=parameters, **kwargs)


def create_local_agent_span(agent_name: str, agent_type: str, action: str = "execute", **kwargs):
    """Create a local agent span for tracking"""
    return LocalAgentSpan(agent_name=agent_name, agent_type=agent_type, action=action, **kwargs)


def get_local_metrics_summary() -> Dict[str, Any]:
    """Get summary of all local metrics"""
    return get_metrics_store().get_summary_stats()


def export_local_metrics(output_path: str = "observability_data/metrics_export.json") -> Dict:
    """Export all local metrics to JSON"""
    return get_metrics_store().export_to_json(Path(output_path))
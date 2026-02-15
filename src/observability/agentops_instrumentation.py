"""
AgentOps Instrumentation Module

Provides proper instrumentation for LLM calls, tools, and agents using AgentOps SDK.
This fixes the issue where only a single session span was visible by creating
proper hierarchical spans for all operations.

Supports both AgentOps v3 and v4 APIs.
"""

import os
import time
import asyncio
import functools
import warnings
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from datetime import datetime
from loguru import logger

# Suppress AgentOps deprecation warnings for v3 API
warnings.filterwarnings("ignore", message=".*is deprecated.*", category=DeprecationWarning)

# AgentOps imports
try:
    import agentops
    # Try to import event classes (v3 API, deprecated in v4)
    try:
        from agentops import LLMEvent, ToolEvent, ActionEvent
        HAS_EVENT_CLASSES = True
    except ImportError:
        HAS_EVENT_CLASSES = False

    # Check for v4 API (start_span, end_span)
    HAS_V4_API = hasattr(agentops, 'start_span')

    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False
    agentops = None
    HAS_EVENT_CLASSES = False
    HAS_V4_API = False

# Global state for session tracking
_agentops_initialized = False
_current_session = None


def init_agentops(api_key: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
    """
    Initialize AgentOps with proper configuration for full tracing.

    Args:
        api_key: AgentOps API key (defaults to AGENT_OPS_API_KEY env var)
        tags: Tags for session identification

    Returns:
        True if initialization was successful
    """
    global _agentops_initialized, _current_session

    if not AGENTOPS_AVAILABLE:
        logger.warning("AgentOps not available. Install with: pip install agentops")
        return False

    api_key = api_key or os.getenv("AGENT_OPS_API_KEY") or os.getenv("AGENTOPS_API_KEY")
    if not api_key:
        logger.warning("AgentOps API key not provided")
        return False

    try:
        # Initialize with default tags for tracing
        default_tags = ["multi-agent-demo", "instrumented"]
        if tags:
            default_tags.extend(tags)

        # Initialize AgentOps - this sets up the session
        agentops.init(
            api_key=api_key,
            tags=default_tags,
            # Enable automatic instrumentations where possible
            auto_start_session=True,
        )

        _agentops_initialized = True
        logger.info(f"AgentOps initialized with tags: {default_tags}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize AgentOps: {e}")
        return False


def end_agentops_session(end_state: str = "Success", end_state_reason: str = "") -> None:
    """
    End the current AgentOps session.

    Args:
        end_state: Final state of the session ("Success", "Fail", "Indeterminate")
        end_state_reason: Reason for the session end state
    """
    global _agentops_initialized

    if not AGENTOPS_AVAILABLE or not _agentops_initialized:
        return

    try:
        agentops.end_session(
            end_state=end_state,
            end_state_reason=end_state_reason
        )
        logger.info(f"AgentOps session ended with state: {end_state}")
    except Exception as e:
        logger.error(f"Failed to end AgentOps session: {e}")


def create_llm_span(
    model: str,
    prompt: str,
    agent_name: str = "unknown",
    **kwargs
):
    """
    Create a context manager for LLM call tracing.

    Usage:
        with create_llm_span(model="llama2", prompt="Hello", agent_name="ResearchAgent") as span:
            response = await llm.generate(prompt)
            span["completion"] = response
    """
    return LLMSpanContext(model=model, prompt=prompt, agent_name=agent_name, **kwargs)


class LLMSpanContext:
    """Context manager for LLM call tracing with AgentOps."""

    def __init__(self, model: str, prompt: str, agent_name: str = "unknown", **kwargs):
        self.model = model
        self.prompt = prompt
        self.agent_name = agent_name
        self.kwargs = kwargs
        self.start_time = None
        self.completion = ""
        self.tokens_prompt = 0
        self.tokens_completion = 0
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        # Record the LLM event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                if HAS_V4_API:
                    # v4 API - use start_span/end_span pattern
                    agentops.record({
                        "event_type": "llm",
                        "prompt": self.prompt,
                        "completion": self.completion,
                        "model": self.model,
                        "agent_name": self.agent_name,
                        "duration": duration,
                        "error": self.error,
                        **self.kwargs
                    })
                elif HAS_EVENT_CLASSES:
                    # v3 API - use LLMEvent
                    event = LLMEvent(
                        prompt=self.prompt,
                        completion=self.completion,
                        model=self.model,
                        agent_name=self.agent_name,
                        duration=duration,
                        error=self.error,
                        **self.kwargs
                    )
                    agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record LLM event: {e}")

        return False  # Don't suppress exceptions

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        # Record the LLM event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    if HAS_EVENT_CLASSES:
                        event = LLMEvent(
                            prompt=self.prompt,
                            completion=self.completion,
                            model=self.model,
                            agent_name=self.agent_name,
                            duration=duration,
                            error=self.error,
                            **self.kwargs
                        )
                        agentops.record(event)
                    else:
                        agentops.track("LLM Call", {
                            "prompt": self.prompt[:500] if self.prompt else "",
                            "completion": self.completion[:500] if self.completion else "",
                            "model": self.model,
                            "agent_name": self.agent_name,
                            "duration": duration,
                            "error": self.error,
                        })
            except Exception as e:
                logger.warning(f"Failed to record LLM event: {e}")

        return False


def create_tool_span(
    tool_name: str,
    agent_name: str = "unknown",
    parameters: Optional[Dict] = None,
    **kwargs
):
    """
    Create a context manager for tool call tracing.

    Usage:
        with create_tool_span(tool_name="web_search", agent_name="ResearchAgent") as span:
            result = await search(query)
            span["result"] = result
    """
    return ToolSpanContext(tool_name=tool_name, agent_name=agent_name, parameters=parameters, **kwargs)


class ToolSpanContext:
    """Context manager for tool call tracing with AgentOps."""

    def __init__(self, tool_name: str, agent_name: str = "unknown", parameters: Optional[Dict] = None, **kwargs):
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

        # Record the Tool event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                event = ToolEvent(
                    name=self.tool_name,
                    agent_name=self.agent_name,
                    parameters=self.parameters,
                    result=str(self.result)[:500] if self.result else None,  # Truncate long results
                    duration=duration,
                    error=self.error,
                    **self.kwargs
                )
                agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record Tool event: {e}")

        return False

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        # Record the Tool event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                event = ToolEvent(
                    name=self.tool_name,
                    agent_name=self.agent_name,
                    parameters=self.parameters,
                    result=str(self.result)[:500] if self.result else None,
                    duration=duration,
                    error=self.error,
                    **self.kwargs
                )
                agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record Tool event: {e}")

        return False


def create_agent_span(
    agent_name: str,
    agent_type: str,
    action: str = "execute",
    **kwargs
):
    """
    Create a context manager for agent action tracing.

    Usage:
        with create_agent_span(agent_name="ResearchAgent", agent_type="research") as span:
            result = await agent.execute_task(task)
            span["result"] = result
    """
    return AgentSpanContext(agent_name=agent_name, agent_type=agent_type, action=action, **kwargs)


class AgentSpanContext:
    """Context manager for agent action tracing with AgentOps."""

    def __init__(self, agent_name: str, agent_type: str, action: str = "execute", **kwargs):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.action = action
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

        # Record the Action event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                event = ActionEvent(
                    action_type=self.action,
                    agent_name=self.agent_name,
                    duration=duration,
                    error=self.error,
                    **self.kwargs
                )
                agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record Action event: {e}")

        return False

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is not None:
            self.error = str(exc_val)

        # Record the Action event with AgentOps
        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                event = ActionEvent(
                    action_type=self.action,
                    agent_name=self.agent_name,
                    duration=duration,
                    error=self.error,
                    **self.kwargs
                )
                agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record Action event: {e}")

        return False


def instrument_llm_call(func: Callable) -> Callable:
    """
    Decorator to automatically instrument LLM calls with AgentOps.

    Usage:
        @instrument_llm_call
        async def generate_response(self, prompt: str) -> str:
            ...
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract common parameters
        self_arg = args[0] if args else None
        agent_name = getattr(self_arg, 'name', getattr(self_arg, 'agent_type', 'unknown'))
        model = getattr(self_arg, 'model_name', 'unknown')
        prompt = kwargs.get('prompt', str(args[1]) if len(args) > 1 else '')

        start_time = time.time()
        error = None
        result = None

        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            if AGENTOPS_AVAILABLE and _agentops_initialized:
                try:
                    event = LLMEvent(
                        prompt=prompt,
                        completion=str(result)[:500] if result else '',
                        model=model,
                        agent_name=agent_name,
                        duration=duration,
                        error=error
                    )
                    agentops.record(event)
                except Exception as e:
                    logger.warning(f"Failed to record LLM event: {e}")

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        self_arg = args[0] if args else None
        agent_name = getattr(self_arg, 'name', getattr(self_arg, 'agent_type', 'unknown'))
        model = getattr(self_arg, 'model_name', 'unknown')
        prompt = kwargs.get('prompt', str(args[1]) if len(args) > 1 else '')

        start_time = time.time()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            if AGENTOPS_AVAILABLE and _agentops_initialized:
                try:
                    event = LLMEvent(
                        prompt=prompt,
                        completion=str(result)[:500] if result else '',
                        model=model,
                        agent_name=agent_name,
                        duration=duration,
                        error=error
                    )
                    agentops.record(event)
                except Exception as e:
                    logger.warning(f"Failed to record LLM event: {e}")

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def instrument_tool_call(tool_name: str) -> Callable:
    """
    Decorator factory to instrument tool calls with AgentOps.

    Usage:
        @instrument_tool_call("web_search")
        async def web_search(self, query: str) -> Dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            agent_name = getattr(self_arg, 'name', getattr(self_arg, 'agent_type', 'unknown'))

            start_time = time.time()
            error = None
            result = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time

                if AGENTOPS_AVAILABLE and _agentops_initialized:
                    try:
                        event = ToolEvent(
                            name=tool_name,
                            agent_name=agent_name,
                            parameters=kwargs,
                            result=str(result)[:500] if result else None,
                            duration=duration,
                            error=error
                        )
                        agentops.record(event)
                    except Exception as e:
                        logger.warning(f"Failed to record Tool event: {e}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            agent_name = getattr(self_arg, 'name', getattr(self_arg, 'agent_type', 'unknown'))

            start_time = time.time()
            error = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time

                if AGENTOPS_AVAILABLE and _agentops_initialized:
                    try:
                        event = ToolEvent(
                            name=tool_name,
                            agent_name=agent_name,
                            parameters=kwargs,
                            result=str(result)[:500] if result else None,
                            duration=duration,
                            error=error
                        )
                        agentops.record(event)
                    except Exception as e:
                        logger.warning(f"Failed to record Tool event: {e}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def instrument_agent_action(action_type: str) -> Callable:
    """
    Decorator factory to instrument agent actions with AgentOps.

    Usage:
        @instrument_agent_action("execute_task")
        async def execute_task(self, task: Task) -> Any:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            agent_name = getattr(self_arg, 'name', 'unknown')
            task = args[1] if len(args) > 1 else None
            task_id = getattr(task, 'id', 'unknown') if task else 'unknown'

            start_time = time.time()
            error = None
            result = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time

                if AGENTOPS_AVAILABLE and _agentops_initialized:
                    try:
                        event = ActionEvent(
                            action_type=action_type,
                            agent_name=agent_name,
                            duration=duration,
                            error=error,
                            task_id=task_id,
                            success=error is None
                        )
                        agentops.record(event)
                    except Exception as e:
                        logger.warning(f"Failed to record Action event: {e}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            self_arg = args[0] if args else None
            agent_name = getattr(self_arg, 'name', 'unknown')
            task = args[1] if len(args) > 1 else None
            task_id = getattr(task, 'id', 'unknown') if task else 'unknown'

            start_time = time.time()
            error = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time

                if AGENTOPS_AVAILABLE and _agentops_initialized:
                    try:
                        event = ActionEvent(
                            action_type=action_type,
                            agent_name=agent_name,
                            duration=duration,
                            error=error,
                            task_id=task_id,
                            success=error is None
                        )
                        agentops.record(event)
                    except Exception as e:
                        logger.warning(f"Failed to record Action event: {e}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class InstrumentedOllamaClient:
    """
    Wrapper around OllamaClient that provides automatic AgentOps instrumentation.
    """

    def __init__(self, base_client, agent_name: str = "OllamaAgent"):
        """
        Args:
            base_client: The underlying OllamaClient instance
            agent_name: Name of the agent using this client
        """
        self._client = base_client
        self.agent_name = agent_name
        self.model_name = getattr(base_client, 'model_name', 'unknown')

    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text with AgentOps instrumentation."""
        start_time = time.time()
        error = None
        result = None

        try:
            result = await self._client.generate(model, prompt, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            if AGENTOPS_AVAILABLE and _agentops_initialized:
                try:
                    response_text = result.get('response', '') if result else ''
                    event = LLMEvent(
                        prompt=prompt,
                        completion=response_text,
                        model=model,
                        agent_name=self.agent_name,
                        duration=duration,
                        error=error,
                        tokens_prompt=len(prompt.split()),  # Approximate token count
                        tokens_completion=len(response_text.split())
                    )
                    agentops.record(event)
                except Exception as e:
                    logger.warning(f"Failed to record LLM event: {e}")

    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat with AgentOps instrumentation."""
        start_time = time.time()
        error = None
        result = None

        # Convert messages to prompt for tracking
        prompt = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in messages])

        try:
            result = await self._client.chat(model, messages, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            if AGENTOPS_AVAILABLE and _agentops_initialized:
                try:
                    response_text = result.get('message', {}).get('content', '') if result else ''
                    event = LLMEvent(
                        prompt=prompt,
                        completion=response_text,
                        model=model,
                        agent_name=self.agent_name,
                        duration=duration,
                        error=error
                    )
                    agentops.record(event)
                except Exception as e:
                    logger.warning(f"Failed to record LLM event: {e}")

    def __getattr__(self, name):
        """Proxy all other attributes to the underlying client."""
        return getattr(self._client, name)


class InstrumentedToolRegistry:
    """
    Wrapper around ToolRegistry that provides automatic AgentOps instrumentation
    for all tool executions.
    """

    def __init__(self, base_registry, agent_name: str = "Agent"):
        """
        Args:
            base_registry: The underlying ToolRegistry instance
            agent_name: Name of the agent using this registry
        """
        self._registry = base_registry
        self.agent_name = agent_name

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with AgentOps instrumentation."""
        start_time = time.time()
        error = None
        result = None

        try:
            result = await self._registry.execute_tool(tool_name, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time

            if AGENTOPS_AVAILABLE and _agentops_initialized:
                try:
                    event = ToolEvent(
                        name=tool_name,
                        agent_name=self.agent_name,
                        parameters=kwargs,
                        result=str(result)[:500] if result else None,
                        duration=duration,
                        error=error
                    )
                    agentops.record(event)
                except Exception as e:
                    logger.warning(f"Failed to record Tool event: {e}")

    def __getattr__(self, name):
        """Proxy all other attributes to the underlying registry."""
        return getattr(self._registry, name)


class InstrumentedBaseAgent:
    """
    Mixin class that adds AgentOps instrumentation to agent operations.

    Inherit from this class to automatically get instrumentation for:
    - Task execution
    - Tool usage
    - Memory operations
    """

    def _instrumented_execute_task(self, task, original_execute):
        """Execute task with instrumentation."""
        return _instrumented_execute_task_impl(self, task, original_execute)


async def _instrumented_execute_task_impl(self, task, original_execute):
    """Implementation of instrumented task execution."""
    agent_name = getattr(self, 'name', 'unknown')
    task_id = getattr(task, 'id', 'unknown')

    start_time = time.time()
    error = None
    result = None

    try:
        result = await original_execute(task)
        return result
    except Exception as e:
        error = str(e)
        raise
    finally:
        duration = time.time() - start_time

        if AGENTOPS_AVAILABLE and _agentops_initialized:
            try:
                event = ActionEvent(
                    action_type="execute_task",
                    agent_name=agent_name,
                    duration=duration,
                    error=error,
                    task_id=task_id,
                    success=error is None
                )
                agentops.record(event)
            except Exception as e:
                logger.warning(f"Failed to record Action event: {e}")


# Convenience function for checking AgentOps status
def is_agentops_initialized() -> bool:
    """Check if AgentOps has been properly initialized."""
    return _agentops_initialized and AGENTOPS_AVAILABLE


def get_agentops_status() -> Dict[str, Any]:
    """Get the current AgentOps status and configuration."""
    return {
        "available": AGENTOPS_AVAILABLE,
        "initialized": _agentops_initialized,
        "api_key_set": bool(os.getenv("AGENT_OPS_API_KEY") or os.getenv("AGENTOPS_API_KEY")),
    }
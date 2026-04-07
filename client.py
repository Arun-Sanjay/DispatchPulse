"""DispatchPulse client. Inherits all functionality from MCPToolClient."""

from __future__ import annotations

try:
    from openenv.core.mcp_client import MCPToolClient
except ImportError:  # pragma: no cover - allow import even if openenv missing
    MCPToolClient = object  # type: ignore


class DispatchPulseEnv(MCPToolClient):  # type: ignore[misc]
    """Client for the DispatchPulse environment.

    Provides the standard MCPToolClient interface:
        - ``list_tools()``: discover available tools
        - ``call_tool(name, **kwargs)``: invoke a tool by name
        - ``reset(**kwargs)``: reset the environment, optionally passing task_name
        - ``step(action)``: low-level step interface

    Example:
        >>> from client import DispatchPulseEnv
        >>> with DispatchPulseEnv(base_url="http://localhost:8000") as env:
        ...     env.reset(task_name="easy", seed=42)
        ...     view = env.call_tool("view_dispatch_center")
        ...     print(view)
        ...     env.call_tool("dispatch", call_id="CALL-001", unit_id="ALS-1")
    """

    pass

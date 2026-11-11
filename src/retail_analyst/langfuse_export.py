"""Optional export of traces to Langfuse Cloud (the client allowed SaaS tracing).

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_BASE_URL picks
the region, e.g. https://us.cloud.langfuse.com). Missing keys, a missing package, wrong keys or
no network never break the agent: Langfuse exports in the background, and the local JSONL trace
is always written. Text goes through the same PII regex as the local trace before it leaves.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from retail_analyst.safety.pii import redact_pii

log = logging.getLogger(__name__)


def _mask(*, data: Any, **_: Any) -> Any:
    """Langfuse mask hook: redact PII in every string before export."""
    if isinstance(data, str):
        return redact_pii(data)[0]
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [_mask(data=v) for v in data]
    return data


class LangfuseExport:
    def __init__(self) -> None:
        self.handler: Any = None
        self._client: Any = None
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return  # not configured: local traces only
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            # Export failures are retried in the background and must not spill into the chat.
            for name in ("langfuse", "opentelemetry"):
                logging.getLogger(name).setLevel(logging.ERROR)
            self._client = Langfuse(mask=_mask)
            self.handler = CallbackHandler()
        except Exception as exc:  # never let observability break the agent
            log.warning("Langfuse export disabled: %s", exc)
            self.handler = self._client = None

    @property
    def enabled(self) -> bool:
        return self.handler is not None

    def run_config(self, session_id: str, user_id: str, trace_id: str | None) -> dict[str, Any]:
        """LangChain run config: the callback plus ids that group traces by session and user."""
        if not self.enabled:
            return {}
        return {
            "callbacks": [self.handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
                "langfuse_tags": ["retail-analyst"],
                "trace_id": trace_id or "",  # the local trace id, to find the same turn in JSONL
            },
        }

    def check(self) -> bool:
        """For `analyst-doctor`: are the keys accepted? (one network call)"""
        try:
            return bool(self._client and self._client.auth_check())
        except Exception:
            return False

    def flush(self) -> None:
        try:
            if self._client:
                self._client.flush()
        except Exception:
            pass

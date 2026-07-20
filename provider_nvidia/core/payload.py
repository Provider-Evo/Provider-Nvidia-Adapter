


from typing import Any, Dict, List

MAX_TOKENS: int = 229376


def build_payload(
    messages: List[Dict[str, Any]],
    model: str = "",
    stream: bool = True,
    **kw: Any,
) -> Dict[str, Any]:
    """构建聊天请求体。

    Args:
        messages: 消息列表。
        model: 模型名。
        stream: 是否流式。
        **kw: 额外参数（max_tokens, temperature, top_p, stop）。

    Returns:
        请求体字典。

    Examples:
        >>> p = build_payload([{"role": "user", "content": "hi"}], "m1")
        >>> p["model"]
        'm1'
        >>> p["stream"]
        True
        >>> "max_tokens" in p
        True
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": kw.get("max_tokens", MAX_TOKENS),
    }
    if kw.get("temperature") is not None:
        payload["temperature"] = kw["temperature"]
    if kw.get("top_p") is not None:
        payload["top_p"] = kw["top_p"]
    if kw.get("stop"):
        payload["stop"] = kw["stop"]
    return payload

# =======================================================================
# 重导出 — 同包内协同模块的公共符号（保持外部 ``from .. import`` 路径稳定）
# =======================================================================

from .headers import (
    build_headers,
)

from .helpers.sse import (
    parse_sse_line,
)

__all__ = [
    "build_headers",
    "parse_sse_line",
]

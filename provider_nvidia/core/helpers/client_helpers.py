from __future__ import annotations

"""Nvidia HTTP 客户端辅助模块。

职责：
    承载 :class:`_KeyState`（单Key运行时状态）与请求/响应级别的纯函数
    （构造请求、解析非流式/流式响应），供 ``client.py`` 中的
    :class:`NvidiaClient` facade 调用。拆分自 ``client.py``，不改变任何
    现有行为。
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

import aiohttp

from src.core.utils.errors import PlatformError
from src.foundation.logger import get_logger
from ..headers import build_headers
from ..payload import build_payload
from .sse import parse_sse_line

logger = get_logger(__name__)

BASE_URL: str = "https://integrate.api.nvidia.com/v1"
CHAT_PATH: str = "/chat/completions"
RECOVERY_INTERVAL: int = 60


class KeyState:
    """单个API Key的运行时状态。"""

    __slots__ = (
        "key", "valid", "busy", "consecutive_failures", "last_error_time",
    )

    def __init__(self, key: str) -> None:
        """初始化Key状态。

        Args:
            key: API Key字符串。
        """
        self.key: str = key
        self.valid: bool = True
        self.busy: bool = False
        self.consecutive_failures: int = 0
        self.last_error_time: float = 0.0

    @property
    def available(self) -> bool:
        """判断是否可用。

        Returns:
            如果Key有效、未繁忙且未超限则返回True。
        """
        if not self.valid:
            if time.time() - self.last_error_time >= RECOVERY_INTERVAL:
                self.valid = True
                self.consecutive_failures = 0
            else:
                return False
        if self.busy:
            return False
        if self.consecutive_failures >= 3:
            if time.time() - self.last_error_time < RECOVERY_INTERVAL:
                return False
            self.consecutive_failures = 0
        return True

    def mark_success(self) -> None:
        """标记请求成功。"""
        self.busy = False
        self.consecutive_failures = 0

    def mark_failure(self, status: int = 0) -> None:
        """根据HTTP状态码处理失败。

        Args:
            status: HTTP响应状态码。
        """
        self.busy = False
        self.last_error_time = time.time()
        if status in (401, 402, 403):
            self.valid = False
            logger.warning(
                "nvidia Key无效 (HTTP%d): %s...", status, self.key[:16]
            )
        else:
            self.consecutive_failures += 1


async def nonstream_response(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """解析非流式 JSON 响应。

    Args:
        resp: HTTP 响应对象。

    Yields:
        文本内容或含 usage 的字典。
    """
    data = await resp.json()
    choice = (data.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content", "")
    if content:
        yield content
    usage = data.get("usage")
    if usage:
        yield {"usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }}


async def stream_response(
    resp: aiohttp.ClientResponse,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """解析流式 SSE 响应。

    Args:
        resp: HTTP 响应对象。

    Yields:
        文本片段或元数据字典。
    """
    async for line in resp.content:
        text = line.decode("utf-8", errors="replace").strip()
        if not text or not text.startswith("data:"):
            continue
        data_str = text[5:].strip()
        if data_str == "[DONE]":
            break
        parsed = parse_sse_line(data_str)
        if parsed is not None:
            yield parsed


def build_chat_request(
    ks: KeyState,
    messages: List[Dict[str, Any]],
    model: str,
    stream: bool,
    **kw: Any,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """构造 nvidia 聊天补全请求的 URL/headers/payload。

    Args:
        ks: 目标 Key 状态。
        messages: 对话消息列表。
        model: 模型名。
        stream: 是否流式输出。
        **kw: 额外参数。

    Returns:
        (url, headers, payload) 三元组。
    """
    headers = build_headers(ks.key)
    payload = build_payload(messages, model, stream=stream, **kw)
    url = "{}{}".format(BASE_URL, CHAT_PATH)
    return url, headers, payload


async def dispatch_response(
    resp: aiohttp.ClientResponse,
    stream: bool,
    ks: KeyState,
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """校验响应状态码并转发至流式/非流式解析器。

    Args:
        resp: HTTP 响应对象。
        stream: 是否流式输出。
        ks: 对应的 Key 状态，非 200 时用于标记失败。

    Yields:
        文本片段(str)或结构化数据(dict)。

    Raises:
        PlatformError: HTTP 状态码非 200 时抛出。
    """
    if resp.status != 200:
        body = await resp.text()
        ks.mark_failure(resp.status)
        raise PlatformError(
            "nvidia HTTP{}: {}".format(resp.status, body[:300])
        )

    if not stream:
        async for chunk in nonstream_response(resp):
            yield chunk
    else:
        async for chunk in stream_response(resp):
            yield chunk

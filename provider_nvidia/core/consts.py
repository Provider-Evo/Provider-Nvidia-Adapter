


from typing import Dict, List

BASE_URL: str = "https://integrate.api.nvidia.com/v1"
CHAT_PATH: str = "/chat/completions"
MAX_TOKENS: int = 229376
RECOVERY_INTERVAL: int = 60

MODELS: List[str] = [
    "qwen/qwen3-coder-480b-a35b-instruct",
]

CAPS: Dict[str, bool] = {
    "chat": True,
    "completions": True,
    "responses": True,
    "tools": True,
    "native_tools": True,
}

FETCH_MODELS_ENABLED: bool = False
MODEL_FETCH_INTERVAL: int = 86400

# =======================================================================
# 重导出 — 同包内协同模块的公共符号（保持外部 ``from .. import`` 路径稳定）
# =======================================================================

from .headers import (
    build_headers,
)

from .payload import (
    build_payload,
)

from .helpers.sse import (
    parse_sse_line,
)

__all__ = [
    "build_headers",
    "build_payload",
    "parse_sse_line",
]

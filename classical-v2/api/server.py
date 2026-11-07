# -*- coding: utf-8 -*-
r"""Qwen3-4B-Classical 私有化推理网关（v2）。

客户端 -> 本服务(:8000, API Key) -> Ollama(:11434) / llama-server(:8080) -> qwen3-classical

相对 v1 的改动：
  1. 流式请求单独记 TTFT（首 token 时间）。v1 只在响应结束时打一条总耗时，
     写简历时拿不出 TTFT，等于把最值钱的指标丢了。
  2. 增加 /metrics（Prometheus 文本格式），暴露请求数、错误数、在途请求、
     TTFT 直方图。不用引第三方依赖。
  3. 明确注释：semaphore 是进程内的，uvicorn --workers>1 会失效。

⚠️ 部署注意
  - asyncio.Semaphore 只在单进程内生效。要开多 worker 必须换 Redis 分布式信号量，
    否则总并发 = workers × MAX_CONCURRENCY，实测会打爆显存。
  - 上游若是 llama-server，MAX_CONCURRENCY 应与它的 --parallel 对齐。
"""
import asyncio
import json
import logging
import os
import secrets
import time
import uuid

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("MODEL", "qwen3-classical")
BASE_MODEL = os.getenv("BASE_MODEL", "qwen3-base")
API_KEY = os.getenv("API_KEY", "").strip()

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "30"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))

if not API_KEY:
    raise RuntimeError("API_KEY is required; set it in environment or .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("classical-api")

app = FastAPI(title="Qwen3 Classical API", version="2.0.0")
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
ALLOWED_MODELS = {MODEL, BASE_MODEL}

TTFT_BUCKETS = [100, 250, 500, 1000, 2000, 5000]
STATS = {
    "requests": 0,
    "errors": 0,
    "in_flight": 0,
    "queued": 0,
    "ttft_sum_ms": 0.0,
    "ttft_count": 0,
    "buckets": {b: 0 for b in TTFT_BUCKETS},
    "inf": 0,
}


def observe_ttft(ms):
    STATS["ttft_sum_ms"] += ms
    STATS["ttft_count"] += 1
    for b in TTFT_BUCKETS:
        if ms <= b:
            STATS["buckets"][b] += 1
            return
    STATS["inf"] += 1


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="invalid api key")


def _error(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    STATS["errors"] += 1
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


def _headers(trace_id: str) -> dict:
    return {"X-Request-ID": trace_id, "Cache-Control": "no-cache"}


def _payload(req: ChatRequest) -> dict:
    model = req.model or MODEL
    if model not in ALLOWED_MODELS:
        raise ValueError(f"unknown model: {model}")
    max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "max_tokens": min(max_tokens, MAX_TOKENS),
    }
    if req.temperature is not None:
        payload["temperature"] = max(0.0, min(2.0, req.temperature))
    if req.top_p is not None:
        payload["top_p"] = max(0.0, min(1.0, req.top_p))
    if req.stop:
        payload["stop"] = req.stop
    return payload


@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
        return {"status": "ok", "ollama": "up", "model": MODEL, "in_flight": STATS["in_flight"]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "ollama": "down", "detail": str(exc)}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = [
        "# HELP classical_requests_total 已处理请求数",
        "# TYPE classical_requests_total counter",
        f"classical_requests_total {STATS['requests']}",
        "# HELP classical_errors_total 返回 4xx/5xx 的次数",
        "# TYPE classical_errors_total counter",
        f"classical_errors_total {STATS['errors']}",
        "# HELP classical_in_flight 正在生成的请求数",
        "# TYPE classical_in_flight gauge",
        f"classical_in_flight {STATS['in_flight']}",
        "# HELP classical_queued 排队等待信号量的请求数",
        "# TYPE classical_queued gauge",
        f"classical_queued {STATS['queued']}",
        "# HELP classical_ttft_ms 首 token 时延",
        "# TYPE classical_ttft_ms histogram",
    ]
    cum = 0
    for b in TTFT_BUCKETS:
        cum += STATS["buckets"][b]
        lines.append(f'classical_ttft_ms_bucket{{le="{b}"}} {cum}')
    lines.append(f'classical_ttft_ms_bucket{{le="+Inf"}} {STATS["ttft_count"]}')
    lines.append(f"classical_ttft_ms_sum {STATS['ttft_sum_ms']:.1f}")
    lines.append(f"classical_ttft_ms_count {STATS['ttft_count']}")
    return "\n".join(lines) + "\n"


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": MODEL, "object": "model", "owned_by": "local"},
            {"id": BASE_MODEL, "object": "model", "owned_by": "local"},
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat(req: ChatRequest, request: Request):
    trace_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    STATS["requests"] += 1

    try:
        payload = _payload(req)
    except ValueError as exc:
        return _error(400, str(exc))

    STATS["queued"] += 1
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        return _error(429, "server busy, retry later", "rate_limit_error")
    finally:
        STATS["queued"] -= 1
    STATS["in_flight"] += 1

    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    if not req.stream:
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                return _error(resp.status_code, resp.text[:500], "upstream_error")
            data = resp.json()
            usage = data.get("usage") or {}
            ms = (time.perf_counter() - started) * 1000
            logger.info(json.dumps({
                "trace_id": trace_id, "model": payload["model"], "stream": False,
                "status": resp.status_code, "ms": round(ms, 1),
                "tokens": usage.get("total_tokens"),
            }, ensure_ascii=False))
            return JSONResponse(data, headers=_headers(trace_id))
        except httpx.TimeoutException:
            return _error(504, "upstream timeout", "timeout_error")
        except Exception as exc:  # noqa: BLE001
            return _error(502, f"upstream error: {exc}", "upstream_error")
        finally:
            STATS["in_flight"] -= 1
            semaphore.release()

    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    try:
        upstream = await client.send(
            client.build_request("POST", url, json=payload), stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        await client.aclose()
        STATS["in_flight"] -= 1
        semaphore.release()
        return _error(502, f"upstream error: {exc}", "upstream_error")

    if upstream.status_code >= 400:
        body = (await upstream.aread()).decode("utf-8", "replace")[:500]
        await upstream.aclose()
        await client.aclose()
        STATS["in_flight"] -= 1
        semaphore.release()
        return _error(upstream.status_code, body, "upstream_error")

    async def stream_body():
        ttft_ms = None
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                        observe_ttft(ttft_ms)
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            STATS["in_flight"] -= 1
            semaphore.release()
            logger.info(json.dumps({
                "trace_id": trace_id, "model": payload["model"], "stream": True,
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }, ensure_ascii=False))

    return StreamingResponse(stream_body(), media_type="text/event-stream",
                             headers=_headers(trace_id))

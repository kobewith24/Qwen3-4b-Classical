# -*- coding: utf-8 -*-
r"""
Qwen3-4B-Classical 私有化推理网关。

客户端 -> 本服务(:8000, API Key) -> Ollama(:11434) -> qwen3-classical
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
from fastapi.responses import JSONResponse, StreamingResponse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("classical-api")

app = FastAPI(title="Qwen3 Classical API", version="1.0.0")
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
ALLOWED_MODELS = {MODEL, BASE_MODEL}


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
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_}},
    )


def _headers(trace_id: str) -> dict:
    return {"X-Request-ID": trace_id, "Cache-Control": "no-cache"}


def _payload(req: ChatRequest) -> dict:
    model = req.model or MODEL
    if model not in ALLOWED_MODELS:
        raise ValueError(f"unknown model: {model}")
    max_tokens = req.max_tokens or DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    max_tokens = min(max_tokens, MAX_TOKENS)

    payload = {
        "model": model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": req.stream,
        "max_tokens": max_tokens,
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
        return {"status": "ok", "ollama": "up", "model": MODEL}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "ollama": "down", "detail": str(exc)}


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

    try:
        payload = _payload(req)
    except ValueError as exc:
        return _error(400, str(exc))

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        return _error(429, "server busy, retry later", "rate_limit_error")

    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    if not req.stream:
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                return _error(resp.status_code, resp.text[:500], "upstream_error")
            data = resp.json()
            usage = data.get("usage") or {}
            logger.info(json.dumps({
                "trace_id": trace_id, "model": payload["model"], "stream": False,
                "status": resp.status_code, "ms": round((time.perf_counter() - started) * 1000, 1),
                "tokens": usage.get("total_tokens"),
            }, ensure_ascii=False))
            return JSONResponse(data, headers=_headers(trace_id))
        except httpx.TimeoutException:
            return _error(504, "upstream timeout", "timeout_error")
        except Exception as exc:  # noqa: BLE001
            return _error(502, f"upstream error: {exc}", "upstream_error")
        finally:
            semaphore.release()

    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    try:
        upstream = await client.send(
            client.build_request("POST", url, json=payload),
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        await client.aclose()
        semaphore.release()
        return _error(502, f"upstream error: {exc}", "upstream_error")

    if upstream.status_code >= 400:
        body = (await upstream.aread()).decode("utf-8", "replace")[:500]
        await upstream.aclose()
        await client.aclose()
        semaphore.release()
        return _error(upstream.status_code, body, "upstream_error")

    async def stream_body():
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            semaphore.release()
            logger.info(json.dumps({
                "trace_id": trace_id, "model": payload["model"], "stream": True,
                "status": upstream.status_code,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }, ensure_ascii=False))

    return StreamingResponse(
        stream_body(),
        media_type="text/event-stream",
        headers=_headers(trace_id),
    )

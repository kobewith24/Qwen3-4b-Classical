# Qwen3 Classical API

把本地 Ollama 的 `qwen3-classical` 包装成带鉴权、限流和日志的 HTTP 接口。

```text
客户端 -> FastAPI :8000 -> Ollama :11434 -> qwen3-classical
```

## 启动

```powershell
cd D:\Projects\Qwen3-4B-Classical
conda activate deepseek
pip install -r api\requirements.txt

$env:API_KEY="dev-key"
$env:MODEL="qwen3-classical"
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

也可以复制 `api/.env.example` 为 `api/.env` 后修改；**不要把真实 key 提交到 Git**。

启动前确认 Ollama 已经运行，且模型存在：

```powershell
ollama list
```

## 接口

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| GET | `/health` | 否 | 检查网关和 Ollama 状态 |
| GET | `/v1/models` | 是 | 可用模型列表 |
| POST | `/v1/chat/completions` | 是 | OpenAI 兼容，支持 stream |

鉴权头：

```text
Authorization: Bearer <API_KEY>
```

## 非流式

```powershell
curl.exe -s http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer dev-key" `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3-classical","messages":[{"role":"user","content":"帮我断句：学而时习之不亦说乎"}]}'
```

## 流式

```powershell
curl.exe -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer dev-key" `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3-classical","stream":true,"messages":[{"role":"user","content":"帮我断句：学而时习之不亦说乎"}]}'
```

返回为标准 `text/event-stream`，以 `data: [DONE]` 结束。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 地址 |
| `MODEL` | `qwen3-classical` | 微调模型名 |
| `BASE_MODEL` | `qwen3-base` | 对照模型名 |
| `API_KEY` | 无（必填） | Bearer Token |
| `MAX_CONCURRENCY` | `1` | 同时生成数；单卡 8GB 建议 1 |
| `QUEUE_TIMEOUT` | `30` | 排队超时秒数，超时返回 429 |
| `REQUEST_TIMEOUT` | `120` | 上游请求超时秒数 |
| `MAX_TOKENS` | `1024` | 单请求最大 token 上限 |
| `DEFAULT_MAX_TOKENS` | `512` | 未传 max_tokens 时使用 |

## 安全与并发

- 默认只监听 `127.0.0.1`；要内网访问再改 `0.0.0.0`，并加防火墙。
- 不要直接暴露 Ollama 的 `11434`。
- 单卡 8GB 下 `MAX_CONCURRENCY=1`，其他请求排队；排队超时返回 429。
- 网关日志只记录 trace_id、模型、耗时、token 数，不默认记录完整 prompt。
- 客户端断开时流式响应会结束并释放信号量，但 Ollama 已开始的生成可能仍会跑完。

## 错误码

| 状态码 | 含义 |
|---|---|
| 400 | 参数错误 / 未知模型 |
| 401 | 缺少或错误的 API Key |
| 429 | 并发队列已满 |
| 502 | Ollama 上游错误 |
| 504 | 上游超时 |

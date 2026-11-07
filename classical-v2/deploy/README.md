# 本地部署（RTX 4060 Laptop 8GB）+ 量化对比

训练、部署、测试全部在这张卡上，所以数字是端到端可复现的。

## 一、显存账

| 档位 | 权重 | 4k 上下文 KV | 合计 | 能否跑 |
|---|---:|---:|---:|---|
| f16 | 8.0 GB | ~0.6 GB | 8.6 GB | ❌ 超 8 GB |
| q8_0 | 4.2 GB | ~0.6 GB | 4.8 GB | ✅ 甜点，质量几乎无损 |
| q4_k_m | 2.5 GB | ~0.6 GB | 3.1 GB | ✅ 省显存，质量见 E3 |

训练时实测峰值 4.52 GB（4bit QLoRA + 8 步梯度累积 + seq 1024），
所以**训练和推理不能同时开**——8 GB 不够两边。

## 二、导出三档 GGUF

```bash
# 1) 合并 LoRA（fp16，约 8 GB 磁盘）
python train/merge.py --run-name final

# 2) 转 GGUF
python convert_hf_to_gguf.py ../weights/Qwen3-4B-Instruct-2507-Classical-final-fp16 \
    --outfile qwen3-4b-classical-final-f16.gguf --outtype f16

# 3) 量化两档
./llama-quantize qwen3-4b-classical-final-f16.gguf qwen3-4b-classical-q8.gguf  Q8_0
./llama-quantize qwen3-4b-classical-final-f16.gguf qwen3-4b-classical-q4.gguf  Q4_K_M
```

## 三、起服务（llama-server，不是 Ollama）

```bash
llama-server -m qwen3-4b-classical-q8.gguf \
    -ngl 99 -c 4096 --flash-attn -np 2 \
    --host 127.0.0.1 --port 8080
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `-ngl 99` | 全量上卡 | q8 的 4B 完全放得下 |
| `-c 4096` | 4096 | 断句样本 p99 只有 305 字，4k 够用；8k 会挤掉并发 |
| `-np 2` | 2 | 两个并行槽位；`-np 4` 会让 KV cache 翻倍，8 GB 容易触顶 |
| `--flash-attn` | 开 | 省显存、提速 |

网关（`api/server.py`）指向它：

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:8080"   # 变量名沿用 v1，其实就是"上游 OpenAI 兼容地址"
$env:MODEL="qwen3-classical"
$env:MAX_CONCURRENCY="2"                        # 必须和 -np 对齐
$env:API_KEY="dev-key"
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

> Qwen3-2507 是非 thinking 默认，不用额外关；如果你换了原版 Qwen3，
> 记得在 `config.py` 里把 `ENABLE_THINKING=False` 传进模板。

## 四、要测出来的数字（E3）

| 指标 | 命令 |
|---|---|
| 解码 tok/s、TTFT p50/p95 | `python api/loadtest.py --url http://127.0.0.1:8000/v1/chat/completions --key dev-key --concurrency 1,2 --requests 24 --tag q8` |
| q4 与 q8 的质量差 | `python eval/eval_v2.py --backend openai --vllm --base-url http://127.0.0.1:8080/v1 --tag q4` 各跑一次 |
| 显存峰值 | `nvidia-smi --query-gpu=memory.used --format=csv -l 1` |

**不要报 4 路以上并发**。8 GB 卡说 16 路一眼假；"2 路并发 p95 = X ms"才是真的。

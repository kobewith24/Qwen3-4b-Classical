# -*- coding: utf-8 -*-
r"""
绕过 `ollama create` 的 chtimes 报错，直接把 GGUF blob 注册成 Ollama 模型。

背景：
  Ollama 0.40.6 在本机 `ollama create` 时，把 GGUF 拷进 blobs 后会调用 chtimes
  设置修改时间，Windows 下报 "Access is denied"（Python os.utime 同一文件却成功），
  导致模型建不出来。blob 本身已经落盘且 digest 正确，只差 manifest。

做法：
  复制已有同架构模型的 manifest（config / template / params 三层完全复用），
  只把 model 层的 digest 与 size 换成新 GGUF 的，写进 manifests-v2。

用法：
  python deploy/import_gguf_manifest.py <新模型名> <gguf路径> [参考模型名]
  python deploy/import_gguf_manifest.py qwen3-classical-q4 weights/qwen3-4b-classical-q4km.gguf qwen3-classical
"""
import hashlib
import json
import os
import shutil
import sys

MODELS = os.environ.get("OLLAMA_MODELS", r"D:\Ollama\models")
MANIFEST_ROOT = os.path.join(MODELS, "manifests-v2", "ollama.com", "library")
BLOB_ROOT = os.path.join(MODELS, "blobs")


def sha256_of(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest_path(name, tag="latest"):
    return os.path.join(MANIFEST_ROOT, name.replace(":", "/"), tag)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    new_name, gguf = sys.argv[1], sys.argv[2]
    ref_name = sys.argv[3] if len(sys.argv) > 3 else "qwen3-classical"

    if not os.path.isfile(gguf):
        print(f"GGUF 不存在: {gguf}")
        return 1

    ref = manifest_path(ref_name)
    if not os.path.isfile(ref):
        print(f"参考 manifest 不存在: {ref}")
        return 1

    size = os.path.getsize(gguf)
    digest = sha256_of(gguf)
    blob = os.path.join(BLOB_ROOT, f"sha256-{digest}")

    if os.path.abspath(gguf) != os.path.abspath(blob):
        print(f">> 拷入 blobs（{size / 1024 ** 3:.2f} GB）")
        shutil.copyfile(gguf, blob)

    man = json.load(open(ref, encoding="utf-8"))
    hit = False
    for layer in man["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            layer["digest"] = f"sha256:{digest}"
            layer["size"] = size
            layer["from"] = os.path.basename(gguf)
            hit = True
    if not hit:
        print("参考 manifest 里没有 model 层")
        return 1

    out = manifest_path(new_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    print(f">> 已注册 {new_name}")
    print(f"   digest sha256:{digest[:16]}...  size {size / 1024 ** 3:.2f} GB")
    print(f"   manifest {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

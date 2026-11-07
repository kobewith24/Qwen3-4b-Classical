# 本地 4060 8GB 一键流程（Windows / PowerShell）
#
#   .\run_all.ps1 -OnlyStats    # 只量 v1 的监督 token 构成（2 分钟，不训练）
#   .\run_all.ps1 -OnlyBuild    # 只造 v2 数据（几分钟）
#   .\run_all.ps1 -Sweep        # E1 数据量扫描 2k/4k/8k（约 6 小时）
#   .\run_all.ps1 -Final        # 最终模型 8k x 2ep（约 6.5 小时）
#
# 实测：160 个样本 229.9s，即 1.44 s/样本；峰值显存 4.5 GB。
param(
    [string]$Py = "D:\Anaconda\envs\deepseek\python.exe",
    [switch]$OnlyStats,
    [switch]$OnlyBuild,
    [switch]$Sweep,
    [switch]$Final,
    [int]$FinalSize = 8000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run($argsArr) {
    & $Py @argsArr
    if ($LASTEXITCODE -ne 0) { throw "失败: $($argsArr -join ' ')" }
}

# 留出典籍：必须两部文件里都有，否则 OOD 的断句集会是 0 条
$Holdout = "北史,北齐书,礼记"

Step "0/4 统计 v1 的监督 token 构成"
Run @("train/finetune.py", "--stats-only", "--run-name", "v1",
      "--data", "../data/classical_dataset", "--model", "../model/Qwen3-4B-Instruct-2507")
if ($OnlyStats) { return }

Step "1/4 构建 v2 数据集"
# 不加通用数据：实测 v1 那 2000 条"通用"是 100% 英文（CJK 比例中位数 = 0），
# 却占了 27.3% 的监督信号，混进来比不混更糟。
Run @("data/build_dataset_v2.py", "--no-general", "--holdout-books", $Holdout,
      "--out", "../data/v2")
if ($OnlyBuild) { return }

if ($Sweep) {
    foreach ($n in @(2000, 4000, 8000)) {
        Step "2/4 E1 数据量=$n（1 epoch，约 $([math]::Round($n * 1.44 / 60)) 分钟）"
        Run @("train/finetune.py", "--run-name", "e1_$n", "--limit", "$n",
              "--epochs", "1", "--grad-accum", "8", "--eval-steps", "250")
    }
}

if ($Final) {
    Step "3/4 最终模型（$FinalSize 条 x 2 epoch）"
    Run @("train/finetune.py", "--run-name", "final", "--limit", "$FinalSize",
          "--epochs", "2", "--grad-accum", "8", "--eval-steps", "250")
    Step "4/4 合并"
    Run @("train/merge.py", "--run-name", "final")
}

Write-Host "`n下一步：" -ForegroundColor Green
Write-Host "  $Py eval/eval_v2.py --backend hf --load-4bit --tag final"

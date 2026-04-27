# Sequentially train four ERRNet variants (separate checkpoint folders under ./checkpoints/<name>).
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File .\scripts\train_four_variants.ps1
# Or from repo root:
#   .\scripts\train_four_variants.ps1
#
# Optional: -Gpu 1 -NoRealDataOnly
param(
    [int] $Gpu = 0,
    [switch] $NoRealDataOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoRoot

# --nThreads 0: avoids Windows RuntimeError on shared file mapping (e.g. after long runs)
$common = @("python", "train_errnet.py", "--gpu_ids", "$Gpu", "--nThreads", "0")
if (-not $NoRealDataOnly) { $common += "--real_data_only" }

function Invoke-Train {
    param([string[]]$ExtraArgs, [string]$Label)
    Write-Host "========== $Label ==========" -ForegroundColor Cyan
    $argv = $common + $ExtraArgs
    & $argv[0] $argv[1..($argv.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $Label (exit $LASTEXITCODE)" }
}

# 0) baseline: neither hyper nor dark_channel
Invoke-Train -Label "ab_00_baseline (3ch in)"        -ExtraArgs @("--name", "ab_00_baseline")
# 1) paper-style hypercolumn
Invoke-Train -Label "ab_01_hyper"                     -ExtraArgs @("--name", "ab_01_hyper", "--hyper")
# 2) dark channel prior
Invoke-Train -Label "ab_02_dark_channel"              -ExtraArgs @("--name", "ab_02_dark_channel", "--dark_channel")
# 3) hyper + dark channel
Invoke-Train -Label "ab_03_hyper_and_dark_channel"   -ExtraArgs @("--name", "ab_03_hyper_and_dark_channel", "--hyper", "--dark_channel")

Write-Host "All four runs finished. Checkpoints under: $RepoRoot\checkpoints\ab_0*\*" -ForegroundColor Green

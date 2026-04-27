# Run test_errnet.py for all four ablation checkpoints. All outputs go under
#   result_dir / save_subdir / <image name> /   with per-variant file names
#   (e.g. errnet_model_ab_00_baseline.png, m_input_ab_00_baseline.png, ...).
# After all runs, prints quality metrics and Conv1 input-group mean |W| (RGB / DCP / hyper)
#   from metrics_*.json (see util/input_channel_weights.py).
#
# Usage (from repo root E:\ERRNet):
#   powershell -ExecutionPolicy Bypass -File .\scripts\test_four_checkpoints.ps1
#
# Optional:
#   .\scripts\test_four_checkpoints.ps1 -Dataset real20 -ResultDir "./results/ablation_compare" -SaveSubdir "real20" -Gpu 0
#
param(
    [string] $Dataset = "real20",
    [string] $DataRoot = "./datasets/processed_data",
    [string] $ResultDir = "./results/ablation_compare",
    [string] $SaveSubdir = "real20",
    [int] $Gpu = 0,
    [string] $CheckpointRoot = "./checkpoints",
    [string] $WeightName = "errnet_latest.pt"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoRoot

$variants = @(
    @{  Tag = "ab_00_baseline";              Extra = @() },
    @{  Tag = "ab_01_hyper";                 Extra = @("--hyper") },
    @{  Tag = "ab_02_dark_channel";          Extra = @("--dark_channel") },
    @{  Tag = "ab_03_hyper_and_dark_channel"; Extra = @("--hyper", "--dark_channel") }
)

# Must match keys from util/index.py quality_assess. Used so Format-Table always
# shows these columns: PS defaults to the *first* object's property set, so a row
# with only { Tag } (e.g. empty metrics json) would otherwise hide PSNR/SSIM/etc.
$MetricKeys = @('PSNR', 'SSIM', 'LMSE', 'NCC')
# test_errnet.py JSON (merged with first-conv group stats + eval_diagnostics)
$WeightKeys = @('w_mean_abs_RGB', 'w_mean_abs_dark', 'w_mean_abs_hyper')
$SaliencyKeys = @('sal_grad_l1_RGB', 'sal_grad_l1_DCP', 'sal_grad_l1_hyper')
$ActivationKeys = @('act_conv1_mean_l1', 'act_conv1_fro_l2')
$CovKeys = @('cov_pooled_tr', 'cov_pooled_fro', 'cov_pooled_off_fro')

$summaryRows = [System.Collections.Generic.List[object]]::new()
$outRoot = Join-Path $RepoRoot (Join-Path $ResultDir $SaveSubdir)

function Build-MetricsRow {
    param([string] $Tag, $JsonObj)
    $row = [ordered]@{ Tag = $Tag }
    $names = if ($null -ne $JsonObj) { $JsonObj.PSObject.Properties.Name } else { @() }
    foreach ($k in $script:MetricKeys) {
        if ($names -contains $k) { $row[$k] = $JsonObj.$k } else { $row[$k] = $null }
    }
    foreach ($k in $script:WeightKeys) {
        if ($names -contains $k) { $row[$k] = $JsonObj.$k } else { $row[$k] = $null }
    }
    foreach ($k in $script:SaliencyKeys) {
        if ($names -contains $k) { $row[$k] = $JsonObj.$k } else { $row[$k] = $null }
    }
    foreach ($k in $script:ActivationKeys) {
        if ($names -contains $k) { $row[$k] = $JsonObj.$k } else { $row[$k] = $null }
    }
    foreach ($k in $script:CovKeys) {
        if ($names -contains $k) { $row[$k] = $JsonObj.$k } else { $row[$k] = $null }
    }
    return [pscustomobject]$row
}

foreach ($v in $variants) {
    $ckpt = Join-Path $RepoRoot (Join-Path $CheckpointRoot (Join-Path $v.Tag $WeightName))
    if (-not (Test-Path -LiteralPath $ckpt)) {
        throw "Missing checkpoint: $ckpt`nTrain that variant first, or set -WeightName to an existing .pt"
    }
    $metricsName = "metrics_$($v.Tag).json"
    Write-Host "========== Testing $($v.Tag) (shared: $ResultDir / $SaveSubdir, tag=$($v.Tag)) ==========" -ForegroundColor Cyan
    $argv = @(
        "python", "test_errnet.py",
        "--dataset", $Dataset,
        "--data_root", $DataRoot,
        "--result_dir", $ResultDir,
        "--save_subdir", $SaveSubdir,
        "--file_tag", $v.Tag,
        "--metrics_out", $metricsName,
        "-r",
        "--icnn_path", $ckpt,
        "--gpu_ids", "$Gpu"
    ) + $v.Extra
    & $argv[0] $argv[1..($argv.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "test_errnet failed for $($v.Tag) (exit $LASTEXITCODE)" }

    $mj = Join-Path $outRoot $metricsName
    if (Test-Path -LiteralPath $mj) {
        $raw = Get-Content -Raw -Encoding UTF8 $mj
        if ([string]::IsNullOrWhiteSpace($raw)) {
            Write-Warning "Empty file: $mj"
        }
        $j = $raw | ConvertFrom-Json
        $summaryRows.Add((Build-MetricsRow -Tag $v.Tag -JsonObj $j)) | Out-Null
    } else {
        Write-Warning "Metrics file not found: $mj"
        $summaryRows.Add((Build-MetricsRow -Tag $v.Tag -JsonObj $null)) | Out-Null
    }
}

Write-Host ""
Write-Host "========== Average image metrics (all variants) ==========" -ForegroundColor Green
# Wide Out-String avoids Format-Table truncating long numeric lines on narrow consoles
$summaryRows | Format-Table -Property (@('Tag') + $MetricKeys) -AutoSize | Out-String -Width 4096 | Write-Host
Write-Host "========== Conv1 mean |W| by input group (RGB / DCP / hyper) ==========" -ForegroundColor Green
Write-Host "(from first-layer weights; DCP/hyper empty if that branch was not used for the checkpoint.)" -ForegroundColor DarkGray
$summaryRows | Format-Table -Property (@('Tag') + $WeightKeys) -AutoSize | Out-String -Width 4096 | Write-Host
Write-Host "========== Saliency |dL/dx| L1 (L1 = L1 loss; grad w.r.t. net first-layer input) ==========" -ForegroundColor Green
$summaryRows | Format-Table -Property (@('Tag') + $SaliencyKeys) -AutoSize | Out-String -Width 4096 | Write-Host
Write-Host "========== Activation @ conv1: mean_c[(1/HW)sum|F|], mean[||F||_2] over images ==========" -ForegroundColor Green
$summaryRows | Format-Table -Property (@('Tag') + $ActivationKeys) -AutoSize | Out-String -Width 4096 | Write-Host
Write-Host "========== Pooled (global-avg) feature cov @ conv1 across N test images ==========" -ForegroundColor Green
Write-Host "tr=mean(diag C), fro=||C||_F, off_fro=||C - diag(C)||_F" -ForegroundColor DarkGray
$summaryRows | Format-Table -Property (@('Tag') + $CovKeys) -AutoSize | Out-String -Width 4096 | Write-Host
Write-Host "Images: $outRoot" -ForegroundColor Green
Write-Host "Per-variant JSON: $outRoot\metrics_*.json" -ForegroundColor DarkGray

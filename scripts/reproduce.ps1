param(
    [string]$Data,
    [string]$OutputRoot = "outputs/reproduction",
    [switch]$Smoke,
    [switch]$IncludeAblations,
    [switch]$IncludeDiscreteTime
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepositoryRoot
$env:PYTHONPATH = Join-Path $RepositoryRoot "src"

$Arguments = @("scripts/reproduce.py", "--output-root", $OutputRoot)
if ($Data) { $Arguments += @("--data", $Data) }
if ($Smoke) { $Arguments += "--smoke" }
if ($IncludeAblations) { $Arguments += "--include-ablations" }
if ($IncludeDiscreteTime) { $Arguments += "--include-discrete-time" }

python @Arguments

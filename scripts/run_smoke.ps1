$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"

python -m p_help_hcc.data make-synthetic --out data/synthetic_hcc.csv --n 120 --seed 42
python -m p_help_hcc.train --config configs/default.yaml --data data/synthetic_hcc.csv --output outputs/smoke --fast
python -m p_help_hcc.test --model outputs/smoke/fold_0/model.joblib --data data/synthetic_hcc.csv --split outputs/smoke/splits_seed_42.json --fold 0
python -m p_help_hcc.validate --data data/synthetic_hcc.csv --model outputs/smoke/fold_0/model.joblib

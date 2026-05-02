<div align="center">

# P-HELP-HCC

**面向肝細胞癌存活分層的平行階層可解釋學習管線**

論文配套參考實作，嚴格遵照 Phase&nbsp;A / C / E / P 四階段方法章節與八分類 HCC 實驗設定。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-research--code-orange.svg)](#)

[English](README.md) &nbsp;|&nbsp; **繁體中文**

</div>

---

## 目錄

1. [專案概述](#專案概述)
2. [架構一覽](#架構一覽)
3. [實作的論文元件](#實作的論文元件)
4. [快速開始](#快速開始)
5. [專案結構](#專案結構)
6. [設定檔與超參數](#設定檔與超參數)
7. [真實資料合約](#真實資料合約)
8. [驗證流程](#驗證流程)
9. [可重現性說明](#可重現性說明)
10. [引用方式](#引用方式)

---

## 專案概述

P-HELP-HCC 是一個面向肝細胞癌（hepatocellular carcinoma, HCC）的平行系統存活分層框架，將八分類預後任務組織為以下四個 ACP 階段：

| 階段 | 主題 | 輸出 |
|------|------|------|
| **A** | 由六個互動智能體組成的人工社會 | 46 維智能體狀態 $\mathbf{S}_i$ |
| **C** | 計算實驗（存活、手術情境、反事實） | 類別機率、RMST 差距、情境效應 |
| **E** | 多層可解釋存活模型 | Cox HR、SHAP、表型、反事實、回饋 |
| **P** | 平行執行與自適應更新 | 串流虛擬–真實誤差控制器 |

私有的 673 名病人佇列**不**會隨此倉庫散佈。本程式碼支援兩種資料模式：

- 符合[資料合約](#真實資料合約)的真實 CSV / TSV / XLSX / Parquet 輸入；
- 以論文公開佇列統計為依據生成的合成 HCC 佇列，用於乾跑與可重現性檢核。

---

## 架構一覽

```
                                                                                    
            +-----------------+         +--------------------+   Phase E            
   x_i  --> | Phase A         | S_i --> | Phase C            | -----------+         
  67-dim    | Society         | 46-dim  | 4 學習器集成       |            v         
            | Transformer     |         | + K-means K=4      |   +-----------------+
            +-----------------+         | + 反事實掃掠       |   | Cox-EN, SHAP,   |
                  ^                     +--------------------+   | cluster, CF,    |
                  |                              |               | feedback layers |
                  |                              v               +-----------------+
            +-----------------+         +--------------------+            |         
            | Phase P         | <------ | 虛擬–真實誤差     |  <---------+         
            | controller      |  retrain| 串流監測          |   IPCW Brier        
            +-----------------+         +--------------------+                      
```

> K-means 表型分支在**67 維精選輸入**（公式 7）上運作；存活集成與反事實滾動則在**46 維智能體狀態**（公式 4）上運作；Phase P 校準損失採用公式 13 的 **IPCW 多項式 Brier**，當分母低於數值底限時自動退化為僅成熟（matured-only）版本。

---

## 實作的論文元件

- **資料層**：資料集載入、規範欄位驗證、存活標籤推導，以及 67 維精選特徵 schema（第 8.1 節）。
- **資料切分**：以五個種子 `{42, 123, 2024, 31415, 65537}` 進行病人層級重複 5-fold 交叉驗證，分層鍵為八分類標籤與手術策略軸（第 8.2 節）。
- **Phase A**：人工社會投影至 $d_S{=}46$ 智能體狀態，並使用表 II 校準的動力學常數（Gompertz 腫瘤成長、AFP 對數域遞推、肝纖維化更新、治療策略、Child–Pugh 可行性上限）。
- **Phase C — 存活集成**：四個異質基學習器以 Brier 最佳融合：正則化多項式邏輯回歸、XGBoost（缺則退化為 Gradient Boosting）、校準隨機森林、加類別權重之焦點損失 PyTorch MLP（第 5.1 節）。
- **Phase C — 表型分支**：保留 95% 變異量的 PCA 加上 $K{=}4$ 之 K-means，輸出 silhouette / Davies–Bouldin / Calinski–Harabasz 等內部有效性指標。
- **Phase C — 反事實掃掠**：六動作集 `{None, Resection, TACE, RFA, Sorafenib, Combo}`、傾向分數閘 $[0.05,\,0.95]$、指南信心門檻 $\rho^{\star}{=}0.6$、$B{=}200$ 自助重抽，並輸出 $P(\mathrm{OS}{>}12\,\mathrm{m})$ 三方案治療臂報告（第 5.4 節 + 演算法 1）。
- **Phase E**：以 PyTorch 偏概似實作的 Cox 彈性網風險層、SHAP 與排列重要性工具、Cox–SHAP 排序對齊、$\tanh$ 銳度 $\kappa{=}5$ 的解釋一致性損失（第 6 節）。
- **Phase P**：串流虛擬–真實誤差控制器，門檻 $\bar{e}_{\text{soft}}{=}0.18$ / $\bar{e}_{\text{hard}}{=}0.32$、線上步長 $\eta{=}10^{-4}$、近端錨定 $\lambda_w{=}10^{-3}$、監測視窗 $n_b{=}30$、再訓練緩衝 $n_r{=}200$、誤差混合權重 $\alpha_e{=}0.5$（第 7 節 + 演算法 2）。

---

## 快速開始

以可編輯模式安裝後，對合成佇列執行一次冒煙測試：

```powershell
python -m pip install -e .

# 1. 生成符合論文類別 / 手術策略比例的小型合成佇列
python -m p_help_hcc.data make-synthetic --out data/synthetic_hcc.csv --n 120

# 2. 以 --fast 冒煙設定訓練完整管線（Phase A -> C -> E）
python -m p_help_hcc.train --config configs/default.yaml `
                           --data   data/synthetic_hcc.csv `
                           --output outputs/smoke `
                           --fast

# 3. 對保留折進行評估
python -m p_help_hcc.test --model outputs/smoke/fold_0/model.joblib `
                          --data  data/synthetic_hcc.csv `
                          --split outputs/smoke/splits_seed_42.json `
                          --fold  0

# 4. 執行佇列層級的驗證程序
python -m p_help_hcc.validate --data  data/synthetic_hcc.csv `
                              --model outputs/smoke/fold_0/model.joblib
```

> 完整的論文規模設定（5 種子 × 5 折 = 25 次執行、完整估計器數量、100 個 MLP epoch）寫在 `configs/default.yaml` 內。當真實佇列與目標工作站準備就緒後，再移除 `--fast` 旗標。

---

## 專案結構

```
code/
├── configs/
│   ├── default.yaml          # 論文規模超參數
│   └── search_grid.yaml      # 第 8.2 節之巢狀搜尋網格
├── data/                     # 不含 PHI 的本機輸入目錄（被 git 忽略）
├── outputs/                  # 執行產物（被 git 忽略）
├── scripts/run_smoke.ps1     # 一鍵冒煙腳本
├── src/p_help_hcc/
│   ├── society.py            # Phase A：SocietyTransformer 與動力學
│   ├── clustering.py         # Phase C：PCA + K-means 表型路由
│   ├── ensemble.py           # Phase C：4 學習器 Brier 最佳堆疊
│   ├── neural.py             # Phase C：焦點損失 MLP
│   ├── counterfactual.py     # Phase C：情境掃掠 + 傾向分數閘
│   ├── cox.py                # Phase E：Cox 彈性網（Torch）
│   ├── explain.py            # Phase E：SHAP + IPCW Brier + L_exp / L_clin
│   ├── parallel.py           # Phase P：串流誤差控制器
│   ├── pipeline.py           # 端到端 PHelpHCCPipeline
│   ├── splits.py             # 病人層級重複 5-fold 切分
│   ├── preprocessing.py      # 67 特徵精選與插補
│   ├── data.py               # 資料載入與合成佇列生成器
│   └── train.py / test.py / validate.py
└── tests/                    # 單元與冒煙測試
```

---

## 設定檔與超參數

`configs/default.yaml` 已將論文最終選定值編碼。最常被查閱的數值如下：

| 區塊 | 符號 | 數值 |
|------|:----:|------:|
| 精選輸入維度 | $d$ | $67$ |
| 智能體狀態維度 | $d_S$ | $46$ |
| 外層 / 內層 CV | folds × seeds | $5 \times 5$ |
| 表型數 | $K_c^{\star}$ | $4$ |
| MLP 主幹 | hidden / dropout / 啟動函式 | $[256,128,64]$ / $0.2$ / GELU |
| 最佳化器 | Adam, lr, batch, epochs, patience | $10^{-3}$, $32$, $100$, $15$ |
| 焦點損失 | $\gamma$ | $1.5$ |
| 隨機森林 | $n_{\text{est}}$, max depth | $500$, $10$ |
| XGBoost | $n_{\text{est}}$, lr, max depth | $500$, $0.05$, $6$ |
| 融合 | $\alpha_{\text{fuse}}$ | $0.60$ |
| 反事實 | $B$, propensity gate, $\rho^{\star}$ | $200$, $[0.05,0.95]$, $0.6$ |
| Phase E 損失 | $\gamma_1{=}\lambda_{\text{cal}}$ / $\gamma_2{=}\lambda_{\text{exp}}$ / $\gamma_3{=}\lambda_{\text{clin}}$ | $1.0$ / $0.2$ / $0.1$ |
| Phase E 損失 | $\tanh$ 銳度 $\kappa$ | $5.0$ |
| Cox 彈性網 | epochs, lr, $\lambda_{\ell_1}$, $\lambda_{\ell_2}$ | $300$, $0.03$, $10^{-3}$, $10^{-3}$ |
| Phase P 門檻 | $\bar e_{\text{soft}}$ / $\bar e_{\text{hard}}$ | $0.18$ / $0.32$ |
| Phase P 視窗 | $n_b$ / $n_r$ | $30$ / $200$ |
| 類別權重 | C1 … C8 | `[1.0, 1.5, 1.7, 2.1, 2.5, 2.7, 2.3, 4.5]` |

巢狀超參搜尋網格寫在 `configs/search_grid.yaml`，與第 8.2 節的描述逐格對應。

---

## 真實資料合約

**必填欄位**

| 欄位 | 型別 | 說明 |
|---|---|---|
| `overall_survival_months` | float | 自診斷起的整體存活月數 |
| `event` | int (0/1) | $1$ 表示死亡 / 事件，$0$ 表示審查 |

**可選欄位（缺漏時自動推導）**

| 欄位 | 允許值 |
|---|---|
| `survival_class` | `0..7` 或 `C1..C8` |
| `surgical_strategy` | `none` / `ablation` / `resection` |
| `dominant_aetiology` | `HBV` / `HCV` / `NBNC` |
| 臨床共變量 | `age`、`sex_male`、`tumor_size_cm`、`afp`、`albumin`、`bilirubin`、`inr`、`ajcc_stage`、治療旗標等 |

預處理管線會把可用的臨床欄位映射到穩定的 67 特徵 schema（`x_00` … `x_66`）。若這些 `x_*` 欄位本就存在，則被視為既有精選特徵矩陣。自由文字或識別欄位永遠不會序列化進入模型成品。

> **資料衛生**：請勿將 PHI 放入 `data/` 進行散佈。常見表格格式以及 `data/raw/`、`data/private/` 目錄已預設由 Git 忽略。`joblib` 與 pickle 模型在載入時可執行任意程式碼，因此僅應載入本機產出或來自可信 release 的模型。

Excel 與 Parquet 載入需要對應的 pandas 引擎，例如 `.xlsx` 需要 `openpyxl`、`.parquet` 需要 `pyarrow`。

---

## 驗證流程

本機測試：

```powershell
python -m unittest discover -s tests
```

冒煙管線（合成佇列 + `--fast` 預算）：

```powershell
python -m p_help_hcc.train --config configs/default.yaml `
                           --data   data/synthetic_hcc.csv `
                           --output outputs/smoke `
                           --fast
```

每折會輸出 `metrics.json`、訓練好的 `model.joblib` 以及該折的切分清單。`metrics.csv` 與 `metrics_summary.json` 會彙總全部 25 次論文執行的結果。

---

## 可重現性說明

- 位元級可重現以主種子 `42` 為起點，傳遞到 NumPy、PyTorch（CPU/CUDA）、Python `random`、scikit-learn 與 XGBoost；確定性 CUDA 由 `torch.use_deterministic_algorithms(True)` 與 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 啟用。
- 25 次執行協定（5 種子 × 5 折）以平均值 ± 標準差呈現，並搭配 $1{,}000$ 次重抽的百分位自助信賴區間，與基於標準差的區間在每個指標上吻合至 $\pm 0.005$ 以內。
- Phase P 控制器是平行執行迴路的**回顧式模擬**。前瞻性部署需要先進行未來的靜默影子執行，再啟用軟 / 硬門檻規則。

---

## 引用方式

若使用本程式碼或在此框架上延伸，請引用論文：

```bibtex
@article{phelp_hcc,
  title   = {P-HELP-HCC: Parallel Hierarchical Explainable Learning Pipeline
             for Hepatocellular Carcinoma Survival Stratification},
  author  = {Anonymous},
  journal = {Manuscript under review},
  year    = {2026}
}
```

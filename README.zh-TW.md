# P-HLPL-HCC

**用於肝細胞癌八類存活分層的平行分層可解釋學習流程**

本專案是論文隨附的研究程式。它提供可執行的分析路徑與非證據性的 smoke test；不公開 673 位病人的私有資料，也不宣稱已在缺少受控執行產物時重現論文數值。

[English](README.md) | **繁體中文**

## 已實作範圍

- 資料端點：從診斷索引日起計算存活時間；在 72 個月以前失訪且尚未死亡者，不會被強制指定八類硬標籤。
- 切分：五個種子下的病人層級五折交叉驗證，預設同時依存活類別與事件指標分層；小樣本 strata 會採用確定性的降級策略。
- 離散時間存活：七個區間 hazard 加上 72 個月後的 C8 尾端機率；訓練、驗證與評估均使用只在訓練折擬合的 reverse-Kaplan--Meier IPCW cell weights。預測表會輸出七個 hazard 欄位，風險／事件／設限表包含 72 個月尾端列。
- 六區塊病人狀態：Patient、Tumor、Liver、Treatment、Guideline 與 Explanation 均有可執行的 drop variant。主分類器只使用標準化的靜態狀態。
- 物理單位動態規則：以公分與月份表示的 Gompertz 與纖維化公式只可用於獨立的 longitudinal back-test；不得套用到標準化 latent state。公開 back-test 未實作獨立 AFP transition。未提供真實縱向輸入時，設定檔本身不是數值證據。
- Phase C：四個 base learners、Brier-optimal fusion、K-means phenotype branch，以及共用 backbone 的六類 scenario auxiliary head。
- 單一病人 scenario：propensity 只使用 Treatment 之前的 Patient/Tumor/Liver 狀態；每個 arm 都會把 factual-treatment-derived auxiliary coordinates 設為共同中性值；臨床顯示需要恰好 `B=200` 個有限的外部病人層級 bootstrap 預測、guideline confidence、propensity overlap 與不跨零的信賴區間。缺少任一輸入時，預設臨床顯示為空；門檻為 `rho*=0.30`。
- Cohort-level observational analysis：輸出 naive、IPTW、cross-fitted AIPW/DR、overlap retention、調整前後 SMD、E-value，以及 IPTW-KM RMST。使用 `B=1000` 病人重抽樣；RMST 每次重抽樣都重新擬合 propensity，AIPW score bootstrap 則明確保留既有的 cross-fitted nuisance predictions。所有結果都只是 observational sensitivity summaries，不是因果治療效果或治療建議。
- Phase E：先在 46 維狀態擬合 Cox direction，再把 `L_cal`、`L_exp` 與 `L_clin` 以 `0.4/0.3/0.2` 權重加入 MLP 並反向傳播。`L_exp` 使用可微的 input-gradient proxy，不是把 SHAP 放進訓練迴圈。三個 named loss-drop variants 會把對應權重設為零。
- Phase P：驗證 replay residual 會影響 censoring-informed 訓練權重與 MLP validation-stream checkpoint selection；one-vs-rest Platt calibrator 會被 `predict_proba` 實際呼叫。Residual 定義為 `1-P(true)+alpha_e*sum_c(P_c-onehot_c)^2`。A6 會同時關閉三條路徑。
- Locked external validation：完整 preprocessing、imputation、模型、校準與 threshold contract 必須在接觸外部資料前凍結；target anchor recalibration 會另列為 sensitivity analysis，不會冒充 strict locked validation。

訓練與重現命令沒有執行 nested hyperparameter search。Within-fold validation stream 只用於 checkpoint／候選評分；候選 grid 是可選的探索工具，不能視為已完成 nested-CV 的證據。

## 快速開始

```powershell
python -m pip install -e .

# 僅供 schema、unit 與 smoke 軟體測試；fixture 記錄不是臨床觀察，
# 不得用於論文實驗、數值重現或科學推論。
python -m p_hlpl_hcc.data make-fixture --out data/fixture_hcc.csv --n 120
python -m p_hlpl_hcc.train --config configs/default.yaml `
  --data data/fixture_hcc.csv --output outputs/smoke --fast

# 設限感知的離散時間路徑。
python -m p_hlpl_hcc.train --config configs/discrete_time.yaml `
  --data data/fixture_hcc.csv --output outputs/discrete --fast

# 單元與整合測試。
python -m pytest -q
```

物理單位動態只能透過獨立 back-test 執行：

```powershell
python scripts/run_dynamics_backtest.py --data <longitudinal.csv> `
  --variant full_dynamics --output-dir outputs/dynamics
```

主分類器明確拒絕 `--dynamics`，避免把公分／月份公式錯誤套到標準化狀態。

## 主要設定

| 區塊 | 設定 |
|---|---|
| 重複 outer CV | 5 folds × 5 seeds |
| Within-fold validation | 20% |
| 輸入／病人狀態 | 67／46 維 |
| 存活區間 | 0、6、12、24、36、48、60、72 個月 |
| Scenario／cohort bootstrap | 200／1000 |
| Propensity trim／guideline threshold | 0.05／0.30 |
| Phase-E loss weights | 0.4／0.3／0.2 |
| Phase-P residual mix | 0.5 |

## 真實資料與證據邊界

必要欄位為 `overall_survival_months` 與二元 `event`。`survival_class` 可提供 `0..7`、`1..8` 或 `C1..C8`；若未提供且端點可確定，程式會依區間產生。自由文字與識別欄位不會寫入模型。

完整論文驗證仍需要受控的私有 split manifests、每折預測與 checkpoints、外部 cohort、縱向物理單位資料、圖表來源表，以及實測裝置功耗／延遲／峰值記憶體 logs。缺少這些產物時，程式通過 smoke test 不等同於重現論文中的數值結果。

## 引用

```bibtex
@article{phelp_hcc,
  title   = {Parallel Explainable Internet of Medical Things Framework with
             a Structured Multi-Agent Patient-State Representation for
             Hepatocellular Carcinoma Survival Prediction},
  author  = {Wen-Dong Jiang and Tsung-Jung Lin and Chih-Yung Chang},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Submitted}
}
```

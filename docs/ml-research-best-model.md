# Titanic 最佳模型解法 — Phase 1 研究筆記

> 研究日期：2026-07-03  
> 實作收尾：2026-07-06  
> 專案：Kaggle [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)  
> 階段：Phase 1 研究完成 → Phase 3 實作完成（Step 5 LB **0.81578**）

---

## Research summary

| 項目 | 內容 |
|------|------|
| **Task** | 二元分類：依乘客特徵預測 `Survived`（0/1），評估指標為 **Accuracy** |
| **Libraries / versions assumed** | `pandas`, `numpy`, `scikit-learn` ≥1.3；進階可選 `xgboost`, `lightgbm`, `catboost`, `optuna` |
| **Key APIs / classes** | `Pipeline`, `ColumnTransformer`, `StratifiedKFold`, `cross_val_score`；進階：`LGBMClassifier`, `CatBoostClassifier`, `StackingClassifier` |
| **Example sources** | [sklearn Titanic ColumnTransformer](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)、[Alexis Cook Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)、[Geeky Codes RF 0.837](https://geekycodes.in/python/titanic-advanced-feature-engineering-tutorial/)、[HF eriksarriegui/titanic-survival-predictor](https://huggingface.co/eriksarriegui/titanic-survival-predictor) |
| **Dataset format rules** | 訓練：`train.csv` 891 列含 `Survived`；測試：`test.csv` 418 列；提交：`PassengerId,Survived` 共 419 行（含 header） |
| **Hardware hint** | 本地 CPU 即可；全資料 <1k 列，無需 GPU / HF Job |
| **Risks / unknowns** | CV 與 public LB 常有 2–5% 落差；過度 stacking 易過擬合；`Cabin`/`Ticket` 高缺失需謹慎處理 |

---

## 1. 問題定義與資料輪廓

### 1.1 任務性質

這是 **小樣本表格二元分類**，不是 NLP / 深度學習任務。最佳解法應以 **特徵工程 + 樹模型集成** 為主，而非 transformers / TRL。

### 1.2 本地資料統計（`data/train.csv`, `data/test.csv`）

| 項目 | 數值 |
|------|------|
| 訓練集 | 891 列 × 12 欄 |
| 測試集 | 418 列 × 11 欄（無標籤） |
| 目標分布 | 未生還 61.6% / 生還 38.4%（略不平衡） |
| `Age` 缺失 | train 177、test 86 |
| `Cabin` 缺失 | train 687 (77%)、test 327 (78%) |
| `Embarked` 缺失 | train 2 |
| `Fare` 缺失 | test 1 |

### 1.3 強訊號（EDA 結論）

| 特徵 | 觀察 | 含義 |
|------|------|------|
| `Sex` | female 生還率 74.2% vs male 18.9% | 「婦女幼兒優先」政策，最強單一因子之一 |
| `Pclass` | 1st 63.0% > 2nd 47.3% > 3rd 24.2% | 艙等與社經地位、甲板位置相關 |
| `Name` | 含 Mr/Mrs/Miss/Master 等稱謂 | 可萃取 Title，代理年齡與社會角色 |
| `SibSp` + `Parch` | 家庭結構 | 可組成 FamilySize、IsAlone |
| `Cabin` | 77% 缺失 | 首字母可當 Deck，但增益有限且模型敏感 |

**Baseline 參考**：僅用 `Sex` 預測（全 female=1, male=0）約 **0.765** public score（`gender_submission.csv` 思路）。

---

## 2. 文獻與社群解法掃描

### 2.1 準確率天花板（經驗區間）

| 層級 | CV Accuracy | Public LB | 說明 |
|------|-------------|-----------|------|
| 入門 baseline | 0.75–0.78 | 0.76–0.78 | Sex + Pclass + 簡單填補 |
| 標準解法 | 0.80–0.84 | 0.78–0.82 | Title/Family + GBDT |
| 進階集成 | 0.84–0.86 | 0.80–0.84 | Optuna 調參 + CatBoost/LightGBM + stacking |
| 頂尖 kernel | — | ~0.837+ | 重度特徵工程 + 多折 OOF 融合 |

> 注意：CV 常比 public LB **高 2–5%**；複雜 stacking 在 OOF 上更好，但 public 分數反而下降的案例不少。

### 2.2 單模型比較（社群共識，10-fold Stratified CV）

| 模型 | 典型 CV Accuracy | 優點 | 缺點 |
|------|------------------|------|------|
| Logistic Regression | 0.78–0.80 | 可解釋、快速 | 難捕捉非線性 |
| Random Forest | 0.82–0.84 | 穩定、特徵重要性 | 小資料易過擬合 |
| Gradient Boosting (sklearn) | 0.82–0.85 | 無額外依賴 | 較慢 |
| XGBoost | 0.84–0.85 | 成熟、調參生態好 | 需安裝 |
| LightGBM | 0.83–0.84 | 快、類別特徵友好 | 需安裝 |
| **CatBoost** | **0.85–0.86** | 類別編碼內建、小資料表現佳 | 需安裝 |
| Stacking (LR meta) | 0.84–0.86 OOF | 融合多模型 | **易過擬合 public LB** |

**來源**：[eriksarriegui/titanic-survival-predictor](https://huggingface.co/eriksarriegui/titanic-survival-predictor)（CatBoost 0.8563 CV）、[Ajayvarmaramineni/titanic-kaggle](https://github.com/Ajayvarmaramineni/titanic-kaggle)（簡單加權優於 stacking on LB）。

### 2.3 官方 / 教學範例

- **sklearn 官方**：`fetch_openml("titanic")` + `ColumnTransformer` + `LogisticRegression` → hold-out **0.798**  
  路徑：https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
- **Kaggle 入門**：Alexis Cook tutorial — Sex/Pclass/Age 填補 + 簡單模型，目標第一次 submission

---

## 3. 特徵工程 — 投資報酬率排序

依社群反覆驗證，**特徵工程貢獻 > 模型換皮**。

### Tier 1（必做，高 ROI）

| 特徵 | 作法 | 理由 |
|------|------|------|
| `Title` | `Name.str.extract(r' ([A-Za-z]+)\.', expand=False)`，合併 Rare/Mlle/Ms/Mme | 代理性別、年齡段、社會地位；Master ≈ 男童 |
| `FamilySize` | `SibSp + Parch + 1` | 家庭逃生行為不同 |
| `IsAlone` | `FamilySize == 1` | 獨行與家庭乘客模式不同 |
| `Age` 填補 | 依 `Title` + `Pclass` 中位數 | 比全局中位數更合理（177 缺失） |
| `Embarked` 填補 | mode (`S`) | 僅 2 筆缺失 |
| `Fare` 填補 | 依 `Pclass` 中位數 | test 有 1 筆缺失 |

### Tier 2（建議，中等 ROI）

| 特徵 | 作法 | 理由 |
|------|------|------|
| `Deck` | `Cabin.str[0]`，缺失填 `U` | 甲板與救生艇距離；對 XGB/LGBM 略有幫助 |
| `FarePerPerson` | `Fare / FamilySize` | 消除家庭共票價偏差 |
| `AgeBin` | 分箱：Child/Teen/Adult/Senior | 捕捉非線性 |
| `Sex × Pclass` | 交互特徵或 one-hot 組合 | 女性頭等艙生還率極高 |
| `TicketGroupSize` | 同 Ticket 人數 | 團體逃生模式 |

### Tier 3（可選，邊際增益 / 過擬合風險）

| 特徵 | 風險 |
|------|------|
| LOO survival rate by Ticket/Surname | 資料洩漏風險高，需嚴格在 fold 內計算 |
| 模型式 Age 填補（RF Regressor） | 略優但複雜度上升 |
| `Cabin` 完整編碼 | 缺失過多，增益不穩定 |
| Pseudo-labeling | 小資料易放大錯誤 |

---

## 4. 推薦解法（分階段）

### 4.1 決策樹

```
問題：表格二元分類，891 樣本
  ├─ 需要可解釋 / 教學？ → LogisticRegression + Pipeline（sklearn 官方路線）
  ├─ 需要穩定 0.78+ submission？ → RF/GBM + Tier 1 特徵
  └─ 目標 0.80–0.84 public LB？ → CatBoost or LightGBM + Tier 1–2 + StratifiedKFold
       └─ 進階：多模型 soft voting（優於複雜 stacking）
```

### 4.2 首選方案（推薦）

**CatBoostClassifier + Tier 1–2 特徵工程 + 5/10-fold Stratified CV**

理由：
1. 多個獨立來源在 CV 上 CatBoost 略優於 XGB/LGBM（~0.85+）
2. 原生處理類別特徵，減少 one-hot 爆炸
3. 小資料、低維特徵場景的訓練成本低（秒級～分鐘級）
4. 無需 GPU

```python
# 概念性 API 形狀（Phase 3 實作參考）
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

model = CatBoostClassifier(
    iterations=300,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    cat_features=["Sex", "Embarked", "Title", "Deck", "Pclass"],
)
# Pipeline: feature_engineering_fn → model
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

### 4.3 備選方案

| 方案 | 適用情境 | 預期 LB |
|------|----------|---------|
| **A. sklearn Pipeline only** | 零額外依賴、教學 | 0.76–0.80 |
| **B. LightGBM + Optuna** | 已有 boosting 生態、要快速迭代 | 0.78–0.82 |
| **C. Soft Voting Ensemble** | 單模型 CV 已穩定 0.83+ | 0.80–0.83 |
| **D. Stacking + LR meta** | 追求 OOF 極致（需警惕 LB 回落） | OOF 0.84+，LB 不穩 |

### 4.4 不推薦

| 方向 | 原因 |
|------|------|
| 深度學習（MLP/TabNet） | 891 樣本太少，難調且無穩定優勢 |
| LLM / TRL SFT | 任務類型不匹配 |
| 複雜 stacking 作為第一步 | 維護成本高，LB 常不如簡單加權 |
| 在 CV 外計算 target encoding | 資料洩漏 |

---

## 5. 驗證與提交策略

### 5.1 交叉驗證

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# scoring="accuracy"
```

- 必須 **Stratified**（目標略不平衡）
- 以 **CV mean ± std** 為主指標，不以單次 hold-out 定論
- 所有填補 / 編碼統計量 **在 fold 內 fit**（或用 `Pipeline` 封裝）

### 5.2 提交管線

```
train.csv + test.csv
  → 合併做特徵工程（注意：填補統計量只從 train 算）
  → 訓練最終模型（可用全 train 或 CV 平均模型）
  → predict test → submission.csv (418 rows + header)
  → Kaggle Submit（每日限 10 次）
```

### 5.3 常見陷阱

| 陷阱 | 後果 | 防法 |
|------|------|------|
| OOF vs LB 落差過大 | 以為 0.85 實際 0.78 | 優先簡化模型；檢查洩漏 |
| 全資料 fit 填補再 CV | 樂觀偏差 | `Pipeline` / `ColumnTransformer` |
| 過度調 threshold | CV 上 +1%，LB 反降 | Titanic 預設 0.5 通常已夠 |
| 忘記 test `Fare` 缺失 | 推論失敗 | 合併填補或單獨處理 |

---

## 6. 建議實作路線圖（Phase 2 → 3）

### Step 1 — Baseline（半天）

- [x] `Sex` + `Pclass` + `Age`/`Fare` 中位數填補
- [x] `RandomForestClassifier` + `StratifiedKFold(5)`
- [x] 產出 `submission_step1.csv`
- [x] CV 0.829；**LB 0.751**

### Step 2 — 特徵工程（半天）

- [x] Tier 1 全部 + Tier 2（`Deck`, `FarePerPerson`）
- [x] CV 0.827；**LB 0.744**（Deck OneHot 在 test 不穩）

### Step 3 — 模型優化（1 天）

- [x] CatBoost + Tier 1–2 特徵
- [x] CV 0.838；**LB 0.768**

### Step 4 — 集成（可選）

- [x] Tier 1 + 正則化 RF / CatBoost soft voting
- [x] CV 0.831；**LB 0.782**

### Step 5 — 參考 notebook（實際最佳）

- [x] 移植 [`titanic-81-57-leaderboard-top-1-no-cheating.ipynb`](../titanic-81-57-leaderboard-top-1-no-cheating.ipynb) → `features_kaggle815.py`
- [x] train+test 合併填補、Status、Deck 多步推斷、`Lucky_family` 等
- [x] CatBoost（depth=4, iter=1000, lr=0.0005）
- [x] CV 0.824；**LB 0.81578**

### Step 6 — Optuna 調參（已完成，LB 反降）

- [x] 50 trials Optuna；CV 0.847（+2.3% vs Step 5）
- [x] **LB 0.79425**（-2.2% vs Step 5）— CV 優化過擬合 train fold
- [x] **結論：生產提交用 Step 5**

---

## 7. 依賴建議

### 最小依賴（教學 / baseline）

```
pandas
numpy
scikit-learn
```

### 推薦依賴（競賽分數）

```
pandas
numpy
scikit-learn
catboost   # 或 lightgbm + xgboost
optuna     # 可選，超參搜尋
```

---

## 8. Phase 1 檢查清單

- [x] 範例程式路徑或 doc 已找到（sklearn 官方 + Kaggle tutorial + 社群 repo）
- [x] Import 與 config 形狀已記錄（`Pipeline` / `CatBoostClassifier` / `StratifiedKFold`）
- [x] 資料欄位與訓練方法需求已說明（表格分類，非 conversational）
- [x] 無需使用者輸入的阻塞項（資料已在 `data/`，公開競賽）

**下一步**：`ml-plan-validate` → 確認依賴與硬體 → Phase 3 實作 baseline pipeline。

---

## 9. 成熟度評估：能否直接參考付現？

> 補充研究：2026-07-03 — 回答「歷史悠久練手專案是否已有成熟高準度架構可直接複製」

### 9.1 短答

| 層面 | 是否成熟 | 能否直接付現 |
|------|----------|--------------|
| **特徵工程配方** | 極成熟（業界共識） | **可以** — Title / FamilySize / Age 填補幾乎所有高分方案相同 |
| **Pipeline 架構模式** | 成熟 | **可以** — `ColumnTransformer` + `Pipeline` + `StratifiedKFold` 是標準答案 |
| **單一官方套件** | 不存在 | **不行** — 沒有 `pip install titanic-best-model` 這種東西 |
| **0.84+ Kaggle LB** | 有案例但無保證 | **部分可以** — 需本地驗證，勿盲抄 kernel |
| **預訓練權重** | 有 | **可以** — HF 上可 `joblib.load` 直接推論 |

**結論**：這題的 **解法空間已收斂**，架構與特徵配方可直接參考；但 **沒有一個「複製即 0.84」的單一標準實作**。務實策略是：**抄架構 + 抄特徵，模型用 CatBoost/LightGBM，本地 CV 驗證後再提交。**

### 9.2 為何說「成熟」

經過 10+ 年、數萬份 Kaggle notebook，解法已收斂到固定模式：

```
train.csv + test.csv
    │
    ▼
特徵工程（幾乎固定的 Tier 1 配方）
  Title, FamilySize, IsAlone, Age/Fare/Embarked 填補, Deck
    │
    ▼
sklearn Pipeline（防資料洩漏）
  ColumnTransformer(impute → encode/scale) → Classifier
    │
    ▼
StratifiedKFold CV（5 或 10 fold）
    │
    ▼
[可選] 多模型 soft voting / 簡單加權
    │
    ▼
submission.csv（418 列）
```

這不是某一人發明的架構，而是 **社群反覆試錯後的共識**。新專案不需要從零設計，只需選一個參考實作對齊。

### 9.3 可直接參考的實作（按用途分級）

#### A 級 — 架構範本（推薦 fork / 對齊）

| 參考 | 用途 | 準確度 | 依賴 | 直接付現度 |
|------|------|--------|------|------------|
| [sklearn 官方 Titanic 範例](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) | `ColumnTransformer` + `Pipeline` 最小正確範例 | hold-out **0.798** | sklearn only | ★★★★★ 架構正確，分數偏低 |
| [jameskoero/titanic-survival-prediction](https://github.com/jameskoero/titanic-survival-prediction) | **Leak-free Pipeline**、pytest、FastAPI/Streamlit | hold-out **0.810** | sklearn only | ★★★★★ 工程品質最高，適合本專案骨架 |
| [Alexis Cook Kaggle Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial) | 官方入門、第一次 submission | ~0.765–0.78 | sklearn | ★★★★ 教學向，非高分 |

#### B 級 — 分數導向（可下載模型或抄特徵）

| 參考 | 用途 | 準確度 | 直接付現度 |
|------|------|--------|------------|
| [HF eriksarriegui/titanic-survival-predictor](https://huggingface.co/eriksarriegui/titanic-survival-predictor) | 預訓練 CatBoost/XGB 等 joblib，25 特徵 | CV **0.856** (CatBoost) | ★★★★ 可直接 `hf_hub_download` + predict |
| [Geeky Codes RF Tutorial](https://geekycodes.in/python/titanic-advanced-feature-engineering-tutorial/) | 特徵工程 + RF 調參 | public LB **0.837** | ★★★ 特徵配方可抄，notebook 需改寫 |
| [Ajayvarmaramineni/titanic-kaggle](https://github.com/Ajayvarmaramineni/titanic-kaggle) | 集成策略對比（簡單加權 vs stacking） | public LB **0.787** | ★★★ 教訓價值高：stacking OOF 高但 LB 反降 |

#### C 級 — 生產級 MLOps（過重，練手不必一開始就上）

| 參考 | 用途 | 準確度 | 直接付現度 |
|------|------|--------|------------|
| [JustaKris/Titanic-Machine-Learning-from-Disaster](https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster) | 完整 MLOps：Pydantic config、82 tests、Docker、CI | CV **0.862** (Voting) | ★★ 架構可參考，專案過重 |
| [Ajandaghian/titanic_ml_kaggle](https://github.com/Ajandaghian/titanic_ml_kaggle) | YAML 驅動、模組化 pipeline | CV **0.80+** | ★★ 適合學 MLOps 結構 |

### 9.4 不建議直接盲抄的來源

| 來源 | 原因 |
|------|------|
| Kaggle Top kernel（0.83+） | 常含 LB 導向的過擬合技巧、in-test group consensus、偽標籤 |
| 複雜 Stacking + 多層 meta-learner | OOF 好看，public LB 常回落 2–3% |
| 深度學習 / TabNet / MLP | 891 樣本無穩定優勢，維護成本高 |
| LLM / HuggingFace TRL | 任務類型完全不匹配 |

### 9.5 本專案建議的「付現路線」

依 Ponytail 原則（最少程式碼、最大收益），建議三條路徑擇一：

#### 路徑 1 — 最快出分（推薦）

1. 從 HF 下載 `eriksarriegui/titanic-survival-predictor` 的 CatBoost joblib
2. 對齊其 25 特徵工程函式（或讀 repo 原始碼）
3. 對 `data/test.csv` 推論 → `submission.csv`
4. 預期：CV ~0.85，LB ~0.80–0.83

#### 路徑 2 — 最佳學習 + 可維護（推薦本 repo）

1. 以 [jameskoero/titanic-survival-prediction](https://github.com/jameskoero/titanic-survival-prediction) 的 Pipeline 結構為骨架
2. 換分類器為 `CatBoostClassifier`（或 `HistGradientBoostingClassifier` 免額外依賴）
3. 套用本文件 Tier 1–2 特徵
4. 預期：CV ~0.82–0.84，LB ~0.78–0.82

#### 路徑 3 — 零依賴 baseline

1. 照 sklearn 官方範例
2. 加上 Title + FamilySize
3. 預期：CV ~0.78–0.80

### 9.6 「高度成熟」的具體含義

| 已成熟（不需再發明） | 仍未標準化（需自己驗證） |
|----------------------|--------------------------|
| Title 萃取 regex | 最佳超參數（資料量小，差異不大） |
| FamilySize = SibSp + Parch + 1 | 是否加 Ticket group LOO 特徵 |
| Age 依 Title/Pclass 填補 | CatBoost vs LightGBM 誰更好（差距 <1%） |
| Pipeline 防洩漏 | CV fold 數（5 vs 10） |
| StratifiedKFold | 集成權重 |
| soft voting 優於複雜 stacking | public LB 能否到 0.84（運氣成分） |

### 9.7 Phase 1 補充檢查清單

- [x] 確認存在可複製的成熟架構模式（Pipeline + FE 配方）
- [x] 列出分級參考實作與付現度評分
- [x] 釐清「無單一官方標準、但有共識配方」
- [x] 給出本專案三條付現路線

**收尾（2026-07-06）**：**最佳提交 `submission_step5.csv`（LB 0.81578）**。Step 6 Optuna CV↑ LB↓，見 §10。

---

## 10. 實作結果與收尾（2026-07-06）

### 10.1 Public Leaderboard 實測

| 提交檔 | 方法摘要 | CV mean | Public LB |
|--------|----------|---------|-----------|
| `submission_step1.csv` | RF + Sex/Pclass/Age/Fare | 0.829 | 0.75119 |
| `submission_step2.csv` | Tier1–2 + RF OneHot | 0.827 | 0.74401 |
| `submission_step3.csv` | Tier1–2 + CatBoost | 0.838 | 0.76794 |
| `submission_step4.csv` | Tier1 + 正則 RF/CB voting | 0.831 | **0.78229** |
| `submission_step5.csv` | **Kaggle 815 notebook 配方** | 0.824 | **0.81578** ← **最佳** |
| `submission_step6.csv` | Step 5 特徵 + Optuna CatBoost | 0.847 | 0.79425 |

性別 baseline（全 female=1）約 **0.765**。LB **1.0** 多為查表作弊，非 ML 目標（見 §9.4）。

**推薦提交**：`submission_step5.csv`

### 10.2 關鍵教訓

1. **CV 高 ≠ LB 高**：Step 3 CV 0.838 但 LB 0.768；應追泛化而非 OOF 極致。
2. **特徵工程 > 換模型**：Step 5 特徵配方單獨拉開 ~3% LB（0.782 → 0.816）。
3. **train+test 合併填補**對 Ticket/Deck/Price 有效（notebook 核心技巧）。
4. **Deck 不可粗暴 OneHot**（Step 2 教訓）；需 domain 多步推斷。
5. **簡單集成 + 正則化**（Step 4）有助 LB，但不及成熟 FE 配方。
6. **Optuna 追 CV 有害**（Step 6）：CV 0.847 → LB 0.794；小資料上超參搜尋易過擬合 fold 模式。

### 10.3 程式碼對照

| 模組 | 對應 Step | 說明 |
|------|-----------|------|
| `train.py` + `features.py` | 1–4（歷史） | Pipeline / Tier1 / voting；Step 5+ 以 `features_kaggle815` 為主 |
| `features_kaggle815.py` | 5+ | notebook 特徵工程移植 |
| `train.py` | 5–6 | 預設 Step 5（最佳 LB）；`--step 6` 跑 Optuna |
| `requirements.txt` | — | pandas, sklearn, catboost, optuna |

執行：

```bash
conda run -n base python train.py
```

產出：`submission_step{N}.csv`（N = `train.py` 內 `STEP`）。

### 10.4 與研究預期對照

| 研究預期 | 實際 | 結論 |
|----------|------|------|
| 標準解法 LB 0.78–0.82 | **0.816** | 達標，靠 notebook 級 FE |
| 路徑 2（自研 Pipeline）LB 0.78–0.82 | Step 4 0.782 | 接近下緣；Tier1 配方不足 |
| HF / 頂尖合法上限 ~0.84 | Step 5 **0.816** | 已達標準上緣；Step 6 未改善 |

### 10.5 參考實作（本專案採用）

- **Step 5 採用**：[titanic-81-57-leaderboard-top-1-no-cheating.ipynb](../titanic-81-57-leaderboard-top-1-no-cheating.ipynb)（作者聲稱無作弊 LB 0.81578，本專案重現一致）
- **LB 1.0 解析**：[How top LB got their score](https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn)

---

## 參考連結

| 資源 | URL |
|------|-----|
| 競賽主頁 | https://www.kaggle.com/competitions/titanic |
| Alexis Cook 教學 | https://www.kaggle.com/code/alexisbcook/titanic-tutorial |
| sklearn ColumnTransformer 範例 | https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html |
| Advanced FE Tutorial (RF 0.837) | https://geekycodes.in/python/titanic-advanced-feature-engineering-tutorial/ |
| **Kaggle 815 notebook（本專案 Step 5）** | `titanic-81-57-leaderboard-top-1-no-cheating.ipynb` |
| LB 1.0 作弊解析 | https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn |
| HF 模型參考（CatBoost ensemble） | https://huggingface.co/eriksarriegui/titanic-survival-predictor |
| GitHub 集成案例 | https://github.com/Ajayvarmaramineni/titanic-kaggle |
| Leak-free Pipeline 範本（推薦骨架） | https://github.com/jameskoero/titanic-survival-prediction |
| MLOps 完整範例 | https://github.com/JustaKris/Titanic-Machine-Learning-from-Disaster |
| YAML 驅動 Pipeline | https://github.com/Ajandaghian/titanic_ml_kaggle |

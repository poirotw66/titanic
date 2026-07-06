# Titanic — Phase 2 計畫與驗證

> 驗證日期：2026-07-03  
> 前置：[`ml-research-best-model.md`](ml-research-best-model.md)（Phase 1 完成）  
> 階段：ml-plan-validate

---

## Phase 2 完成檢查清單

- [x] 執行計畫已定義（見下方 Step 1–4）
- [x] 資料集 schema 已驗證（本地 CSV，非 Hub）
- [x] 任務格式確認：表格二元分類，`Survived` 目標，Accuracy 指標
- [x] 本地 vs Job 路徑已決定：**本地 CPU**，不需 HF Job / GPU
- [x] 依賴已選定並通過 smoke test

**下一步**：Phase 3–5 完成。最終交付 **blend**（Step 5 + 7b）；見 [`CLOSE.md`](CLOSE.md)。

---

## 1. 執行計畫（Phase 3）

| # | 任務 | 狀態 | 實測 |
|---|------|------|------|
| 1 | Baseline RF | done | CV 0.829 / LB 0.751 |
| 2 | Tier 1–2 特徵 | done | CV 0.827 / LB 0.744 |
| 3 | CatBoost | done | CV 0.838 / LB 0.768 |
| 4 | RF/CB soft voting | done | CV 0.831 / LB 0.782 |
| 5 | Kaggle 815 notebook | done | CV 0.824 / **LB 0.816** |
| 6 | Optuna 調參 | done | CV 0.847 / LB 0.794（CV↑ LB↓） |
| 7 | Geeky 鬆散 / 7b 嚴格 | done | 7b LB 0.816；與 Step 5 平手 |
| **blend** | Step 5 + 7b 平均機率 | done | 最終提交，見 `train.py` |

**最佳 public LB**：0.81578（Step 5 / 7b）。**推薦提交**：`submission_step_blend.csv`。

採用研究筆記 **路徑 2**：sklearn `Pipeline` 骨架 + `CatBoostClassifier` + Tier 1–2 特徵。

參考骨架：[jameskoero/titanic-survival-prediction](https://github.com/jameskoero/titanic-survival-prediction)  
可選對照（非必須）：[eriksarriegui/titanic-survival-predictor](https://huggingface.co/eriksarriegui/titanic-survival-predictor)

---

## 2. 資料集驗證

| 檔案 | 形狀 | 說明 |
|------|------|------|
| `data/train.csv` | 891 × 12 | 含 `Survived` 標籤 |
| `data/test.csv` | 418 × 11 | 無標籤，欄位 = train 去掉 `Survived` |
| `data/gender_submission.csv` | 418 × 2 | 提交格式參考：`PassengerId,Survived` |

**目標分布**：未生還 549 (61.6%) / 生還 342 (38.4%)

**缺失值**：

| 欄位 | train | test |
|------|-------|------|
| Age | 177 | 86 |
| Cabin | 687 | 327 |
| Embarked | 2 | 0 |
| Fare | 0 | 1 |

**格式檢查**：通過 — `test` 欄位 ∪ `{Survived}` = `train` 欄位；提交需 418 列 + header。

---

## 3. 硬體路徑

| 項目 | 決策 | 理由 |
|------|------|------|
| 執行環境 | **本地** | 891 列小表；CatBoost/sklearn 訓練秒級～分鐘級 |
| 運算裝置 | **CPU** | 研究結論與實測均不需 GPU；樹模型在此規模無 GPU 優勢 |
| HF Job | **不需要** | 非 LLM 微調；無 Hub 訓練任務 |
| 預估資源 | < 1 GB RAM，< 1 分鐘/次 CV | 遠低於本機上限 |

**本機實測（2026-07-03）**：

| 資源 | 規格 |
|------|------|
| CPU | Intel Xeon w3-2423，12 核 |
| RAM | 471 GiB（可用 ~448 GiB） |
| GPU | 2× NVIDIA RTX 6000 Ada 49 GB（**本任務不使用**） |

---

## 4. 依賴決策

### 4.1 選定 stack

| 套件 | 納入 | 版本（驗證環境） | 理由 |
|------|------|------------------|------|
| pandas | **是** | 2.2.2 | 資料載入與特徵工程 |
| numpy | **是** | 1.26.4 | sklearn / catboost 基礎 |
| scikit-learn | **是** | 1.7.1 | Pipeline、CV、baseline RF |
| catboost | **是** | 1.2.10 | Phase 1 首選分類器；CV 表現最佳 |
| joblib | **是** | 1.4.2 | 模型序列化 |
| lightgbm / xgboost | **否** | — | 與 CatBoost 重疊；Ponytail：先單一 booster |
| optuna | **暫不** | 4.0.0（base 已有） | Step 3 再啟用；baseline 不需 |

鎖定於 [`requirements.txt`](../requirements.txt)。

### 4.2 Python 環境現況

| 環境 | 狀態 | 建議 |
|------|------|------|
| 系統 `python3` (3.12.3) | **無 ML 套件**；`python3-venv` 未安裝 | 不直接用 |
| conda `base` (3.11.5) | **全部依賴已安裝**，smoke test 通過 | 開發期可直接使用 |

**建議執行方式（擇一）**：

```bash
# A. 使用現有 conda base（已驗證可用）
conda run -n base python train.py

# B. 專用環境（較乾淨，首次需建立）
conda create -n titanic python=3.11 -y
conda activate titanic
pip install -r requirements.txt
python train.py
```

### 4.3 Smoke test 結果

在 conda `base` 上，以 `Pclass + Sex + Age + Fare`（簡易填補）：

| 模型 | 訓練集 accuracy（未 CV，僅驗證可跑） |
|------|--------------------------------------|
| RandomForest (50 trees) | 0.976 |
| CatBoost (50 iter) | 0.827 |

結論：依賴可 import、可 fit；進入 Phase 3 無阻塞。

---

## 5. 風險與備註

| 風險 | 緩解 |
|------|------|
| 系統 Python 無套件 | 用 conda；或 `apt install python3.12-venv python3-pip` 後建 `.venv` |
| CV 與 public LB 落差 2–5% | StratifiedKFold；避免 test 洩漏；不盲抄 LB 導向 kernel |
| CatBoost 類別特徵索引 | Pipeline 內明確傳 `cat_features` 欄位名 |

---

## 參考

- Phase 1 研究：[`ml-research-best-model.md`](ml-research-best-model.md)
- 競賽：https://www.kaggle.com/competitions/titanic

# Titanic — 鐵達尼號生存預測

> Kaggle 入門競賽完整實作：研究 → 漸進式特徵工程 → 模型集成。  
> **Public LB：0.81578**（合法解法區間上緣）

**English** → [readme.md](readme.md)

**競賽頁面：** [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic)

---

## 成果摘要

| 提交檔 | 方法 | CV（5-fold） | Public LB |
|--------|------|--------------|-----------|
| `submission_step5.csv` | Kaggle 815 notebook 特徵工程 + CatBoost | 0.824 | **0.81578** |
| `submission_step7b.csv` | Geeky 進階 FE 教學（嚴格 ipynb 移植）+ RF | 0.836 | **0.81578** |
| `submission_step_blend.csv` | Step 5 與 7b **存活機率平均** | 0.833 | **0.81578** |

三條不同 pipeline 在 public 榜上同分（約 341 / 418 題正確）。**對外展示以 Step 5 為主**；blend 為集成實驗（較 Step 5 改動 6 筆硬標籤，分數不變——對錯抵銷）。

性別 baseline（全判女性存活）約 **0.765**。排行榜 **1.0** 多為查表作弊，非合理 ML 目標。

---

## 快速開始

```bash
pip install -r requirements.txt

python train.py              # 預設 blend → submission_step_blend.csv
python train.py --step 5     # CatBoost 單模 → submission_step5.csv
python train.py --step 7b    # RF 嚴格移植 → submission_step7b.csv
python train.py --step 6     # Optuna 實驗（CV↑ LB↓）
python train.py --step 7     # Geeky 鬆散版（LB 較差）
```

若使用 conda：`conda run -n base python train.py`

提交檔已加入 `.gitignore`，請在本機執行後產生。

---

## 專案結構

```
train.py                       # 入口：--step 5 | 6 | 7 | 7b | blend
features_kaggle815.py          # Step 5 — 815 notebook 特徵工程
features_geeky837b.py          # Step 7b — 教學 ipynb 嚴格移植
features_geeky837.py           # Step 7 — 鬆散移植（實驗）
features.py                    # Step 1–4 — Pipeline / Tier1 / voting

data/train.csv, data/test.csv  # Kaggle 競賽資料

docs/
  CLOSE.md                     # 收尾清單與 portfolio 摘要
  ml-research-best-model.md    # Phase 1 研究與全步驟紀錄
  ml-plan-validate.md          # Phase 2 計畫與環境驗證

titanic-81-57-leaderboard-top-1-no-cheating.ipynb      # Step 5 參考
titanic-advanced-feature-engineering-tutorial.ipynb    # Step 7b 參考
```

---

## 演進路線

| Step | 變更內容 | Public LB | 心得 |
|------|----------|-----------|------|
| 1–2 | 基礎特徵 / OneHot RF | 0.74–0.75 | 特徵不足 |
| 3–4 | CatBoost / soft voting | 0.77–0.78 | 換模型但 FE 沒跟上 |
| **5** | **815 notebook FE + CatBoost** | **0.81578** | **突破（+3% LB）** |
| 6 | Step 5 特徵 + Optuna | 0.794 | CV 0.847 但 LB 下降 |
| 7 | Geeky target encoding（鬆散） | 0.734 | notebook 移植不完整 |
| 7b | Geeky 嚴格 ipynb 移植 | 0.81578 | scaler/encoder 分開 fit 很關鍵 |
| blend | (p₅ + p₇b) / 2 | 0.81578 | 集成實驗；public 與單模平手 |

---

## 關鍵教訓

1. **特徵工程 > 調參** — Step 5 單靠成熟 FE 配方將 LB 從 ~0.782 拉到 0.816；Optuna 反而有害。
2. **CV 高 ≠ LB 高** — 訓練 891 筆、測試 418 筆，OOF 常與 leaderboard 脫鉤。
3. **嚴格 reproduce 很重要** — Step 7 與 7b 差在 scaler/encoder 是否分開 fit，CV 差約 7%。
4. **知道何時停** — 0.81578 是誠實解法的實務天花板；追 1.0 沒有意義。

完整收尾：[docs/CLOSE.md](docs/CLOSE.md) · 研究筆記：[docs/ml-research-best-model.md](docs/ml-research-best-model.md)

---

## 資料說明

| 檔案 | 說明 |
|------|------|
| `data/train.csv` | 891 位乘客，含 `Survived` 標籤 |
| `data/test.csv` | 418 位乘客，需預測存活與否 |
| `data/gender_submission.csv` | 範例提交檔 |

| 欄位 | 說明 |
|------|------|
| `PassengerId` | 乘客編號 |
| `Survived` | 目標（僅 train）：1 = 存活，0 = 未存活 |
| `Pclass` | 艙等（1 / 2 / 3） |
| `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked` | 特徵 |

**評分指標：** 測試集準確率（Accuracy）。

---

## 依賴套件

`pandas`、`numpy`、`scikit-learn`、`catboost`、`optuna`（僅 Step 6）— 見 [requirements.txt](requirements.txt)。

---

## 參考資源

- [Kaggle 815 notebook](titanic-81-57-leaderboard-top-1-no-cheating.ipynb) — Step 5 配方
- [進階 FE 教學](titanic-advanced-feature-engineering-tutorial.ipynb) — Step 7b 配方
- [Alexis Cook 教學](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)
- [LB 1.0 如何作弊](https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn)

---

## 授權

競賽資料與規則依 [Kaggle 條款](https://www.kaggle.com/competitions/titanic/rules)。

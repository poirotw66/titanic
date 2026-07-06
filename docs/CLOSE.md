# Titanic 專案收尾（2026-07-06）

## 狀態：完成

本專案從研究、漸進式實作到 Kaggle 提交已走完一輪完整 ML 工程循環。合法解法區間（public LB ~0.78–0.82）已達上緣；後續邊際效益低，**建議以本文件與 `readme.md` 作 portfolio 展示**。

---

## 最終推薦提交

| 檔案 | 說明 | Public LB |
|------|------|-----------|
| **`submission_step_blend.csv`** | Step 5 + Step 7b **平均存活機率**（預設 `train.py`）；OOF CV **0.833** | **0.81578** |
| `submission_step5.csv` | Kaggle 815 notebook 單模 | **0.81578** |
| `submission_step7b.csv` | Geeky ipynb 嚴格移植單模 | **0.81578** |

**Public LB 結論**：三份提交同分；blend 改動 6 筆預測但對錯抵銷，分數不變。Portfolio 以 **Step 5** 為代表解法，**blend** 為集成實驗。

```bash
conda run -n base python train.py              # 預設：blend
conda run -n base python train.py --step 5     # 單模 CatBoost（與 blend public 同分）
conda run -n base python train.py --step 7b    # 單模 RF
```

Blend 動機：兩條獨立特徵路線在 10 筆上分歧；public 上未優於單模，private 仍可事後對照（非必要）。

---

## Pipeline 架構（Portfolio）

```
data/
  train.csv, test.csv          # Kaggle 原始資料

train.py                       # 單一入口：--step 5|6|7|7b|blend

features.py                    # Steps 1–4（Tier1 Pipeline、RF/CatBoost voting）
features_kaggle815.py          # Step 5：815 notebook 特徵工程
features_geeky837.py           # Step 7：Geeky 鬆散版（實驗，LB 差）
features_geeky837b.py          # Step 7b：ipynb 嚴格移植

docs/
  ml-research-best-model.md    # Phase 1 研究 + 全步驟 LB 紀錄
  ml-plan-validate.md          # Phase 2 計畫與環境驗證
  CLOSE.md                     # 本文件

titanic-81-57-leaderboard-top-1-no-cheating.ipynb   # Step 5 參考
titanic-advanced-feature-engineering-tutorial.ipynb # Step 7b 參考
```

### 演進路線（摘要）

| Step | 方法 | CV | Public LB | 結論 |
|------|------|-----|-----------|------|
| 1–2 | 基礎 / OneHot RF | ~0.83 | 0.74–0.75 | 特徵不足 |
| 3–4 | CatBoost / voting | ~0.83 | 0.77–0.78 | 模型換了，FE 沒跟上 |
| **5** | 815 notebook FE + CatBoost | 0.824 | **0.81578** | **突破點** |
| 6 | Step 5 + Optuna | 0.847 | 0.794 | CV↑ LB↓ |
| 7 | Geeky 鬆散版 | 0.763 | 0.734 | 移植不完整 |
| 7b | Geeky 嚴格 ipynb | 0.836 | **0.81578** | 與 Step 5 LB 平手 |
| **blend** | (p5 + p7b) / 2 | **0.833** | **0.81578** | public 與單模平手；集成實驗 |

---

## 關鍵教訓（可寫進履歷 / 面試）

1. **特徵工程 > 調參**：Step 5 單靠成熟 FE 配方 +3% LB，Optuna 反而有害。
2. **CV 高 ≠ LB 高**：891 筆訓練、418 筆測試，OOF 與 leaderboard 脫鉤常見。
3. **嚴格 reproduce 很重要**：Step 7 與 7b 差在 scaler/encoder 是否分開 fit，CV 差 ~7%。
4. **集成要 diversify**：Step 5 與 7b 預測 97.6% 一致、LB 同分；blend 改 6 筆仍 **0.81578**（對錯抵銷）。
5. **知道何時停**：0.81578 ≈ 341/418，已達合法解法平台；追 1.0 無意義。

---

## 收尾步驟（建議順序）

### 今天可完成

- [x] 跑完 Step 1–7b + blend，記錄 public LB
- [ ] **Git**：commit + push blend 與 `docs/CLOSE.md`（本地尚有未提交變更）
- [ ] **README 對外**：確認 GitHub repo 描述一句話（`readme.md` / `readme-TW.md` 已更新）
- [ ] **Portfolio 一頁摘要**（面試 / 履歷用，見下方模板）

### 可選（不必再刷分）

- [ ] Kaggle 競賽結束後對照 **private LB**（Step 5 vs blend，記一筆在 `CLOSE.md` 即可）
- [ ] 不再新增 Step；邊際效益極低

### 明確不做

- 追 public LB 0.84+ 或 1.0
- 再跑 Optuna / 新 FE 路線（已證實 CV↑ LB↓ 或無增益）

---

## Portfolio 一頁摘要（可直接複用）

**Titanic — Machine Learning from Disaster**  
從研究、漸進式特徵工程到 Kaggle 提交的完整 ML 小專案。

- **成果**：Public LB **0.81578**（合法解法上緣；891 train / 418 test）
- **方法**：重現 Kaggle 815 notebook 特徵配方 + CatBoost；對照 Geeky RF 嚴格移植；最終 blend 集成
- **亮點**：Step 5 特徵配方單獨 +3.3% LB；證實 CV 與 LB 脫鉤；嚴格 reproduce 差 7% CV
- **技術**：pandas FE、CatBoost、sklearn RF、StratifiedKFold、機率 blend
- **程式**：`train.py --step 5|7b|blend`，模組化 `features_*.py`，文件 `docs/ml-research-best-model.md`

---

## 環境

```bash
pip install -r requirements.txt
# 或 conda base（本機已驗證）
```

依賴：`pandas`, `numpy`, `scikit-learn`, `catboost`, `optuna`（僅 Step 6）, `joblib`

---

## 後續（可選，非必要）

- [x] 提交 `submission_step_blend.csv` — public LB **0.81578**（與 Step 5 / 7b 同分）
- [ ] 競賽 private 公布後，在本文補一行 private 分數即可

---

## 參考

- 研究全文：`docs/ml-research-best-model.md`
- Kaggle 競賽：https://www.kaggle.com/competitions/titanic
- LB 1.0 作弊說明：https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn

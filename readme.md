# Titanic — Machine Learning from Disaster

> End-to-end Kaggle survival prediction: research → iterative feature engineering → ensemble.  
> **Public LB: 0.81578** (top of the legitimate-solution range)

**繁體中文** → [readme-TW.md](readme-TW.md)

**Competition:** [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic)

---

## Results

| Submission | Method | CV (5-fold) | Public LB |
|------------|--------|-------------|-----------|
| `submission_step5.csv` | Kaggle 815 notebook FE + CatBoost | 0.824 | **0.81578** |
| `submission_step7b.csv` | Geeky advanced FE tutorial (strict ipynb port) + RF | 0.836 | **0.81578** |
| `submission_step_blend.csv` | Average of Step 5 & 7b survival probabilities | 0.833 | **0.81578** |

Three different pipelines converge on the same public score (~341 / 418 correct). **Step 5** is the primary portfolio story; **blend** is the ensemble experiment (6 hard-label changes vs Step 5, score unchanged due to offsetting errors).

Gender baseline (predict all female = survived) is ~0.765. Leaderboard **1.0** scores are mostly lookup cheating, not a realistic ML target.

---

## Quick start

```bash
pip install -r requirements.txt

python train.py              # blend (default) → submission_step_blend.csv
python train.py --step 5     # CatBoost single model → submission_step5.csv
python train.py --step 7b    # RF strict port   → submission_step7b.csv
python train.py --step 6     # Optuna experiment (CV↑ LB↓)
python train.py --step 7     # Geeky loose port (failed LB)
```

With conda: `conda run -n base python train.py`

Submission CSVs are gitignored; generate them locally.

---

## Project structure

```
train.py                       # CLI: --step 5 | 6 | 7 | 7b | blend
features_kaggle815.py          # Step 5 — Kaggle 815 notebook feature engineering
features_geeky837b.py          # Step 7b — strict port of advanced FE tutorial
features_geeky837.py           # Step 7 — loose port (experiment)
features.py                    # Steps 1–4 — Pipeline / Tier1 / voting

data/train.csv, data/test.csv  # Kaggle competition data

docs/
  CLOSE.md                     # Wrap-up checklist & portfolio summary
  ml-research-best-model.md    # Phase 1 research + full step log
  ml-plan-validate.md          # Phase 2 plan & environment validation
```

Reference notebooks (external, not vendored in this repo):

- [Kaggle 815 notebook](https://www.kaggle.com/code/eu1234/titanic-81-57-leaderboard-top-1-no-cheating) — Step 5 recipe
- [Advanced FE tutorial](https://geekycodes.in/python/titanic-advanced-feature-engineering-tutorial/) — Step 7b recipe

---

## Approach (iterative steps)

| Step | What changed | Public LB | Takeaway |
|------|----------------|-----------|----------|
| 1–2 | Basic features / OneHot RF | 0.74–0.75 | Features too weak |
| 3–4 | CatBoost / soft voting | 0.77–0.78 | Model change without strong FE |
| **5** | **815 notebook FE + CatBoost** | **0.81578** | **Breakthrough (+3% LB)** |
| 6 | Optuna on Step 5 features | 0.794 | CV 0.847 but LB dropped |
| 7 | Geeky target encoding (loose) | 0.734 | Incomplete notebook port |
| 7b | Geeky strict ipynb port | 0.81578 | Separate scaler/encoder matters |
| blend | (p₅ + p₇b) / 2 | 0.81578 | Ensemble experiment; public tie |

---

## Lessons learned

1. **Feature engineering > hyperparameter tuning** — Step 5 FE alone moved LB ~0.782 → 0.816; Optuna hurt LB.
2. **High CV ≠ high LB** — small train set (891) vs test set (418); OOF can mislead.
3. **Strict reproduction matters** — Step 7 vs 7b differed ~7% CV on scaler/encoder fit details.
4. **Know when to stop** — 0.81578 is the practical ceiling for honest solutions; chasing 1.0 is pointless.

Full write-up: [docs/CLOSE.md](docs/CLOSE.md) · Research: [docs/ml-research-best-model.md](docs/ml-research-best-model.md)

---

## Data

| File | Description |
|------|-------------|
| `data/train.csv` | 891 passengers with `Survived` label |
| `data/test.csv` | 418 passengers — predict survival |
| `data/gender_submission.csv` | Example submission |

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique ID |
| `Survived` | Target (train only): 1 = survived, 0 = did not |
| `Pclass` | Ticket class (1st / 2nd / 3rd) |
| `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked` | Features |

**Metric:** Accuracy on the test set.

---

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `catboost`, `optuna` (Step 6 only) — see [requirements.txt](requirements.txt).

---

## References

- [Kaggle 815 notebook](https://www.kaggle.com/code/eu1234/titanic-81-57-leaderboard-top-1-no-cheating) — Step 5 recipe
- [Advanced FE tutorial](https://geekycodes.in/python/titanic-advanced-feature-engineering-tutorial/) — Step 7b recipe
- [Alexis Cook tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)
- [How top LB got 1.0 (cheating)](https://www.kaggle.com/tarunpaparaju/how-top-lb-got-their-score-use-titanic-to-learn)

---

## License

Competition data and rules: [Kaggle terms](https://www.kaggle.com/competitions/titanic/rules).

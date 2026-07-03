# Titanic — Machine Learning from Disaster

> Kaggle Getting Started competition: predict which passengers survived the Titanic shipwreck.

**Competition:** [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic)  
**Citation:** Will Cukierski. Titanic - Machine Learning from Disaster. https://kaggle.com/competitions/titanic, 2012. Kaggle.

---

## Overview

This is a classic introductory machine learning competition on Kaggle. The goal is to build a model that predicts whether each passenger in the test set survived the sinking of the RMS Titanic.

The competition runs indefinitely with a **rolling leaderboard** (submissions older than two months are invalidated).

---

## The Challenge

On April 15, 1912, during her maiden voyage, the RMS Titanic sank after colliding with an iceberg. There were not enough lifeboats for everyone onboard, resulting in the deaths of **1,502 out of 2,224** passengers and crew.

While luck played a role, some groups of people were more likely to survive than others. Your task is to answer: **what sorts of people were more likely to survive?** using passenger data (name, age, gender, socio-economic class, etc.).

---

## Data

This project includes the competition datasets under `data/`:

| File | Description |
|------|-------------|
| `data/train.csv` | 891 passengers with features and the ground-truth `Survived` label |
| `data/test.csv` | 418 passengers without labels — predict survival for these |
| `data/gender_submission.csv` | Example submission file |

### Features (`train.csv` / `test.csv`)

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Target (train only): `1` = survived, `0` = did not survive |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

Use patterns in `train.csv` to predict survival for the 418 passengers in `test.csv`.

---

## Evaluation

**Metric:** Accuracy — the percentage of passengers correctly predicted.

For each passenger in the test set, predict `0` (deceased) or `1` (survived).

---

## Submission Format

Submit a CSV with **exactly 418 rows plus a header**. No extra columns or rows.

| Column | Description |
|--------|-------------|
| `PassengerId` | Passenger ID (any order) |
| `Survived` | Binary prediction: `1` = survived, `0` = deceased |

Example:

```csv
PassengerId,Survived
892,0
893,1
894,0
```

You may submit up to **10 predictions per day** on Kaggle.

---

## Getting Started

1. **Explore the data** — inspect `data/train.csv` and `data/test.csv`.
2. **Build a model** — train on labeled data, tune features, and validate locally.
3. **Generate predictions** — output a submission CSV for the test set.
4. **Submit on Kaggle** — upload via **Submit Predictions** and check the leaderboard.

### Recommended Resources

- [Alexis Cook's Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial) — step-by-step first submission
- [Competition Notebooks](https://www.kaggle.com/competitions/titanic/code) — community insights and approaches
- [Titanic Discussion Forum](https://www.kaggle.com/competitions/titanic/discussion) — help and tips from other competitors
- [Kaggle Discord](https://discord.gg/kaggle) — competitions, careers, and community

---

## FAQ

**What is a Getting Started competition?**  
Designed for newcomers to data science and Kaggle. No cash prize, rolling timeline, and focused on learning the platform.

**Why did my team disappear from the leaderboard?**  
Submissions older than two months are invalidated. Teams with no recent submissions also drop off to keep the leaderboard fresh.

**Where do I get help?**  
Use the [Titanic Discussion Forum](https://www.kaggle.com/competitions/titanic/discussion). Kaggle does not have a dedicated support team for individual code issues.

---

## License

Data and competition rules are governed by [Kaggle's competition terms](https://www.kaggle.com/competitions/titanic/rules).

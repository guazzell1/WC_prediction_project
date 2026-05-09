# ⚽ International Football Match Predictor: A Chronological Feature Engineering Approach

## 📌 Project Overview
This project aims to build a Machine Learning model capable of predicting the outcomes of international football matches (Home Win, Draw, or Away Win). 

While many sports prediction models suffer from **Data Leakage** by inadvertently using future data to predict the past (e.g., using static historical averages), this project stands out by implementing a strictly **chronological data engineering pipeline**. The algorithm dynamically processes the dataset match by match, maintaining a "living notebook" of team statistics. This ensures that the model only makes predictions based on the exact information available right before kickoff.

## 📊 Dataset
The dataset used for this project contains historical international football results and was sourced from Kaggle:
🔗 [International football results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

## 🧠 Core Architecture & Feature Engineering

To capture the true strength of a team without leaking future match results into the training data, two main sets of features were engineered at runtime:

### 1. Head-to-Head (H2H) History
The model extracts the direct confrontation history between two playing teams. By sorting the dataset chronologically, the algorithm reads the H2H strength *before* the match occurs, saves this "snapshot" as a feature, and only updates its internal tracker *after* the match is concluded.

### 2. The "Momentum" Factor (8-Year Sliding Window)
A national team's performance from decades ago holds little relevance today. To capture the current form of each squad, a **Sliding Window algorithm** was implemented. For every match, the algorithm scans the history and filters out outdated data, counting only the goals scored and conceded by the team within the **last 8 years**. This dynamic memory ensures the strength metric is always fresh and representative of the current generation of players.

## ⚙️ Data Leakage Prevention
To ensure the mathematical integrity of the model, a rigorous feature selection process was applied prior to training:
* **Removed Target Leaks:** Features like `home_score` and `away_score` from the current match were immediately dropped. Passing these to the model would result in artificial accuracy, as the algorithm would just memorize the score rather than learning patterns.
* **Removed Textual/Redundant Data:** Variables like `tournament`, `city`, `country`, and team names were dropped after their intelligence was successfully extracted into the numerical H2H and Momentum features.

## 🚀 Baseline Model & Performance

Due to the significant scale variance between the engineered features (e.g., H2H wins ranging from 0-5 vs. Momentum goals ranging from 0-150+), the data was standardized using `StandardScaler` to optimize the Gradient Descent.

* **Algorithm:** Logistic Regression (Multiclass)
* **Baseline Accuracy:** ~55%

**Why 55% is a solid baseline:** 
Unlike sports with binary outcomes, football has three possible results (Home, Away, Draw), making random guessing accurate only 33% of the time. Given the extremely high variance and unpredictability of football, achieving 55% accuracy using only 8 engineered historical features establishes a highly profitable and robust baseline model.

## 💻 Production Inference & The 'Error' Anomaly

A production-ready function `predict_game()` was built to simulate real-world usage. It requires only the team names as inputs, dynamically calculating their past history and current momentum before running the standardized array through the trained model.

**Known Anomaly (The 'Error' Class):**
When interpreting the model's probabilities, a 4th class labeled `Error` (with ~0.0% probability) appears. This is a documented byproduct of our data leakage prevention pipeline. Matches in the historical dataset with missing score values (`NaN`) were labeled as `Error` during target creation. Because the score columns were aggressively dropped to prevent data leaks *before* the general `dropna()` was executed, these rows bypassed the null-cleaning phase. This anomaly was intentionally kept in this version to demonstrate the critical impact of execution order in data preprocessing pipelines.

## 🔮 Next Steps & Future Work (v2.0)
As a continuously evolving project, the following improvements are mapped for future iterations:
1. **TimeSeries Validation:** Implementing `TimeSeriesSplit` instead of standard Cross-Validation to rigorously test the model while respecting the chronological timeline of the matches.
2. **Advanced Algorithms:** Transitioning from linear models to tree-based ensemble methods, such as **Random Forest** or **XGBoost**, to capture complex non-linear patterns.
3. **Business Logic Filtering:** Filtering the training dataset to weight competitive tournament matches (e.g., World Cup, Euros) heavier than friendly matches.
4. **Pipeline Refactoring:** Adjusting the execution order of the `dropna()` method to permanently eliminate the `Error` class artifact prior to target encoding.

---
*Developed as a portfolio project showcasing Data Engineering, Feature Selection, and Machine Learning best practices.*

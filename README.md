# Heart Disease Classification — Tree Models & Ensembles

Week 5 deliverable for my AI & Data Science internship at Arcana Info Pvt Ltd. This project predicts whether a patient has heart disease using the UCI Heart Disease dataset, comparing Decision Tree, Random Forest, and XGBoost against a Logistic Regression baseline, then tuning the top models with cross-validation.

## 🎯 Project Goal

Build and compare classification models to predict heart disease presence (`target`: 0 = no disease, 1 = disease), evaluate them properly using precision, recall, F1, and AUC (not just accuracy), and tune the strongest models using cross-validation.

## 🚩 A Data Leakage Catch (the most important part of this project)

My first run of this notebook produced suspiciously perfect results — AUC of 1.0 and ~98% accuracy across every model. That's not realistic for real-world medical data, so I investigated instead of celebrating.

Turns out the raw dataset (1025 rows) contained **723 duplicate rows** — the same patient records repeated multiple times. Since I hadn't removed them before splitting into train/test, identical rows ended up on both sides of the split, meaning the model had effectively already "seen" parts of the test set during training.

**Fix:** dropped duplicates with `df.drop_duplicates()`, reducing the dataset to **302 unique patient records**, then re-ran the entire pipeline from scratch. Scores dropped to a realistic 80–90% range — which is the correct, trustworthy outcome, not a worse one.

**Takeaway:** always check for duplicates *before* splitting the data, not after.

## 🗂️ Dataset

- **Source:** UCI Heart Disease dataset (Cleveland subset, via Kaggle)
- **Size:** 302 rows after deduplication, 13 features + target
- **Features:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, number of major vessels, thalassemia

## 🧠 Models Trained

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.803 | 0.730 | 0.931 | 0.818 | 0.847 |
| Decision Tree | 0.738 | 0.741 | 0.690 | 0.714 | 0.736 |
| Random Forest | 0.820 | 0.765 | 0.897 | 0.825 | 0.880 |
| XGBoost | 0.803 | 0.758 | 0.862 | 0.807 | 0.852 |
| **Tuned Random Forest** ⭐ | 0.820 | 0.781 | 0.862 | 0.820 | **0.903** |
| Tuned XGBoost | 0.803 | 0.758 | 0.862 | 0.807 | 0.856 |

**Best model: Tuned Random Forest** — highest AUC (0.903) of everything tested, using `RandomizedSearchCV` with 5-fold cross-validation (`n_estimators=150, min_samples_split=10, max_depth=5`).

## 📊 Key Findings

- **Decision Tree performed worst across every metric** — expected, since a single unpruned tree overfits and doesn't get the averaging benefit that ensemble methods do.
- **Tuning Random Forest gave a genuine AUC improvement** (0.880 → 0.903), trading a small amount of recall for better precision.
- **Tuning XGBoost had almost no effect** (AUC 0.852 → 0.856) — the default settings were likely already close to optimal for this dataset, and/or the search space needed to be wider.
- **False negatives matter most here** — in a medical context, missing an actual diagnosis is worse than an unnecessary follow-up test. The Tuned Random Forest still misses ~14% of actual disease cases (4 out of 29 in the test set), which is a real limitation worth stating plainly rather than hiding behind a good AUC score.
- **Small dataset caveat:** only 302 total rows (~60 in the test set) means single-split metrics carry some noise. 5-fold CV scores (Random Forest: 0.835 ± 0.006, XGBoost: 0.805 ± 0.026) are the more reliable estimates — and Random Forest's low standard deviation suggests it's consistently accurate, not just accurate on this one split.

## 🛠️ Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Decision Tree, Random Forest, `RandomizedSearchCV`, cross-validation, metrics)
- XGBoost
- Matplotlib / Seaborn (EDA, confusion matrix, ROC curve visualizations)
- Streamlit (interactive prediction app)

## 💻 Streamlit App

An interactive app lets you enter patient details and get a real-time risk prediction from the Tuned Random Forest model.

### Run it locally:
```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```

Make sure `heart_disease_model.pkl` and `model_features.pkl` (generated at the end of the notebook) are in the same folder as `app.py`.

⚠️ **Disclaimer:** This model is trained on a small dataset (302 patients) for educational purposes. It is **not** a diagnostic tool. Always consult a medical professional for real clinical decisions.

## 📁 Repository Structure

```
├── classification.ipynb          # Full notebook: EDA, models, tuning, evaluation
├── app.py                        # Streamlit prediction app
├── heart_disease_model.pkl       # Saved Tuned Random Forest model
├── model_features.pkl            # Feature column order for the model
└── README.md
```

## 🙋 About This Project

Part of my Week 5 ML curriculum (decision trees, ensembles, evaluation metrics, cross-validation) during my AI & Data Science internship at Arcana Info Pvt Ltd. Follows on from Week 3 (SQL analysis) and Week 4 (Linear Regression pipeline) in the same internship series.

## 👤 Author

**Ayesha Iqball**
BS Computer Science, NCBA&E, Lahore
AI & Data Science Intern @ Arcana Info Pvt Ltd

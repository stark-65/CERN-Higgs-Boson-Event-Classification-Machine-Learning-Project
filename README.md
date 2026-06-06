# ⚛️ Higgs Boson Signal Classifier

End-to-end ML pipeline to classify CERN Higgs Boson events as **Signal** or **Background**.

## Models
| Model | ROC-AUC |
|---|---|
| XGBoost | ~0.928 |
| LightGBM | ~0.926 |
| Logistic Regression | ~0.882 |

## Features
- 📊 EDA — distributions, correlation heatmap, weighted class balance
- 🏋️ Train & Compare — all 3 models, ROC curves, feature importance
- 📈 Metrics — confusion matrix, PR curve, threshold slider
- 🔍 SHAP — beeswarm + bar plots, top feature table
- 🎯 Predict — upload any CSV, get signal probabilities + download results

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
CERN Higgs Boson ML Challenge — [Kaggle](https://www.kaggle.com/c/higgs-boson/data)  
Upload `training.csv` and `test.csv` directly in the app sidebar.

## Deploy
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → select repo → Deploy

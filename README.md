# Enterprise Credit Risk & Loan Portfolio Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20SHAP-green.svg)

An enterprise-grade credit risk analytics platform that enables banks and financial institutions to assess borrower-level default risk and monitor portfolio-level exposure using machine learning and explainable AI.

## 🎯 Key Features

- **ML-Powered Risk Prediction**: Logistic Regression, Random Forest, and XGBoost models
- **Explainable AI**: SHAP-based feature importance and individual prediction explanations
- **Portfolio Analytics**: Expected Loss calculations, concentration risk, stress testing
- **Interactive Dashboard**: Streamlit-based UI with rich visualizations
- **Real-Time Assessment**: Individual loan risk scoring with instant feedback

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
├─────────────────────────────────────────────────────────────┤
│  Executive    Individual    Portfolio    Risk      Model    │
│  Dashboard    Risk          Analysis    Segment   Insights  │
├─────────────────────────────────────────────────────────────┤
│              Analytics & Explainability Layer                │
│         (SHAP, Portfolio Metrics, Expected Loss)            │
├─────────────────────────────────────────────────────────────┤
│                    ML Model Layer                            │
│      (Logistic Regression, Random Forest, XGBoost)          │
├─────────────────────────────────────────────────────────────┤
│               Data Processing Pipeline                       │
│    (Preprocessing, Feature Engineering, Encoding)           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-credit-risk.git
cd enterprise-credit-risk

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
# Run with demo data (no training required)
streamlit run app.py
```

### Train Models (Optional)

```bash
# Download Lending Club data and train models
python train_model.py --samples 100000
```

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Dashboard** | Portfolio KPIs, risk distribution, stress test results |
| **Individual Risk** | Single loan risk assessment with SHAP explanations |
| **Portfolio Analysis** | Segment breakdown, concentration risk, risk bands |
| **Risk Segmentation** | 2D heatmaps, cohort analysis by income/DTI |
| **Model Insights** | Feature importance, model performance metrics |

## 🧮 Risk Metrics

- **Probability of Default (PD)**: ML-predicted likelihood of loan default
- **Expected Loss (EL)**: PD × Exposure at Default × Loss Given Default
- **Risk Tiers**: Low (<20% PD), Medium (20-50% PD), High (>50% PD)
- **Concentration Risk**: HHI index and top exposure analysis

## 📁 Project Structure

```
enterprise_credit_risk/
├── app.py                    # Main Streamlit application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── src/
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering
│   ├── models/               # ML model training & evaluation
│   ├── explainability/       # SHAP explanations
│   ├── analytics/            # Portfolio risk metrics
│   └── dashboard/            # Streamlit UI components
├── data/raw/                 # Downloaded dataset
└── models/saved/             # Trained model artifacts
```

## 🔧 Technologies

- **ML/Data**: scikit-learn, XGBoost, pandas, numpy, imbalanced-learn
- **Explainability**: SHAP
- **Visualization**: Plotly, Streamlit
- **Data Source**: [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## 📈 Model Performance

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | ~0.68 | ~0.62 | ~0.65 |
| Random Forest | ~0.72 | ~0.66 | ~0.68 |
| XGBoost | ~0.74 | ~0.68 | ~0.72 |

*Results may vary based on data sample and hyperparameters*

## 📜 License

MIT License - See LICENSE file for details.

## 👤 Author

Built as an enterprise credit risk analytics demonstration project.

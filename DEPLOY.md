# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud (FREE)

### Step 1: Push Code to GitHub
```bash
cd "e:\a - Copy\enterprise_credit_risk"
git init
git add .
git commit -m "Initial commit: Credit Risk Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-platform.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - **Repository:** `YOUR_USERNAME/credit-risk-platform`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

### Step 3: Wait 2-3 minutes
Your app will be live at: `https://your-app-name.streamlit.app`

---

## Data Loading Options in the App

| Option | Description |
|--------|-------------|
| **Demo Data** | Synthetic 10,000 loans for quick visualization |
| **Kaggle Lending Club** | Downloads real data (requires kagglehub) |
| **Upload CSV** | Upload your own loan data file |

### CSV File Requirements
For custom uploads, your CSV should have these columns:
- `loan_amnt` - Loan amount
- `grade` - Credit grade (A-G) [optional]
- `loan_status` - Loan outcome [optional]
- `purpose`, `home_ownership`, `term` [optional for analysis]

---

## Environment Variables (Optional)
If using Kaggle data on Streamlit Cloud, add secrets:
1. Go to app settings → Secrets
2. Add:
```toml
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_api_key"
```

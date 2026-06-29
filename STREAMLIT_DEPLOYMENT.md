# Deploy Supply Chain Pipeline on Streamlit Cloud

## Quick Start (5 minutes)

### Step 1: Push to GitHub ✅
Your repository is already public and contains all the data files.

### Step 2: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Fill in:
   - **Repository**: `snehashrotriya0424/-supply-chain-stockout-risk-pipeline`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **"Deploy!"**

Streamlit will automatically:
- Install dependencies from `requirements.txt`
- Run your app
- Generate a live URL (typically `https://[app-name].streamlit.app`)

**Your app will be live in ~2-3 minutes! 🚀**

---

## What's Deployed

The Streamlit app includes 4 interactive pages:

### 📊 Dashboard Overview
- Key metrics (high-risk SKUs, days of supply, warehouse/supplier count)
- Risk score distribution histogram
- Risk by category analysis
- Days of supply by warehouse
- Top 10 highest-risk SKUs snapshot

### 🎯 Risk Scoring
- Configurable risk thresholds (High/Medium/Low)
- High-risk SKU list with action items
- Specific recommendations per SKU
- Buy/expedite guidance based on inventory levels

### 📈 Model Performance
- Toggle between Logistic Regression vs Random Forest
- Performance metrics: ROC-AUC, Recall, Precision, F1
- Confusion matrix visualization
- Business interpretation of false positives/negatives

### 🔍 Deep Dive Analysis
- Feature importance (logistic regression coefficients)
- Performance by product category
- Correlation of features with stockout risk

---

## Data Files Included

All CSV files are committed to the repo and loaded by `streamlit_app.py`:
- `model_output.csv` — Risk scores for all SKUs
- `model_features.csv` — Feature set used by the model
- `feature_importance.csv` — Coefficient weights
- `model_metrics.json` — Performance metrics
- `products.csv` — Product catalog

---

## After Deployment

### Share Your App
Once deployed, Streamlit generates a public URL. Share it with:
- Teammates/stakeholders
- Hiring managers
- Portfolio contacts

### Troubleshooting

**App won't load?**
- Check that all `.csv` and `.json` files are in the repo root
- Verify `requirements.txt` includes `streamlit>=1.28` and `plotly>=5.17`
- Check Streamlit Cloud logs for error details

**Data looks wrong?**
- Ensure you ran the full pipeline locally first:
  ```bash
  python generate_data.py
  python feature_engineering.py
  python model.py
  ```
- Commit the generated CSV files to GitHub

**Slow performance?**
- Streamlit Cloud free tier has resource limits
- On-demand loading via `@st.cache_data` decorator is included
- For production, upgrade to Streamlit Pro or deploy to Heroku/Railway

---

## Alternative Deployments

### Heroku (older, now paid, not recommended)
Use Railway or Render instead — they offer free tiers and better pricing.

### Railway.app (Recommended Alternative)
1. Connect GitHub repo
2. Set environment to Python
3. Create `Procfile`:
   ```
   web: streamlit run streamlit_app.py --logger.level=error
   ```
4. Deploy

### Google Cloud Run (Production)
1. Create `Dockerfile` (provided in repo)
2. Push to Google Container Registry
3. Deploy to Cloud Run (free tier: 2M requests/month)

---

## Next Steps

1. ✅ All files are ready — just deploy!
2. Share the live URL with stakeholders
3. Iterate on dashboard based on feedback
4. Consider adding:
   - SQL query results (supplier anomalies, lead time trends)
   - Real-time data refresh
   - Export reports to PDF
   - User authentication (Streamlit Community Cloud doesn't support this free tier, but Cloud Run does)

---

**Questions?** Check the [Streamlit docs](https://docs.streamlit.io) or [Community forums](https://discuss.streamlit.io)

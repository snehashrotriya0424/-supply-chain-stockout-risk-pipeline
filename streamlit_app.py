import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Supply Chain Stockout Risk Pipeline",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & description
st.title("🚚 Supply Chain Stockout Risk Intelligence Platform")
st.markdown("""
An end-to-end analytics pipeline — SQL → Python/ML → Interactive Dashboard
Built on simulated supply chain data: **40 SKUs × 4 warehouses × 10 suppliers × 1 year of daily data**
""")

# Load data
@st.cache_data
def load_data():
    model_output = pd.read_csv("model_output.csv")
    model_features = pd.read_csv("model_features.csv")
    feature_importance = pd.read_csv("feature_importance.csv", index_col=0)
    products = pd.read_csv("products.csv")
    
    with open("model_metrics.json") as f:
        metrics = json.load(f)
    
    # Debug: print actual columns
    print("Columns in model_output:", model_output.columns.tolist())
    
    return model_output, model_features, feature_importance, products, metrics

try:
    model_output, model_features, feature_importance, products, metrics = load_data()
    
    # Rename stockout_risk to risk_score if it exists
    if "stockout_risk" in model_output.columns:
        model_output.rename(columns={"stockout_risk": "risk_score"}, inplace=True)
    
    # Verify risk_score exists
    if "risk_score" not in model_output.columns:
        st.error("❌ Column 'risk_score' or 'stockout_risk' not found in model_output.csv")
        st.write("Available columns:", model_output.columns.tolist())
        st.stop()
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("**Steps to fix:**\n1. Run: `python generate_data.py`\n2. Run: `python feature_engineering.py`\n3. Run: `python model.py`")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a view",
    ["📊 Dashboard Overview", "🎯 Risk Scoring", "📈 Model Performance", "🔍 Deep Dive Analysis"]
)

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if page == "📊 Dashboard Overview":
    st.header("Executive Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = (model_output["risk_score"] >= 0.5).sum()
    total_skus = len(model_output)
    high_risk_pct = (high_risk / total_skus) * 100
    avg_days_supply = model_output["days_of_supply"].mean()
    
    col1.metric("🚨 High-Risk SKUs", f"{high_risk}/{total_skus}", f"{high_risk_pct:.1f}%")
    col2.metric("📦 Avg Days of Supply", f"{avg_days_supply:.1f} days", "All SKUs")
    col3.metric("🏢 Warehouses", int(model_output["warehouse_id"].nunique()))
    col4.metric("👥 Suppliers in Data", int(model_output["supplier_id"].nunique()))
    
    st.divider()
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            model_output,
            x="risk_score",
            nbins=30,
            color_discrete_sequence=["#FF6B6B"],
            title="Stockout Risk Score Distribution"
        )
        fig.update_xaxes(title="Risk Score")
        fig.update_yaxes(title="Number of SKU-Warehouse Combinations")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk by Category")
        risk_by_cat = model_output.groupby("category").agg({
            "risk_score": "mean",
            "sku": "count"
        }).rename(columns={"sku": "count"}).reset_index()
        
        fig = px.bar(
            risk_by_cat,
            x="category",
            y="risk_score",
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
            title="Average Risk Score by Category",
            labels={"risk_score": "Avg Risk Score", "category": "Category"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Days of supply analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Days of Supply by Warehouse")
        dos_by_wh = model_output.groupby("warehouse_id")["days_of_supply"].mean().reset_index()
        dos_by_wh.columns = ["Warehouse", "Avg Days of Supply"]
        dos_by_wh["Warehouse"] = dos_by_wh["Warehouse"].astype(str)
        
        fig = px.bar(
            dos_by_wh,
            x="Warehouse",
            y="Avg Days of Supply",
            color="Avg Days of Supply",
            color_continuous_scale="Blues",
            title="Average Days of Supply by Warehouse"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Highest Risk SKU-Warehouse Combinations")
        latest_snapshot = model_output.nlargest(10, "risk_score")[
            ["sku", "category", "warehouse_id", "risk_score", "days_of_supply", "on_hand_qty"]
        ].copy()
        latest_snapshot = latest_snapshot.round(3)
        
        st.dataframe(latest_snapshot, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 2: RISK SCORING
# ============================================================================
elif page == "🎯 Risk Scoring":
    st.header("SKU Risk Scoring & Recommendations")
    
    # Risk thresholds
    col1, col2 = st.columns(2)
    high_threshold = col1.slider("High Risk Threshold", 0.0, 1.0, 0.5)
    medium_threshold = col2.slider("Medium Risk Threshold", 0.0, 1.0, 0.3)
    
    # Categorize risk
    model_output["risk_category"] = pd.cut(
        model_output["risk_score"],
        bins=[0, medium_threshold, high_threshold, 1.0],
        labels=["Low", "Medium", "High"]
    )
    
    # Risk summary
    risk_counts = model_output["risk_category"].value_counts()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk", int(risk_counts.get("Low", 0)))
    col2.metric("🟡 Medium Risk", int(risk_counts.get("Medium", 0)))
    col3.metric("🔴 High Risk", int(risk_counts.get("High", 0)))
    
    st.divider()
    
    # Filter and display high-risk SKUs
    st.subheader("High-Risk SKUs - Action Items")
    high_risk_skus = model_output[model_output["risk_category"] == "High"].sort_values(
        "risk_score", ascending=False
    )
    
    if len(high_risk_skus) > 0:
        display_cols = ["sku", "category", "warehouse_id", "supplier_id", 
                       "risk_score", "days_of_supply", "on_hand_qty", "on_order_qty"]
        
        st.dataframe(
            high_risk_skus[display_cols].round(2),
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Recommendations
        st.subheader("Recommended Actions")
        for idx, row in high_risk_skus.head(5).iterrows():
            action_text = f"{row['sku']} ({row['category']}) - Risk: {row['risk_score']:.2%}"
            with st.expander(f"📌 {action_text}"):
                col1, col2 = st.columns(2)
                col1.metric("Days of Supply", f"{row['days_of_supply']:.1f}")
                col2.metric("On Hand", f"{row['on_hand_qty']:.0f} units")
                
                recommendations = []
                if row["days_of_supply"] < 5:
                    recommendations.append("⚠️ **CRITICAL**: Days of supply < 5. Expedite purchase order immediately.")
                if row["on_hand_qty"] < row["avg_sales_7d"] * 3:
                    recommendations.append("📦 **Stock Level Low**: On-hand inventory < 3 days of average sales.")
                if row["reliability_score"] < 0.7:
                    recommendations.append("🚚 **Supplier Concern**: Reliability score below target. Consider alternate supplier.")
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.info("✓ Standard monitoring recommended")
    else:
        st.success("✅ No high-risk SKUs detected!")

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.header("ML Model Performance Metrics")
    
    # Model selection
    model_type = st.radio("Select Model", ["Logistic Regression", "Random Forest"])
    model_key = "logistic_regression" if model_type == "Logistic Regression" else "random_forest"
    
    perf_metrics = metrics[model_key]
    
    st.divider()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 ROC-AUC", f"{perf_metrics['roc_auc']:.3f}")
    col2.metric("✅ Recall", f"{perf_metrics['recall']:.3f}")
    col3.metric("🎯 Precision", f"{perf_metrics['precision']:.3f}")
    col4.metric("⚖️ F1-Score", f"{perf_metrics['f1']:.3f}")
    
    st.divider()
    
    # Confusion matrix visualization
    st.subheader("Confusion Matrix")
    cm = np.array(perf_metrics["confusion_matrix"])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No Stockout", "Predicted: Stockout"],
        y=["Actual: No Stockout", "Actual: Stockout"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    st.info(f"""
    **Interpretation:**
    - True Negatives (✅): {tn} — Correctly predicted no stockout
    - False Positives (⚠️): {fp} — Incorrectly predicted stockout (safe but conservative)
    - False Negatives (❌): {fn} — Missed actual stockouts (worst case)
    - True Positives (🎯): {tp} — Correctly predicted stockouts
    
    **Business Impact:** With recall of {perf_metrics['recall']:.1%}, the model catches **{perf_metrics['recall']:.0%} of actual stockouts**.
    """)

# ============================================================================
# PAGE 4: DEEP DIVE ANALYSIS
# ============================================================================
elif page == "🔍 Deep Dive Analysis":
    st.header("Feature Importance & Advanced Analysis")
    
    # Feature importance
    st.subheader("Model Feature Importance (Logistic Regression Coefficients)")
    
    fi_sorted = feature_importance.sort_values("coefficient", key=abs, ascending=True)
    
    fig = px.barh(
        x=fi_sorted["coefficient"],
        y=fi_sorted.index,
        color=fi_sorted["coefficient"],
        color_continuous_scale="RdBu",
        title="Feature Coefficients (Impact on Stockout Risk)",
        labels={"coefficient": "Coefficient", "index": "Feature"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Insights:**
    - **Days of Supply** (-6.04): Strongest predictor. ↑ days = ↓ risk
    - **Demand Volatility** (+0.62): Higher volatility increases risk
    - **Average Sales** (-0.56): Stable sales reduce risk
    - **Reliability Score** (-0.23): Reliable suppliers reduce risk
    """)
    
    st.divider()
    
    # Category analysis
    st.subheader("Performance by Category")
    
    cat_analysis = model_output.groupby("category").agg({
        "risk_score": ["mean", "std"],
        "days_of_supply": "mean",
        "sku": "count"
    }).round(2)
    
    cat_analysis.columns = ["Avg Risk", "Risk Std Dev", "Avg DOS", "# Records"]
    st.dataframe(cat_analysis, use_container_width=True)
    
    st.divider()
    
    # Correlation with risk
    st.subheader("Feature Correlation with Risk Score")
    
    correlation_cols = [col for col in ["days_of_supply", "demand_volatility_30d", "avg_sales_30d",
                       "on_hand_qty", "base_lead_time_days", "reliability_score"] if col in model_output.columns]
    
    if correlation_cols:
        correlations = model_output[correlation_cols + ["risk_score"]].corr()["risk_score"].drop("risk_score").sort_values()
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation="h",
            color=correlations.values,
            color_continuous_scale="RdBu",
            title="Feature Correlation with Stockout Risk"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required correlation columns not found in data")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
---
**About this Pipeline:**
- **Data**: Simulated 1 year of daily demand, inventory, and purchase orders (40 SKUs × 4 warehouses × 10 suppliers)
- **Architecture**: SQL business logic → Python feature engineering → ML classification
- **Model**: Logistic Regression (ROC-AUC: 0.84, Recall: 0.76)
- **Output**: Real-time risk scores for stockout prediction

📖 [View Repository](https://github.com/snehashrotriya0424/-supply-chain-stockout-risk-pipeline)
""")

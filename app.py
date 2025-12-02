import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

# ------------------------------
# Helper functions
# ------------------------------
@st.cache_data
def load_csv(uploaded_file):
    """Load CSV file with proper encoding"""
    df = pd.read_csv(uploaded_file, encoding='unicode_escape', low_memory=False)
    return df

def preprocess(df):
    """Clean and preprocess the retail data"""
    df = df.copy()
    
    # Show original columns for debugging
    st.write("**Original columns found:**", df.columns.tolist())
    
    # FIXED: Check if columns already have correct case
    # If columns already match our target names, don't rename them
    expected_columns = {
        'InvoiceNo': ['InvoiceNo', 'invoiceno', 'invoice_no'],
        'StockCode': ['StockCode', 'stockcode', 'stock_code'],
        'Description': ['Description', 'description', 'desc'],
        'Quantity': ['Quantity', 'quantity', 'qty'],
        'InvoiceDate': ['InvoiceDate', 'invoicedate', 'invoice_date'],
        'UnitPrice': ['UnitPrice', 'unitprice', 'unit_price'],
        'CustomerID': ['CustomerID', 'customerid', 'customer_id'],
        'Country': ['Country', 'country']
    }
    
    # Create mapping only if needed
    rename_map = {}
    for target, variants in expected_columns.items():
        for col in df.columns:
            if col in variants:
                if col != target:  # Only rename if different
                    rename_map[col] = target
                break
    
    if rename_map:
        st.write("**Renaming columns:**", rename_map)
        df = df.rename(columns=rename_map)
    else:
        st.write("✅ **Columns already have correct names**")
    
    # Verify all required columns exist
    required = ['InvoiceNo', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        return None
    
    st.success("✅ All required columns found!")
    
    # Parse dates
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        invalid_dates = df['InvoiceDate'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ Found {invalid_dates} invalid dates, removing them")
            df = df[~df['InvoiceDate'].isna()]
    
    # Remove rows with missing CustomerID
    if 'CustomerID' in df.columns:
        before = len(df)
        df = df[~df['CustomerID'].isna()]
        removed = before - len(df)
        if removed > 0:
            st.write(f"🗑️ Removed {removed} rows with missing CustomerID")
    
    # Remove zero/negative quantities
    if 'Quantity' in df.columns:
        before = len(df)
        df = df[df['Quantity'] > 0]
        removed = before - len(df)
        if removed > 0:
            st.write(f"🗑️ Removed {removed} rows with zero/negative quantity")
    
    # Remove zero/negative prices
    if 'UnitPrice' in df.columns:
        before = len(df)
        df = df[df['UnitPrice'] > 0]
        removed = before - len(df)
        if removed > 0:
            st.write(f"🗑️ Removed {removed} rows with zero/negative price")
    
    # Create TotalPrice column
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        st.write("✅ Created TotalPrice column")
    else:
        st.error("❌ Cannot create TotalPrice - missing Quantity or UnitPrice")
        return None
    
    # Remove cancellation invoices (those starting with 'C')
    if 'InvoiceNo' in df.columns:
        before = len(df)
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)]
        removed = before - len(df)
        if removed > 0:
            st.write(f"🗑️ Removed {removed} cancellation records")
    
    # Remove negative TotalPrice
    if 'TotalPrice' in df.columns:
        before = len(df)
        df = df[df['TotalPrice'] > 0]
        removed = before - len(df)
        if removed > 0:
            st.write(f"🗑️ Removed {removed} rows with negative total price")
    
    return df

@st.cache_data
def compute_rfm(df, snapshot_date=None):
    """Calculate RFM (Recency, Frequency, Monetary) metrics for each customer"""
    df = df.copy()
    df['CustomerID'] = df['CustomerID'].astype(str)
    
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    st.write(f"📅 Snapshot date for RFM calculation: {snapshot_date.date()}")
    
    # Group by customer and calculate RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': lambda x: x.nunique(),  # Frequency
        'TotalPrice': lambda x: x.sum()  # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Ensure no negative monetary values
    rfm['Monetary'] = rfm['Monetary'].clip(lower=0.01)
    
    st.write(f"✅ RFM calculated for {len(rfm)} customers")
    
    return rfm

def rfm_score(df):
    """Convert RFM values to scores (1-5)"""
    r = df.copy()
    
    # Recency: Lower is better, so reverse the labels
    r['R_score'] = pd.qcut(r['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
    
    # Frequency: Higher is better
    r['F_score'] = pd.qcut(r['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    
    # Monetary: Higher is better
    r['M_score'] = pd.qcut(r['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    
    # Combined RFM score
    r['RFM_score'] = r['R_score']*100 + r['F_score']*10 + r['M_score']
    
    return r

def scale_features(X):
    """Standardize features to zero mean and unit variance"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def run_pca(X, n_components=2):
    """Reduce dimensionality using PCA"""
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_.sum()
    st.write(f"PCA: {explained_var:.1%} variance explained with {n_components} components")
    return Xp, pca

def fit_kmeans(X, n_clusters=3):
    """K-Means clustering"""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def fit_gmm(X, n_clusters=3):
    """Gaussian Mixture Model clustering"""
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def fit_agglomerative(X, n_clusters=3):
    """Hierarchical Agglomerative clustering"""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels, model

def fit_birch(X, n_clusters=3):
    """BIRCH clustering"""
    model = Birch(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels, model

def fit_dbscan(X, eps=0.5, min_samples=5):
    """DBSCAN clustering"""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def evaluate_labels(X, labels):
    """Calculate clustering quality metrics"""
    unique_labels = set(labels)
    label_set = [l for l in unique_labels if l != -1]  # Exclude noise for DBSCAN
    
    if len(label_set) >= 2:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        return sil, db
    else:
        return None, None

def cluster_summary(rfm_df, labels):
    """Summarize each cluster's characteristics"""
    df = rfm_df.copy()
    df['Cluster'] = labels
    
    summary = df.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Recency': 'median',
        'Frequency': 'median',
        'Monetary': 'median'
    }).rename(columns={'CustomerID':'Count'}).reset_index()
    
    return summary

def assign_personas(summary):
    """Assign marketing personas to clusters based on RFM characteristics"""
    personas = []
    
    for _, row in summary.iterrows():
        cluster = row['Cluster']
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        
        # Define personas based on RFM patterns
        if recency <= 30 and frequency >= summary['Frequency'].median() and monetary >= summary['Monetary'].median():
            persona = "🌟 Champions"
            recommendation = "Reward them. Offer exclusive products, early access, and VIP programs."
        
        elif recency <= 30 and frequency < summary['Frequency'].median():
            persona = "🌱 New Customers"
            recommendation = "Build relationships. Provide onboarding, educational content, and first-purchase incentives."
        
        elif recency > 90 and monetary >= summary['Monetary'].median():
            persona = "💤 Hibernating High-Value"
            recommendation = "Re-engage with personalized offers, 'We miss you' campaigns, and special discounts."
        
        elif recency > 90:
            persona = "👋 Lost Customers"
            recommendation = "Win-back campaigns with aggressive discounts or product recommendations based on past purchases."
        
        elif frequency >= summary['Frequency'].median():
            persona = "🔄 Loyal Customers"
            recommendation = "Keep them engaged with loyalty programs, exclusive content, and appreciation rewards."
        
        else:
            persona = "🤔 Potential Loyalists"
            recommendation = "Upsell and cross-sell. Offer membership programs and frequency incentives."
        
        personas.append({
            'Cluster': cluster,
            'Persona': persona,
            'Recommendation': recommendation,
            'Count': row['Count']
        })
    
    return pd.DataFrame(personas)

def to_excel_download(df):
    """Convert dataframe to Excel file for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Customer_Segments')
    processed_data = output.getvalue()
    return processed_data

# ------------------------------
# Main Streamlit App
# ------------------------------

st.title("🧭 Customer Segmentation Dashboard")
st.markdown("Analyze customer behavior using RFM analysis and machine learning clustering")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file",
        type=['csv'],
        help="Expected columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
    )
    
    sample_data = st.checkbox("Use sample data (for testing)", value=False)
    
    st.divider()
    
    st.subheader("Clustering Settings")
    algorithm = st.selectbox(
        "Algorithm",
        options=["KMeans", "GaussianMixture (GMM)", "Agglomerative", "BIRCH", "DBSCAN"]
    )
    
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    
    if algorithm == "DBSCAN":
        eps = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 3, 50, 5)
    
    use_pca_for_model = st.checkbox("Apply PCA before clustering", value=True)
    
    st.divider()
    
    visual_method = st.selectbox("Visualization method", options=["PCA (2D)", "UMAP (2D)"])

# Check if data is provided
if uploaded_file is None and not sample_data:
    st.info("""
    ### 👋 Welcome! Get started by uploading your data
    
    **Expected CSV format:**
    - InvoiceNo: Transaction identifier
    - StockCode: Product code
    - Description: Product description
    - Quantity: Number of items purchased
    - InvoiceDate: Date and time of purchase
    - UnitPrice: Price per item
    - CustomerID: Unique customer identifier
    - Country: Customer's country
    
    Or check "Use sample data" in the sidebar to try with synthetic data.
    """)
    st.stop()

# Load data
if uploaded_file:
    with st.spinner("Loading data..."):
        raw = load_csv(uploaded_file)
elif sample_data:
    # Generate sample data
    st.info("📝 Using synthetic sample data")
    rng = np.random.RandomState(42)
    n = 5000
    raw = pd.DataFrame({
        'InvoiceNo': rng.choice([f"{10000+i}" for i in range(2000)], size=n),
        'StockCode': rng.randint(1000, 1100, size=n),
        'Description': rng.choice(['Widget A', 'Gadget B', 'Thing C', 'Product X', 'Item Y'], size=n),
        'Quantity': rng.randint(1, 15, size=n),
        'InvoiceDate': pd.to_datetime('2021-01-01') + pd.to_timedelta(rng.randint(0, 365, size=n), unit='D'),
        'UnitPrice': rng.choice([5.0, 9.99, 15.5, 2.5, 20.0, 30.0], size=n),
        'CustomerID': rng.choice([f"C{1000+i}" for i in range(500)], size=n),
        'Country': rng.choice(['United Kingdom', 'France', 'Germany', 'USA'], size=n)
    })

# Display raw data
st.subheader("📊 Raw Data Preview")
st.dataframe(raw.head(10), use_container_width=True)
st.write(f"**Total records:** {len(raw):,}")
st.write(f"**Date range:** {raw['InvoiceDate'].min()} to {raw['InvoiceDate'].max()}")
st.write(f"**Unique customers:** {raw['CustomerID'].nunique():,}")

# Preprocess
st.subheader("🔧 Data Preprocessing")
df_clean = preprocess(raw)

if df_clean is None:
    st.error("❌ Preprocessing failed. Please check your data format.")
    st.stop()

st.write(f"**✅ Clean records:** {len(df_clean):,}")
with st.expander("View cleaned data sample"):
    st.dataframe(df_clean.head(10))

# Compute RFM
st.subheader("📈 RFM Analysis")
rfm = compute_rfm(df_clean)
rfm_scores = rfm_score(rfm)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
with col2:
    st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f}")
with col3:
    st.metric("Avg Monetary ($)", f"{rfm['Monetary'].mean():.2f}")

with st.expander("View RFM scores"):
    st.dataframe(rfm_scores.head(20))

# Feature selection
st.subheader("🎯 Feature Selection")
feature_mode = st.radio(
    "Choose features for clustering:",
    options=["RFM continuous values", "RFM scores (1-5)"],
    horizontal=True
)

if feature_mode == "RFM continuous values":
    X = rfm[['Recency', 'Frequency', 'Monetary']].copy()
else:
    X = rfm_scores[['R_score', 'F_score', 'M_score']].copy()

# Scale features
X_scaled, scaler = scale_features(X)

# Prepare visualization
if visual_method == "PCA (2D)":
    X_vis, pca_vis = run_pca(X_scaled, n_components=2)
    vis_df = pd.DataFrame(X_vis, columns=['Component 1', 'Component 2'])
else:
    try:
        import umap
        reducer = umap.UMAP(random_state=42)
        X_vis = reducer.fit_transform(X_scaled)
        vis_df = pd.DataFrame(X_vis, columns=['UMAP 1', 'UMAP 2'])
    except ImportError:
        st.warning("⚠️ UMAP not installed. Falling back to PCA. Install with: `pip install umap-learn`")
        X_vis, pca_vis = run_pca(X_scaled, n_components=2)
        vis_df = pd.DataFrame(X_vis, columns=['Component 1', 'Component 2'])

# Run clustering
run_button = st.button("🚀 Run Clustering", type="primary", use_container_width=True)

if run_button:
    with st.spinner(f"Running {algorithm} clustering..."):
        
        # Apply clustering
        if algorithm == "GaussianMixture (GMM)":
            if use_pca_for_model:
                X_model, _ = run_pca(X_scaled, n_components=min(5, X_scaled.shape[1]))
            else:
                X_model = X_scaled
            labels, model = fit_gmm(X_model, n_clusters=n_clusters)
        
        elif algorithm == "KMeans":
            labels, model = fit_kmeans(X_scaled, n_clusters=n_clusters)
        
        elif algorithm == "Agglomerative":
            labels, model = fit_agglomerative(X_scaled, n_clusters=n_clusters)
        
        elif algorithm == "BIRCH":
            labels, model = fit_birch(X_scaled, n_clusters=n_clusters)
        
        elif algorithm == "DBSCAN":
            labels, model = fit_dbscan(X_scaled, eps=eps, min_samples=min_samples)
        
        # Add cluster labels
        rfm['Cluster'] = labels
        vis_df['Cluster'] = labels.astype(str)
        vis_df['CustomerID'] = rfm['CustomerID'].values
        vis_df['Recency'] = rfm['Recency'].values
        vis_df['Frequency'] = rfm['Frequency'].values
        vis_df['Monetary'] = rfm['Monetary'].values
        
        # Evaluation metrics
        st.subheader("📊 Clustering Quality Metrics")
        sil, db = evaluate_labels(X_scaled, labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", len(set(labels)) - (1 if -1 in labels else 0))
        with col2:
            st.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A",
                     help="Higher is better (range: -1 to 1)")
        with col3:
            st.metric("Davies-Bouldin Index", f"{db:.3f}" if db is not None else "N/A",
                     help="Lower is better")
        
        # Visualization
        st.subheader("🎨 Cluster Visualization")
        fig = px.scatter(
            vis_df,
            x=vis_df.columns[0],
            y=vis_df.columns[1],
            color='Cluster',
            hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
            title=f"Customer Segments ({algorithm})",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster summary
        st.subheader("📋 Cluster Summary")
        summary = cluster_summary(rfm, labels)
        st.dataframe(summary, use_container_width=True)
        
        # Personas and recommendations
        st.subheader("👥 Customer Personas & Marketing Strategies")
        persona_df = assign_personas(summary)
        
        for _, row in persona_df.iterrows():
            with st.expander(f"{row['Persona']} - {row['Count']} customers"):
                st.write(f"**Cluster {row['Cluster']}**")
                st.write(f"**Strategy:** {row['Recommendation']}")
                
                # Show sample customers
                cluster_customers = rfm[rfm['Cluster'] == row['Cluster']].head(5)
                st.dataframe(
                    cluster_customers[['CustomerID', 'Recency', 'Frequency', 'Monetary']],
                    use_container_width=True
                )
        
        # Download results
        st.subheader("💾 Download Results")
        download_df = rfm.copy()
        xlsx = to_excel_download(download_df)
        
        st.download_button(
            label="📥 Download Excel with Cluster Labels",
            data=xlsx,
            file_name=f"customer_segments_{algorithm}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.success("✅ Clustering complete! Review the personas and download the results.")

st.divider()
st.caption("Customer Segmentation Dashboard | Built with Streamlit")

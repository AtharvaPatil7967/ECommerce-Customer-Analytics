# ECommerce-Customer-Analytics
An interactive Streamlit dashboard for e-commerce customer segmentation using RFM analysis and clustering algorithms (K-Means, GMM, DBSCAN).
A comparative analysis tool for customer segmentation. Features automated RFM calculation, multiple clustering algorithms (GMM, K-Means, DBSCAN), and interactive 3D visualizations to identify 'Champions', 'Loyal', and 'At-Risk' customer groups.


# 🧭 Customer Segmentation Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange)

## 📄 Abstract
This project focuses on the development of an intelligent customer segmentation model that helps businesses group customers effectively and design targeted marketing strategies. Using **RFM (Recency, Frequency, Monetary)** analysis combined with machine learning clustering algorithms, this dashboard provides actionable insights into customer behavior.

## 🎯 Objectives
* To design an interactive dashboard for visualizing customer segments.
* To compare different clustering algorithms (K-Means, GMM, DBSCAN, etc.) to find the most effective model.
* To identify key customer personas like "Champions," "Loyal Customers," and "At-Risk" users.

## ⚙️ Key Features
* **Data Preprocessing:** Automated handling of missing values, negative quantities, and duplicate records.
* **RFM Analysis:** Automatic calculation of Recency, Frequency, and Monetary scores.
* **Multi-Algorithm Support:** Compare performance between:
    * K-Means Clustering
    * Gaussian Mixture Models (GMM)
    * DBSCAN
    * Agglomerative Clustering
    * BIRCH
* **Interactive Visualization:** 2D/3D Scatter plots using PCA and UMAP.
* **Business Personas:** Auto-generation of marketing strategies (e.g., "Upsell to Potential Loyalists").
* **Export:** Download segmented data as Excel files.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (PCA, StandardScaler, Clustering algorithms)
* **Visualization:** Plotly Express, Matplotlib, Seaborn

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name

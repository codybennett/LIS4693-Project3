"""
LIS4693 - IR & Text Mining - Project 3

This module implements the K-Means clustering exploration interface.
It provides functionality for:
* **TF-IDF Computation**: Generate document-term matrices for clustering.
* **K-Means Clustering**: Group documents into clusters based on similarity.
* **PCA Visualization**: Reduce dimensionality for 2D cluster visualization.
* **Cluster Insights**: Display top keywords and category distributions per cluster.

Key Features:
* **Interactive Clustering**: Adjust the number of clusters dynamically.
* **Cluster Analysis**: Explore cluster sizes, keywords, and category distributions.
* **Data Persistence**: Save cluster assignments to the database.

Date: 2025-04-26
Version: 2.0
Authors: Cody Bennett
         - codybennett@ou.edu
"""

import pandas as pd
import streamlit as st
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.set_page_config(layout="wide", page_title="K-Means Exploration")
# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.write("Use the links below to navigate:")
    st.markdown("[🏠 Home](/)")
    st.markdown("[📄 Application](/app)")
    st.markdown("[📊 K-Means Exploration](/kmeans_exploration)")
    st.markdown("[🤖 Naive Bayes Classification](/nbc_exploration)")

# --- PAGE HEADER ---
st.title("K-Means Exploration")
st.markdown(
    """
    This page allows you to explore and visualize document clusters using **K-Means Clustering** and **PCA**.

    - **K-Means Clustering**: A machine learning algorithm that groups documents into clusters based on their similarity.
    - **PCA (Principal Component Analysis)**: A dimensionality reduction technique used to project high-dimensional data into 2D for visualization.
    - **Clusters**: Groups of similar documents, where each cluster represents a distinct topic or theme.

    Use the options in the sidebar to adjust the number of clusters and explore the results below.
    """
)

cwd = os.getcwd()


# --- HELPER FUNCTIONS ---
@st.cache_data
def load_processed_corpus(database="analysis.db"):
    database_path = os.path.join(cwd, database)
    with sqlite3.connect(database_path) as db:
        df = pd.read_sql_query(
            "SELECT fileid, processed_content, categories FROM documents", db
        )
    return df


@st.cache_data
def compute_tfidf(corpus_df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus_df["processed_content"])
    return tfidf_matrix, vectorizer


# --- LOAD DATA ---
with st.spinner("Loading data..."):
    df = load_processed_corpus()
    if df.empty:
        st.error("No documents found. Please process documents first.")
        st.stop()

# --- UNIQUE CATEGORIES WIDGET ---
if "categories" in df.columns:
    unique_categories = {cat for cats in df["categories"] for cat in cats}
    st.metric(label="Unique Categories", value=len(unique_categories))
else:
    st.warning("No category data available in the corpus.")

# --- CLUSTERING OPTIONS ---
st.sidebar.header("Clustering Options")
st.sidebar.info(
    """
    **Clustering Options**:
    - Adjust the number of clusters to group documents into distinct topics.
    - The clustering results are visualized using PCA and bar charts.
    """
)
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

# --- COMPUTE TF-IDF ---
tfidf_matrix, vectorizer = compute_tfidf(df)

# --- K-MEANS CLUSTERING ---
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

df["Cluster"] = cluster_labels

# --- SAVE CLUSTER ASSIGNMENTS ---
database_path = os.path.join(cwd, "analysis.db")
try:
    with sqlite3.connect(database_path) as db:
        cur = db.cursor()
        for fileid, cluster in zip(df["fileid"], cluster_labels):
            cur.execute(
                "UPDATE documents SET cluster = ? WHERE fileid = ?",
                (int(cluster), fileid),
            )
        db.commit()
    st.success("Cluster assignments saved to database.")
except Exception as e:
    st.error(f"Error saving clusters: {e}")

# --- VISUALIZATIONS ---

st.subheader("📊 Cluster Size Distribution")
st.markdown(
    """
    This bar chart shows the number of documents in each cluster. Larger clusters indicate more documents grouped under a similar topic.
    """
)
cluster_counts = df["Cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

st.subheader("📈 2D PCA Scatter Plot of Clusters")
st.markdown(
    """
    This scatter plot visualizes the clusters in 2D space using PCA. Each point represents a document, and its position is determined by its similarity to other documents.
    """
)

# PCA Projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(tfidf_matrix.toarray())

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=50, alpha=0.7
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Scatter Plot of Document Clusters")
plt.colorbar(scatter, ticks=range(num_clusters))
st.pyplot(fig)

st.subheader("🌟 Top Keywords per Cluster")
st.markdown(
    """
    The top keywords for each cluster provide insights into the main topics or themes represented by the documents in that cluster.
    """
)

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for cluster_num in range(num_clusters):
    st.markdown(f"**Cluster {cluster_num}:**")
    keywords = ", ".join(terms[ind] for ind in order_centroids[cluster_num, :10])
    st.write(keywords)

# --- CATEGORY DISTRIBUTION BY CLUSTER ---
st.subheader("📂 Category Distribution by Cluster")
st.markdown(
    """
    This section shows the distribution of document categories within each cluster. It helps identify the dominant categories in each group.
    """
)

if "categories" in df.columns:
    category_distribution = {}
    for cluster_num in range(num_clusters):
        cluster_docs = df[df["Cluster"] == cluster_num]
        categories = cluster_docs["categories"].explode().value_counts()
        category_distribution[cluster_num] = categories

    for cluster_num, categories in category_distribution.items():
        st.markdown(f"**Cluster {cluster_num} Categories:**")
        st.bar_chart(categories)
else:
    st.warning("No category data available in the corpus.")

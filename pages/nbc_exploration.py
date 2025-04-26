"""
LIS4693 - IR & Text Mining - Project 3

This module implements the Naive Bayes classification exploration interface.
It provides functionality for:
* **TF-IDF Computation**: Generate document-term matrices for classification.
* **Naive Bayes Classifier**: Train and evaluate a classifier on K-Means clusters.
* **Performance Metrics**: Display accuracy, confusion matrix, and classification report.

Key Features:
* **Interactive Training**: Split data into training and testing sets.
* **Visualization**: Display confusion matrix and classification results.
* **Cluster-Based Classification**: Use K-Means clusters as labels for classification.

Date: 2025-04-26
Version: 2.0
Authors: Cody Bennett
         - codybennett@ou.edu
"""

import streamlit as st
import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Naive Bayes Classifier Exploration")
st.title("🤖 Naive Bayes Classifier on K-Means Clusters")
# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.write("Use the links below to navigate:")
    st.markdown("[🏠 Home](/)")
    st.markdown("[📄 Application](/app)")
    st.markdown("[📊 K-Means Exploration](/kmeans_exploration)")
    st.markdown("[🤖 Naive Bayes Classification](/nbc_exploration)")

cwd = os.getcwd()


# --- HELPER FUNCTIONS ---
@st.cache_data
def compute_tfidf(corpus_df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus_df["processed_content"])
    return tfidf_matrix, vectorizer


# --- LOAD DATA ---
with st.spinner("Loading processed corpus..."):
    database_path = os.path.join(cwd, "analysis.db")
    with sqlite3.connect(database_path) as db:
        df = pd.read_sql_query("SELECT * FROM documents", db)
        if df.empty:
            st.error("No documents found. Please process documents first.")
            st.stop()

if "cluster" not in df.columns or df["cluster"].isnull().all():
    st.error("No cluster labels found. Please run K-means clustering first!")
    st.stop()

# --- COMPUTE TF-IDF ---
tfidf_matrix, vectorizer = compute_tfidf(df)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, df["cluster"], test_size=0.2, random_state=42
)

# --- TRAIN NAIVE BAYES ---
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- RESULTS ---
st.subheader("📊 Classifier Results")

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {accuracy:.2f}")

# Display results in two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["cluster"].unique()))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(df["cluster"].unique()),
        yticklabels=sorted(df["cluster"].unique()),
    )
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Cluster")
    st.pyplot(fig)

with col2:
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

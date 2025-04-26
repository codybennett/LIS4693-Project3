"""
LIS4693 - IR & Text Mining - Project 3

This module implements the main application interface for searching and exploring the processed corpus.
It provides functionality for:
* **Corpus Search**: Perform keyword-based searches using TF-IDF and cosine similarity.
* **Cluster Visualization**: Display cluster distributions and filter results by cluster.
* **Document Exploration**: View document snippets, metadata, and full content.
* **Dataset Management**: Reinitialize, reindex, and reload the dataset.

Key Features:
* **Enhanced Search**: Supports relevance-based and fallback full-text search.
* **Interactive Filters**: Filter results by cluster or categories.
* **Pagination**: Paginate search results for better navigation.

Date: 2025-04-26
Version: 2.0
Authors: Cody Bennett
         - codybennett@ou.edu
"""

import json
import logging
import os

import sqlite3

import pandas as pd
import streamlit as st
from nltk import pos_tag,download
from nltk.corpus import stopwords, wordnet
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from data_collection import DataCollector

st.set_page_config(layout="wide", page_title="Search Corpus")
st.title("Search Corpus")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.write("Use the links below to navigate:")
    st.markdown("[🏠 Home](/)")
    st.markdown("[📄 Application](/app)")
    st.markdown("[📊 K-Means Exploration](/kmeans_exploration)")
    st.markdown("[🤖 Naive Bayes Classification](/nbc_exploration)")

cwd = os.getcwd()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Ensure NLTK packages are available
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    filename="output.log",
    filemode="a",
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
)


@st.cache_data
def load_tfidf_and_corpus(database="analysis.db"):
    collector = DataCollector()
    tfidf_matrix, vectorizer = collector.load_tfidf_from_database()

    if tfidf_matrix is None or vectorizer is None:
        collector.compute_and_store_tfidf()
        tfidf_matrix, vectorizer = collector.load_tfidf_from_database()

    database_path = os.path.join(os.getcwd(), database)
    with sqlite3.connect(database_path) as db:
        df = pd.read_sql_query("SELECT * FROM documents", db)
        if "categories" in df.columns:
            df["categories"] = df["categories"].apply(json.loads)
        else:
            df["categories"] = [[]] * len(df)

    df["Snippet"] = df["processed_content"].apply(
        lambda x: generate_relevant_snippet(x, [], snippet_length=30)
    )
    return df, tfidf_matrix, vectorizer


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Default noun lemmatization
    return " ".join(word for word in tokens if word not in stop_words and len(word) > 1)



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def generate_relevant_snippet(content, query_terms, snippet_length=30):
    lines = content.splitlines()
    content_without_title = "\n".join(line for line in lines if not line.isupper())
    words = content_without_title.split()
    for i, word in enumerate(words):
        if word.lower() in query_terms:
            start = max(0, i - snippet_length // 2)
            end = min(len(words), i + snippet_length // 2)
            return " ".join(words[start:end])
    return " ".join(words[:snippet_length])


# --- CLUSTER COLORING ---
cluster_colors = {
    0: "blue",
    1: "green",
    2: "orange",
    3: "purple",
    4: "red",
    5: "cyan",
    6: "pink",
    7: "brown",
    8: "olive",
    9: "teal",
}


def get_cluster_color(cluster):
    return cluster_colors.get(cluster, "gray")


# --- SEARCH FUNCTION ---
def enhanced_search_corpus(
    query, corpus_df, tfidf_matrix, vectorizer, relevance_threshold=0.2
):
    preprocessed_query = preprocess_text(query)
    query_vector = vectorizer.transform([preprocessed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []
    for idx, similarity in enumerate(similarities):
        if similarity >= relevance_threshold:
            row = corpus_df.iloc[idx]
            snippet = generate_relevant_snippet(
                row["content"], preprocessed_query.split()
            )
            results.append(
                {
                    "Document ID": row["fileid"],
                    "Title": row.get("title", "Untitled Document"),
                    "Categories": row.get("categories", []),
                    "Snippet": snippet,
                    "Word Count": row["word_count"],
                    "Relevance": similarity,
                    "Cluster": row.get("cluster", "Unknown"),
                    "Content": row.get("content", "Full content not available."),
                }
            )

    if results:
        return pd.DataFrame(results).sort_values(by="Relevance", ascending=False)
    else:
        fallback_results = []
        for idx, row in corpus_df.iterrows():
            content_lower = row["content"].lower()
            title_lower = row.get("title", "").lower()
            query_lower = query.lower()

            # Check for exact text match in content or partial match in title
            if query_lower in content_lower or query_lower in title_lower:
                snippet = generate_relevant_snippet(
                    row["content"], query_lower.split()
                )
                relevance = 0.3 if query_lower in title_lower else 0.2  # Higher relevance for title match
                fallback_results.append(
                    {
                        "Document ID": row["fileid"],
                        "Title": row.get("title", "Untitled Document"),
                        "Categories": row.get("categories", []),
                        "Snippet": snippet,
                        "Word Count": row["word_count"],
                        "Relevance": relevance,
                        "Cluster": row.get("cluster", "Unknown"),
                        "Content": row.get("content", "Full content not available."),
                    }
                )
        return pd.DataFrame(fallback_results).sort_values(by="Relevance", ascending=False) if fallback_results else pd.DataFrame()


# --- PAGINATED RESULTS DISPLAY ---
def display_paginated_results(results_df, page_size=10):
    if "Cluster" not in results_df.columns:
        results_df["Cluster"] = "Unknown"

    total_results = len(results_df)
    total_pages = (total_results + page_size - 1) // page_size
    current_page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )
    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_results = results_df.iloc[start_idx:end_idx]

    selected_clusters = st.multiselect(
        "Filter by Cluster (optional):", sorted(results_df["Cluster"].dropna().unique())
    )
    selected_categories = st.multiselect(
        "Filter by Category (optional):",
        sorted({cat for cats in results_df["Categories"] for cat in cats}),
    )

    filtered_df = paginated_results
    if selected_clusters:
        filtered_df = filtered_df[filtered_df["Cluster"].isin(selected_clusters)]
    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["Categories"].apply(
                lambda cats: any(cat in selected_categories for cat in cats)
            )
        ]

    for _, row in filtered_df.iterrows():
        relevance = row.get("Relevance", 0.0)
        color = "gray"  # Default color for relevance (can be customized if needed)

        with st.expander(f"📄 {row['Title']}  |  Relevance {relevance:.2f}", expanded=False):
            st.markdown(
                f"**<span style='color:{color}'>Relevance {relevance:.2f}</span>**",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Categories:** {', '.join(row['Categories']) if row['Categories'] else 'None'}"
            )
            st.markdown(f"**Snippet:** {row['Snippet']}")
            st.markdown(f"**Word Count:** {row['Word Count']}")
            st.markdown("### Full Content")
            st.text(row["Content"])


# --- MAIN PAGE CONTENT ---
st.markdown(
    """
## How to Use the App
1. **Search**: Enter a keyword or phrase in the search bar.
   - The app first looks for exact matches in document titles and content.
   - If no exact matches are found, it switches to an enhanced search using TF-IDF and cosine similarity.
2. **Filter Results**: Use the sidebar to narrow down results by Cluster or Category.
3. **Explore Documents**: Click on a document to view its snippet, metadata, and full content.
4. **Fallback Search**: If the enhanced search doesn't find anything, the app performs a broader full-text search.

**Relevance Scoring:**
- Calculated via Cosine Similarity (TF-IDF).
- If no TF-IDF match, fallback searches document content directly.
- Exact matches in titles or content are given the highest relevance.
- Enhanced search results are ranked based on cosine similarity with the query.
- Full-text matches are included as a fallback with lower relevance.
"""
)

query = st.text_input(
    "Enter your search query:", help="Type a keyword or phrase to search the corpus."
)

# --- LOAD AND SEARCH ---
with st.spinner("Loading processed corpus and TF-IDF matrix..."):
    df, tfidf_matrix, vectorizer = load_tfidf_and_corpus()

if query:
    with st.spinner("Performing search..."):
        # Attempt exact search first
        exact_results = []
        query_lower = query.lower()
        for idx, row in df.iterrows():
            content_lower = row["content"].lower()
            title_lower = row.get("title", "").lower()

            if query_lower in content_lower or query_lower in title_lower:
                snippet = generate_relevant_snippet(
                    row["content"], query_lower.split()
                )
                relevance = 1.0 if query_lower in title_lower else 0.8  # Higher relevance for title match
                exact_results.append(
                    {
                        "Document ID": row["fileid"],
                        "Title": row.get("title", "Untitled Document"),
                        "Categories": row.get("categories", []),
                        "Snippet": snippet,
                        "Word Count": row["word_count"],
                        "Relevance": relevance,
                        "Cluster": row.get("cluster", "Unknown"),
                        "Content": row.get("content", "Full content not available."),
                    }
                )

        if exact_results:
            results = pd.DataFrame(exact_results).sort_values(by="Relevance", ascending=False)
            st.success(f"Found {len(results)} exact matching documents.")
        else:
            # Fallback to enhanced search
            results = enhanced_search_corpus(query, df, tfidf_matrix, vectorizer)
            if not results.empty:
                st.success(f"Found {len(results)} matching documents.")
            else:
                st.warning("No matching documents found.")

        display_paginated_results(results)

        st.subheader("📊 Category Distribution of Results")
        if "Categories" in results.columns:
            category_counts = (
                pd.Series([cat for cats in results["Categories"] for cat in cats])
                .value_counts()
                .sort_index()
            )
            st.bar_chart(category_counts)
else:
    st.write("No query provided. Explore the corpus below:")
    st.dataframe(df[["fileid", "title", "categories", "Snippet", "word_count"]])

    st.subheader("📊 Category Distribution")
    if "categories" in df.columns:
        category_counts = (
            pd.Series([cat for cats in df["categories"] for cat in cats])
            .value_counts()
            .sort_index()
        )
        st.bar_chart(category_counts)

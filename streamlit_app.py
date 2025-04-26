"""
LIS4693 - IR & Text Mining - Project 3

This module serves as the entry point for the Streamlit application. It ensures the database is initialized,
required NLTK packages are available, and provides navigation to different pages of the application.

Key Features:
* **Database Initialization**: Ensures the SQLite database and required tables are created.
* **NLTK Package Management**: Downloads necessary NLTK packages if not already available.
* **Navigation**: Provides links to different pages for search, clustering, and classification.
* **Corpus Management**: Automatically processes documents and computes TF-IDF if missing.

This application is designed to support information retrieval and text mining tasks, including:
* Searching the corpus for relevant documents.
* Exploring document clusters using K-Means.
* Evaluating classification performance using Naive Bayes.

Date: 2025-04-26
Version: 1.1
Authors: Cody Bennett
         - codybennett@ou.edu
"""

import streamlit as st
import nltk
from nltk.data import find
import pandas as pd
import sqlite3
import os
from pages.data_collection import DataCollector  # Import the DataCollector class

def ensure_nltk_packages():
    """Ensure required NLTK packages are downloaded."""
    packages = [
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "wordnet",
    ]
    for package in packages:
        try:
            find(f"corpora/{package}")  # Check if the package is already downloaded
        except LookupError:
            nltk.download(package, quiet=True)  # Download the package if not found

# Ensure NLTK packages are available
ensure_nltk_packages()

st.set_page_config(layout="wide", page_title="LIS 4693 - IR & Text Mining - Project 3")

@st.cache_data
def initialize_database(database="analysis.db"):
    """Create the SQLite database file and tables if they do not exist, and process documents if needed.

    If the TF-IDF matrix is not found, calculate and store it.

    :param database: Path to the SQLite database
    :type database: str
    """
    try:
        # Ensure the mycorpus directory exists
        corpus_dir = os.path.join(os.getcwd(), "mycorpus")
        os.makedirs(corpus_dir, exist_ok=True)

        # Ensure the database is created in the current working directory
        database_path = os.path.join(os.getcwd(), database)
        with sqlite3.connect(database_path) as db:
            cur = db.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                            id TEXT PRIMARY KEY,
                            fileid TEXT UNIQUE,
                            content TEXT,
                            processed_content TEXT,
                            tokens TEXT,
                            word_count INTEGER,
                            categories TEXT,
                            title TEXT,
                            cluster INTEGER DEFAULT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tfidf (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matrix BLOB,
                    vectorizer BLOB
                )
            """)
            db.commit()

        # Check if the database contains any processed documents
        with sqlite3.connect(database_path) as db:
            cur = db.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            document_count = cur.fetchone()[0]

        # If no documents are processed, kick off processing
        collector = DataCollector()  # Initialize the DataCollector
        if document_count == 0:
            st.warning("No processed documents found in the database. Starting data collection...")
            collector.process_all_documents()  # Process all documents
            st.success("Data collection completed successfully.")

        # Check if TF-IDF data exists
        tfidf_matrix, vectorizer = collector.load_tfidf_from_database()
        if tfidf_matrix is None or vectorizer is None:
            st.warning("TF-IDF data not found in the database. Calculating and storing TF-IDF...")
            collector.compute_and_store_tfidf()
            st.success("TF-IDF calculation and storage completed successfully.")
    except Exception as e:
        st.error(f"An error occurred while initializing the database: {e}")

@st.cache_data
def check_database(database="analysis.db"):
    """Check if the SQLite database is initialized and contains processed data.

    :param database: Path to the SQLite database
    :type database: str
    :return: Boolean indicating whether the database is ready
    :rtype: bool
    """
    database_path = os.path.join(os.getcwd(), database)  # Reference the database in the current working directory
    with sqlite3.connect(database_path) as db:
        cur = db.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        return cur.fetchone() is not None

# Ensure the database is initialized before checking readiness
initialize_database()

if not check_database():
    st.error("The SQLite database is not initialized or does not contain the required table. Please run the data collection process first.")
else:
    st.success("The SQLite database is ready for use.")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.write("Use the links below to navigate:")
    st.markdown("[🏠 Home](/)")
    st.markdown("[📄 Application](/app)")
    st.markdown("[📊 K-Means Exploration](/kmeans_exploration)")
    st.markdown("[🤖 Naive Bayes Classification](/nbc_exploration)")

# Homepage content
st.title("Welcome to the IR & Text Mining Project")

st.markdown(
    """
# About The Project

_This application is an advanced information retrieval system designed for analyzing and exploring text data._

------------

* Accepts user queries and returns a ranked list of relevant results.
* Highlights matching text snippets for better context and understanding.
* Provides tools for filtering, exploring, and exporting corpus data.

## Features

* **Search Functionality**: Perform keyword or phrase searches across the corpus with relevance scoring.
* **Corpus Exploration**: Filter and explore documents by clusters or categories.
* **K-Means Clustering**: Group documents into clusters to uncover hidden patterns.
* **Naive Bayes Classification**: Evaluate classification performance using precomputed clusters.
* **Export Capability**: Save search results or the entire corpus for offline analysis.

## Built With

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Getting Started

1. Navigate to the [Application](/app) to search the corpus and explore document metadata.
2. Use the [K-Means Exploration](/kmeans_exploration.py) page to visualize document clusters.
3. Visit the [Naive Bayes Classification](/nbc_exploration) page to analyze classification performance.

## Contributing

Contributions are welcome! If you have suggestions for improvements, please fork the repository and create a pull request. You can also open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development

Source code for this project is located in a private GitHub repository.

The application is deployed as a Streamlit application via [Community Cloud](https://streamlit.io/cloud).

## License

Distributed under the GNU GPL3 License.

## Contact

* Cody Bennett - <codybennett@ou.edu>

Project Link: [https://github.com/codybennett/LIS4693-Project3](https://github.com/codybennett/LIS4693-Project3)
"""
)

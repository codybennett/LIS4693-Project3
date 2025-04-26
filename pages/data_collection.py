"""
LIS4693 - IR & Text Mining - Project 3

This module implements the `DataCollector` class, which is responsible for collecting, processing, and storing documents
from the Reuters corpus. It provides functionality for tokenization, lemmatization, stopword removal, and TF-IDF computation.

Key Features:
* **Corpus Processing**: Processes documents from the Reuters corpus and stores them in an SQLite database.
* **Metadata Extraction**: Extracts metadata such as word count, categories, and titles for each document.
* **TF-IDF Computation**: Computes and stores the TF-IDF matrix for the processed corpus.
* **Database Management**: Handles the creation and management of SQLite tables for storing document data and TF-IDF matrices.

Date: 2025-04-26
Version: 2.0
Authors: Cody Bennett
         - codybennett@ou.edu
"""

from pathlib import Path
import os
import sqlite3
import nltk
import hashlib
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  # For hashing the TF-IDF matrix
import html  # For HTML unescaping

import log_config

logger = log_config.create_logger()

nltk.download("reuters", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

corpus = nltk.corpus.reuters
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Collect all documents containing > 50 words
tmp_documents = [
    tmp_document
    for tmp_document in corpus.fileids()
    if len(corpus.words(tmp_document)) > 50
]


class DataCollector:
    """A class for collecting and processing documents from the Reuters corpus.

    This class retrieves documents from the Reuters corpus that contain more than
    50 words, organizes them into a directory structure, and provides methods to
    access document content, metadata, and word counts. It also handles caching
    and persistence using SQLite for processed data and tokens.
    """

    def __init__(self):
        """Initialize the DataCollector and set up the SQLite database."""
        logger.debug("Initializing DataCollector and downloading Reuters Corpus")
        logger.info("Retrieving %d documents matching criteria", len(tmp_documents))

        # Initialize metadata dictionary
        self.document_metadata = {}
        self.documents = tmp_documents

        # Process documents
        for fileid in tmp_documents:
            doc_root = fileid.split("/")[0]
            # Create directory tree if needed
            if doc_root not in self.document_metadata:
                Path(f"mycorpus/{doc_root}").mkdir(parents=True, exist_ok=True)
                self.document_metadata[doc_root] = {"num_documents": 0, "num_words": 0}

            # Update metadata
            self.document_metadata[doc_root]["num_documents"] += 1
            self.document_metadata[doc_root]["num_words"] += len(corpus.words(fileid))

        logger.info("Corpus Metadata: %s", self.document_metadata)

        # Initialize SQLite database
        self.database = os.getenv("LISDB", "analysis.db")
        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute(
                """
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
                    """
            )

            db.commit()

            # Check if the 'categories' column exists, and add it if not
            cur.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cur.fetchall()]
            if "categories" not in columns:
                logger.info("Adding 'categories' column to 'documents' table.")
                cur.execute("ALTER TABLE documents ADD COLUMN categories TEXT")
                db.commit()

            # Check if the 'title' column exists, and add it if not
            if "title" not in columns:
                logger.info("Adding 'title' column to 'documents' table.")
                cur.execute("ALTER TABLE documents ADD COLUMN title TEXT")
                db.commit()

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tfidf (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matrix BLOB,
                    vectorizer BLOB
                )
            """
            )
            db.commit()

    def _hash_content(self, content):
        """Generate a hash for the content to use as a unique ID.

        :param content: Document content
        :type content: str
        :return: Hash of the content
        :rtype: str
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _process_document(self, content):
        """Process a document by tokenizing, lemmatizing, and removing stopwords.

        :param content: Document content
        :type content: str
        :return: Tuple containing processed content and tokens
        :rtype: tuple
        """
        tokens = word_tokenize(content)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        pos_tags = pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(word, self._get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]
        processed_content = " ".join(
            word for word in tokens if word not in stop_words and len(word) > 1
        )
        return processed_content, tokens

    def _get_wordnet_pos(self, treebank_tag):
        """Map POS tag to the format used by WordNetLemmatizer.

        :param treebank_tag: POS tag in Treebank format
        :type treebank_tag: str
        :return: Corresponding WordNet POS tag
        :rtype: str
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        if treebank_tag.startswith("V"):
            return wordnet.VERB
        if treebank_tag.startswith("N"):
            return wordnet.NOUN
        if treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def _extract_title(self, content):
        """Extract the title as the first uppercase HTML-escaped text from the content.

        :param content: Document content
        :type content: str
        :return: Extracted title
        :rtype: str
        """
        lines = content.splitlines()
        for line in lines:
            line = html.unescape(line.strip())
            if line.isupper():
                return line
        return "Untitled Document"

    def upsert_document(self, fileid):
        """Insert or update a document in the SQLite database.

        :param fileid: Document file ID
        :type fileid: str
        """
        content = corpus.raw(fileid)
        doc_id = self._hash_content(content)
        word_count = len(corpus.words(fileid))
        categories = json.dumps(
            corpus.categories(fileid)
        )  # Get categories as a JSON string
        title = self._extract_title(content)  # Extract the title

        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
            if cur.fetchone():
                logger.info("Document %s already processed. Skipping.", fileid)
                return

            processed_content, tokens = self._process_document(content)
            cur.execute(
                """
                INSERT INTO documents (id, fileid, content, processed_content, tokens, word_count, categories, title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    doc_id,
                    fileid,
                    content,
                    processed_content,
                    json.dumps(tokens),
                    word_count,
                    categories,
                    title,
                ),
            )
            db.commit()
            logger.info("Document %s processed and stored.", fileid)

    def process_all_documents(self):
        """Process and store all documents in the corpus."""
        for fileid in self.documents:
            self.upsert_document(fileid)

    def get_processed_document(self, fileid):
        """Retrieve a processed document from the SQLite database.

        :param fileid: Document file ID
        :type fileid: str
        :return: Processed document data
        :rtype: dict or None
        """
        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute("SELECT * FROM documents WHERE fileid = ?", (fileid,))
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "fileid": row[1],
                    "content": row[2],
                    "processed_content": row[3],
                    "tokens": json.loads(row[4]),
                    "word_count": row[5],
                    "categories": json.loads(row[6]),
                    "title": row[7],
                }
            logger.error("Processed document %s not found in database.", fileid)
            return None

    def get_all_processed_documents(self):
        """Retrieve all processed documents from the SQLite database.

        :return: List of processed document data
        :rtype: list
        """
        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute("SELECT * FROM documents")
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "fileid": row[1],
                    "content": row[2],
                    "processed_content": row[3],
                    "tokens": json.loads(row[4]),
                    "word_count": row[5],
                    "categories": json.loads(row[6]),
                    "title": row[7],
                }
                for row in rows
            ]

    def compute_and_store_tfidf(self):
        """Compute the TF-IDF matrix and store it in the database."""
        # Load processed content from the database
        with sqlite3.connect(self.database) as db:
            query = "SELECT processed_content FROM documents"
            df = pd.read_sql_query(query, db)

        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df["processed_content"])

        # Serialize the matrix and vectorizer
        serialized_matrix = pickle.dumps(tfidf_matrix)
        serialized_vectorizer = pickle.dumps(vectorizer)

        # Store in the database
        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute("DELETE FROM tfidf")  # Clear existing data
            cur.execute(
                "INSERT INTO tfidf (matrix, vectorizer) VALUES (?, ?)",
                (serialized_matrix, serialized_vectorizer),
            )
            db.commit()

    def load_tfidf_from_database(self):
        """Load the TF-IDF matrix and vectorizer from the database.

        :return: Tuple containing the TF-IDF matrix and vectorizer
        :rtype: tuple
        """
        with sqlite3.connect(self.database) as db:
            cur = db.cursor()
            cur.execute("SELECT matrix, vectorizer FROM tfidf LIMIT 1")
            row = cur.fetchone()
            if row:
                tfidf_matrix = pickle.loads(row[0])
                vectorizer = pickle.loads(row[1])
                return tfidf_matrix, vectorizer
            else:
                logger.error("TF-IDF data not found in the database.")
                return None, None

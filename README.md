<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable first-line-heading  -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/codybennett/LIS4693-Project3">
    <img src="Testing Screenshot.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">LIS4693-Project3</h3>

  <p align="center">
    Project 3 for University of Oklahoma LIS4693
    <br />
    <a href="https://github.com/codybennett/LIS4693-Project3"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/codybennett/LIS4693-Project3">View Demo</a>
    ·
    <a href="https://github.com/codybennett/LIS4693-Project3/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/codybennett/LIS4693-Project3/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#running-the-streamlit-application">Running the Streamlit Application</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is an advanced information retrieval and text mining system designed for analyzing and exploring text data. It allows users to search, filter, cluster, and explore a corpus of documents interactively. The system provides tools for efficient text retrieval, clustering, classification, and document exploration.

Key features include:

* **Search Functionality**: Perform keyword or phrase searches across the corpus using TF-IDF for relevance scoring and fallback full-text search.
* **K-Means Clustering**: Group documents into clusters and visualize them using PCA scatter plots and category distributions.
* **Naive Bayes Classification**: Train a supervised learning model to predict document clusters and evaluate its performance.
* **Corpus Exploration**: Filter and explore documents interactively by clusters or categories.
* **Export Capability**: Save the corpus or search results to a file for offline use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
* ![SQLite](https://img.shields.io/badge/sqlite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
* ![NLTK](https://img.shields.io/badge/nltk-85C1E9?style=for-the-badge&logo=python&logoColor=white)
* ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This project uses the Reuters dataset from the NLTK library. Follow these steps to set up the project locally.

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required Python packages:

```sh
pip install -r requirements.txt
```

### Installation

1. Clone the repo:

   ```sh
   git clone https://github.com/codybennett/LIS4693-Project3.git
   ```

2. Install Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Generate the corpus:

   Run the `data_collection.py` script to process the Reuters dataset and populate the SQLite database:

   ```sh
   python data_collection.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

This utility analyzes a directory of text documents (corpus). Users can:

1. **Generate Corpus**: Use the `data_collection.py` script to create the corpus from the Reuters dataset.
2. **Search**: Enter a keyword or phrase in the search bar to find relevant documents.
3. **Cluster**: Use K-Means clustering to group documents into clusters and visualize them.
4. **Filter**: Use sidebar options to narrow down results by clusters or categories.
5. **Explore Results**: Expand search results to view document snippets and full content.
6. **Export**: Save the corpus or search results to a text file for offline analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

### Search Functionality
- Perform keyword or phrase searches using TF-IDF for relevance scoring.
- Fallback to full-text search if no TF-IDF matches are found.

### K-Means Clustering
- Group documents into clusters based on their similarity.
- Visualize clusters using PCA scatter plots.
- Analyze category distributions within each cluster.

### Naive Bayes Classification
- Train a Naive Bayes Classifier on K-Means clusters.
- Predict the cluster for new or unseen documents.
- Evaluate the classifier's performance using metrics like accuracy, confusion matrix, and classification report.

### Corpus Exploration
- Filter documents by clusters or categories.
- View document snippets and full content interactively.

### Export Capability
- Save the corpus or search results to a file for offline use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running the Streamlit Application

To interact with the corpus and perform searches, you can run the Streamlit application locally or access it via Streamlit Cloud:

#### Local Setup

1. Ensure all dependencies are installed:
   ```sh
   pip install -r requirements.txt
   ```

2. Start the Streamlit application:
   ```sh
   streamlit run streamlit_app.py
   ```

3. Open the provided URL in your browser to access the application.

#### Streamlit Cloud

The application is also deployed on [Streamlit Cloud](https://lis4693-project3.streamlit.app/). You can access it directly without setting up a local environment by visiting the following link:

[Streamlit Cloud Deployment](https://streamlit.io/cloud)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU GPL3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

* Cody Bennett - <codybennett@ou.edu>

Project Link: [https://github.com/codybennett/LIS4693-Project3](https://github.com/codybennett/LIS4693-Project3)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was developed as part of the LIS4693 course at the University of Oklahoma. Special thanks to the course instructors and teaching assistants for their guidance and support.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/codybennett/LIS4693-Project3.svg?style=for-the-badge
[contributors-url]: https://github.com/codybennett/LIS4693-Project3/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/codybennett/LIS4693-Project3.svg?style=for-the-badge
[forks-url]: https://github.com/codybennett/LIS4693-Project3/network/members
[stars-shield]: https://img.shields.io/github/stars/codybennett/LIS4693-Project3.svg?style=for-the-badge
[stars-url]: https://github.com/codybennett/LIS4693-Project3/stargazers
[issues-shield]: https://img.shields.io/github/issues/codybennett/LIS4693-Project3.svg?style=for-the-badge
[issues-url]: https://github.com/codybennett/LIS4693-Project3/issues
[license-shield]: https://img.shields.io/badge/license-GPLv3-blue
[license-url]: https://github.com/codybennett/LIS4693-Project3/blob/master/LICENSE.txt

# SmartSearchify: AI-Enabled Semantic Search for eSankhyiki Portal

**[GitHub Repository](https://github.com/rahulkhichar7/SmartSearchify-AI-Enabled-Semantic-Search-for-eSankhyiki-Portal.git)**

---

## Overview

SmartSearchify is an AI-powered semantic search system designed to enhance the usability and accessibility of the [eSankhyiki Portal](https://www.esankhyiki.gov.in/), India's official statistics platform. The project leverages advanced Natural Language Processing (NLP) and Machine Learning (ML) to allow users to search for statistical datasets using simple, natural language queries—eliminating the need for complex dropdowns or prior knowledge of indicator names.

This solution bridges the gap between human language and structured government data, making data discovery faster, more accurate, and inclusive for researchers, policymakers, journalists, and the general public.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Data and Model Preparation](#data-and-model-preparation)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Features

- **Semantic Search**: Users can enter queries in plain English (e.g., "female literacy rate in Rajasthan in 2020").
- **Product Classification**: Automatically classifies queries into relevant statistical products (e.g., CPI, NAS, PLFS).
- **Hybrid Retrieval**: Combines fast cosine similarity filtering with deep re-ranking using transformer-based cross-encoders for high accuracy.
- **Interactive Web App**: Built with Streamlit for ease of use and rapid prototyping.
- **Efficient & Scalable**: Optimized for speed and memory usage, suitable for production deployment.
- **Memory and Performance Tracking**: Tracks memory usage and inference time for each query.

---

## Project Structure

```
.
├── app.py                      # Main Streamlit app
├── config.py                   # Configuration constants
├── services/
│   ├── data_loader.py          # Data/model loading utilities
│   ├── search_engine.py        # Core search and classification logic
│   └── utils.py                # Helper functions (UI, memory tracking)
├── final_dataset.csv           # Preprocessed dataset with embeddings
├── LinearSVC_classifier.joblib # Trained classifier model
├── Resources/
│   ├── Processed_csv_files/    # Data used for training/validation
│   ├── classification_model_selection.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Final_search_model_testing.ipynb
│   ├── Project_Presentation.pptx
│   ├── eSankhyiki Dataset/     # Raw data per product
│   ├── Project_Report.pdf      # Detailed technical report
│   └── Project_Proposal.pdf
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── ...
```

---

## How It Works

1. **Data Preparation**:  
   - Metadata and descriptions are scraped and processed from the eSankhyiki portal.
   - Text fields are cleaned, rephrased, and embedded using the `all-MiniLM-L6-v2` transformer model.
   - The dataset is stored in `final_dataset.csv` with precomputed embeddings for fast retrieval.

2. **Query Processing**:  
   - User enters a natural language query in the Streamlit web app.
   - Query is embedded using the same sentence transformer.
   - A LinearSVC classifier predicts the top 3 most relevant statistical products.

3. **Hybrid Semantic Search**:  
   - **Stage 1**: Cosine similarity filters the top 50 candidate records per product.
   - **Stage 2**: A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks these candidates for deep semantic alignment.
   - The app displays both product-specific and overall top matches, with detailed metadata and download links.

4. **Performance Tracking**:  
   - Memory usage and inference time are logged for optimization and debugging.

For a detailed explanation of the methodology, see the [Project Report][1].

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rahulkhichar7/SmartSearchify-AI-Enabled-Semantic-Search-for-eSankhyiki-Portal.git
cd SmartSearchify-AI-Enabled-Semantic-Search-for-eSankhyiki-Portal
```

### 2. Install Dependencies

Recommended: Use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install all required packages:

```bash
pip install -r requirements.txt
```

If you need to generate `requirements.txt`, use:

```bash
pip install streamlit pandas numpy sentence-transformers scikit-learn joblib psutil
pip freeze > requirements.txt
```

---

## Usage

1. Ensure `final_dataset.csv` and `LinearSVC_classifier.joblib` are present in the project root.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the provided local URL in your browser.  
4. Enter your search query in natural language and explore the results!

---

## Data and Model Preparation

- The `Resources/Processed_csv_files/` and `eSankhyiki Dataset/` directories contain raw and processed data used for training and evaluation.
- All preprocessing, model training, and evaluation steps are documented in the provided Jupyter notebooks.
- The classifier and embeddings are precomputed for fast inference.

---

## Results & Performance

- **Classification Accuracy**: 100% (LinearSVC)
- **Search Latency**: ~45 ms per query
- **Memory Usage**: ~1.1 GB on first load, ~200 MB per query
- **Hybrid Search**: Cosine similarity for initial filtering, cross-encoder for re-ranking ensures both speed and accuracy

For detailed experiments, benchmarks, and methodology, refer to the [Project Report][1].

---

## Contributing

Contributions, issues, and feature requests are welcome!  
Please open an issue or submit a pull request on [GitHub](https://github.com/rahulkhichar7/SmartSearchify-AI-Enabled-Semantic-Search-for-eSankhyiki-Portal.git).

---

## License

This project is for academic and demonstration purposes.  
For licensing details, please contact the repository owner.

---

## References

- [HuggingFace Transformers](https://huggingface.co/)
- [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [Sentence Transformers](https://www.sbert.net/)
- [eSankhyiki Portal](https://www.esankhyiki.gov.in/)
- [Project Report][1]

---

[1]: Project_Report.pdf "SmartSearchify Project Report"

---

**Contact:**  
Rahul Kichar, IIT Gandhinagar  
[GitHub](https://github.com/rahulkhichar7)

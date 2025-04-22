# SmartSearchify-AI-Enabled-Semantic-Search-for-eSankhyiki-Portal

Sure! Here's a professional `README.md` file tailored for your **Data Catalog Semantic Search** project, ready to showcase on GitHub:

---

```markdown
# 🔍 Data Catalog Semantic Search App

A powerful Streamlit-based application for semantically searching structured data catalogs using deep learning techniques like Sentence Transformers, CrossEncoders, and classification models.

---

## ✨ Features

- 🔎 **Semantic Query Search**: Understands user intent using sentence embeddings and re-ranks results for accuracy.
- 🎯 **Product Classification**: Predicts the most relevant data product using a Linear SVC classifier.
- ⚡ **Hybrid Search Strategy**: Combines cosine similarity for fast filtering with CrossEncoder for deep semantic re-ranking.
- 📦 **Streamlit UI**: Simple and interactive front-end for non-technical users.
- 📉 **Optimized Performance**: Filters top 50 candidates before deep semantic scoring to save time.

---

## 📁 Project Structure

```
├── your_data.csv               # Dataset with precomputed embeddings
├── LinearSVC_classifier.joblib # Pre-trained classification model
├── app.py                      # Main Streamlit application
├── README.md                   # You're reading this :)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/data-catalog-semantic-search.git
cd data-catalog-semantic-search
```

### 2. Install dependencies
Create a virtual environment (recommended) and install:

```bash
pip install -r requirements.txt
```

Or manually install key packages:
```bash
pip install streamlit pandas numpy sentence-transformers scikit-learn joblib psutil
```

### 3. Add your data
Make sure your `your_data.csv` file:
- Includes columns like `search_emb`, `title_emb`, `Title`, `Product`, `Description`, etc.
- Has precomputed sentence embeddings stored as stringified lists.

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🧠 Model Details

- **Sentence Embedding Model**: `all-MiniLM-L6-v2` from `sentence-transformers`
- **Semantic Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` from `sentence-transformers`
- **Classifier**: Linear SVC model trained to map queries to one of several data products

---

## 🧪 Memory Profiling

Memory usage is printed in the console:
- Resident memory used (`RSS`)
- Python-level memory tracking (`tracemalloc`)

Example:
```
[After loading models and data] Memory RSS (physical): 1068.53 MB
[After loading models and data] Tracemalloc Current: 38.37 MB; Peak: 43.08 MB
```

---

## 📸 UI Preview

> Coming soon — or include your own screenshots of the app UI.

---

## 🛠 Future Improvements

- Add batch inference for faster processing
- Live memory usage panel in UI
- Dockerfile for containerized deployment
- Add search result export/download

---

## 🧾 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Credits

Built using:
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Scikit-learn](https://scikit-learn.org/)
```

---

Let me know if you want a **`requirements.txt`**, **screenshot**, or **badge styling** too!
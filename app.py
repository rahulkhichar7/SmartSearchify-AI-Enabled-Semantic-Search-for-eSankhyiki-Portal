import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import ast
import time
import psutil
import os
import tracemalloc

# Tracking Start
tracemalloc.start()

process = psutil.Process(os.getpid())

st.set_page_config(layout="wide", page_title="Data Catalog Search")

# Load data and models
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('final_dataset.csv')  

    def convert_embedding(embed_str):
        try:
            if isinstance(embed_str, str):
                return np.array(ast.literal_eval(embed_str), dtype=np.float32)
            return embed_str
        except:
            return np.zeros(384, dtype=np.float32)

    df['search_emb'] = df['search_emb'].apply(convert_embedding)
    df['title_emb'] = df['title_emb'].apply(convert_embedding)

    classifier = load('LinearSVC_classifier.joblib')
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    return df, classifier, embedder, cross_encoder

df, classifier, embedder, cross_encoder = load_data_and_models()

# Unique product list
products = np.array(['ASI', 'ASUSE', 'CAMS', 'CPI', 'HCES', 'IIP', 'MIS', 'NAS', 'PLFS', 'WMI'])

# Classification function
def classify_query(query, embedder, classifier):
    query_embedding = embedder.encode([query])
    decision_scores = classifier.decision_function(query_embedding)
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    probabilities = probabilities[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_products = products[top3_indices]
    top3_probs = probabilities[top3_indices]
    return list(zip(top3_products, top3_probs))

# Deep semantic search using cosine + CrossEncoder
def deep_semantic_search(query, df_slice, top_k):
    # First: reduce to top 50 by cosine
    query_embedding = embedder.encode([query])
    doc_embeddings = np.stack(df_slice['search_emb'].values)
    cosine_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    top50_indices = np.argsort(cosine_scores)[-50:][::-1]
    top50_df = df_slice.iloc[top50_indices]

    # Then: re-rank using CrossEncoder
    query_doc_pairs = [(query, text) for text in top50_df['search_text'].tolist()]
    cross_scores = cross_encoder.predict(query_doc_pairs)
    top_indices = np.argsort(cross_scores)[-top_k:][::-1]
    return top50_df.iloc[top_indices]

def display_result_card(result, key_suffix=""):
    with st.expander(f"**{result['Title']}**", expanded=False):
        st.markdown(f"**Product:** {result['Product']}")
        st.markdown(f"**Category:** {result['Category']}")
        st.markdown(f"**Geography:** {result['Geography']}")
        st.markdown(f"**Frequency:** {result['Frequency']}")
        st.markdown(f"**Reference Period:** {result['Reference Period']}")
        st.markdown(f"**Release Date:** {result['Release Date']}")
        st.markdown(f"**Table No:** {result['Table No']}")
        st.markdown(f"**Data Source:** {result['Data Source']}")
        st.markdown(f"[Download Data]({result['Download URL']})", unsafe_allow_html=True)
        st.markdown(f"**Description:** {result['Description']}")

if 'last_query' not in st.session_state:
    st.session_state['last_query'] = ""

st.title("Data Catalog Search")
query = st.text_input("Enter your search query:", key="search_query", 
                     value=st.session_state.get('last_query', ''))

if query and (query != st.session_state.get('last_query', '') or 'classified_results' not in st.session_state):
    with st.spinner("Searching..."):
        query_embedding = embedder.encode([query])
        top_products = classify_query(query, embedder, classifier)

        classified_results = {}
        n_results = [3, 2, 1]  # For top 3 products

        for i, (product, prob) in enumerate(top_products):
            product_df = df[df['Product'] == product]
            top_results = deep_semantic_search(query, product_df, n_results[i])
            classified_results[product] = {
                'probability': prob,
                'results': top_results
            }

        # Apply deep semantic search across all products
        overall_results = deep_semantic_search(query, df, 10)
        st.session_state['classified_results'] = classified_results
        st.session_state['overall_results'] = overall_results
        st.session_state['last_query'] = query

# Display results
if 'classified_results' in st.session_state and query:
    t = time.time()
    classified_results = st.session_state['classified_results']
    overall_results = st.session_state['overall_results']

    col1, col2 = st.columns([7, 3])

    with col1:
        st.header("Product-Specific Results")
        for product, data in classified_results.items():
            st.subheader(f"{product} ({(data['probability']*100):.1f}% match)")
            for i, (_, result) in enumerate(data['results'].iterrows(), 1):
                display_result_card(result, f"{product}_{i}")

    with col2:
        st.header("Overall Top Matches")
        for i, (_, result) in enumerate(overall_results.iterrows(), 1):
            display_result_card(result, f"overall_{i}")
    print("\n\n\nTotal Time: ",(time.time()-t)*1000, ' ms\n')



# Print physical memory usage
def print_memory_usage(label=""):
    rss = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[{label}] Memory RSS (physical): {rss:.2f} MB")

    # Peak from tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    print(f"[{label}] Tracemalloc Current: {current / (1024**2):.2f} MB; Peak: {peak / (1024**2):.2f} MB")

print_memory_usage("After loading models and data")

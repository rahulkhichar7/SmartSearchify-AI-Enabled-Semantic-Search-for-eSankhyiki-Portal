import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import ast

# Set page config must be first command
st.set_page_config(layout="wide", page_title="Data Catalog Search")

# Load data and models with proper embedding conversion
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('your_data.csv')  # Update with your actual data source
    
    # Convert string representations of lists to actual numpy arrays
    def convert_embedding(embed_str):
        try:
            if isinstance(embed_str, str):
                return np.array(ast.literal_eval(embed_str), dtype=np.float32)
            return embed_str
        except:
            return np.zeros(384, dtype=np.float32)  # Default embedding if conversion fails
    
    df['search_emb'] = df['search_emb'].apply(convert_embedding)
    df['title_emb'] = df['title_emb'].apply(convert_embedding)
    
    classifier = load('LinearSVC_classifier.joblib')
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return df, classifier, embedder

df, classifier, embedder = load_data_and_models()

# Unique products
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

# Semantic search function with proper embedding handling
def semantic_search(query_embedding, product_df, n_results):
    doc_embeddings = np.stack(product_df['search_emb'].values)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    return product_df.iloc[top_indices]

# Overall search function
def overall_search(query_embedding, df, n_results=10):
    doc_embeddings = np.stack(df['search_emb'].values)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-n_results:][::-1]
    return df.iloc[top_indices]

# Display result card (used for both product-specific and overall results)
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

# Initialize session state
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = ""

# Main app interface
st.title("Data Catalog Search")
query = st.text_input("Enter your search query:", key="search_query", 
                     value=st.session_state.get('last_query', ''))

if query and (query != st.session_state.get('last_query', '') or 
             'classified_results' not in st.session_state):
    with st.spinner("Searching..."):
        # Classify and get product-specific results
        query_embedding = embedder.encode([query])
        top_products = classify_query(query, embedder, classifier)
        
        classified_results = {}
        n_results = [3, 2, 1]  # Results for 1st, 2nd, 3rd product
        
        for i, (product, prob) in enumerate(top_products):
            product_df = df[df['Product'] == product]
            num_rows = len(product_df)
            
            if num_rows <= 50:
                results = semantic_search(query_embedding, product_df, n_results[i])
            else:
                doc_embeddings = np.stack(product_df['search_emb'].values)
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                top50_indices = np.argsort(similarities)[-50:][::-1]
                top50_df = product_df.iloc[top50_indices]
                results = semantic_search(query_embedding, top50_df, n_results[i])
            
            classified_results[product] = {
                'probability': prob,
                'results': results
            }
        
        # Get overall results
        overall_results = overall_search(query_embedding, df, 10)
        
        # Store in session state
        st.session_state['classified_results'] = classified_results
        st.session_state['overall_results'] = overall_results
        st.session_state['last_query'] = query

if 'classified_results' in st.session_state and query:
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
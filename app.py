import streamlit as st
import numpy as np
from services.data_loader import load_data_and_models
from services.search_engine import classify_query, deep_semantic_search
from services.utils import display_result_card, init_memory_tracking, print_memory_usage
from config import N_RESULTS, TOP_K_OVERALL

# Initialize memory tracking
process = init_memory_tracking()
st.set_page_config(layout="wide", page_title="Data Catalog Search")

# Load data and models
df, classifier, embedder, cross_encoder = load_data_and_models()

# Session state management
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'classified_results' not in st.session_state:
    st.session_state.classified_results = None
if 'overall_results' not in st.session_state:
    st.session_state.overall_results = None

# UI Components
st.title("Data Catalog Search")
query = st.text_input(
    "Enter your search query:",
    key="search_query",
    value=st.session_state.last_query
)

# Search execution
if query and (query != st.session_state.last_query or not st.session_state.classified_results):
    with st.spinner("Searching..."):
        # Classify query into products
        top_products = classify_query(query, embedder, classifier)
        
        # Product-specific searches
        classified_results = {}
        for i, (product, prob) in enumerate(top_products):
            product_df = df[df['Product'] == product]
            top_results = deep_semantic_search(
                query, product_df, embedder, cross_encoder, N_RESULTS[i]
            )
            classified_results[product] = {
                'probability': prob,
                'results': top_results
            }
        
        # Overall search
        overall_results = deep_semantic_search(
            query, df, embedder, cross_encoder, TOP_K_OVERALL
        )
        
        # Update session state
        st.session_state.classified_results = classified_results
        st.session_state.overall_results = overall_results
        st.session_state.last_query = query

# Display results
if st.session_state.classified_results and query:
    classified_results = st.session_state.classified_results
    overall_results = st.session_state.overall_results

    col1, col2 = st.columns([7, 3])

    with col1:
        st.header("Product-Specific Results")
        for product, data in classified_results.items():
            st.subheader(f"{product} ({(data['probability']*100):.1f}% match)")
            for _, result in data['results'].iterrows():
                display_result_card(result)

    with col2:
        st.header("Overall Top Matches")
        for _, result in overall_results.iterrows():
            display_result_card(result)

# Memory usage logging
print_memory_usage(process, "After search execution")

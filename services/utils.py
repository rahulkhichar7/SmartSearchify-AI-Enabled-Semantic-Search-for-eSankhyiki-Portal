import time
import psutil
import os
import tracemalloc
import streamlit as st

def init_memory_tracking():
    """Initialize memory tracking"""
    tracemalloc.start()
    return psutil.Process(os.getpid())

def print_memory_usage(process, label=""):
    """Log memory usage statistics"""
    rss = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[{label}] Memory RSS (physical): {rss:.2f} MB")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"[{label}] Tracemalloc Current: {current / (1024**2):.2f} MB; Peak: {peak / (1024**2):.2f} MB")

def display_result_card(result):
    """Render result card in Streamlit UI"""
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

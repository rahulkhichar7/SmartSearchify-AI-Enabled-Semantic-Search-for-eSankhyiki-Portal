import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer, CrossEncoder
from joblib import load
import streamlit as st
from config import DATA_FILE, MODEL_FILE, EMBEDDER_MODEL, CROSS_ENCODER_MODEL

@st.cache_resource
def load_data_and_models():
    """Load dataset and ML models with caching"""
    df = pd.read_csv(DATA_FILE)
    
    def convert_embedding(embed_str):
        try:
            if isinstance(embed_str, str):
                return np.array(ast.literal_eval(embed_str), dtype=np.float32)
            return embed_str
        except:
            return np.zeros(384, dtype=np.float32)
    
    df['search_emb'] = df['search_emb'].apply(convert_embedding)
    df['title_emb'] = df['title_emb'].apply(convert_embedding)
    
    classifier = load(MODEL_FILE)
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    
    return df, classifier, embedder, cross_encoder

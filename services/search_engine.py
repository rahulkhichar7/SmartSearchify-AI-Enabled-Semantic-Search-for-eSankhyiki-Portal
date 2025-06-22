import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import PRODUCTS, N_RESULTS, TOP_K_OVERALL, TOP_K_CANDIDATES

def classify_query(query, embedder, classifier):
    """Classify query into product categories"""
    query_embedding = embedder.encode([query])
    decision_scores = classifier.decision_function(query_embedding)
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    probabilities = probabilities[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_products = np.array(PRODUCTS)[top3_indices]
    top3_probs = probabilities[top3_indices]
    return list(zip(top3_products, top3_probs))

def deep_semantic_search(query, df_slice, embedder, cross_encoder, top_k):
    """Two-stage semantic search (cosine + cross-encoder)"""
    # First stage: cosine similarity
    query_embedding = embedder.encode([query])
    doc_embeddings = np.stack(df_slice['search_emb'].values)
    cosine_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_candidates = np.argsort(cosine_scores)[-TOP_K_CANDIDATES:][::-1]
    candidate_df = df_slice.iloc[top_candidates]

    # Second stage: cross-encoder re-ranking
    query_doc_pairs = [(query, text) for text in candidate_df['search_text'].tolist()]
    cross_scores = cross_encoder.predict(query_doc_pairs)
    top_indices = np.argsort(cross_scores)[-top_k:][::-1]
    return candidate_df.iloc[top_indices]

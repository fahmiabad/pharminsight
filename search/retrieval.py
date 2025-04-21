"""
Search and retrieval functionality for PharmInsight.
"""
import faiss
import numpy as np
from .embeddings import get_embedding
from .indexing import load_search_index
from ..config import SIMILARITY_THRESHOLD, K_RETRIEVE

def search_documents(query, k=K_RETRIEVE, threshold=SIMILARITY_THRESHOLD):
    """
    Search for relevant documents.
    
    Args:
        query (str): The search query
        k (int): Number of results to retrieve
        threshold (float): Minimum similarity score (0-1)
        
    Returns:
        list: Relevant document chunks with scores
    """
    # Load index and metadata
    index, metadata = load_search_index()
    
    if index.ntotal == 0 or not metadata:
        return []
    
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    k = min(k, len(metadata))  # Don't request more results than we have
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata) and dist >= threshold:
            doc = metadata[idx].copy()
            doc["score"] = float(dist)
            results.append(doc)
    
    return results

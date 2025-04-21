"""
Vector index management for PharmInsight.
"""
import os
import faiss
import numpy as np
import pickle
import streamlit as st
import sqlite3
from config import INDEX_PATH, DOCS_METADATA_PATH, DB_PATH

def rebuild_index_from_db():
    """
    Rebuild the search index from database records
    
    Returns:
        tuple: (success, count)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all active document chunks
    c.execute('''
    SELECT c.chunk_id, c.text, c.embedding, d.filename, d.category, d.doc_id
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE d.is_active = 1
    ''')
    
    results = c.fetchall()
    conn.close()
    
    all_embeddings = []
    metadata = []
    
    for chunk_id, text, embedding_bytes, filename, category, doc_id in results:
        # Deserialize embedding
        if embedding_bytes:
            embedding = pickle.loads(embedding_bytes)
            all_embeddings.append(embedding)
            
            # Create metadata entry
            meta = {
                "chunk_id": chunk_id,
                "text": text,
                "source": filename,
                "category": category,
                "doc_id": doc_id
            }
            metadata.append(meta)
    
    # Create and save FAISS index
    if all_embeddings:
        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        dimension = all_embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(all_embeddings)
        index.add(all_embeddings)
        
        # Save index and metadata
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
            
        return True, len(metadata)
    
    return False, 0

def load_search_index():
    """
    Load the search index and metadata
    
    Returns:
        tuple: (index, metadata)
    """
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_METADATA_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(DOCS_METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            return index, metadata
        else:
            # Try to rebuild the index
            success, _ = rebuild_index_from_db()
            if success:
                index = faiss.read_index(INDEX_PATH)
                with open(DOCS_METADATA_PATH, "rb") as f:
                    metadata = pickle.load(f)
                return index, metadata
    except Exception as e:
        st.error(f"Error loading search index: {e}")
        
    # Return empty index and metadata as fallback
    dimension = 1536  # Default for OpenAI embeddings
    empty_index = faiss.IndexFlatIP(dimension)
    return empty_index, []

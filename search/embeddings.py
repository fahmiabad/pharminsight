"""
Embedding generation for PharmInsight.
"""
import streamlit as st
import numpy as np
from openai import OpenAI
import os
from config import DEFAULT_EMBEDDING_MODEL, get_openai_api_key

def get_openai_client():
    """
    Get an OpenAI client with proper error handling
    
    Returns:
        OpenAI: Client instance
    """
    try:
        api_key = get_openai_api_key()
        if not api_key:
            st.error("❌ OPENAI_API_KEY is missing. Please add it in your environment or Streamlit Secrets.")
            st.stop()
            
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        st.stop()

def get_embedding(text, model=DEFAULT_EMBEDDING_MODEL):
    """
    Get embedding vector for text using OpenAI API with proper error handling.
    
    Args:
        text (str): Text to get embedding for
        model (str): Embedding model to use
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    client = get_openai_client()
    
    try:
        # Clean and prepare text
        text = text.replace("\n", " ").strip()
        
        # Handle empty text
        if not text:
            # Default dimensions for different models
            dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            dim = dimensions.get(model, 1536)
            return np.zeros(dim, dtype=np.float32)
            
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
        
    except Exception as e:
        st.error(f"Error creating embedding: {e}")
        # Try fallback to older model if the specified one fails
        if model != "text-embedding-ada-002":
            st.warning(f"Falling back to text-embedding-ada-002 model")
            return get_embedding(text, "text-embedding-ada-002")
        raise

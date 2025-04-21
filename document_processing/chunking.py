"""
Text chunking utilities for PharmInsight.
"""
import streamlit as st

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks with semantic boundaries.
    
    Args:
        text (str): The document text to chunk
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text or len(text.strip()) == 0:
        st.warning("Empty text provided for chunking")
        return []
        
    if len(text) < chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Find the end of the chunk
        end = min(start + chunk_size, text_len)
        
        # If we're at the end of the text, just take what's left
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Try to find a good breaking point - prioritize paragraph breaks, then sentences
        # Look for semantic boundaries
        last_paragraph = text.rfind('\n\n', start + chunk_size // 2, end)
        last_period = text.rfind('. ', start + chunk_size // 2, end)
        last_newline = text.rfind('\n', start + chunk_size // 2, end)
        
        # Choose the best break point
        if last_paragraph > start:
            break_point = last_paragraph + 2  # Include the double newline
        elif last_period > start:
            break_point = last_period + 1  # Include the period
        elif last_newline > start:
            break_point = last_newline + 1  # Include the newline
        else:
            # No good boundary found, break at chunk_size
            break_point = end
            
        # Add the chunk
        chunk = text[start:break_point].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Calculate the next starting point with overlap
        start = break_point - overlap
        if start < 0:
            start = 0
            
    return chunks

"""
Document management functionality for PharmInsight.
"""
import sqlite3
import uuid
import datetime
import pickle
import pandas as pd
import streamlit as st
from config import DB_PATH
from utils.logging import log_action
from document_processing.extraction import extract_text_from_file
from document_processing.chunking import chunk_text
from search.embeddings import get_embedding
from search.indexing import rebuild_index_from_db

def process_document(file, metadata=None):
    """
    Process a document file and store in database.
    
    Args:
        file: Uploaded file object
        metadata (dict): Additional metadata about the document
        
    Returns:
        tuple: (success, message)
    """
    if not metadata:
        metadata = {}
        
    try:
        # Extract text from file
        text, extraction_success, extraction_message = extract_text_from_file(file)
        
        if not extraction_success:
            return False, extraction_message
            
        if not text or not text.strip():
            return False, "No text content could be extracted from the file"
        
        # Create database connection
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Store document info
        doc_id = str(uuid.uuid4())
        
        c.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                file.name,
                datetime.datetime.now().isoformat(),
                st.session_state["username"],
                metadata.get("category", "Uncategorized"),
                metadata.get("description", ""),
                metadata.get("expiry_date", ""),
                1  # is_active
            )
        )
        
        # Verify document insertion
        c.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if not c.fetchone():
            conn.rollback()
            conn.close()
            return False, "Failed to insert document record"
        
        # Chunk the text and create embeddings
        chunks = chunk_text(
            text, 
            chunk_size=metadata.get("chunk_size", 1000), 
            overlap=metadata.get("chunk_overlap", 200)
        )
        
        # Process chunks
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            chunk_id = f"{doc_id}_{i+1}"
            
            try:
                embedding = get_embedding(
                    chunk_text, 
                    model=metadata.get("embedding_model", "text-embedding-3-small")
                )
                
                # Serialize embedding
                embedding_bytes = pickle.dumps(embedding)
                
                # Store chunk
                c.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (chunk_id, doc_id, chunk_text, embedding_bytes)
                )
            except Exception as e:
                conn.rollback()
                conn.close()
                return False, f"Error processing chunk {i+1}: {str(e)}"
        
        # Verify chunks were inserted
        c.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
        chunk_count = c.fetchone()[0]
        
        if chunk_count == 0:
            conn.rollback()
            conn.close()
            return False, "No chunks were successfully processed and stored"
        
        conn.commit()
        conn.close()
        
        # Log action
        log_action(
            st.session_state["username"],
            "upload_document",
            f"Uploaded document: {file.name}, ID: {doc_id}, Chunks: {chunk_count}"
        )
        
        return True, f"Document processed successfully with {chunk_count} chunks"
        
    except Exception as e:
        return False, f"Error processing document: {str(e)}"

def list_documents():
    """
    Get list of all documents
    
    Returns:
        pandas.DataFrame: Documents data
    """
    conn = sqlite3.connect(DB_PATH)
    
    docs_df = pd.read_sql_query(
        """
        SELECT d.doc_id, d.filename, d.category, d.upload_date, d.is_active,
               u.username as uploader, COUNT(c.chunk_id) as chunks
        FROM documents d
        JOIN users u ON d.uploader = u.username
        LEFT JOIN chunks c ON d.doc_id = c.doc_id
        GROUP BY d.doc_id
        ORDER BY d.upload_date DESC
        """,
        conn
    )
    
    conn.close()
    
    return docs_df

def get_document_details(doc_id):
    """
    Get detailed information about a document
    
    Args:
        doc_id (str): Document ID
        
    Returns:
        tuple: (doc_info, chunks_df) - Document info and chunks data
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Get document info
    doc_df = pd.read_sql_query(
        """
        SELECT d.*, u.username as uploader_name
        FROM documents d
        JOIN users u ON d.uploader = u.username
        WHERE d.doc_id = ?
        """,
        conn,
        params=(doc_id,)
    )
    
    if doc_df.empty:
        conn.close()
        return None, None
    
    # Get chunks
    chunks_df = pd.read_sql_query(
        "SELECT chunk_id, text FROM chunks WHERE doc_id = ?",
        conn,
        params=(doc_id,)
    )
    
    conn.close()
    
    return doc_df.iloc[0].to_dict(), chunks_df

def toggle_document_status(doc_id, active_status):
    """
    Enable or disable a document
    
    Args:
        doc_id (str): Document ID
        active_status (bool): New active status
        
    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute(
        "UPDATE documents SET is_active = ? WHERE doc_id = ?",
        (1 if active_status else 0, doc_id)
    )
    
    conn.commit()
    conn.close()
    
    status_str = "activated" if active_status else "deactivated"
    log_action(
        st.session_state["username"],
        f"document_{status_str}",
        f"Document {doc_id} was {status_str}"
    )
    
    # Rebuild index after toggling document status
    rebuild_index_from_db()
    
    return True

def delete_document(doc_id):
    """
    Delete a document and all its chunks
    
    Args:
        doc_id (str): Document ID
        
    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get filename first for logging
    c.execute("SELECT filename FROM documents WHERE doc_id = ?", (doc_id,))
    result = c.fetchone()
    filename = result[0] if result else "unknown"
    
    # Delete chunks first (foreign key constraint)
    c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    
    # Delete the document
    c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    
    conn.commit()
    conn.close()
    
    log_action(
        st.session_state["username"],
        "delete_document",
        f"Deleted document: {filename}, ID: {doc_id}"
    )
    
    # Rebuild index after deletion
    rebuild_index_from_db()
    
    return True

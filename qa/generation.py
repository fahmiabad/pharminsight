"""
Answer generation functionality for PharmInsight.
"""
import sqlite3
import uuid
import datetime
import json
import streamlit as st
from openai import OpenAI
from config import DB_PATH, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE
from search.embeddings import get_openai_client
from search.retrieval import search_documents
from qa.history import record_search_history

def generate_answer(query, model=DEFAULT_LLM_MODEL, include_explanation=True, temp=DEFAULT_TEMPERATURE):
    """
    Generate an answer to a query based on retrieved documents.
    
    Args:
        query (str): The user query
        model (str): LLM model to use
        include_explanation (bool): Whether to include detailed explanation
        temp (float): Temperature for generation
        
    Returns:
        dict: Answer data including sources and metadata
    """
    # Record search in history
    record_search_history(query)
    
    # Search for relevant documents
    k_retrieve = st.session_state.get("k_retrieve", 5)
    similarity_threshold = st.session_state.get("similarity_threshold", 0.75)
    results = search_documents(query, k=k_retrieve, threshold=similarity_threshold)
    
    # Initialize response structure
    answer_data = {
        "question_id": str(uuid.uuid4()),
        "query": query,
        "answer": "",
        "sources": results,
        "model": model,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    client = get_openai_client()
    
    try:
        if results:
            # Prepare context from retrieved chunks
            max_context_length = 4000
            context_pieces = []
            current_length = 0
            
            for result in results:
                # Add source information to each chunk
                chunk = f"Source: {result['source']} (Category: {result.get('category', 'Uncategorized')})\n\n{result['text']}"
                chunk_length = len(chunk)
                
                # Check if adding this chunk would exceed max context length
                if current_length + chunk_length > max_context_length:
                    # If this is the first chunk, add it even if it exceeds length (truncated)
                    if not context_pieces:
                        context_pieces.append(chunk[:max_context_length])
                    break
                
                context_pieces.append(chunk)
                current_length += chunk_length
                
            context = "\n\n---\n\n".join(context_pieces)
            
            # Build prompt based on whether explanation is requested
            if include_explanation:
                prompt = f"""
You are a clinical expert assistant for pharmacists. Answer the following question based ONLY on the provided document excerpts. 
If the document excerpts don't contain the information needed to answer the question, 
say "I don't have enough information about this in the provided documents."

**Document Excerpts:**
{context}

**Question:**
{query}

Provide your answer in this format:

**Answer:**
[A direct and concise answer to the question]

**Explanation:**
[A detailed explanation with specific information from the documents]

**Sources:**
[List the sources of information used]
"""
            else:
                prompt = f"""
You are a clinical expert assistant for pharmacists. Answer the following question based ONLY on the provided document excerpts.
If the document excerpts don't contain the information needed to answer the question, 
say "I don't have enough information about this in the provided documents."

**Document Excerpts:**
{context}

**Question:**
{query}

Provide your answer in this format:

**Answer:**
[A direct and concise answer to the question]
"""
            answer_source = "Retrieved Documents"
        else:
            # No relevant documents found
            prompt = f"""
You are a clinical expert assistant for pharmacists. No relevant document was found for the following question:

Question: {query}

Please respond with:
"I don't have specific information about this in the provided documents. Please consider consulting official clinical guidelines or pharmacist resources for accurate information."
"""
            answer_source = "No Relevant Documents"

        # Generate the answer using the LLM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert clinical assistant for pharmacists."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Update answer data
        answer_data["answer"] = answer
        answer_data["source_type"] = answer_source
        
        # Store the Q&A pair in database
        store_qa_pair(answer_data)
        
        return answer_data
            
    except Exception as e:
        error_message = f"Error generating answer: {str(e)}"
        answer_data["answer"] = error_message
        answer_data["source_type"] = "Error"
        return answer_data

def store_qa_pair(answer_data):
    """
    Store a question-answer pair in the database
    
    Args:
        answer_data (dict): Answer data including question, answer, and sources
        
    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Store the QA pair
    c.execute(
        "INSERT INTO qa_pairs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            answer_data["question_id"],
            st.session_state["username"],
            answer_data["query"],
            answer_data["answer"],
            answer_data["timestamp"],
            json.dumps([s.get("source", "") for s in answer_data["sources"]]),
            answer_data["model"]
        )
    )
    
    conn.commit()
    conn.close()
    
    return True

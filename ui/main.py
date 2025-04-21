"""
Main search page UI for PharmInsight.
"""
import streamlit as st
from ..qa.generation import generate_answer
from ..qa.feedback import feedback_ui
from .common import display_source_documents
from ..search.indexing import load_search_index

def main_page():
    """Render the main search page"""
    st.title("PharmInsight for Pharmacists")
    st.markdown("Ask questions related to clinical guidelines, drug information, or policies.")
    
    # Check if we have documents
    index, metadata = load_search_index()
    has_documents = index.ntotal > 0 and metadata
    
    if not has_documents:
        st.warning("⚠️ No documents are currently available in the system. Please contact an administrator.")
        # Add debug button to check document database
        if st.button("Debug: Check Document Database"):
            from ..database.setup import get_db_connection
            conn = get_db_connection()
            c = conn.cursor()
            
            # Check documents table
            c.execute("SELECT COUNT(*) FROM documents")
            doc_count = c.fetchone()[0]
            
            # Check chunks table
            c.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = c.fetchone()[0]
            
            st.info(f"Documents in database: {doc_count}, Chunks in database: {chunk_count}")
            
            if doc_count > 0 and chunk_count == 0:
                st.error("Documents exist but no chunks were created. Check the document processing logic.")
            elif doc_count == 0:
                st.error("No documents in the database. Try uploading some documents first.")
            
            conn.close()
        
    # Search interface
    query = st.text_input("Ask a clinical question", placeholder="e.g., What is the recommended monitoring plan for amiodarone?")
    
    # Recent searches suggestions
    if "user_history" in st.session_state and st.session_state["user_history"]:
        recent_searches = [h["query"] for h in st.session_state["user_history"][-3:]]
        if recent_searches:
            st.caption("Recent searches:")
            cols = st.columns(len(recent_searches))
            for i, col in enumerate(cols):
                with col:
                    if st.button(recent_searches[i], key=f"recent_{i}"):
                        query = recent_searches[i]
                        # Force re-run with this query
                        st.session_state["current_query"] = query
                        st.experimental_rerun()
    
    # Check for a query from session state (from clicking a recent search)
    if "current_query" in st.session_state:
        query = st.session_state["current_query"]
        del st.session_state["current_query"]
    
    # Submit button
    search_col, clear_col = st.columns([5, 1])
    with search_col:
        search_clicked = st.button("Search", type="primary", disabled=not query, use_container_width=True)
    with clear_col:
        clear_clicked = st.button("Clear", use_container_width=True)
        if clear_clicked:
            query = ""
            
    # Process search
    if search_clicked and query:
        with st.spinner("Searching documents and generating answer..."):
            # Get model settings from sidebar
            llm_model = st.session_state.get("llm_model", "gpt-3.5-turbo")
            include_explanation = st.session_state.get("include_explanation", True)
            temperature = st.session_state.get("temperature", 0.2)
            
            # Generate answer
            result = generate_answer(
                query, 
                model=llm_model, 
                include_explanation=include_explanation,
                temp=temperature
            )
        
        # Display answer
        if result:
            if result["source_type"] == "Retrieved Documents":
                st.success("Answer based on the provided documents:")
            elif result["source_type"] == "No Relevant Documents":
                st.warning("No matching information found in the documents.")
            else:
                st.error("Error generating answer.")
                
            st.markdown(result["answer"])
            
            # Add feedback UI for this answer with 1-3 rating
            feedback_ui(result["question_id"])
            
            # Display sources if available
            display_source_documents(result["sources"])

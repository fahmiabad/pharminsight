"""
Document management admin page UI for PharmInsight.
"""
import streamlit as st
from ...auth.auth import is_admin
from ...document_processing.management import (
    list_documents,
    get_document_details,
    toggle_document_status,
    delete_document,
    process_document
)
from ...search.indexing import rebuild_index_from_db

def admin_documents_page():
    """Admin document management page"""
    if not is_admin():
        st.warning("Admin access required")
        return
        
    st.title("Document Management")
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["📄 Documents List", "⬆️ Upload Documents", "🔄 Rebuild Index"])
    
    # Tab 1: Documents List
    with tab1:
        st.subheader("All Documents")
        
        # Get all documents
        docs_df = list_documents()
        
        if docs_df.empty:
            st.info("No documents in the system yet.")
        else:
            # Format the status column
            docs_df['status'] = docs_df['is_active'].apply(lambda x: "✅ Active" if x else "❌ Inactive")
            
            # Display as a dataframe with action buttons
            st.dataframe(
                docs_df[['filename', 'category', 'upload_date', 'uploader', 'chunks', 'status']],
                column_config={
                    "upload_date": st.column_config.DatetimeColumn(
                        "Upload Date",
                        format="YYYY-MM-DD HH:mm"
                    )
                }
            )
            
            # Document details and actions
            st.subheader("Document Details")
            selected_doc_id = st.selectbox(
                "Select a document",
                docs_df['doc_id'].tolist(),
                format_func=lambda x: docs_df.loc[docs_df['doc_id'] == x, 'filename'].iloc[0]
            )
            
            if selected_doc_id:
                doc_info, chunks_df = get_document_details(selected_doc_id)
                
                if doc_info:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Filename:** {doc_info['filename']}")
                        st.write(f"**Category:** {doc_info['category']}")
                        st.write(f"**Upload Date:** {doc_info['upload_date'][:10]}")
                        
                    with col2:
                        st.write(f"**Uploaded by:** {doc_info['uploader_name']}")
                        st.write(f"**Status:** {'Active' if doc_info['is_active'] else 'Inactive'}")
                        st.write(f"**Chunks:** {len(chunks_df) if chunks_df is not None else 0}")
                        
                    with col3:

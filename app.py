import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
Document management admin page UI for PharmInsight.
"""
import streamlit as st
from auth.auth import is_admin
from document_processing.management import (
    list_documents,
    get_document_details,
    toggle_document_status,
    delete_document,
    process_document
)
from search.indexing import rebuild_index_from_db

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
                        is_active = bool(doc_info['is_active'])
                        if is_active:
                            if st.button("Deactivate Document"):
                                if toggle_document_status(selected_doc_id, False):
                                    st.success("Document deactivated")
                                    st.experimental_rerun()
                        else:
                            if st.button("Activate Document"):
                                if toggle_document_status(selected_doc_id, True):
                                    st.success("Document activated")
                                    st.experimental_rerun()
                                    
                        if st.button("Delete Document", type="primary"):
                            confirm = st.checkbox("Confirm deletion?")
                            if confirm and delete_document(selected_doc_id):
                                st.success("Document deleted successfully")
                                st.experimental_rerun()
                    
                    # Show document chunks
                    if chunks_df is not None and not chunks_df.empty:
                        with st.expander("View Document Chunks"):
                            for i, row in chunks_df.iterrows():
                                st.markdown(f"**Chunk {i+1}**")
                                st.text_area(
                                    f"Text for chunk {i+1}",
                                    value=row['text'],
                                    height=150,
                                    key=f"chunk_{row['chunk_id']}",
                                    disabled=True
                                )
                                st.divider()
    
    # Tab 2: Upload Documents
    with tab2:
        st.subheader("Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or text files)", 
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )
        
        if uploaded_files:
            # Meta-information for uploads
            with st.expander("Document Information", expanded=True):
                category = st.selectbox(
                    "Document Category",
                    ["Clinical Guidelines", "Drug Information", "Policy", "Protocol", "Research", "Educational", "Other"],
                    index=0
                )
                
                description = st.text_area("Description (optional)")
                expiry_date = st.date_input("Expiry Date (if applicable)", value=None)
                
            # Processing options
            with st.expander("Processing Options", expanded=True):
                chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000, 100)
                chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, 200, 50)
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                    index=0
                )
                st.info("A smaller chunk size and larger overlap may improve search quality but increases processing time and storage requirements.")
            
            # Debug option for document processing
            debug_mode = st.checkbox("Enable debug mode", value=True)
            
            # Process files
            if st.button("Process and Upload", type="primary"):
                metadata = {
                    "category": category,
                    "description": description,
                    "expiry_date": expiry_date.isoformat() if expiry_date else None,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_model": embedding_model
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                    progress_bar.progress((i) / len(uploaded_files))
                    
                    if debug_mode:
                        st.write(f"Processing file: {file.name}")
                        st.write(f"File size: {file.size} bytes")
                        st.write(f"File type: {file.type}")
                        
                        # Extract a sample of the file content
                        if file.name.lower().endswith(".pdf"):
                            with st.expander("File Preview"):
                                st.write("PDF file detected. Attempting text extraction...")
                                from ...document_processing.extraction import extract_text_from_pdf
                                sample_text = extract_text_from_pdf(file)
                                file.seek(0)  # Reset file pointer
                                preview = sample_text[:500] + "..." if len(sample_text) > 500 else sample_text
                                st.text(preview)
                    
                    success, message = process_document(file, metadata)
                    
                    if success:
                        st.success(f"✅ {file.name}: {message}")
                    else:
                        st.error(f"❌ {file.name}: {message}")
                        
                progress_bar.progress(1.0)
                status_text.text("Processing complete")
                
                # Rebuild index
                with st.spinner("Rebuilding search index..."):
                    success, count = rebuild_index_from_db()
                    if success:
                        st.success(f"Search index rebuilt with {count} document chunks")
                    else:
                        st.error("Failed to rebuild search index")
                        
    # Tab 3: Rebuild Index
    with tab3:
        st.subheader("Rebuild Search Index")
        st.markdown("""
        Rebuilding the search index will update the vector database used for document retrieval.
        This is useful if documents were added, modified, or removed directly in the database.
        """)
        
        if st.button("Rebuild Index", type="primary"):
            with st.spinner("Rebuilding search index..."):
                success, count = rebuild_index_from_db()
                if success:
                    st.success(f"Search index rebuilt successfully with {count} document chunks")
                else:
                    st.error("Failed to rebuild search index")

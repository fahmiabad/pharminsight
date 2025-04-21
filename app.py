"""
PharmInsight - A clinical knowledge base for pharmacists.

This is the main entry point for the PharmInsight application.
"""
import streamlit as st
import sys
import os
import sqlite3
import hashlib
import datetime
import uuid
import pandas as pd

# Add the current directory to Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
APP_VERSION = "2.1.0"
DB_PATH = "pharminsight.db"

# Database initialization function
def init_database():
    """Initialize all database tables"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()
        
        # Users table
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')
        
        # Documents table
        c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP,
            uploader TEXT NOT NULL,
            category TEXT,
            description TEXT,
            expiry_date TEXT,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # Document chunks table
        c.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        ''')
        
        # Questions and answers table
        c.execute('''
        CREATE TABLE IF NOT EXISTS qa_pairs (
            question_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TIMESTAMP,
            sources TEXT,
            model_used TEXT
        )
        ''')
        
        # Feedback table with 3-point rating scale
        c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            question_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 3),
            comment TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES qa_pairs(question_id)
        )
        ''')
        
        # Audit log table
        c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP
        )
        ''')
        
        # User search history
        c.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            history_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            timestamp TIMESTAMP,
            num_results INTEGER
        )
        ''')
        
        # Create default admin if it doesn't exist
        c.execute("SELECT username FROM users WHERE username = 'admin'")
        if not c.fetchone():
            # Create with default password 'adminpass'
            password_hash = hashlib.sha256(f"adminpass:salt_key".encode()).hexdigest()
            c.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("admin", password_hash, "admin", "admin@example.com", "System Administrator", 
                 datetime.datetime.now().isoformat(), None)
            )
            
            # Create regular user account
            user_hash = hashlib.sha256(f"password:salt_key".encode()).hexdigest()
            c.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("user", user_hash, "user", "user@example.com", "Regular User", 
                 datetime.datetime.now().isoformat(), None)
            )
            
            # Log the creation
            log_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            c.execute(
                "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)",
                (log_id, "system", "create_user", "Created default admin and regular user", timestamp)
            )
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return False

def log_action(user_id, action, details=None):
    """Log an action to the audit log"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()
        
        log_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        c.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)",
            (log_id, user_id, action, details, timestamp)
        )
        
        conn.commit()
        conn.close()
        return log_id
    except Exception as e:
        st.error(f"Logging error: {str(e)}")
        return None

def verify_password(username, password):
    """Verify username and password"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()
        
        c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result:
            stored_hash, role = result
            calculated_hash = hashlib.sha256(f"{password}:salt_key".encode()).hexdigest()
            
            if calculated_hash == stored_hash:
                # Update last login time
                c.execute(
                    "UPDATE users SET last_login = ? WHERE username = ?",
                    (datetime.datetime.now().isoformat(), username)
                )
                conn.commit()
                
                # Log successful login
                log_action(username, "login", "Successful login")
                
                conn.close()
                return True, role
        
        # Log failed login attempt
        if username:
            log_action("system", "failed_login", f"Failed login attempt for user: {username}")
            
        conn.close()
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        
    if "username" not in st.session_state:
        st.session_state["username"] = None
        
    if "role" not in st.session_state:
        st.session_state["role"] = None
        
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
        
    if "user_history" not in st.session_state:
        st.session_state["user_history"] = []
        
    # Search settings
    if "k_retrieve" not in st.session_state:
        st.session_state["k_retrieve"] = 5
        
    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = 0.75
        
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "gpt-3.5-turbo"
        
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.2
        
    if "include_explanation" not in st.session_state:
        st.session_state["include_explanation"] = True

def is_admin():
    """Check if current user is an admin"""
    return st.session_state.get("role") == "admin"

def render_sidebar():
    """Render the application sidebar"""
    with st.sidebar:
        st.image("https://www.svgrepo.com/show/530588/chemical-laboratory.svg", width=50)
        st.title("PharmInsight")
        st.caption(f"v{APP_VERSION}")
        
        # Show user info
        if st.session_state["authenticated"]:
            st.write(f"Logged in as: **{st.session_state['username']}** ({st.session_state['role']})")
            if st.button("Logout"):
                log_action(st.session_state["username"], "logout", "User logged out")
                st.session_state["authenticated"] = False
                st.session_state["username"] = None
                st.session_state["role"] = None
                st.session_state["page"] = "main"
                st.rerun()
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            
            if st.button("📚 Home", use_container_width=True):
                st.session_state["page"] = "main"
                st.rerun()
                
            if st.button("👤 Your Profile", use_container_width=True):
                st.session_state["page"] = "profile"
                st.rerun()
                
            if st.button("📜 Search History", use_container_width=True):
                st.session_state["page"] = "history"
                st.rerun()
                
            # Admin-only options
            if is_admin():
                st.divider()
                st.subheader("Admin")
                
                if st.button("📄 Document Management", use_container_width=True):
                    st.session_state["page"] = "admin_docs"
                    st.rerun()
                    
                if st.button("👥 User Management", use_container_width=True):
                    st.session_state["page"] = "admin_users"
                    st.rerun()
                    
                if st.button("📊 Analytics", use_container_width=True):
                    st.session_state["page"] = "admin_analytics"
                    st.rerun()
            
            # Settings
            st.divider()
            st.subheader("Settings")
            
            with st.expander("Search Settings"):
                st.slider("Results to retrieve", 3, 10, 
                        st.session_state.get("k_retrieve", 5), 1, 
                        key="k_retrieve")
                
                st.slider("Similarity threshold", 0.0, 1.0, 
                        st.session_state.get("similarity_threshold", 0.75), 0.05, 
                        key="similarity_threshold")
                
            with st.expander("Model Settings"):
                st.selectbox(
                    "LLM Model",
                    ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
                    index=0,
                    key="llm_model"
                )
                st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature")
                st.checkbox("Include detailed explanations", value=True, key="include_explanation")

def login_form():
    """Display login form"""
    st.header("Welcome to PharmInsight")
    st.markdown("Please login to continue")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            authenticated, role = verify_password(username, password)
            if authenticated:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = role
                st.success(f"Login successful! Welcome back, {username}.")
                
                # Clear any previous page selection
                st.session_state["page"] = "main"
                
                # Force page refresh
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.markdown("""
        ### PharmInsight
        A clinical knowledge base for pharmacists.
        
        * Ask questions about drugs and clinical guidelines
        * Get evidence-based answers with citations
        * Access information from your organization's documents
        
        Demo login:
        - Regular user: user/password
        - Admin: admin/adminpass
        """)

def main_page():
    """Render the main search page"""
    st.title("PharmInsight for Pharmacists")
    st.markdown("Ask questions related to clinical guidelines, drug information, or policies.")
    
    # Simple check if database is initialized
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM documents")
        doc_count = c.fetchone()[0]
        conn.close()
        
        if doc_count == 0:
            st.warning("⚠️ No documents are currently available in the system. Please contact an administrator.")
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        
    # Search interface
    query = st.text_input("Ask a clinical question", placeholder="e.g., What is the recommended monitoring plan for amiodarone?")
    
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
            # This would normally call your search and answer generation functions
            # For now, just display a placeholder response
            st.success("Answer based on the provided documents:")
            st.markdown("""
            ## Example Answer
            
            This is a placeholder answer since the full search functionality is not implemented in this simplified version.
            
            In a fully functional system, this would show:
            
            1. The answer to your query based on the documents
            2. Relevant sources and citations
            3. A feedback mechanism to rate the answer
            
            To implement the complete functionality, you'll need to add the search, embedding, and answer generation modules.
            """)

def profile_page():
    """Render a simple profile page"""
    st.title("Your Profile")
    st.write(f"Username: {st.session_state['username']}")
    st.write(f"Role: {st.session_state['role']}")
    
    # This would show more profile information and settings in the full implementation
    st.info("This is a simplified profile page. The full implementation would include more details and settings.")

def history_page():
    """Render a simple history page"""
    st.title("Search History")
    st.info("This is a simplified history page. The full implementation would show your search history.")

def admin_docs_page():
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
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
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
                
                if not docs_df.empty:
                    selected_doc_id = st.selectbox(
                        "Select a document",
                        docs_df['doc_id'].tolist(),
                        format_func=lambda x: docs_df.loc[docs_df['doc_id'] == x, 'filename'].iloc[0]
                    )
                    
                    if selected_doc_id:
                        # Get document details
                        conn = sqlite3.connect(DB_PATH, timeout=20.0)
                        doc_df = pd.read_sql_query(
                            """
                            SELECT d.*, u.username as uploader_name
                            FROM documents d
                            JOIN users u ON d.uploader = u.username
                            WHERE d.doc_id = ?
                            """,
                            conn,
                            params=(selected_doc_id,)
                        )
                        
                        # Get chunks
                        chunks_df = pd.read_sql_query(
                            "SELECT chunk_id, text FROM chunks WHERE doc_id = ?",
                            conn,
                            params=(selected_doc_id,)
                        )
                        
                        conn.close()
                        
                        if not doc_df.empty:
                            doc_info = doc_df.iloc[0].to_dict()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Filename:** {doc_info['filename']}")
                                st.write(f"**Category:** {doc_info['category']}")
                                st.write(f"**Upload Date:** {doc_info['upload_date'][:10]}")
                                
                            with col2:
                                st.write(f"**Uploaded by:** {doc_info['uploader_name']}")
                                st.write(f"**Status:** {'Active' if doc_info['is_active'] else 'Inactive'}")
                                st.write(f"**Chunks:** {len(chunks_df) if not chunks_df.empty else 0}")
                                
                            with col3:
                                is_active = bool(doc_info['is_active'])
                                if is_active:
                                    if st.button("Deactivate Document"):
                                        try:
                                            conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                            c = conn.cursor()
                                            c.execute(
                                                "UPDATE documents SET is_active = ? WHERE doc_id = ?",
                                                (0, selected_doc_id)
                                            )
                                            conn.commit()
                                            conn.close()
                                            
                                            status_str = "deactivated"
                                            log_action(
                                                st.session_state["username"],
                                                f"document_{status_str}",
                                                f"Document {selected_doc_id} was {status_str}"
                                            )
                                            
                                            st.success("Document deactivated")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deactivating document: {str(e)}")
                                else:
                                    if st.button("Activate Document"):
                                        try:
                                            conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                            c = conn.cursor()
                                            c.execute(
                                                "UPDATE documents SET is_active = ? WHERE doc_id = ?",
                                                (1, selected_doc_id)
                                            )
                                            conn.commit()
                                            conn.close()
                                            
                                            status_str = "activated"
                                            log_action(
                                                st.session_state["username"],
                                                f"document_{status_str}",
                                                f"Document {selected_doc_id} was {status_str}"
                                            )
                                            
                                            st.success("Document activated")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error activating document: {str(e)}")
                                            
                                if st.button("Delete Document", type="primary"):
                                    confirm = st.checkbox("Confirm deletion?")
                                    if confirm:
                                        try:
                                            conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                            c = conn.cursor()
                                            
                                            # Get filename first for logging
                                            c.execute("SELECT filename FROM documents WHERE doc_id = ?", (selected_doc_id,))
                                            result = c.fetchone()
                                            filename = result[0] if result else "unknown"
                                            
                                            # Delete chunks first (foreign key constraint)
                                            c.execute("DELETE FROM chunks WHERE doc_id = ?", (selected_doc_id,))
                                            
                                            # Delete the document
                                            c.execute("DELETE FROM documents WHERE doc_id = ?", (selected_doc_id,))
                                            
                                            conn.commit()
                                            conn.close()
                                            
                                            log_action(
                                                st.session_state["username"],
                                                "delete_document",
                                                f"Deleted document: {filename}, ID: {selected_doc_id}"
                                            )
                                            
                                            st.success("Document deleted successfully")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting document: {str(e)}")
                            
                            # Show document chunks
                            if not chunks_df.empty:
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
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    
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
                    "chunk_overlap": chunk_overlap
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
                                
                                try:
                                    from io import BytesIO
                                    from PyPDF2 import PdfReader
                                    
                                    reader = PdfReader(file)
                                    sample_text = ""
                                    
                                    # Extract first page as sample
                                    if len(reader.pages) > 0:
                                        sample_text = reader.pages[0].extract_text()
                                    
                                    file.seek(0)  # Reset file pointer
                                    preview = sample_text[:500] + "..." if len(sample_text) > 500 else sample_text
                                    st.text(preview)
                                except Exception as e:
                                    st.error(f"Error previewing PDF: {str(e)}")
                    
                    try:
                        # Extract text from file
                        text = ""
                        if file.name.lower().endswith(".pdf"):
                            try:
                                from io import BytesIO
                                from PyPDF2 import PdfReader
                                
                                reader = PdfReader(file)
                                text = ""
                                for page in reader.pages:
                                    text += page.extract_text() + "\n\n"
                                file.seek(0)
                            except Exception as e:
                                st.error(f"Error extracting PDF: {str(e)}")
                                continue
                        elif file.name.lower().endswith((".txt", ".csv", ".md")):
                            try:
                                text = file.read().decode("utf-8")
                                file.seek(0)
                            except UnicodeDecodeError:
                                # Try other encodings
                                file.seek(0)
                                for encoding in ["latin-1", "windows-1252", "iso-8859-1"]:
                                    try:
                                        text = file.read().decode(encoding)
                                        file.seek(0)
                                        break
                                    except:
                                        file.seek(0)
                                        continue
                        else:
                            st.error(f"Unsupported file type: {file.name}")
                            continue
                        
                        if not text.strip():
                            st.error(f"No content extracted from {file.name}")
                            continue
                        
                        # Chunk the text
                        chunks = []
                        if len(text) < chunk_size:
                            chunks = [text]
                        else:
                            start = 0
                            text_len = len(text)
                            
                            while start < text_len:
                                end = min(start + chunk_size, text_len)
                                
                                if end >= text_len:
                                    chunks.append(text[start:])
                                    break
                                    
                                # Find semantic boundaries
                                last_paragraph = text.rfind('\n\n', start + chunk_size // 2, end)
                                last_period = text.rfind('. ', start + chunk_size // 2, end)
                                last_newline = text.rfind('\n', start + chunk_size // 2, end)
                                
                                # Choose the best break point
                                if last_paragraph > start:
                                    break_point = last_paragraph + 2
                                elif last_period > start:
                                    break_point = last_period + 1
                                elif last_newline > start:
                                    break_point = last_newline + 1
                                else:
                                    break_point = end
                                    
                                chunk = text[start:break_point].strip()
                                if chunk:
                                    chunks.append(chunk)
                                
                                start = break_point - chunk_overlap
                                if start < 0:
                                    start = 0
                        
                        # Store document in database
                        conn = sqlite3.connect(DB_PATH, timeout=20.0)
                        c = conn.cursor()
                        
                        # Create document record
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
                        
                        # Store chunks
                        for i, chunk_text in enumerate(chunks):
                            if not chunk_text.strip():
                                continue
                                
                            chunk_id = f"{doc_id}_{i+1}"
                            
                            # For now, we're not generating embeddings
                            c.execute(
                                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                (chunk_id, doc_id, chunk_text, None)
                            )
                        
                        conn.commit()
                        conn.close()
                        
                        log_action(
                            st.session_state["username"],
                            "upload_document",
                            f"Uploaded document: {file.name}, ID: {doc_id}, Chunks: {len(chunks)}"
                        )
                        
                        st.success(f"✅ {file.name}: Processed successfully with {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"❌ {file.name}: Error processing document: {str(e)}")
                        
                progress_bar.progress(1.0)
                status_text.text("Processing complete")
                
                # Rebuild index placeholder
                with st.spinner("Rebuilding search index..."):
                    st.info("Search index rebuilding would happen here in the full implementation")
                    
    # Tab 3: Rebuild Index
    with tab3:
        st.subheader("Rebuild Search Index")
        st.markdown("""
        Rebuilding the search index will update the vector database used for document retrieval.
        This is useful if documents were added, modified, or removed directly in the database.
        """)
        
        if st.button("Rebuild Index", type="primary"):
            with st.spinner("Rebuilding search index..."):
                try:
                    # Just count chunks for now
                    conn = sqlite3.connect(DB_PATH, timeout=20.0)
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM chunks")
                    count = c.fetchone()[0]
                    conn.close()
                    
                    st.success(f"Search index would be rebuilt with {count} document chunks")
                    st.info("In the full implementation, this would generate embeddings and build a vector index")
                except Exception as e:
                    st.error(f"Error accessing database: {str(e)}")

def admin_users_page():
    """Render a simple user management page"""
    if not is_admin():
        st.warning("Admin access required")
        return
    
    st.title("User Management")
    
    # Show basic user list
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        users_df = pd.read_sql_query("SELECT username, role, email, full_name, created_at, last_login FROM users", conn)
        conn.close()
        
        st.subheader("User Accounts")
        st.dataframe(users_df)
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
    
    st.info("This is a simplified user management page. The full implementation would allow creating and managing users.")

def admin_analytics_page():
    """Render a simple analytics page"""
    if not is_admin():
        st.warning("Admin access required")
        return
    
    st.title("Analytics Dashboard")
    st.info("This is a simplified analytics page. The full implementation would show usage statistics and visualization.")

# Initialize the database
init_database()

# Initialize session state variables
initialize_session_state()

# Display sidebar if authenticated
if st.session_state["authenticated"]:
    render_sidebar()

# Page routing based on authentication state
if not st.session_state["authenticated"]:
    # Show login form
    login_form()
else:
    # Route to the appropriate page based on session state
    page = st.session_state.get("page", "main")
    
    if page == "main":
        main_page()
    elif page == "profile":
        profile_page()
    elif page == "history":
        history_page()
    elif page == "admin_docs":
        admin_docs_page()
    elif page == "admin_users":
        admin_users_page()
    elif page == "admin_analytics":
        admin_analytics_page()
    else:
        st.warning(f"Page '{page}' not found")
        st.session_state["page"] = "main"
        st.rerun()

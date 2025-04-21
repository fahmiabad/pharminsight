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
    """Render the user profile page"""
    st.header("Your Profile")
    
    try:
        # Get user data
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()
        
        c.execute("""
        SELECT username, role, email, full_name, created_at, last_login
        FROM users WHERE username = ?
        """, (st.session_state["username"],))
        
        user_data = c.fetchone()
        
        if not user_data:
            st.error("User data not found")
            return
            
        username, role, email, full_name, created_at, last_login = user_data
        
        # Get user activity statistics
        questions_df = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM qa_pairs WHERE user_id = ?",
            conn, params=(username,)
        )
        question_count = questions_df['count'].iloc[0]
        
        feedback_df = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM feedback WHERE user_id = ?", 
            conn, params=(username,)
        )
        feedback_count = feedback_df['count'].iloc[0]
        
        recent_searches = pd.read_sql_query(
            """
            SELECT query, timestamp 
            FROM search_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
            LIMIT 5
            """, 
            conn, params=(username,)
        )
        
        conn.close()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Account Information")
            st.write(f"**Username:** {username}")
            st.write(f"**Role:** {role}")
            st.write(f"**Email:** {email or 'Not set'}")
            st.write(f"**Full Name:** {full_name or 'Not set'}")
            st.write(f"**Account Created:** {created_at[:10] if created_at else 'Unknown'}")
            st.write(f"**Last Login:** {last_login[:10] if last_login else 'First login'}")
            
            st.divider()
            st.subheader("Update Profile")
            
            new_email = st.text_input("Email", value=email or "")
            new_name = st.text_input("Full Name", value=full_name or "")
            
            if st.button("Update Profile"):
                try:
                    conn = sqlite3.connect(DB_PATH, timeout=20.0)
                    c = conn.cursor()
                    
                    c.execute(
                        "UPDATE users SET email = ?, full_name = ? WHERE username = ?",
                        (new_email, new_name, username)
                    )
                    conn.commit()
                    conn.close()
                    
                    log_action(username, "update_profile", "Updated user profile")
                    st.success("Profile updated successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating profile: {str(e)}")
                
            st.divider()
            st.subheader("Change Password")
            
            current_pwd = st.text_input("Current Password", type="password")
            new_pwd = st.text_input("New Password", type="password")
            confirm_pwd = st.text_input("Confirm New Password", type="password")
            
            if st.button("Change Password"):
                if not current_pwd or not new_pwd or not confirm_pwd:
                    st.error("All password fields are required")
                    return
                    
                if new_pwd != confirm_pwd:
                    st.error("New passwords don't match")
                    return
                    
                # Verify current password
                calculated_hash = hashlib.sha256(f"{current_pwd}:salt_key".encode()).hexdigest()
                
                try:
                    conn = sqlite3.connect(DB_PATH, timeout=20.0)
                    c = conn.cursor()
                    
                    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
                    stored_hash = c.fetchone()[0]
                    
                    if stored_hash != calculated_hash:
                        st.error("Current password is incorrect")
                        conn.close()
                        return
                    
                    # Update password
                    new_hash = hashlib.sha256(f"{new_pwd}:salt_key".encode()).hexdigest()
                    c.execute(
                        "UPDATE users SET password_hash = ? WHERE username = ?",
                        (new_hash, username)
                    )
                    conn.commit()
                    conn.close()
                    
                    log_action(username, "change_password", "User changed their password")
                    st.success("Password changed successfully")
                except Exception as e:
                    st.error(f"Error changing password: {str(e)}")
        
        with col2:
            st.subheader("Your Activity")
            
            st.metric("Questions Asked", question_count)
            st.metric("Feedback Given", feedback_count)
            
            st.subheader("Recent Searches")
            if not recent_searches.empty:
                for _, row in recent_searches.iterrows():
                    formatted_time = row['timestamp'][:10] if row['timestamp'] else ""
                    st.write(f"• {row['query']} ({formatted_time})")
            else:
                st.write("No recent searches")
                
    except Exception as e:
        st.error(f"Error loading profile: {str(e)}")

def history_page():
    """Render search history page"""
    st.title("Your Search History")
    
    try:
        # Get search history
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        
        history_df = pd.read_sql_query(
            """
            SELECT query, timestamp, num_results
            FROM search_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """,
            conn,
            params=(st.session_state["username"],)
        )
        
        # Get answered questions
        qa_df = pd.read_sql_query(
            """
            SELECT question_id, query, timestamp
            FROM qa_pairs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """,
            conn,
            params=(st.session_state["username"],)
        )
        
        conn.close()
        
        # Display history
        tab1, tab2 = st.tabs(["🔍 Search History", "❓ Questions Asked"])
        
        with tab1:
            if history_df.empty:
                st.info("You haven't made any searches yet.")
            else:
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.write(f"Showing your last {len(history_df)} searches")
                
                with col2:
                    if st.button("Clear History"):
                        try:
                            conn = sqlite3.connect(DB_PATH, timeout=20.0)
                            c = conn.cursor()
                            
                            c.execute(
                                "DELETE FROM search_history WHERE user_id = ?",
                                (st.session_state["username"],)
                            )
                            
                            conn.commit()
                            conn.close()
                            
                            log_action(
                                st.session_state["username"],
                                "clear_history",
                                "User cleared search history"
                            )
                            
                            st.session_state["user_history"] = []
                            st.success("Search history cleared")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing history: {str(e)}")
                
                # Display history items
                for i, row in history_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{row['query']}**")
                            # Format timestamp for display
                            timestamp = row['timestamp']
                            if timestamp:
                                formatted_time = timestamp[:19].replace('T', ' ') if 'T' in timestamp else timestamp[:16]
                                st.caption(f"{formatted_time}")
                            # Show if search had results
                            if 'num_results' in row and row['num_results'] > 0:
                                st.caption(f"Found {row['num_results']} results")
                            else:
                                st.caption("No results found")
                        with col2:
                            if st.button("Search Again", key=f"again_{i}"):
                                st.session_state["current_query"] = row['query']
                                st.session_state["page"] = "main"
                                st.rerun()
                        st.divider()
        
        with tab2:
            if qa_df.empty:
                st.info("You haven't asked any questions yet.")
            else:
                st.write(f"Showing your last {len(qa_df)} questions")
                
                # Display questions
                for i, row in qa_df.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{row['query']}**")
                            # Format timestamp for display
                            timestamp = row['timestamp']
                            if timestamp:
                                formatted_time = timestamp[:19].replace('T', ' ') if 'T' in timestamp else timestamp[:16]
                                st.caption(f"{formatted_time}")
                        with col2:
                            if st.button("View Answer", key=f"view_{i}"):
                                # This would normally link to the answer, but for now just search again
                                st.session_state["current_query"] = row['query']
                                st.session_state["page"] = "main"
                                st.rerun()
                        st.divider()
                
    except Exception as e:
        st.error(f"Error loading search history: {str(e)}")

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
    """Admin user management page"""
    if not is_admin():
        st.warning("Admin access required")
        return
        
    st.title("User Management")
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["👥 Users List", "➕ Create User", "📊 User Activity"])
    
    try:
        # Tab 1: Users List
        with tab1:
            st.subheader("All Users")
            
            # Get all users
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
            users_df = pd.read_sql_query(
                """
                SELECT username, role, email, full_name, created_at, last_login 
                FROM users
                ORDER BY username
                """, 
                conn
            )
            
            if users_df.empty:
                st.info("No users found in the system.")
            else:
                # Format dates
                if 'created_at' in users_df.columns:
                    users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                if 'last_login' in users_df.columns:
                    users_df['last_login'] = pd.to_datetime(users_df['last_login']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Display as a dataframe
                st.dataframe(
                    users_df,
                    column_config={
                        "username": st.column_config.TextColumn("Username"),
                        "role": st.column_config.TextColumn("Role"),
                        "email": st.column_config.TextColumn("Email"),
                        "full_name": st.column_config.TextColumn("Full Name"),
                        "created_at": st.column_config.TextColumn("Created"),
                        "last_login": st.column_config.TextColumn("Last Login")
                    },
                    use_container_width=True
                )
                
                # User details and actions
                st.subheader("User Details")
                selected_username = st.selectbox(
                    "Select a user",
                    users_df['username'].tolist()
                )
                
                if selected_username:
                    # Get user details
                    user_row = users_df[users_df['username'] == selected_username].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Username:** {user_row['username']}")
                        st.write(f"**Role:** {user_row['role']}")
                        st.write(f"**Email:** {user_row['email'] or 'Not set'}")
                        
                    with col2:
                        st.write(f"**Full Name:** {user_row['full_name'] or 'Not set'}")
                        st.write(f"**Created:** {user_row['created_at']}")
                        st.write(f"**Last Login:** {user_row['last_login'] or 'Never'}")
                    
                    # User actions
                    st.subheader("User Actions")
                    
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("Reset Password", disabled=(selected_username == st.session_state["username"])):
                            st.session_state["show_reset_password"] = True
                    
                    with action_col2:
                        if st.button("Change Role", disabled=(selected_username == st.session_state["username"])):
                            st.session_state["show_change_role"] = True
                            
                    with action_col3:
                        if st.button("Delete User", disabled=(selected_username == st.session_state["username"])):
                            st.session_state["show_delete_user"] = True
                    
                    # Reset password form
                    if st.session_state.get("show_reset_password", False):
                        st.subheader("Reset Password")
                        new_password = st.text_input("New Password", type="password")
                        confirm_password = st.text_input("Confirm Password", type="password")
                        
                        if st.button("Submit Password Reset"):
                            if not new_password or not confirm_password:
                                st.error("Please enter both password fields")
                            elif new_password != confirm_password:
                                st.error("Passwords don't match")
                            else:
                                try:
                                    # Hash new password
                                    password_hash = hashlib.sha256(f"{new_password}:salt_key".encode()).hexdigest()
                                    
                                    # Update in database
                                    conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                    c = conn.cursor()
                                    c.execute(
                                        "UPDATE users SET password_hash = ? WHERE username = ?",
                                        (password_hash, selected_username)
                                    )
                                    conn.commit()
                                    conn.close()
                                    
                                    log_action(
                                        st.session_state["username"],
                                        "reset_password",
                                        f"Password reset for user: {selected_username}"
                                    )
                                    
                                    st.success(f"Password for {selected_username} has been reset")
                                    st.session_state["show_reset_password"] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error resetting password: {str(e)}")
                    
                    # Change role form
                    if st.session_state.get("show_change_role", False):
                        st.subheader("Change User Role")
                        new_role = st.selectbox(
                            "New Role",
                            ["user", "admin"],
                            index=0 if user_row['role'] == "user" else 1
                        )
                        
                        if st.button("Submit Role Change"):
                            try:
                                # Update role in database
                                conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                c = conn.cursor()
                                c.execute(
                                    "UPDATE users SET role = ? WHERE username = ?",
                                    (new_role, selected_username)
                                )
                                conn.commit()
                                conn.close()
                                
                                log_action(
                                    st.session_state["username"],
                                    "change_role",
                                    f"Changed role for user {selected_username} from {user_row['role']} to {new_role}"
                                )
                                
                                st.success(f"Role for {selected_username} changed to {new_role}")
                                st.session_state["show_change_role"] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error changing role: {str(e)}")
                    
                    # Delete user form
                    if st.session_state.get("show_delete_user", False):
                        st.subheader("Delete User")
                        st.warning(f"Are you sure you want to delete user '{selected_username}'? This action cannot be undone.")
                        
                        confirm = st.checkbox("I understand that this will permanently delete this user and all associated data")
                        
                        if st.button("Confirm Delete") and confirm:
                            try:
                                conn = sqlite3.connect(DB_PATH, timeout=20.0)
                                c = conn.cursor()
                                
                                # Delete user's data from related tables
                                c.execute("DELETE FROM search_history WHERE user_id = ?", (selected_username,))
                                c.execute("DELETE FROM feedback WHERE user_id = ?", (selected_username,))
                                c.execute("DELETE FROM qa_pairs WHERE user_id = ?", (selected_username,))
                                
                                # Finally delete the user
                                c.execute("DELETE FROM users WHERE username = ?", (selected_username,))
                                
                                conn.commit()
                                conn.close()
                                
                                log_action(
                                    st.session_state["username"],
                                    "delete_user",
                                    f"Deleted user: {selected_username}"
                                )
                                
                                st.success(f"User '{selected_username}' has been deleted")
                                st.session_state["show_delete_user"] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting user: {str(e)}")
                    
                    # Get user activity
                    st.subheader("User Activity")
                    
                    user_activity = pd.read_sql_query(
                        """
                        SELECT action, timestamp, details 
                        FROM audit_log
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 10
                        """,
                        conn,
                        params=(selected_username,)
                    )
                    
                    if not user_activity.empty:
                        user_activity['timestamp'] = pd.to_datetime(user_activity['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                        
                        st.dataframe(
                            user_activity,
                            column_config={
                                "timestamp": st.column_config.TextColumn("Time"),
                                "action": st.column_config.TextColumn("Action"),
                                "details": st.column_config.TextColumn("Details")
                            },
                            use_container_width=True
                        )
                    else:
                        st.info("No activity records found for this user")
        
        # Tab 2: Create User
        with tab2:
            st.subheader("Create New User")
            
            # User creation form
            with st.form("create_user_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                role = st.selectbox("Role", ["user", "admin"], index=0)
                email = st.text_input("Email (optional)")
                full_name = st.text_input("Full Name (optional)")
                
                submitted = st.form_submit_button("Create User")
                
                if submitted:
                    if not username or not password or not confirm_password:
                        st.error("Username and password are required")
                    elif password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        try:
                            # Check if username already exists
                            conn = sqlite3.connect(DB_PATH, timeout=20.0)
                            c = conn.cursor()
                            
                            c.execute("SELECT username FROM users WHERE username = ?", (username,))
                            if c.fetchone():
                                st.error("Username already exists")
                                conn.close()
                            else:
                                # Create password hash
                                password_hash = hashlib.sha256(f"{password}:salt_key".encode()).hexdigest()
                                
                                # Create user
                                c.execute(
                                    "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (username, password_hash, role, email, full_name, 
                                    datetime.datetime.now().isoformat(), None)
                                )
                                
                                conn.commit()
                                conn.close()
                                
                                log_action(
                                    st.session_state["username"],
                                    "create_user",
                                    f"Created user: {username} with role: {role}"
                                )
                                
                                st.success(f"User '{username}' created successfully")
                        except Exception as e:
                            st.error(f"Error creating user: {str(e)}")
        
        # Tab 3: User Activity
        with tab3:
            st.subheader("User Activity Overview")
            
            # Time period selection
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 days", "Last 30 days", "All time"],
                index=1
            )
            
            # Convert time period to SQL date filter
            date_filters = {
                "Last 7 days": "date('now', '-7 days')",
                "Last 30 days": "date('now', '-30 days')",
                "All time": "date('1970-01-01')"  # Unix epoch
            }
            
            date_filter = date_filters[time_period]
            
            # Most active users
            active_users = pd.read_sql_query(
                f"""
                SELECT 
                    user_id,
                    COUNT(*) as action_count
                FROM audit_log
                WHERE date(timestamp) > {date_filter}
                GROUP BY user_id
                ORDER BY action_count DESC
                LIMIT 10
                """,
                conn
            )
            
            # User logins
            user_logins = pd.read_sql_query(
                f"""
                SELECT 
                    user_id,
                    COUNT(*) as login_count
                FROM audit_log
                WHERE action = 'login'
                AND date(timestamp) > {date_filter}
                GROUP BY user_id
                ORDER BY login_count DESC
                LIMIT 10
                """,
                conn
            )
            
            # Action distribution
            action_dist = pd.read_sql_query(
                f"""
                SELECT 
                    action,
                    COUNT(*) as count
                FROM audit_log
                WHERE date(timestamp) > {date_filter}
                GROUP BY action
                ORDER BY count DESC
                """,
                conn
            )
            
            # Display visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Most Active Users")
                if not active_users.empty:
                    st.bar_chart(active_users.set_index('user_id')['action_count'])
                else:
                    st.info("No user activity data for the selected period")
                    
            with col2:
                st.subheader("Most Frequent Logins")
                if not user_logins.empty:
                    st.bar_chart(user_logins.set_index('user_id')['login_count'])
                else:
                    st.info("No login data for the selected period")
            
            # Action distribution
            st.subheader("Action Distribution")
            if not action_dist.empty:
                st.bar_chart(action_dist.set_index('action')['count'])
            else:
                st.info("No action data for the selected period")
            
        conn.close()
    except Exception as e:
        st.error(f"Error in user management: {str(e)}")

def admin_analytics_page():
    """Admin analytics dashboard page"""
    if not is_admin():
        st.warning("Admin access required")
        return
        
    st.title("Analytics Dashboard")
    
    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Usage Statistics", 
        "💬 Feedback Analysis",
        "🔍 Search Patterns", 
        "📑 Document Analytics"
    ])
    
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        
        # Tab 1: Usage Statistics
        with tab1:
            st.subheader("System Usage")
            
            # Get counts from different tables
            query_counts = pd.read_sql_query(
                """
                SELECT 
                    (SELECT COUNT(*) FROM users) as users_count,
                    (SELECT COUNT(*) FROM documents) as documents_count,
                    (SELECT COUNT(*) FROM chunks) as chunks_count,
                    (SELECT COUNT(*) FROM qa_pairs) as qa_count,
                    (SELECT COUNT(*) FROM search_history) as searches_count,
                    (SELECT COUNT(*) FROM feedback) as feedback_count
                """,
                conn
            )
            
            # Get recent activity
            recent_activity = pd.read_sql_query(
                """
                SELECT action, user_id, timestamp, details
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT 50
                """,
                conn
            )
            
            # Get user activity over time
            user_activity = pd.read_sql_query(
                """
                SELECT 
                    date(timestamp) as date,
                    COUNT(*) as count,
                    action
                FROM audit_log
                GROUP BY date(timestamp), action
                ORDER BY date
                """,
                conn
            )
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Users", query_counts['users_count'].iloc[0])
                st.metric("Total Documents", query_counts['documents_count'].iloc[0])
                
            with col2:
                st.metric("Document Chunks", query_counts['chunks_count'].iloc[0])
                st.metric("Total Queries", query_counts['qa_count'].iloc[0])
                
            with col3:
                st.metric("Search Count", query_counts['searches_count'].iloc[0])
                st.metric("Feedback Collected", query_counts['feedback_count'].iloc[0])
            
            # Activity over time visualization
            if not user_activity.empty:
                st.subheader("Activity Over Time")
                # Group actions into categories for better visualization
                grouped_activity = user_activity.groupby('date')['count'].sum().reset_index()
                
                st.bar_chart(grouped_activity.set_index('date'))
                
            # Recent activity table
            st.subheader("Recent Activity")
            if not recent_activity.empty:
                # Format timestamp
                recent_activity['formatted_time'] = pd.to_datetime(recent_activity['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    recent_activity[['formatted_time', 'user_id', 'action', 'details']],
                    column_config={
                        "formatted_time": st.column_config.TextColumn("Time"),
                        "user_id": st.column_config.TextColumn("User"),
                        "action": st.column_config.TextColumn("Action"),
                        "details": st.column_config.TextColumn("Details")
                    },
                    use_container_width=True
                )
            else:
                st.info("No recent activity recorded")
        
        # Tab 2: Feedback Analysis
        with tab2:
            st.subheader("Feedback Analysis")
            
            # Get feedback stats
            stats_df = pd.read_sql_query(
                """
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(rating) as avg_rating
                FROM feedback
                """,
                conn
            )
            
            # Rating distribution
            rating_dist = pd.read_sql_query(
                """
                SELECT rating, COUNT(*) as count
                FROM feedback
                GROUP BY rating
                ORDER BY rating
                """,
                conn
            )
            
            # Recent feedback
            recent_feedback = pd.read_sql_query(
                """
                SELECT f.rating, f.comment, f.timestamp, q.query
                FROM feedback f
                JOIN qa_pairs q ON f.question_id = q.question_id
                ORDER BY f.timestamp DESC
                LIMIT 10
                """,
                conn
            )
            
            if not stats_df.empty:
                avg_rating = stats_df['avg_rating'].iloc[0]
                total_feedback = stats_df['total_feedback'].iloc[0]
                
                st.metric("Average Rating", f"{avg_rating:.2f}/3.00" if avg_rating is not None else "No ratings")
                st.metric("Total Feedback Collected", total_feedback)
                
                # Rating distribution visualization
                if not rating_dist.empty:
                    st.subheader("Rating Distribution")
                    
                    rating_labels = {
                        1: "Not Helpful",
                        2: "Somewhat Helpful",
                        3: "Very Helpful"
                    }
                    
                    # Add rating labels
                    rating_dist['rating_label'] = rating_dist['rating'].map(rating_labels)
                    
                    # Create a horizontal bar chart
                    st.bar_chart(rating_dist.set_index('rating_label')['count'])
                    
                # Recent feedback
                st.subheader("Recent Feedback")
                if not recent_feedback.empty:
                    for _, row in recent_feedback.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            rating_emoji = {
                                1: "❌ Not Helpful",
                                2: "⚠️ Somewhat Helpful",
                                3: "✅ Very Helpful"
                            }
                            
                            with col1:
                                st.write(f"**Query:** {row['query']}")
                                if row['comment']:
                                    st.write(f"**Comment:** {row['comment']}")
                            
                            with col2:
                                st.write(f"**Rating:** {rating_emoji.get(row['rating'], 'Unknown')}")
                                st.caption(f"{row['timestamp'][:10] if row['timestamp'] else ''}")
                            
                            st.divider()
                else:
                    st.info("No feedback collected yet")
            else:
                st.info("No feedback data available")
        
        # Tab 3: Search Patterns
        with tab3:
            st.subheader("Search Patterns")
            
            # Get popular searches
            popular_searches = pd.read_sql_query(
                """
                SELECT query, COUNT(*) as count
                FROM search_history
                GROUP BY query
                ORDER BY count DESC
                LIMIT 15
                """,
                conn
            )
            
            # Get search success rate - assuming num_results > 0 means successful search
            search_success = pd.read_sql_query(
                """
                SELECT 
                    date(timestamp) as date,
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN num_results > 0 THEN 1 ELSE 0 END) as successful_searches
                FROM search_history
                GROUP BY date(timestamp)
                ORDER BY date
                """,
                conn
            )
            
            # Query length analysis
            query_length = pd.read_sql_query(
                """
                SELECT 
                    length(query) as query_length,
                    COUNT(*) as count
                FROM search_history
                GROUP BY length(query)
                ORDER BY length(query)
                """,
                conn
            )
            
            # Popular searches
            st.subheader("Popular Searches")
            if not popular_searches.empty:
                # Create a horizontal bar chart for popular searches
                popular_sorted = popular_searches.sort_values(by='count')
                st.bar_chart(popular_sorted.set_index('query')['count'])
            else:
                st.info("No search data available")
                
            # Search success rate
            if not search_success.empty and not search_success.empty.all():
                st.subheader("Search Success Rate")
                
                # Calculate success rate
                search_success['success_rate'] = (
                    search_success['successful_searches'] / search_success['total_searches']
                ) * 100
                
                # Line chart for success rate
                st.line_chart(search_success.set_index('date')['success_rate'])
                
            # Query length distribution
            if not query_length.empty:
                st.subheader("Query Length Distribution")
                st.bar_chart(query_length.set_index('query_length')['count'])
        
        # Tab 4: Document Analytics
        with tab4:
            st.subheader("Document Analytics")
            
            # Document categories
            doc_categories = pd.read_sql_query(
                """
                SELECT 
                    category,
                    COUNT(*) as count
                FROM documents
                GROUP BY category
                ORDER BY count DESC
                """,
                conn
            )
            
            # Document upload timeline
            doc_timeline = pd.read_sql_query(
                """
                SELECT 
                    date(upload_date) as upload_date,
                    COUNT(*) as count
                FROM documents
                GROUP BY date(upload_date)
                ORDER BY upload_date
                """,
                conn
            )
            
            # Document status counts
            doc_status = pd.read_sql_query(
                """
                SELECT 
                    is_active,
                    COUNT(*) as count
                FROM documents
                GROUP BY is_active
                """,
                conn
            )
            
            # Document categories
            st.subheader("Document Categories")
            if not doc_categories.empty:
                st.bar_chart(doc_categories.set_index('category')['count'])
            else:
                st.info("No document category data available")
                
            # Document status
            st.subheader("Document Status")
            if not doc_status.empty:
                # Add status labels
                doc_status['status'] = doc_status['is_active'].map({
                    0: "Inactive",
                    1: "Active"
                })
                st.bar_chart(doc_status.set_index('status')['count'])
            else:
                st.info("No document status data available")
                
            # Document upload timeline
            if not doc_timeline.empty:
                st.subheader("Document Upload Timeline")
                st.line_chart(doc_timeline.set_index('upload_date')['count'])
            
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")

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

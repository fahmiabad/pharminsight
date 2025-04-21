"""
Common UI components for PharmInsight.
"""
import streamlit as st
from config import APP_VERSION
from auth.auth import is_admin, logout_user

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
                logout_user()
                st.experimental_rerun()
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            
            if st.button("📚 Home", use_container_width=True):
                st.session_state["page"] = "main"
                st.experimental_rerun()
                
            if st.button("👤 Your Profile", use_container_width=True):
                st.session_state["page"] = "profile"
                st.experimental_rerun()
                
            if st.button("📜 Search History", use_container_width=True):
                st.session_state["page"] = "history"
                st.experimental_rerun()
                
            # Admin-only options
            if is_admin():
                st.divider()
                st.subheader("Admin")
                
                if st.button("📄 Document Management", use_container_width=True):
                    st.session_state["page"] = "admin_docs"
                    st.experimental_rerun()
                    
                if st.button("👥 User Management", use_container_width=True):
                    st.session_state["page"] = "admin_users"
                    st.experimental_rerun()
                    
                if st.button("📊 Analytics", use_container_width=True):
                    st.session_state["page"] = "admin_analytics"
                    st.experimental_rerun()
            
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
                
            # This will be imported in app.py to avoid circular imports
            from ..auth.auth import login_user
            
            if login_user(username, password):
                st.success(f"Login successful! Welcome back, {username}.")
                
                # Clear any previous page selection
                st.session_state["page"] = "main"
                
                # Force page refresh
                st.experimental_rerun()
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

def display_source_documents(sources):
    """
    Display source documents with relevance scores
    
    Args:
        sources (list): List of source documents with scores
    """
    if sources:
        with st.expander("📄 Sources Used", expanded=True):
            for idx, doc in enumerate(sources):
                score = doc.get("score", 0)
                score_percent = int(score * 100) if score <= 1 else int(score)
                
                st.markdown(f"**Source {idx+1}: {doc['source']} (Relevance: {score_percent}%)**")
                
                # Display a snippet of the text
                text_snippet = doc["text"]
                if len(text_snippet) > 300:
                    text_snippet = text_snippet[:300] + "..."
                
                st.markdown(f"""
                <div style="padding: 10px; border: 1px solid #f0f0f0; border-radius: 5px; 
                          margin-bottom: 10px; background-color: #fafafa;">
                    {text_snippet}
                </div>
                """, unsafe_allow_html=True)

"""
Configuration settings for the PharmInsight application.
"""
import streamlit as st
import os

# ============================================
# Application Constants
# ============================================
APP_VERSION = "2.1.0"
APP_NAME = "PharmInsight"

# ============================================
# File Paths
# ============================================
DB_PATH = "pharminsight.db"
INDEX_PATH = "vector.index"
DOCS_METADATA_PATH = "docs_metadata.pkl"

# ============================================
# Search Settings
# ============================================
SIMILARITY_THRESHOLD = 0.75
K_RETRIEVE = 5

# ============================================
# OpenAI Settings
# ============================================
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.2

# ============================================
# Document Processing Settings
# ============================================
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# ============================================
# Initialize Session State
# ============================================
def initialize_session_state():
    """Initialize all session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        
    if "username" not in st.session_state:
        st.session_state["username"] = None
        
    if "role" not in st.session_state:
        st.session_state["role"] = None
        
    if "show_feedback_form" not in st.session_state:
        st.session_state["show_feedback_form"] = False
        
    if "current_question_id" not in st.session_state:
        st.session_state["current_question_id"] = None
        
    if "page" not in st.session_state:
        st.session_state["page"] = "main"
        
    if "user_history" not in st.session_state:
        st.session_state["user_history"] = []
        
    # Search settings
    if "k_retrieve" not in st.session_state:
        st.session_state["k_retrieve"] = K_RETRIEVE
        
    if "similarity_threshold" not in st.session_state:
        st.session_state["similarity_threshold"] = SIMILARITY_THRESHOLD
        
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = DEFAULT_LLM_MODEL
        
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = DEFAULT_TEMPERATURE
        
    if "include_explanation" not in st.session_state:
        st.session_state["include_explanation"] = True

def get_openai_api_key():
    """Get OpenAI API key from environment or secrets."""
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

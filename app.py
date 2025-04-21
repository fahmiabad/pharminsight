import streamlit as st

st.title("Testing PharmInsight")
st.write("This is a test to see if basic Streamlit functionality works")

# Try to import each module one at a time
try:
    from config import APP_VERSION
    st.success(f"Successfully imported config, version: {APP_VERSION}")
except Exception as e:
    st.error(f"Failed to import config: {str(e)}")

try:
    from database.setup import init_database
    st.success("Successfully imported database.setup")
    try:
        init_database()
        st.success("Successfully initialized database")
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
except Exception as e:
    st.error(f"Failed to import database.setup: {str(e)}")

# Test auth module
try:
    from auth.auth import is_admin
    st.success("Successfully imported auth.auth")
except Exception as e:
    st.error(f"Failed to import auth.auth: {str(e)}")

# Test UI modules
try:
    from ui.common import render_sidebar
    st.success("Successfully imported ui.common")
except Exception as e:
    st.error(f"Failed to import ui.common: {str(e)}")

try:
    from ui.main import main_page
    st.success("Successfully imported ui.main")
except Exception as e:
    st.error(f"Failed to import ui.main: {str(e)}")

# Test document processing
try:
    from document_processing.management import list_documents
    st.success("Successfully imported document_processing.management")
except Exception as e:
    st.error(f"Failed to import document_processing.management: {str(e)}")

# Test search
try:
    from search.indexing import load_search_index
    st.success("Successfully imported search.indexing")
except Exception as e:
    st.error(f"Failed to import search.indexing: {str(e)}")

# Initialize session state variables for testing
try:
    from config import initialize_session_state
    initialize_session_state()
    st.success("Successfully initialized session state")
except Exception as e:
    st.error(f"Failed to initialize session state: {str(e)}")

# Now try to render a simple UI element from one of your modules
st.header("Testing UI Rendering")
st.session_state["authenticated"] = True
st.session_state["username"] = "test_user"
st.session_state["role"] = "admin"

try:
    render_sidebar()
    st.success("Successfully rendered sidebar")
except Exception as e:
    st.error(f"Failed to render sidebar: {str(e)}")

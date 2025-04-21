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

# Continue with other imports...

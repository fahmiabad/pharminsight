"""
Search history page UI for PharmInsight.
"""
import streamlit as st
from ..qa.history import get_user_search_history, clear_search_history

def history_page():
    """Render search history page"""
    st.title("Your Search History")
    
    # Get search history
    history_df = get_user_search_history(limit=50)
    
    if history_df.empty:
        st.info("You haven't made any searches yet.")
    else:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.write(f"Showing your last {len(history_df)} searches")
        
        with col2:
            if st.button("Clear History"):
                if clear_search_history():
                    st.success("Search history cleared")
                    st.experimental_rerun()
        
        # Display history
        for i, row in history_df.iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['query']}**")
                    st.caption(f"{row['timestamp'][:19].replace('T', ' ')}")
                with col2:
                    if st.button("Search Again", key=f"again_{i}"):
                        st.session_state["current_query"] = row['query']
                        st.session_state["page"] = "main"
                        st.experimental_rerun()
                st.divider()

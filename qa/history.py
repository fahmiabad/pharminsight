"""
Search history tracking for PharmInsight.
"""
import sqlite3
import uuid
import datetime
import pandas as pd
import streamlit as st
from config import DB_PATH
from utils.logging import log_action

def record_search_history(query):
    """
    Record a search query in user history
    
    Args:
        query (str): The search query
        
    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    history_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    try:
        c.execute(
            "INSERT INTO search_history VALUES (?, ?, ?, ?, ?)",
            (
                history_id,
                st.session_state["username"],
                query,
                timestamp,
                0  # Will update this with results count
            )
        )
        
        conn.commit()
        conn.close()
        
        # Also store in session state for current session access
        if "user_history" in st.session_state:
            st.session_state["user_history"].append({
                "query": query,
                "timestamp": timestamp
            })
        
        return True
    except Exception as e:
        st.error(f"Error recording search history: {str(e)}")
        conn.close()
        return False

def get_user_search_history(limit=10):
    """
    Get the search history for current user
    
    Args:
        limit (int): Maximum number of history entries to return
        
    Returns:
        pandas.DataFrame: Search history data
    """
    conn = sqlite3.connect(DB_PATH)
    
    history_df = pd.read_sql_query(
        """
        SELECT query, timestamp, num_results
        FROM search_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        conn,
        params=(st.session_state["username"], limit)
    )
    
    conn.close()
    
    return history_df

def clear_search_history():
    """
    Clear the user's search history
    
    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute(
            "DELETE FROM search_history WHERE user_id = ?",
            (st.session_state["username"],)
        )
        
        conn.commit()
        conn.close()
        
        # Clear session history too
        st.session_state["user_history"] = []
        
        log_action(
            st.session_state["username"],
            "clear_history",
            "User cleared search history"
        )
        
        return True
    except Exception as e:
        st.error(f"Error clearing search history: {str(e)}")
        conn.close()
        return False

def get_popular_searches(limit=10):
    """
    Get the most popular search queries across all users
    
    Args:
        limit (int): Maximum number of results to return
        
    Returns:
        pandas.DataFrame: Popular search data
    """
    conn = sqlite3.connect(DB_PATH)
    
    popular_df = pd.read_sql_query(
        """
        SELECT query, COUNT(*) as count
        FROM search_history
        GROUP BY query
        ORDER BY count DESC
        LIMIT ?
        """,
        conn,
        params=(limit,)
    )
    
    conn.close()
    
    return popular_df

"""
Feedback collection and management for PharmInsight.
"""
import sqlite3
import uuid
import datetime
import pandas as pd
import streamlit as st
from config import DB_PATH
from utils.logging import log_action

def submit_feedback(question_id, rating, comment=None):
    """
    Submit feedback for an answer with simplified 1-3 rating scale.
    
    Args:
        question_id (str): ID of the question
        rating (int): Rating (1-3)
        comment (str, optional): Additional comments
        
    Returns:
        bool: Success status
    """
    # Validate rating is in the 1-3 range
    if rating < 1 or rating > 3:
        st.error("Rating must be between 1 and 3")
        return False
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    try:
        c.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (
                feedback_id,
                question_id,
                st.session_state["username"],
                rating,
                comment,
                timestamp
            )
        )
        
        conn.commit()
        
        log_action(
            st.session_state["username"],
            "submit_feedback",
            f"User submitted feedback for question {question_id} with rating {rating}"
        )
        
        conn.close()
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"Error submitting feedback: {str(e)}")
        return False

def get_feedback_stats():
    """
    Get overall feedback statistics for 1-3 rating scale
    
    Returns:
        tuple: (stats_df, rating_dist, recent) - Overall stats, rating distribution, and recent feedback
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Overall stats
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
    recent = pd.read_sql_query(
        """
        SELECT f.rating, f.comment, f.timestamp, q.query
        FROM feedback f
        JOIN qa_pairs q ON f.question_id = q.question_id
        ORDER BY f.timestamp DESC
        LIMIT 10
        """,
        conn
    )
    
    conn.close()
    
    return stats_df, rating_dist, recent

def feedback_ui(question_id):
    """
    Display feedback collection UI with 1-3 rating scale
    
    Args:
        question_id (str): The question ID to collect feedback for
    """
    st.divider()
    
    with st.expander("📊 Rate this answer", expanded=False):
        st.write("Your feedback helps us improve the system")
        
        col1, col2, col3 = st.columns(3)
        
        rating = None
        
        with col1:
            if st.button("1 - Not Helpful", key=f"rate_1_{question_id}"):
                rating = 1
        with col2:
            if st.button("2 - Somewhat Helpful", key=f"rate_2_{question_id}"):
                rating = 2
        with col3:
            if st.button("3 - Very Helpful", key=f"rate_3_{question_id}"):
                rating = 3
                
        if rating:
            st.session_state["show_detailed_feedback"] = True
            st.session_state["selected_rating"] = rating
            
        # Show comment field if a rating was selected
        if st.session_state.get("show_detailed_feedback", False):
            selected_rating = st.session_state.get("selected_rating", 2)
            
            # Show the selected rating
            if selected_rating == 1:
                st.write("You selected: Not Helpful")
            elif selected_rating == 2:
                st.write("You selected: Somewhat Helpful")
            else:
                st.write("You selected: Very Helpful")
                
            comments = st.text_area("Additional comments (optional)", key=f"comments_{question_id}")
            
            if st.button("Submit Feedback", key=f"submit_{question_id}"):
                success = submit_feedback(
                    question_id,
                    selected_rating,
                    comments
                )
                
                if success:
                    st.success("Thank you for your feedback!")
                    # Reset the feedback state
                    st.session_state["show_detailed_feedback"] = False
                    st.session_state["selected_rating"] = None
                else:
                    st.error("Error submitting feedback. Please try again.")

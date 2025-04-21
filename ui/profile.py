"""
User profile page UI for PharmInsight.
"""
import streamlit as st
import pandas as pd
from ..auth.auth import get_user_profile, update_user_profile, change_password
from ..config import DB_PATH

def profile_page():
    """Render the user profile page"""
    st.header("Your Profile")
    
    user_data = get_user_profile(st.session_state["username"])
    
    if not user_data:
        st.error("User data not found")
        return
        
    username = user_data["username"]
    role = user_data["role"]
    email = user_data["email"]
    full_name = user_data["full_name"]
    created_at = user_data["created_at"]
    last_login = user_data["last_login"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Account Information")
        st.write(f"**Username:** {username}")
        st.write(f"**Role:** {role}")
        st.write(f"**Email:** {email or 'Not set'}")
        st.write(f"**Full Name:** {full_name or 'Not set'}")
        st.write(f"**Account Created:** {created_at[:10]}")
        st.write(f"**Last Login:** {last_login[:10] if last_login else 'First login'}")
        
        st.divider()
        st.subheader("Update Profile")
        
        new_email = st.text_input("Email", value=email or "")
        new_name = st.text_input("Full Name", value=full_name or "")
        
        if st.button("Update Profile"):
            if update_user_profile(username, new_email, new_name):
                st.success("Profile updated successfully")
                st.experimental_rerun()
            else:
                st.error("Failed to update profile")
            
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
                
            success, message = change_password(username, current_pwd, new_pwd)
            if success:
                st.success("Password changed successfully")
            else:
                st.error(message)
    
    with col2:
        st.subheader("Your Activity")
        
        import sqlite3
        
        conn = sqlite3.connect(DB_PATH)
        
        # Get question count
        questions_df = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM qa_pairs WHERE user_id = ?",
            conn, params=(username,)
        )
        question_count = questions_df['count'].iloc[0]
        
        # Get feedback count
        feedback_df = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM feedback WHERE user_id = ?", 
            conn, params=(username,)
        )
        feedback_count = feedback_df['count'].iloc[0]
        
        # Recent searches
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
        
        st.metric("Questions Asked", question_count)
        st.metric("Feedback Given", feedback_count)
        
        st.subheader("Recent Searches")
        if not recent_searches.empty:
            for _, row in recent_searches.iterrows():
                st.write(f"• {row['query']} ({row['timestamp'][:10]})")
        else:
            st.write("No recent searches")

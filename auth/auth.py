"""
Authentication functionality for PharmInsight.
"""
import sqlite3
import hashlib
import hmac
import datetime
import uuid
import streamlit as st
from config import DB_PATH
from utils.logging import log_action
from database.setup import get_db_connection

def verify_password(username, password):
    """
    Verify username and password
    
    Args:
        username (str): The username to verify
        password (str): The password to verify
        
    Returns:
        tuple: (is_authenticated, role)
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    
    if result:
        stored_hash, role = result
        calculated_hash = hashlib.sha256(f"{password}:salt_key".encode()).hexdigest()
        
        if hmac.compare_digest(calculated_hash, stored_hash):
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

def create_user(username, password, role, email=None, full_name=None):
    """
    Create a new user
    
    Args:
        username (str): The username of the new user
        password (str): The password for the new user
        role (str): The role of the new user ('admin' or 'user')
        email (str, optional): Email address of the new user
        full_name (str, optional): Full name of the new user
        
    Returns:
        tuple: (success, message)
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if username already exists
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists"
    
    # Create password hash
    password_hash = hashlib.sha256(f"{password}:salt_key".encode()).hexdigest()
    
    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, password_hash, role, email, full_name, 
             datetime.datetime.now().isoformat(), None)
        )
        conn.commit()
        
        # Log user creation
        admin_user = st.session_state.get("username", "system")
        log_action(admin_user, "create_user", f"Created user: {username} with role: {role}")
        
        conn.close()
        return True, "User created successfully"
    except Exception as e:
        conn.close()
        return False, f"Error creating user: {str(e)}"

def login_user(username, password):
    """
    Log in a user and set session state
    
    Args:
        username (str): The username to log in
        password (str): The password to verify
        
    Returns:
        bool: Success status
    """
    authenticated, role = verify_password(username, password)
    if authenticated:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["role"] = role
        return True
    return False

def logout_user():
    """
    Log out the current user
    
    Returns:
        bool: Success status
    """
    if st.session_state.get("authenticated"):
        log_action(st.session_state["username"], "logout", "User logged out")
        
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.session_state["page"] = "main"
        
        return True
    return False

def is_admin():
    """
    Check if the current user is an admin
    
    Returns:
        bool: True if user is admin, False otherwise
    """
    return st.session_state.get("role") == "admin"

def get_user_profile(username):
    """
    Get user profile data
    
    Args:
        username (str): Username to get profile for
        
    Returns:
        dict: User profile data or None if not found
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("""
    SELECT username, role, email, full_name, created_at, last_login
    FROM users WHERE username = ?
    """, (username,))
    
    user_data = c.fetchone()
    conn.close()
    
    if not user_data:
        return None
        
    return {
        "username": user_data[0],
        "role": user_data[1],
        "email": user_data[2],
        "full_name": user_data[3],
        "created_at": user_data[4],
        "last_login": user_data[5]
    }

def update_user_profile(username, email=None, full_name=None):
    """
    Update user profile
    
    Args:
        username (str): Username to update
        email (str, optional): New email
        full_name (str, optional): New full name
        
    Returns:
        bool: Success status
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        c.execute(
            "UPDATE users SET email = ?, full_name = ? WHERE username = ?",
            (email, full_name, username)
        )
        conn.commit()
        conn.close()
        
        log_action(username, "update_profile", "Updated user profile")
        return True
    except Exception:
        conn.close()
        return False

def change_password(username, current_password, new_password):
    """
    Change user password
    
    Args:
        username (str): Username to change password for
        current_password (str): Current password
        new_password (str): New password
        
    Returns:
        tuple: (success, message)
    """
    # Verify current password
    authenticated, _ = verify_password(username, current_password)
    if not authenticated:
        return False, "Current password is incorrect"
    
    # Update password
    password_hash = hashlib.sha256(f"{new_password}:salt_key".encode()).hexdigest()
    
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        c.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username)
        )
        conn.commit()
        conn.close()
        
        log_action(username, "change_password", "User changed their password")
        return True, "Password changed successfully"
    except Exception as e:
        conn.close()
        return False, f"Error changing password: {str(e)}"

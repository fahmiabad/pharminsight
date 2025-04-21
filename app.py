"""
PharmInsight - A clinical knowledge base for pharmacists.

This is the main entry point for the PharmInsight application.
"""
import streamlit as st
import os
import time

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "main"

# Basic login form to test functionality
def simple_login_form():
    st.header("Welcome to PharmInsight")
    st.markdown("Please login to continue")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Hard-coded admin credentials for testing
        if username == "admin" and password == "adminpass":
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = "admin"
            st.success(f"Login successful! Welcome back, {username}.")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# Simple main page
def simple_main_page():
    st.title("PharmInsight for Pharmacists")
    st.write(f"Logged in as: {st.session_state['username']} ({st.session_state['role']})")
    
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None
        st.experimental_rerun()

# Page routing based on authentication state
if not st.session_state["authenticated"]:
    # Show login form
    simple_login_form()
else:
    # Show main page
    simple_main_page()

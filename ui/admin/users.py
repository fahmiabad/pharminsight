"""
User management admin page UI for PharmInsight.
"""
import streamlit as st
import pandas as pd
import sqlite3
from ...auth.auth import is_admin, create_user
from ...config import DB_PATH
from ...database.operations import get_all_users

def admin_users_page():
    """Admin user management page"""
    if not is_admin():
        st.warning("Admin access required")
        return
        
    st.title("User Management")
    
    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["👥 Users List", "➕ Create User", "📊 User Activity"])
    
    # Tab 1: Users List
    with tab1:
        st.subheader("All Users")
        
        # Get all users
        users_df = get_all_users()
        
        if users_df.empty:
            st.info("No users found in the system.")
        else:
            # Format dates
            if 'created_at' in users_df.columns:
                users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            if 'last_login' in users_df.columns:
                users_df['last_login'] = pd.to_datetime(users_df['last_login']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display as a dataframe
            st.dataframe(
                users_df,
                column_config={
                    "username": st.column_config.TextColumn("Username"),
                    "role": st.column_config.TextColumn("Role"),
                    "email": st.column_config.TextColumn("Email"),
                    "full_name": st.column_config.TextColumn("Full Name"),
                    "created_at": st.column_config.TextColumn("Created"),
                    "last_login": st.column_config.TextColumn("Last Login")
                },
                use_container_width=True
            )
            
            # User details and actions
            st.subheader("User Details")
            selected_username = st.selectbox(
                "Select a user",
                users_df['username'].tolist()
            )
            
            if selected_username:
                # Get user details
                user_row = users_df[users_df['username'] == selected_username].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Username:** {user_row['username']}")
                    st.write(f"**Role:** {user_row['role']}")
                    st.write(f"**Email:** {user_row['email'] or 'Not set'}")
                    
                with col2:
                    st.write(f"**Full Name:** {user_row['full_name'] or 'Not set'}")
                    st.write(f"**Created:** {user_row['created_at']}")
                    st.write(f"**Last Login:** {user_row['last_login'] or 'Never'}")
                
                # User actions
                st.subheader("User Actions")
                
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("Reset Password", disabled=(selected_username == st.session_state["username"])):
                        st.session_state["show_reset_password"] = True
                
                with action_col2:
                    if st.button("Change Role", disabled=(selected_username == st.session_state["username"])):
                        st.session_state["show_change_role"] = True
                        
                with action_col3:
                    if st.button("Disable User", disabled=(selected_username == st.session_state["username"])):
                        st.warning("This feature is not yet implemented")
                
                # Reset password form
                if st.session_state.get("show_reset_password", False):
                    st.subheader("Reset Password")
                    new_password = st.text_input("New Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    
                    if st.button("Submit Password Reset"):
                        if not new_password or not confirm_password:
                            st.error("Please enter both password fields")
                        elif new_password != confirm_password:
                            st.error("Passwords don't match")
                        else:
                            # This would call a function to reset the password
                            st.info(f"Password reset for {selected_username} would happen here")
                            st.session_state["show_reset_password"] = False
                
                # Change role form
                if st.session_state.get("show_change_role", False):
                    st.subheader("Change User Role")
                    new_role = st.selectbox(
                        "New Role",
                        ["user", "admin"],
                        index=0 if user_row['role'] == "user" else 1
                    )
                    
                    if st.button("Submit Role Change"):
                        # This would call a function to change the role
                        st.info(f"Role change for {selected_username} from {user_row['role']} to {new_role} would happen here")
                        st.session_state["show_change_role"] = False
                
                # Get user activity
                st.subheader("User Activity")
                
                conn = sqlite3.connect(DB_PATH)
                
                user_activity = pd.read_sql_query(
                    """
                    SELECT action, timestamp, details 
                    FROM audit_log
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                    """,
                    conn,
                    params=(selected_username,)
                )
                
                if not user_activity.empty:
                    user_activity['timestamp'] = pd.to_datetime(user_activity['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.dataframe(
                        user_activity,
                        column_config={
                            "timestamp": st.column_config.TextColumn("Time"),
                            "action": st.column_config.TextColumn("Action"),
                            "details": st.column_config.TextColumn("Details")
                        },
                        use_container_width=True
                    )
                else:
                    st.info("No activity records found for this user")
                
                conn.close()
    
    # Tab 2: Create User
    with tab2:
        st.subheader("Create New User")
        
        # User creation form
        with st.form("create_user_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            role = st.selectbox("Role", ["user", "admin"], index=0)
            email = st.text_input("Email (optional)")
            full_name = st.text_input("Full Name (optional)")
            
            submitted = st.form_submit_button("Create User")
            
            if submitted:
                if not username or not password or not confirm_password:
                    st.error("Username and password are required")
                elif password != confirm_password:
                    st.error("Passwords don't match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    success, message = create_user(username, password, role, email, full_name)
                    
                    if success:
                        st.success(f"User '{username}' created successfully")
                    else:
                        st.error(f"Failed to create user: {message}")
    
    # Tab 3: User Activity
    with tab3:
        st.subheader("User Activity Overview")
        
        # Time period selection
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=1
        )
        
        # Convert time period to SQL date filter
        date_filters = {
            "Last 7 days": "date('now', '-7 days')",
            "Last 30 days": "date('now', '-30 days')",
            "Last 90 days": "date('now', '-90 days')",
            "All time": "date('1970-01-01')"  # Unix epoch
        }
        
        date_filter = date_filters[time_period]
        
        conn = sqlite3.connect(DB_PATH)
        
        # Most active users
        active_users = pd.read_sql_query(
            f"""
            SELECT 
                user_id,
                COUNT(*) as action_count
            FROM audit_log
            WHERE date(timestamp) > {date_filter}
            GROUP BY user_id
            ORDER BY action_count DESC
            LIMIT 10
            """,
            conn
        )
        
        # User logins
        user_logins = pd.read_sql_query(
            f"""
            SELECT 
                user_id,
                COUNT(*) as login_count
            FROM audit_log
            WHERE action = 'login'
            AND date(timestamp) > {date_filter}
            GROUP BY user_id
            ORDER BY login_count DESC
            LIMIT 10
            """,
            conn
        )
        
        # Action distribution
        action_dist = pd.read_sql_query(
            f"""
            SELECT 
                action,
                COUNT(*) as count
            FROM audit_log
            WHERE date(timestamp) > {date_filter}
            GROUP BY action
            ORDER BY count DESC
            """,
            conn
        )
        
        conn.close()
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Active Users")
            if not active_users.empty:
                # Convert to a bar chart
                st.bar_chart(active_users.set_index('user_id'))
            else:
                st.info("No user activity data for the selected period")
                
        with col2:
            st.subheader("Most Frequent Logins")
            if not user_logins.empty:
                st.bar_chart(user_logins.set_index('user_id'))
            else:

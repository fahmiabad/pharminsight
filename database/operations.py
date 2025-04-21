"""
Database CRUD operations for PharmInsight.
"""
import sqlite3
import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from database.setup import get_db_connection
from config import DB_PATH

def execute_query(query: str, params: tuple = (), fetchone: bool = False):
    """
    Execute a database query with parameters.
    
    Args:
        query (str): SQL query
        params (tuple): Query parameters
        fetchone (bool): Whether to fetch one result or all
        
    Returns:
        The query results or None
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params)
        
        if query.strip().upper().startswith(("SELECT", "PRAGMA")):
            if fetchone:
                result = cursor.fetchone()
            else:
                result = cursor.fetchall()
            conn.close()
            return result
        else:
            conn.commit()
            conn.close()
            return cursor.rowcount
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e

def get_user_by_username(username: str) -> Dict[str, Any]:
    """
    Get user data by username.
    
    Args:
        username (str): Username to look up
        
    Returns:
        dict: User data or None if not found
    """
    conn = get_db_connection()
    query = """
    SELECT username, role, email, full_name, created_at, last_login
    FROM users
    WHERE username = ?
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (username,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return None
    
    return {
        "username": result[0],
        "role": result[1],
        "email": result[2],
        "full_name": result[3],
        "created_at": result[4],
        "last_login": result[5]
    }

def get_all_users() -> pd.DataFrame:
    """
    Get all users in the system.
    
    Returns:
        pandas.DataFrame: All users data
    """
    conn = get_db_connection()
    query = """
    SELECT username, role, email, full_name, created_at, last_login
    FROM users
    ORDER BY username
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def query_to_dataframe(query: str, params: tuple = ()) -> pd.DataFrame:
    """
    Execute a query and return results as a DataFrame.
    
    Args:
        query (str): SQL query
        params (tuple): Query parameters
        
    Returns:
        pandas.DataFrame: Query results
    """
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

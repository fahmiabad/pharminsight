"""
Database setup and initialization for PharmInsight.
"""
import sqlite3
import datetime
import hashlib
import uuid
import os
import time
from utils.logging import log_action
from config import DB_PATH

def get_db_connection():
    """Get a database connection with retry logic for locks"""
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Set timeout for acquiring lock (default is 5 seconds)
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                attempt += 1
                if attempt < max_attempts:
                    # Wait before retrying
                    time.sleep(1)
                else:
                    raise e
            else:
                raise e

def init_database():
    """Initialize all database tables"""
    # First check if database exists
    db_exists = os.path.exists(DB_PATH)
    
    # If database file exists but is locked, we might not need to initialize
    if db_exists:
        try:
            # Quick test query to see if database is initialized
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if c.fetchone():
                # Table exists, database is initialized
                conn.close()
                return True
        except Exception:
            # If error occurs, continue with initialization attempt
            pass
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Users table
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            created_at TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')
        
        # Rest of your table creation code...
        # [Include all your other CREATE TABLE statements here]
        
        # Create default admin if it doesn't exist
        c.execute("SELECT username FROM users WHERE username = 'admin'")
        if not c.fetchone():
            # Create with default password 'adminpass'
            password_hash = hashlib.sha256(f"adminpass:salt_key".encode()).hexdigest()
            c.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("admin", password_hash, "admin", "admin@example.com", "System Administrator", 
                 datetime.datetime.now().isoformat(), None)
            )
            
            # Log the creation
            try:
                log_action("system", "create_user", "Created default admin user")
            except Exception:
                # Skip logging if it fails during initialization
                pass
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        if conn:
            conn.close()
        raise e

"""
Database setup and initialization for PharmInsight.
"""
import sqlite3
import datetime
import hashlib
import uuid
from ..utils.logging import log_action
from ..config import DB_PATH

def init_database():
    """Initialize all database tables"""
    conn = sqlite3.connect(DB_PATH)
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
    
    # Documents table
    c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        upload_date TIMESTAMP,
        uploader TEXT NOT NULL,
        category TEXT,
        description TEXT,
        expiry_date TEXT,
        is_active INTEGER DEFAULT 1
    )
    ''')
    
    # Document chunks table
    c.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding BLOB,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    )
    ''')
    
    # Questions and answers table
    c.execute('''
    CREATE TABLE IF NOT EXISTS qa_pairs (
        question_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        timestamp TIMESTAMP,
        sources TEXT,
        model_used TEXT
    )
    ''')
    
    # Modified Feedback table with 3-point rating scale
    c.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id TEXT PRIMARY KEY,
        question_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 3),
        comment TEXT,
        timestamp TIMESTAMP,
        FOREIGN KEY (question_id) REFERENCES qa_pairs(question_id)
    )
    ''')
    
    # Audit log table
    c.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        log_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TIMESTAMP
    )
    ''')
    
    # User search history
    c.execute('''
    CREATE TABLE IF NOT EXISTS search_history (
        history_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        query TEXT NOT NULL,
        timestamp TIMESTAMP,
        num_results INTEGER
    )
    ''')
    
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
        log_action("system", "create_user", "Created default admin user")
        
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a database connection"""
    return sqlite3.connect(DB_PATH)

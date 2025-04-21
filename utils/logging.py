"""
Logging utilities for PharmInsight.
"""
import sqlite3
import datetime
import uuid
from config import DB_PATH

def log_action(user_id, action, details=None):
    """
    Log an action to the audit log
    
    Args:
        user_id (str): The ID of the user performing the action
        action (str): The type of action being performed
        details (str, optional): Additional details about the action
        
    Returns:
        str: The log ID
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    log_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    c.execute(
        "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)",
        (log_id, user_id, action, details, timestamp)
    )
    
    conn.commit()
    conn.close()
    
    return log_id

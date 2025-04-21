"""
Database models and schemas for PharmInsight.
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional, List
import datetime

@dataclass
class User:
    """User model representing a system user."""
    username: str
    role: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: Optional[str] = None
    last_login: Optional[str] = None

@dataclass
class Document:
    """Document model representing an uploaded document."""
    doc_id: str
    filename: str
    uploader: str
    upload_date: str
    category: Optional[str] = None
    description: Optional[str] = None
    expiry_date: Optional[str] = None
    is_active: bool = True

@dataclass
class DocumentChunk:
    """Document chunk model representing a section of a document."""
    chunk_id: str
    doc_id: str
    text: str
    embedding: Optional[bytes] = None

@dataclass
class QAPair:
    """Question-answer pair model."""
    question_id: str
    user_id: str
    query: str
    answer: str
    timestamp: str
    sources: Optional[str] = None
    model_used: Optional[str] = None

@dataclass
class Feedback:
    """Feedback model for answer ratings."""
    feedback_id: str
    question_id: str
    user_id: str
    rating: int  # 1-3 scale
    comment: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class AuditLog:
    """Audit log model for system actions."""
    log_id: str
    user_id: str
    action: str
    details: Optional[str] = None
    timestamp: str

@dataclass
class SearchHistory:
    """Search history model for user queries."""
    history_id: str
    user_id: str
    query: str
    timestamp: str
    num_results: int = 0

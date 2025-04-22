import os
from openai import OpenAI
import faiss
import numpy as np
import pickle
import json
import streamlit as st
import sys
import os
import sqlite3
import hashlib
import datetime
import uuid
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
import traceback # For detailed error logging
import time # For small delays

# Add the current directory to Python's path (use with caution)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
APP_VERSION = "2.1.1" # Incremented version
DB_PATH = "pharminsight.db"
INDEX_PATH = "vector.index"
DOCS_METADATA_PATH = "docs_metadata.pkl"
# SIMILARITY_THRESHOLD and K_RETRIEVE are now managed in session state

# --- Database Initialization ---
def init_database():
    """Initialize all database tables"""
    try:
        # Use context manager for connection handling
        with sqlite3.connect(DB_PATH, timeout=20.0) as conn:
            c = conn.cursor()

            # Users table
            c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                email TEXT UNIQUE,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            ''')

            # Documents table
            c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                uploader TEXT NOT NULL,
                category TEXT,
                description TEXT,
                expiry_date TEXT,
                is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1)),
                FOREIGN KEY (uploader) REFERENCES users(username) ON DELETE SET NULL
            )
            ''')

            # Document chunks table
            c.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB, -- Store embedding if needed, but FAISS index is primary
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
            ''')
            # Add index for faster chunk retrieval by doc_id
            c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")


            # Questions and answers table
            c.execute('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                question_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sources TEXT, -- JSON list of source filenames or identifiers
                model_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
            )
            ''')
             # Add index for faster QA retrieval by user_id
            c.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_user_id ON qa_pairs (user_id)")


            # Feedback table with 3-point rating scale
            c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                question_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 3),
                comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES qa_pairs(question_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
            )
            ''')
            # Add index for faster feedback retrieval
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_question_id ON feedback (question_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback (user_id)")


            # Audit log table
            c.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL, -- Can be 'system' for system actions
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
             # Add index for faster audit log retrieval by user_id and timestamp
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log (user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log (timestamp)")


            # User search history
            c.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                history_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_results INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(username) ON DELETE CASCADE
            )
            ''')
            # Add index for faster history retrieval by user_id and timestamp
            c.execute("CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history (user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history (timestamp)")


            # Create default admin if it doesn't exist
            c.execute("SELECT username FROM users WHERE username = 'admin'")
            if not c.fetchone():
                # Use a more secure way to handle salt in a real application
                salt = "salt_key" # Example salt, should be unique and stored securely
                admin_pass = "adminpass"
                user_pass = "password"

                admin_hash = hashlib.sha256(f"{admin_pass}:{salt}".encode()).hexdigest()
                user_hash = hashlib.sha256(f"{user_pass}:{salt}".encode()).hexdigest()

                # Create admin user
                c.execute(
                    "INSERT INTO users (username, password_hash, role, email, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("admin", admin_hash, "admin", "admin@example.com", "System Administrator", datetime.datetime.now().isoformat())
                )

                # Create regular user account
                c.execute(
                    "INSERT INTO users (username, password_hash, role, email, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("user", user_hash, "user", "user@example.com", "Regular User", datetime.datetime.now().isoformat())
                )

                # Log the creation (using the function ensures consistent logging)
                log_action("system", "create_user", "Created default admin and regular user", conn) # Pass connection

            conn.commit() # Commit changes
        return True
    except sqlite3.Error as e:
        st.error(f"❌ Database initialization error: {str(e)}")
        # Log the detailed error for backend debugging
        print(f"Database initialization failed: {e}\n{traceback.format_exc()}")
        return False
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during database initialization: {str(e)}")
        print(f"Unexpected error during DB init: {e}\n{traceback.format_exc()}")
        return False

# --- Logging ---
def log_action(user_id, action, details=None, conn=None):
    """Log an action to the audit log. Can accept an existing connection."""
    log_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    log_entry = (log_id, user_id, action, details, timestamp)

    # Flag to check if we need to close the connection
    should_close_conn = False
    if conn is None:
        try:
            conn = sqlite3.connect(DB_PATH, timeout=20.0)
            should_close_conn = True
        except sqlite3.Error as e:
            st.error(f"🚨 Logging error (connection): {str(e)}")
            print(f"Logging connection error: {e}")
            return None

    try:
        c = conn.cursor()
        c.execute("INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)", log_entry)
        if should_close_conn: # Only commit if we opened the connection here
             conn.commit()
        return log_id
    except sqlite3.Error as e:
        st.error(f"🚨 Logging error (execution): {str(e)}")
        print(f"Logging execution error: {e}")
        return None
    finally:
        if should_close_conn and conn:
            conn.close()


# --- OpenAI & Embeddings ---
def get_openai_client():
    """Get an OpenAI client with proper error handling"""
    api_key = st.session_state.get("openai_api_key")

    # If not in session state, try environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # If not in environment, try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (AttributeError, KeyError): # Handle cases where secrets aren't available or key missing
             pass # Keep api_key as None or empty

    if not api_key:
        # Don't show error here, let the caller handle it if needed
        # print("DEBUG: OpenAI API key is missing.") # Log for debugging
        return None # Return None to indicate failure

    try:
        client = OpenAI(api_key=api_key)
        # Perform a simple test call to validate the key
        client.models.list()
        # Store the validated key in session state if it wasn't there before
        if not st.session_state.get("openai_api_key"):
             st.session_state["openai_api_key"] = api_key
        return client
    except Exception as e:
        st.error(f"❌ Error initializing or validating OpenAI client: {e}")
        print(f"OpenAI client initialization error: {e}")
        # Clear potentially invalid key from session state
        if "openai_api_key" in st.session_state:
             del st.session_state["openai_api_key"]
        return None

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for text using OpenAI API"""
    client = get_openai_client()
    if not client:
        st.error("⚠️ OpenAI client not available. Cannot generate embeddings.")
        raise ConnectionError("OpenAI client failed to initialize") # Raise specific error

    try:
        # Clean and prepare text
        text = text.replace("\n", " ").strip()

        # Handle empty text after cleaning
        if not text:
            st.warning("⚠️ Attempted to get embedding for empty text.")
            # Return zero vector matching the expected dimension for the model
            # Dimension for text-embedding-3-small is 1536
            return np.zeros(1536, dtype=np.float32)

        response = client.embeddings.create(
            input=[text],
            model=model
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # Normalize for cosine similarity (important for IndexFlatIP)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            # Handle zero vector case if it occurs, though unlikely for non-empty text
            st.warning("⚠️ Generated a zero vector embedding for non-empty text.")

        return embedding

    except Exception as e:
        st.error(f"❌ Error creating embedding: {e}")
        print(f"Embedding generation error for text snippet '{text[:50]}...': {e}")
        raise # Re-raise the exception to be caught by the caller

# --- Index Management ---
def rebuild_index_from_db():
    """Rebuild the search index from database records"""
    st.info("Attempting to rebuild the search index...")
    client = get_openai_client()
    if not client:
         st.error("❌ Cannot rebuild index: OpenAI client is not configured or key is invalid.")
         return False, 0

    all_embeddings = []
    metadata = []
    processed_chunk_ids = set() # To track processed chunks

    try:
        with sqlite3.connect(DB_PATH, timeout=30.0) as conn: # Increased timeout
            c = conn.cursor()

            # Get all active document chunks with necessary metadata
            c.execute('''
            SELECT c.chunk_id, c.text, d.filename, d.category, d.doc_id
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE d.is_active = 1
            ''')

            results = c.fetchall()

        if not results:
            st.warning("⚠️ No active document chunks found in the database to build the index.")
             # If no results, create empty index files
            dimension = 1536 # Default for text-embedding-3-small
            empty_index = faiss.IndexFlatIP(dimension)
            faiss.write_index(empty_index, INDEX_PATH)
            with open(DOCS_METADATA_PATH, "wb") as f:
                pickle.dump([], f)
            st.success("✅ Created empty index files as no active documents were found.")
            return True, 0 # Indicate success but with 0 items

        st.write(f"Found {len(results)} active chunks to process.")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (chunk_id, text, filename, category, doc_id) in enumerate(results):
            status_text.text(f"Processing chunk {i+1}/{len(results)} (ID: {chunk_id})")
            progress = (i + 1) / len(results)
            progress_bar.progress(progress)

            if not text or not text.strip():
                st.warning(f"Skipping empty chunk: {chunk_id} from {filename}")
                continue

            try:
                # Generate embedding for each chunk
                embedding = get_embedding(text) # Use the robust function
                all_embeddings.append(embedding)

                # Create metadata entry
                meta = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": filename,
                    "category": category,
                    "doc_id": doc_id
                }
                metadata.append(meta)
                processed_chunk_ids.add(chunk_id)

            except ConnectionError as ce: # Catch specific error from get_embedding
                 st.error(f"❌ Failed to get embedding for chunk {chunk_id} due to OpenAI connection issue. Stopping rebuild.")
                 print(f"OpenAI connection error during rebuild: {ce}")
                 return False, 0 # Abort rebuild on critical API error
            except Exception as e:
                st.error(f"⚠️ Error processing chunk {chunk_id} from {filename}: {str(e)}. Skipping chunk.")
                print(f"Error processing chunk {chunk_id}: {e}\n{traceback.format_exc()}")
                # Optionally: Log this chunk ID for later review
                continue # Skip this chunk and continue with others

        # --- Create and save FAISS index ---
        if not all_embeddings or not metadata:
             st.warning("⚠️ No embeddings were generated (possibly due to errors). Index cannot be built.")
             # Create empty files if needed
             if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_METADATA_PATH):
                 dimension = 1536
                 empty_index = faiss.IndexFlatIP(dimension)
                 faiss.write_index(empty_index, INDEX_PATH)
                 with open(DOCS_METADATA_PATH, "wb") as f:
                    pickle.dump([], f)
                 st.success("✅ Created empty index files.")
             return True, 0 # Success, but 0 items indexed


        all_embeddings = np.array(all_embeddings, dtype=np.float32)

        # Check for NaN or Inf values in embeddings (optional but good practice)
        if np.isnan(all_embeddings).any() or np.isinf(all_embeddings).any():
            st.error("❌ Invalid values (NaN or Inf) found in generated embeddings. Index build failed.")
            print("ERROR: NaN or Inf found in embeddings.")
            return False, 0

        dimension = all_embeddings.shape[1]
        st.write(f"Building FAISS index with dimension {dimension} for {len(metadata)} items.")

        # Use IndexFlatIP for cosine similarity after normalization
        index = faiss.IndexFlatIP(dimension)

        # FAISS expects L2 normalized vectors for cosine similarity with IndexFlatIP
        # Our get_embedding function already normalizes, so we just add.
        index.add(all_embeddings)
        st.write(f"FAISS index created. Total vectors in index: {index.ntotal}")

        # Save index and metadata
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)

        status_text.text("Index rebuild complete.")
        progress_bar.progress(1.0)
        st.success(f"✅ Index rebuilt successfully with {len(metadata)} document chunks.")
        log_action("system", "rebuild_index", f"Index rebuilt with {len(metadata)} chunks.")
        return True, len(metadata)

    except FileNotFoundError as e:
        st.error(f"❌ File not found error during index rebuild: {str(e)}")
        print(f"FileNotFoundError during rebuild: {e}\n{traceback.format_exc()}")
        return False, 0
    except pickle.PickleError as e:
        st.error(f"❌ Error saving metadata (pickle error): {str(e)}")
        print(f"PickleError during rebuild: {e}\n{traceback.format_exc()}")
        return False, 0
    except faiss.FaissException as e:
         st.error(f"❌ FAISS error during index build/save: {str(e)}")
         print(f"FAISS error during rebuild: {e}\n{traceback.format_exc()}")
         return False, 0
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"❌ An unexpected error occurred during index rebuild: {str(e)}")
        print(f"Unexpected error during rebuild: {e}\n{traceback.format_exc()}") # Log traceback
        return False, 0

# --- UPDATED load_search_index ---
def load_search_index():
    """Load the search index and metadata, attempting rebuild if necessary."""
    # Check cache first
    if "faiss_index" in st.session_state and "index_metadata" in st.session_state:
        index = st.session_state["faiss_index"]
        metadata = st.session_state["index_metadata"]
        # Optional: Add a check for staleness if needed
        # st.sidebar.caption(f"Index in memory ({index.ntotal} vectors)") # Can be noisy
        return index, metadata

    index_exists = os.path.exists(INDEX_PATH)
    metadata_exists = os.path.exists(DOCS_METADATA_PATH)
    index_empty = index_exists and os.path.getsize(INDEX_PATH) == 0
    metadata_empty = metadata_exists and os.path.getsize(DOCS_METADATA_PATH) == 0

    try:
        if index_exists and metadata_exists and not index_empty and not metadata_empty:
            st.sidebar.info("Loading index from files...")
            index = faiss.read_index(INDEX_PATH)
            with open(DOCS_METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)

            if index.ntotal != len(metadata):
                st.sidebar.warning(f"⚠️ Index size ({index.ntotal}) mismatch with metadata ({len(metadata)}). Rebuilding recommended (Admin Panel).")
                # Load what we have, but warn
            else:
                st.sidebar.success(f"✅ Index loaded ({index.ntotal} vectors)")

            # Store loaded index in session state
            st.session_state["faiss_index"] = index
            st.session_state["index_metadata"] = metadata
            return index, metadata

        # --- Conditions to attempt rebuild ---
        elif not index_exists or not metadata_exists or index_empty or metadata_empty:
            if not index_exists or not metadata_exists:
                 st.sidebar.warning("⚠️ Index files not found. Attempting automatic rebuild...")
            elif index_empty or metadata_empty:
                 st.sidebar.warning("⚠️ Index or metadata file is empty. Attempting automatic rebuild...")

            # Try to rebuild the index automatically
            success, count = rebuild_index_from_db() # This function has its own spinner/messages in main area
            if success:
                st.sidebar.success(f"✅ Index rebuilt ({count} vectors). Reloading...")
                # Now try loading again *after* successful rebuild
                if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_METADATA_PATH):
                    index = faiss.read_index(INDEX_PATH)
                    with open(DOCS_METADATA_PATH, "rb") as f:
                        metadata = pickle.load(f)
                    # Sanity check after rebuild
                    if index.ntotal != len(metadata):
                         st.sidebar.error("❌ Rebuild success, but index/metadata mismatch after reload.")
                         # Fall through to return empty
                    else:
                        st.sidebar.success(f"✅ Index reloaded successfully after rebuild ({index.ntotal} vectors).")
                        st.session_state["faiss_index"] = index
                        st.session_state["index_metadata"] = metadata
                        return index, metadata
                else:
                    st.sidebar.error("❌ Rebuild success, but index files still missing after rebuild.")
                    # Fall through to return empty
            else:
                st.sidebar.error("❌ Automatic index rebuild failed. Check logs/Admin Panel.")
                # Fall through to return empty
        else:
             # Should not happen given the checks above, but as a safeguard
             st.sidebar.error("Internal error: Unexpected state in load_search_index.")

    except FileNotFoundError:
        st.sidebar.error("❌ Index/metadata file vanished during load attempt.")
    except pickle.UnpicklingError:
        st.sidebar.error("❌ Error reading metadata file (corrupted?). Please rebuild.")
    except faiss.FaissException as e:
        st.sidebar.error(f"❌ FAISS error loading index: {e}. Please rebuild.")
    except Exception as e:
        st.sidebar.error(f"❌ Unexpected error loading index: {e}")
        print(f"Unexpected error loading index: {e}\n{traceback.format_exc()}")

    # --- Fallback: Return Empty Index ---
    st.sidebar.error("⚠️ Index is not available. Returning empty index.")
    dimension = 1536  # Default for OpenAI embeddings
    empty_index = faiss.IndexFlatIP(dimension)
    empty_metadata = []
    # Store empty structures in session state to prevent repeated failed load attempts
    st.session_state["faiss_index"] = empty_index
    st.session_state["index_metadata"] = empty_metadata
    return empty_index, empty_metadata


# --- Search & Answer Generation ---
def search_documents(query, k=5, threshold=0.7):
    """Search for relevant documents using FAISS index."""
    # Load index and metadata (uses session state cache now)
    index, metadata = load_search_index()

    if index.ntotal == 0 or not metadata:
        # This case is handled by the caller (main_page) now, but good to keep check
        print("DEBUG: search_documents called with empty index/metadata.")
        return [] # Return empty list if index is not usable

    # Get query embedding
    try:
        query_embedding = get_embedding(query)
        # Ensure it's 2D for FAISS search
        query_embedding = query_embedding.reshape(1, -1)
        # Ensure normalization (get_embedding should do this, but double-check)
        if abs(np.linalg.norm(query_embedding) - 1.0) > 1e-5:
             st.warning("⚠️ Debug: Query embedding norm is not 1 after get_embedding. Normalizing again.")
             faiss.normalize_L2(query_embedding)

    except Exception as e:
        st.error(f"❌ Error during query embedding generation: {e}")
        print(f"Error getting query embedding: {e}\n{traceback.format_exc()}")
        return [] # Cannot search without query embedding

    # Search the index
    results = []
    try:
        # Determine the actual number of neighbors to search for
        actual_k = min(k, index.ntotal)
        if actual_k <= 0:
             return []

        # `index.search` returns distances (inner product scores for IndexFlatIP) and indices
        distances, indices = index.search(query_embedding, actual_k)

        # Process results if any were found
        if indices.size > 0 and distances.size > 0:
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                 if idx == -1:
                     continue # Skip invalid indices

                 # Check index bounds rigorously
                 if 0 <= idx < len(metadata):
                     # Compare distance (score) with the threshold
                     if dist >= threshold:
                         # Make a copy to avoid modifying the original metadata list
                         doc = metadata[idx].copy()
                         doc["score"] = float(dist) # Add the score to the result
                         results.append(doc)
                 else:
                     # This should ideally not happen if index and metadata are in sync
                     st.warning(f"🚨 Search Warning: Result index {idx} is out of bounds for metadata (length {len(metadata)}). Index might be corrupted or out of sync!")

    except faiss.FaissException as e:
         st.error(f"❌ Error during FAISS index search: {e}")
         print(f"FAISS search error: {e}\n{traceback.format_exc()}")
         return [] # Return empty on search error
    except Exception as e:
        st.error(f"❌ Unexpected error during index search processing: {e}")
        print(f"Unexpected search processing error: {e}\n{traceback.format_exc()}")
        return []

    # Sort results by score (descending) before returning
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return results


def generate_answer(query, model="gpt-3.5-turbo", include_explanation=True, temp=0.2):
    """Generate an answer to a query based on retrieved documents."""
    client = get_openai_client()
    if not client:
         st.error("❌ Cannot generate answer: OpenAI client not available.")
         # Return a structured error message
         return {
            "question_id": str(uuid.uuid4()),
            "query": query,
            "answer": "Error: Could not connect to the language model service.",
            "sources": [],
            "model": model,
            "timestamp": datetime.datetime.now().isoformat(),
            "source_type": "Error"
        }

    history_id = str(uuid.uuid4())
    username = st.session_state.get("username", "unknown_user") # Handle missing username

    # --- Record search attempt in history ---
    conn_hist = None
    try:
        conn_hist = sqlite3.connect(DB_PATH, timeout=20.0)
        c_hist = conn_hist.cursor()
        c_hist.execute(
            "INSERT INTO search_history (history_id, user_id, query, timestamp, num_results) VALUES (?, ?, ?, ?, ?)",
            (history_id, username, query, datetime.datetime.now().isoformat(), 0) # Initial results = 0
        )
        conn_hist.commit()

        # Update user history in session state (optional, for UI)
        if "user_history" not in st.session_state:
            st.session_state["user_history"] = []
        st.session_state["user_history"].append({
            "query": query,
            "timestamp": datetime.datetime.now().isoformat()
        })
        # Limit session history size if needed
        max_hist_len = 10
        if len(st.session_state["user_history"]) > max_hist_len:
             st.session_state["user_history"] = st.session_state["user_history"][-max_hist_len:]

    except sqlite3.Error as e:
        st.error(f"⚠️ Error recording search history: {str(e)}")
        print(f"History recording error: {e}")
    finally:
        if conn_hist:
            conn_hist.close()

    # --- Search for relevant documents ---
    # Get search parameters from session state
    k_retrieve = st.session_state.get("k_retrieve", 5)
    similarity_threshold = st.session_state.get("similarity_threshold", 0.75)

    try:
        results = search_documents(query, k=k_retrieve, threshold=similarity_threshold)
    except Exception as e:
         st.error(f"❌ Error during document search: {e}")
         results = [] # Ensure results is an empty list on error

    # --- Update search history with result count ---
    conn_update = None
    try:
        conn_update = sqlite3.connect(DB_PATH, timeout=20.0)
        c_update = conn_update.cursor()
        c_update.execute(
            "UPDATE search_history SET num_results = ? WHERE history_id = ?",
            (len(results), history_id)
        )
        conn_update.commit()
    except sqlite3.Error as e:
        st.error(f"⚠️ Error updating search history result count: {str(e)}")
        print(f"History update error: {e}")
    finally:
        if conn_update:
            conn_update.close()

    # --- Prepare response structure ---
    answer_data = {
        "question_id": str(uuid.uuid4()),
        "query": query,
        "answer": "",
        "sources": results, # Include full results with scores
        "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
        "source_type": "No Relevant Documents" # Default assumption
    }

    # --- Generate Answer using LLM ---
    try:
        prompt = ""
        if results:
            answer_data["source_type"] = "Retrieved Documents"
            # Prepare context from retrieved chunks, sorted by score
            context_pieces = []
            for i, result in enumerate(results):
                # Ensure essential keys exist
                source = result.get('source', 'Unknown Source')
                category = result.get('category', 'Uncategorized')
                text = result.get('text', 'No text available')
                score = result.get('score', 0.0)
                chunk = f"Source {i+1}: {source} (Category: {category}, Score: {score:.3f})\n\n{text}"
                context_pieces.append(chunk)

            context = "\n\n---\n\n".join(context_pieces)

            # Build prompt based on settings
            if include_explanation:
                prompt = f"""
You are PharmInsight, a clinical expert assistant for pharmacists. Your knowledge is based *only* on the provided document excerpts.
Answer the user's question concisely and accurately using *only* the information within these excerpts.
If the excerpts do not contain the information needed, clearly state that the information is not available in the provided documents.
Do not add any information not present in the excerpts.

**Document Excerpts:**
--- START OF EXCERPTS ---
{context}
--- END OF EXCERPTS ---

**User's Question:**
{query}

**Instructions:**
1.  Provide a direct answer to the question based *only* on the excerpts.
2.  If the answer is found, provide a detailed explanation, citing the specific source number(s) (e.g., "Source 1", "Source 3") for each piece of information in your explanation.
3.  List the source number(s) used for the answer.
4.  If the information is not in the excerpts, state: "I cannot answer this question based on the provided documents."

**Output Format:**

**Answer:**
[Your direct answer based *only* on the excerpts, or the "cannot answer" statement.]

**Explanation:**
[Detailed explanation citing sources, e.g., "According to Source 1, ... . Source 2 adds that ... ."]
[If you cannot answer, state: "No explanation available as the information was not found in the provided documents."]

**Sources Used:**
[List source numbers, e.g., "Source 1, Source 3"]
[If you cannot answer, state: "None"]
"""
            else:
                prompt = f"""
You are PharmInsight, a clinical expert assistant for pharmacists. Answer the following question based *only* on the provided document excerpts.
If the document excerpts don't contain the information needed to answer the question, state: "I cannot answer this question based on the provided documents."
Do not add any information not present in the excerpts.

**Document Excerpts:**
--- START OF EXCERPTS ---
{context}
--- END OF EXCERPTS ---

**User's Question:**
{query}

**Output Format:**

**Answer:**
[Your direct answer based *only* on the excerpts, or the "cannot answer" statement.]
"""

        else:
            # No relevant documents found - Construct a simple response directly
            answer_data["answer"] = "I don't have specific information about this in the provided documents. Please consider consulting official clinical guidelines or pharmacist resources for accurate information."
            # Skip LLM call if no results
            return answer_data


        # --- Call LLM API ---
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are PharmInsight, an expert clinical assistant for pharmacists, answering questions based *only* on provided text excerpts."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=1000 # Adjust as needed
        )

        answer = response.choices[0].message.content

        # Update answer data
        answer_data["answer"] = answer.strip()

        # --- Store the Q&A pair in database ---
        conn_qa = None
        try:
            conn_qa = sqlite3.connect(DB_PATH, timeout=20.0)
            c_qa = conn_qa.cursor()
            # Extract just the source filenames for storage
            source_filenames = json.dumps([s.get("source", "") for s in answer_data["sources"]])
            c_qa.execute(
                "INSERT INTO qa_pairs (question_id, user_id, query, answer, timestamp, sources, model_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    answer_data["question_id"],
                    username,
                    answer_data["query"],
                    answer_data["answer"],
                    answer_data["timestamp"],
                    source_filenames,
                    answer_data["model"]
                )
            )
            conn_qa.commit()
        except sqlite3.Error as e:
            st.error(f"⚠️ Error storing Q&A pair: {str(e)}")
            print(f"QA storage error: {e}")
        finally:
            if conn_qa:
                conn_qa.close()

        return answer_data

    except Exception as e:
        error_message = f"❌ Error generating answer: {str(e)}"
        st.error(error_message)
        print(f"Answer generation error: {e}\n{traceback.format_exc()}")
        answer_data["answer"] = f"An error occurred while generating the answer. Details: {str(e)}"
        answer_data["source_type"] = "Error"
        return answer_data

# --- Feedback ---
def submit_feedback(question_id, rating, comment=None):
    """Submit feedback for an answer"""
    # Validate rating
    if not 1 <= rating <= 3:
        st.error("Rating must be between 1 and 3")
        return False

    feedback_id = str(uuid.uuid4())
    username = st.session_state.get("username", "unknown_user")
    timestamp = datetime.datetime.now().isoformat()

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()

        c.execute(
            "INSERT INTO feedback (feedback_id, question_id, user_id, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (feedback_id, question_id, username, rating, comment, timestamp)
        )
        conn.commit()

        log_action(
            username,
            "submit_feedback",
            f"Feedback submitted for QID {question_id}. Rating: {rating}. Comment: '{comment[:50]}...'",
            conn # Pass connection to log_action
        )
        # No need to commit again here, log_action doesn't commit if conn is passed

        return True
    except sqlite3.Error as e:
        st.error(f"❌ Error submitting feedback: {str(e)}")
        print(f"Feedback submission error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def feedback_ui(question_id):
    """Display feedback collection UI, managing state within the function."""
    st.divider()

    # Use unique keys based on question_id for state isolation
    feedback_state_key = f"feedback_state_{question_id}"
    comment_key = f"comment_{question_id}"

    # Initialize state for this specific feedback instance if not present
    if feedback_state_key not in st.session_state:
        st.session_state[feedback_state_key] = {"rating": None, "submitted": False}

    # Prevent showing feedback UI if already submitted for this question_id
    if st.session_state[feedback_state_key]["submitted"]:
        st.success("✅ Thank you for your feedback on this answer!")
        return

    with st.expander("📊 Rate this answer", expanded=not st.session_state[feedback_state_key]["submitted"]): # Keep expanded until submitted
        st.write("Your feedback helps improve PharmInsight!")

        cols = st.columns(3)
        rating_buttons = {1: "1 - Not Helpful", 2: "2 - Somewhat Helpful", 3: "3 - Very Helpful"}

        # Display buttons and update rating state on click
        for r, label in rating_buttons.items():
            button_key = f"rate_{r}_{question_id}"
            # Highlight selected button
            button_type = "primary" if st.session_state[feedback_state_key].get("rating") == r else "secondary"
            if cols[r-1].button(label, key=button_key, use_container_width=True, type=button_type):
                st.session_state[feedback_state_key]["rating"] = r
                # Rerun to update the UI showing the comment box / highlighting
                st.rerun()

        # Show comment area and submit button if a rating is selected
        selected_rating = st.session_state[feedback_state_key].get("rating")
        if selected_rating:
            comment = st.text_area("Additional comments (optional)", key=comment_key)

            if st.button("Submit Feedback", key=f"submit_btn_{question_id}"):
                success = submit_feedback(question_id, selected_rating, comment)
                if success:
                    st.session_state[feedback_state_key]["submitted"] = True
                    st.success("✅ Feedback submitted successfully!")
                    st.rerun() # Rerun to show the "Thank you" message and collapse expander
                else:
                    st.error("❌ Error submitting feedback. Please try again.")


# --- Authentication ---
def verify_password(username, password):
    """Verify username and password against the database."""
    if not username or not password:
        return False, None # Basic validation

    # Use a more secure way to handle salt in a real application
    salt = "salt_key" # Example salt, should match the one used in init_database/user creation
    calculated_hash = hashlib.sha256(f"{password}:{salt}".encode()).hexdigest()

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()

        c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
        result = c.fetchone()

        if result:
            stored_hash, role = result
            if calculated_hash == stored_hash:
                # --- Password verified ---
                # Update last login time
                try:
                    c.execute(
                        "UPDATE users SET last_login = ? WHERE username = ?",
                        (datetime.datetime.now().isoformat(), username)
                    )
                    conn.commit() # Commit login time update
                except sqlite3.Error as e:
                    st.warning(f"⚠️ Could not update last login time for {username}: {e}")
                    print(f"Login time update failed for {username}: {e}")
                    # Continue with login even if timestamp update fails

                # Log successful login (pass connection to avoid re-opening)
                log_action(username, "login", "Successful login", conn)
                # No need to commit again here

                return True, role
            else:
                 # --- Password mismatch ---
                 log_action(username, "failed_login", f"Failed login attempt (wrong password) for user: {username}", conn)
                 return False, None
        else:
            # --- User not found ---
            log_action("system", "failed_login", f"Failed login attempt (user not found): {username}", conn)
            return False, None

    except sqlite3.Error as e:
        st.error(f"❌ Authentication error: {str(e)}")
        print(f"Authentication DB error: {e}")
        return False, None
    finally:
        if conn:
            conn.close()


# --- Streamlit UI & Pages ---

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "page": "login", # Start at login page
        "user_history": [],
        "openai_api_key": None, # Will be populated by get_openai_client logic
        "embedding_model": "text-embedding-3-small",
        # Feedback state is handled per-question_id in feedback_ui
        "k_retrieve": 5,
        "similarity_threshold": 0.75,
        "llm_model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "include_explanation": True,
        "current_query": None, # Used for re-running searches
        "query_input_value": "", # To preserve input field value across reruns
        # Admin page states (managed locally now with unique keys)
        "faiss_index": None, # Cache for loaded index
        "index_metadata": None, # Cache for loaded metadata
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_admin():
    """Check if current user is an admin"""
    return st.session_state.get("role") == "admin"

def render_sidebar():
    """Render the application sidebar"""
    with st.sidebar:
        # Use columns for better layout
        col1, col2 = st.columns([1, 3])
        with col1:
             st.image("https://www.svgrepo.com/show/530588/chemical-laboratory.svg", width=50)
        with col2:
             st.title("PharmInsight")
             st.caption(f"v{APP_VERSION}")

        # Show user info if logged in
        if st.session_state.get("authenticated"):
            st.write(f"Logged in as: **{st.session_state.get('username', 'N/A')}**")
            st.write(f"Role: *{st.session_state.get('role', 'N/A')}*")
            if st.button("Logout", key="logout_button"):
                username = st.session_state.get("username", "unknown_user")
                log_action(username, "logout", "User logged out")
                # Reset session state completely on logout
                keys_to_delete = list(st.session_state.keys())
                for key in keys_to_delete:
                     if not key.startswith('_'): # Avoid internal keys
                         del st.session_state[key]
                # Re-initialize with defaults, setting page to login
                initialize_session_state()
                st.session_state["page"] = "login"
                st.success("Logged out successfully.")
                st.rerun()

            st.divider()

            # --- Navigation ---
            st.subheader("Navigation")
            nav_buttons = {
                "main": "📚 Home / Search",
                "profile": "👤 Your Profile",
                "history": "📜 Search History",
            }
            for page_key, label in nav_buttons.items():
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state["page"] = page_key
                    st.rerun()

            # --- Admin Navigation ---
            if is_admin():
                st.divider()
                st.subheader("Admin Panel")
                admin_nav_buttons = {
                    "admin_docs": "📄 Document Management",
                    "admin_users": "👥 User Management",
                    "admin_analytics": "📊 Analytics",
                }
                for page_key, label in admin_nav_buttons.items():
                    if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                        st.session_state["page"] = page_key
                        st.rerun()

            # --- Settings ---
            st.divider()
            st.subheader("Settings")

            # OpenAI API Key Setting (Allow override)
            with st.expander("API Key"):
                 # Use a separate key for the input widget to avoid conflicts
                 api_key_input_value = st.session_state.get("openai_api_key", "")
                 new_key = st.text_input("OpenAI API Key", value=api_key_input_value, type="password", key="api_key_input_widget", help="Overrides environment/secrets if set.")
                 # Update session state only if the input value changes and is not empty
                 if new_key != api_key_input_value and new_key:
                     st.session_state["openai_api_key"] = new_key
                     st.success("API Key updated in session.")
                     # Clear cached client and index to force re-validation/reload on next use
                     if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                     if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                     # Attempt re-validation immediately
                     get_openai_client() # This will show error if invalid
                     st.rerun() # Rerun to reflect potential changes
                 elif not new_key and "openai_api_key" in st.session_state:
                     # If user clears the key, remove it from session state
                     del st.session_state["openai_api_key"]
                     st.warning("API Key cleared from session.")
                     # Clear cached client and index
                     if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                     if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                     st.rerun()


            with st.expander("Search Settings"):
                # Use session state keys directly for widgets
                st.slider("Results to retrieve (k)", min_value=1, max_value=15, value=st.session_state.get("k_retrieve", 5), step=1, key="k_retrieve")
                st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=st.session_state.get("similarity_threshold", 0.75), step=0.05, key="similarity_threshold", format="%.2f")

            with st.expander("Model Settings"):
                available_models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"] # Add more models if needed
                current_model = st.session_state.get("llm_model", "gpt-3.5-turbo")
                # Handle case where saved model is no longer available
                if current_model not in available_models:
                     current_model = available_models[0] # Default to first available
                     st.session_state["llm_model"] = current_model
                st.selectbox(
                    "LLM Model",
                    available_models,
                    index=available_models.index(current_model),
                    key="llm_model"
                )
                st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=st.session_state.get("temperature", 0.2), step=0.1, key="temperature", format="%.1f")
                st.checkbox("Include detailed explanations", value=st.session_state.get("include_explanation", True), key="include_explanation")

        else:
            # Sidebar content for logged-out users (minimal)
            st.info("Please log in to access PharmInsight features.")


def login_form():
    """Display login form"""
    st.header("Welcome to PharmInsight")
    st.markdown("Please log in to continue.")

    col1, col2 = st.columns([1, 1]) # Adjust column ratio if needed

    with col1:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    authenticated, role = verify_password(username, password)
                    if authenticated:
                        # Set auth state AFTER verification
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["role"] = role
                        st.session_state["page"] = "main" # Redirect to main page
                        st.success(f"Login successful! Welcome back, {username}.")
                        # Short delay before rerun can improve UX
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

    with col2:
        st.markdown("""
        ### About PharmInsight
        A clinical knowledge base designed for pharmacists.

        * Ask questions about drugs, guidelines, and policies.
        * Receive evidence-based answers with sources.
        * Leverages your organization's internal documents.

        ---
        **Demo Logins:**
        * Regular user: `user` / `password`
        * Admin: `admin` / `adminpass`
        """)

# --- UPDATED main_page ---
def main_page():
    """Render the main search page"""
    st.title("📚 PharmInsight Search")
    st.markdown("Ask questions related to clinical guidelines, drug information, or policies.")

    client = get_openai_client()
    if not client:
        st.error("🚨 Configuration Error: OpenAI API key is missing or invalid. Please check settings (sidebar). Search functionality is disabled.")
        return

    # Load index - function now provides more sidebar feedback
    index, metadata = load_search_index()

    # Check the state of the loaded index
    is_index_empty = index is None or index.ntotal == 0 or metadata is None or len(metadata) == 0

    if is_index_empty:
        # Display a more informative warning
        st.warning("⚠️ No documents are currently loaded in the search index. Search is disabled.")
        st.caption("""
            This can happen if:
            * No documents have been uploaded and indexed yet.
            * The index files (`vector.index`, `docs_metadata.pkl`) are missing or corrupted.
            * The index rebuild process failed (check sidebar messages or Admin > Rebuild Index).
            * An error occurred while loading the index (check sidebar messages).
            Please check the **Admin Panel > Document Management** or contact an administrator.
            """)
        # The search bar will be disabled below

    # --- Search Input Area ---
    # Use st.session_state.get to initialize the text input value
    initial_query = st.session_state.get("query_input_value", "")
    with st.form(key="search_form"):
        query = st.text_input(
            "Ask a clinical question:",
            placeholder="e.g., What is the recommended monitoring for amiodarone?" if not is_index_empty else "Search disabled - index not loaded",
            key="query_input_widget",
            value=initial_query,
            disabled=is_index_empty # Disable input if index is empty
        )
        search_col, clear_col = st.columns([4, 1])
        with search_col:
            # Disable button if query is empty OR index is empty
            search_submitted = st.form_submit_button("Search", type="primary", use_container_width=True, disabled=not query or is_index_empty)
        with clear_col:
            clear_submitted = st.form_submit_button("Clear", use_container_width=True)

        if clear_submitted:
            # Clear the stored value and the trigger
            st.session_state.query_input_value = ""
            if "current_query" in st.session_state: del st.session_state["current_query"]
            if 'last_search_result' in st.session_state: del st.session_state['last_search_result']
            st.rerun() # Rerun to clear the text input

        if search_submitted and query and not is_index_empty: # Check index again before processing
            # Store query for processing and potentially pre-filling next time
            st.session_state["query_input_value"] = query
            st.session_state["current_query"] = query # Set the trigger
            # Clear previous results before starting new search
            if 'last_search_result' in st.session_state: del st.session_state['last_search_result']
            st.rerun() # Rerun to trigger processing below
        elif search_submitted and is_index_empty:
             # This case should ideally not be reachable due to disabled button, but good failsafe
             st.error("Cannot search because the document index is not available.")


    # --- Display Recent Searches (Optional) ---
    if "user_history" in st.session_state and st.session_state["user_history"] and not is_index_empty: # Also hide if index empty
        # Get unique recent searches, preserving order (most recent first)
        unique_recent = []
        seen = set()
        for h in reversed(st.session_state["user_history"]):
            q = h["query"]
            if q not in seen:
                unique_recent.append(q)
                seen.add(q)
                if len(unique_recent) >= 3: # Limit to 3 unique recent searches
                    break

        if unique_recent:
            st.caption("Recent searches:")
            cols = st.columns(len(unique_recent))
            for i, recent_q in enumerate(unique_recent):
                with cols[i]:
                    button_text = f"'{recent_q[:25]}...'" if len(recent_q) > 25 else f"'{recent_q}'"
                    if st.button(button_text, key=f"recent_{i}", help=f"Search again for: {recent_q}"):
                        # Set the value for the text input for the *next* run
                        st.session_state.query_input_value = recent_q
                        # Set the trigger for processing
                        st.session_state["current_query"] = recent_q
                        # Clear previous results before starting new search
                        if 'last_search_result' in st.session_state: del st.session_state['last_search_result']
                        st.rerun() # Rerun will process the query below


    # --- Process Search and Display Results ---
    # Check if a search was triggered AND index is usable
    if st.session_state.get("current_query") and not is_index_empty:
        current_query_to_process = st.session_state["current_query"]
        # Clear the trigger state immediately after reading it
        st.session_state["current_query"] = None

        st.markdown("---") # Separator
        with st.spinner("🧠 Searching documents and generating answer..."):
            # Get model settings from session state
            llm_model = st.session_state.get("llm_model", "gpt-3.5-turbo")
            include_explanation = st.session_state.get("include_explanation", True)
            temperature = st.session_state.get("temperature", 0.2)

            # Generate answer using the query we stored
            result_data = generate_answer(
                current_query_to_process,
                model=llm_model,
                include_explanation=include_explanation,
                temp=temperature
            )
            # Store result in session state to persist across reruns if needed
            st.session_state['last_search_result'] = result_data
            # Rerun needed to display the results cleanly after spinner
            st.rerun()
    elif st.session_state.get("current_query") and is_index_empty:
         # Clear trigger if search was attempted with no index
         st.session_state["current_query"] = None
         # Error was already shown in the form logic


    # Display the result if it exists in session state
    if 'last_search_result' in st.session_state:
        result_data = st.session_state['last_search_result']
        st.subheader(f"Results for: \"{result_data['query']}\"")

        # Display status message based on how the answer was generated
        source_type = result_data.get("source_type", "Error")
        if source_type == "Retrieved Documents":
            st.success("✅ Answer generated based on the retrieved documents:")
        elif source_type == "No Relevant Documents":
            # This case can happen even with an index, if the query doesn't match
            st.warning("⚠️ No matching information found in the documents for this specific query.")
        elif source_type == "Error":
            st.error("❌ An error occurred while processing your request.")
        else: # Fallback for unexpected source_type
            st.info("Displaying generated response:")


        # Display the main answer content
        st.markdown(result_data.get("answer", "No answer generated."))

        # Display sources if available and relevant
        if source_type == "Retrieved Documents" and result_data.get("sources"):
            with st.expander("📄 Sources Used", expanded=False): # Start collapsed
                for idx, doc in enumerate(result_data["sources"]):
                    score = doc.get("score", 0.0)
                    # Ensure score is float before formatting
                    try:
                        score_percent = int(float(score) * 100)
                    except (ValueError, TypeError):
                        score_percent = 0 # Default if score is invalid

                    source_name = doc.get('source', 'Unknown Source')
                    category = doc.get('category', 'N/A')
                    text_snippet = doc.get("text", "")
                    if len(text_snippet) > 250: # Shorter snippet
                        text_snippet = text_snippet[:250] + "..."

                    st.markdown(f"**Source {idx+1}: {source_name}** (Category: {category}, Relevance: {score_percent}%)")
                    # Use a blockquote or styled div for the snippet
                    st.markdown(f"> {text_snippet}")
                    st.markdown("---") # Separator between sources

        # Display feedback UI only if an answer was generated (not on error/no results)
        if source_type != "Error" and result_data.get("question_id"):
             feedback_ui(result_data["question_id"])


def profile_page():
    """Render the user profile page"""
    st.header(f"👤 Profile: {st.session_state.get('username', '')}")

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        c = conn.cursor()

        # Get user data
        c.execute("""
        SELECT username, role, email, full_name, created_at, last_login
        FROM users WHERE username = ?
        """, (st.session_state["username"],))
        user_data = c.fetchone()

        if not user_data:
            st.error("User data not found.")
            return

        # Unpack user data
        username, role, email, full_name, created_at_raw, last_login_raw = user_data

        # Format dates nicely
        created_at = pd.to_datetime(created_at_raw).strftime('%Y-%m-%d %H:%M') if created_at_raw else 'Unknown'
        last_login = pd.to_datetime(last_login_raw).strftime('%Y-%m-%d %H:%M') if last_login_raw else 'Never'

        # Get user activity statistics
        c.execute("SELECT COUNT(*) FROM qa_pairs WHERE user_id = ?", (username,))
        question_count = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM feedback WHERE user_id = ?", (username,))
        feedback_count = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM search_history WHERE user_id = ?", (username,))
        search_count = c.fetchone()[0]


        # --- Display Profile Information ---
        col1, col2 = st.columns([2, 1]) # Left wider for info/forms

        with col1:
            st.subheader("Account Information")
            st.text_input("Username", value=username, disabled=True)
            st.text_input("Role", value=role, disabled=True)
            st.text_input("Account Created", value=created_at, disabled=True)
            st.text_input("Last Login", value=last_login, disabled=True)

            st.divider()

            # --- Update Profile Form ---
            st.subheader("Update Profile")
            with st.form("update_profile_form"):
                new_email = st.text_input("Email", value=email or "")
                new_name = st.text_input("Full Name", value=full_name or "")
                update_submitted = st.form_submit_button("Update Profile")

                if update_submitted:
                    try:
                        # Use a separate connection context for update
                        with sqlite3.connect(DB_PATH, timeout=20.0) as update_conn:
                            uc = update_conn.cursor()
                            uc.execute(
                                "UPDATE users SET email = ?, full_name = ? WHERE username = ?",
                                (new_email, new_name, username)
                            )
                            update_conn.commit()
                        log_action(username, "update_profile", f"Updated profile. Email: {new_email}, Name: {new_name}")
                        st.success("Profile updated successfully!")
                        st.rerun() # Rerun to reflect changes
                    except sqlite3.IntegrityError:
                         st.error("❌ Error: Email address might already be in use by another account.")
                    except Exception as e:
                        st.error(f"❌ Error updating profile: {str(e)}")
                        print(f"Profile update error for {username}: {e}")


            st.divider()

            # --- Change Password Form ---
            st.subheader("Change Password")
            with st.form("change_password_form"):
                current_pwd = st.text_input("Current Password", type="password")
                new_pwd = st.text_input("New Password", type="password")
                confirm_pwd = st.text_input("Confirm New Password", type="password")
                pwd_submitted = st.form_submit_button("Change Password")

                if pwd_submitted:
                    if not current_pwd or not new_pwd or not confirm_pwd:
                        st.error("All password fields are required.")
                    elif new_pwd != confirm_pwd:
                        st.error("New passwords do not match.")
                    elif len(new_pwd) < 6:
                         st.error("New password must be at least 6 characters long.")
                    else:
                        # Verify current password first
                        verified, _ = verify_password(username, current_pwd)
                        if not verified:
                             # verify_password already logs the failed attempt
                             st.error("Incorrect current password.")
                        else:
                            # Hash new password
                            salt = "salt_key" # Ensure this matches
                            new_hash = hashlib.sha256(f"{new_pwd}:{salt}".encode()).hexdigest()
                            try:
                                # Use a separate connection context
                                with sqlite3.connect(DB_PATH, timeout=20.0) as pwd_conn:
                                    pc = pwd_conn.cursor()
                                    pc.execute(
                                        "UPDATE users SET password_hash = ? WHERE username = ?",
                                        (new_hash, username)
                                    )
                                    pwd_conn.commit()
                                log_action(username, "change_password", "User changed their password successfully.")
                                st.success("Password changed successfully!")
                                # Optionally clear form fields after success
                            except Exception as e:
                                st.error(f"❌ Error changing password: {str(e)}")
                                print(f"Password change error for {username}: {e}")

        with col2:
            # --- Display Activity Stats ---
            st.subheader("Your Activity")
            st.metric("Questions Answered", question_count)
            st.metric("Searches Made", search_count)
            st.metric("Feedback Given", feedback_count)

            st.divider()

            # --- Recent Searches (Profile Page) ---
            st.subheader("Recent Searches")
            recent_searches_df = pd.read_sql_query(
                """
                SELECT query, timestamp
                FROM search_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 5
                """,
                conn, params=(username,)
            )

            if not recent_searches_df.empty:
                for _, row in recent_searches_df.iterrows():
                    ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                    q_short = row['query'][:40] + '...' if len(row['query']) > 40 else row['query']
                    st.markdown(f"* `{ts}`: {q_short}")
            else:
                st.info("No recent searches found.")

    except Exception as e:
        st.error(f"❌ Error loading profile page: {str(e)}")
        print(f"Profile page load error: {e}")
    finally:
        if conn:
            conn.close()


def history_page():
    """Render search history page"""
    st.title("📜 Your Activity History")
    username = st.session_state.get("username", "unknown_user")

    try:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)

        # --- Fetch Data ---
        history_df = pd.read_sql_query(
            """
            SELECT history_id, query, timestamp, num_results
            FROM search_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 100 -- Limit history display
            """,
            conn, params=(username,)
        )

        qa_df = pd.read_sql_query(
            """
            SELECT question_id, query, answer, timestamp, sources, model_used
            FROM qa_pairs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 100 -- Limit display
            """,
            conn, params=(username,)
        )

        feedback_df = pd.read_sql_query(
             """
             SELECT f.rating, f.comment, f.timestamp, q.query as original_query
             FROM feedback f
             JOIN qa_pairs q ON f.question_id = q.question_id
             WHERE f.user_id = ?
             ORDER BY f.timestamp DESC
             LIMIT 100 -- Limit display
             """,
             conn, params=(username,)
        )

        # --- Display in Tabs ---
        tab1, tab2, tab3 = st.tabs(["🔍 Search History", "❓ Questions & Answers", "📊 Feedback Given"])

        with tab1:
            st.subheader("Your Recent Searches")
            if history_df.empty:
                st.info("You haven't performed any searches yet.")
            else:
                # Option to clear history
                if st.button("Clear Search History", key="clear_hist_btn"):
                     try:
                          with sqlite3.connect(DB_PATH, timeout=20.0) as clear_conn:
                              cc = clear_conn.cursor()
                              cc.execute("DELETE FROM search_history WHERE user_id = ?", (username,))
                              clear_conn.commit()
                          log_action(username, "clear_history", "User cleared their search history.")
                          st.session_state["user_history"] = [] # Clear session state too
                          st.success("Search history cleared.")
                          st.rerun()
                     except Exception as e:
                          st.error(f"Error clearing history: {e}")

                st.write(f"Showing your last {len(history_df)} searches:")
                for i, row in history_df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([4, 2, 1])
                        with col1:
                             st.markdown(f"**Query:** `{row['query']}`")
                        with col2:
                            ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                            results_text = f"({row['num_results']} results)" if row['num_results'] is not None else "(Results N/A)"
                            st.caption(f"{ts} {results_text}")
                        with col3:
                            if st.button("Search Again", key=f"again_{row['history_id']}"):
                                 st.session_state["query_input_value"] = row['query'] # Pre-fill input for next run
                                 st.session_state["current_query"] = row['query'] # Set trigger
                                 st.session_state["page"] = "main"
                                 st.rerun()
                        st.divider()

        with tab2:
             st.subheader("Your Recent Questions & Answers")
             if qa_df.empty:
                  st.info("You haven't asked any questions that resulted in an answer yet.")
             else:
                  st.write(f"Showing your last {len(qa_df)} answered questions:")
                  for i, row in qa_df.iterrows():
                       with st.expander(f"Q: {row['query'][:80]}... ({pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')})"):
                            st.markdown("**Answer:**")
                            st.markdown(row['answer'])
                            st.caption(f"Model: {row['model_used']} | Sources: {row['sources']}")
                            # Add button to ask again?
                            if st.button("Ask Again", key=f"ask_again_{row['question_id']}"):
                                 st.session_state["query_input_value"] = row['query']
                                 st.session_state["current_query"] = row['query']
                                 st.session_state["page"] = "main"
                                 st.rerun()


        with tab3:
             st.subheader("Feedback You've Given")
             if feedback_df.empty:
                  st.info("You haven't submitted any feedback yet.")
             else:
                  st.write(f"Showing your last {len(feedback_df)} feedback entries:")
                  rating_labels = {1: "Not Helpful", 2: "Somewhat Helpful", 3: "Very Helpful"}
                  for i, row in feedback_df.iterrows():
                       with st.container():
                            st.markdown(f"**Rating:** {rating_labels.get(row['rating'], 'Unknown')} ({row['rating']}/3)")
                            if row['comment']:
                                 st.markdown(f"**Comment:** {row['comment']}")
                            st.caption(f"For query: '{row['original_query'][:60]}...' | On: {pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                            st.divider()

    except Exception as e:
        st.error(f"❌ Error loading history page: {str(e)}")
        print(f"History page load error: {e}")
    finally:
        if conn:
            conn.close()


# --- Admin Pages ---

def admin_docs_page():
    """Admin document management page"""
    if not is_admin():
        st.warning("⛔ Admin access required for this page.")
        return

    st.title("📄 Document Management")

    # Create tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Documents List", "Upload Documents", "Rebuild Index"])

    try:
        # Connect here once for the page, pass connection if needed
        # Using context manager ensures it's closed
        with sqlite3.connect(DB_PATH, timeout=30.0) as conn:

            # --- Tab 1: Documents List ---
            with tab1:
                st.subheader("All Documents")

                docs_df = pd.read_sql_query(
                    """
                    SELECT d.doc_id, d.filename, d.category, d.upload_date, d.is_active,
                           d.uploader, COUNT(c.chunk_id) as chunks
                    FROM documents d
                    LEFT JOIN chunks c ON d.doc_id = c.doc_id
                    GROUP BY d.doc_id
                    ORDER BY d.upload_date DESC
                    """, conn
                )

                if docs_df.empty:
                    st.info("No documents found in the system.")
                else:
                    # Format data for display
                    docs_df['upload_date'] = pd.to_datetime(docs_df['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
                    docs_df['status'] = docs_df['is_active'].apply(lambda x: "✅ Active" if x else "❌ Inactive")

                    # Display DataFrame
                    st.dataframe(
                        docs_df[['filename', 'category', 'upload_date', 'uploader', 'chunks', 'status']],
                        use_container_width=True,
                         column_config={
                             "upload_date": st.column_config.TextColumn("Uploaded"),
                             "chunks": st.column_config.NumberColumn("Chunks", format="%d")
                        }
                    )

                    st.divider()
                    st.subheader("Document Actions")

                    # Select document for actions
                    doc_options = {row['doc_id']: f"{row['filename']} ({row['status']})" for _, row in docs_df.iterrows()}
                    selected_doc_id = st.selectbox(
                        "Select document for actions:",
                        options=list(doc_options.keys()),
                        format_func=lambda x: doc_options.get(x, "Unknown Document")
                    )

                    if selected_doc_id:
                        selected_doc_info = docs_df[docs_df['doc_id'] == selected_doc_id].iloc[0]
                        is_active = bool(selected_doc_info['is_active'])
                        filename = selected_doc_info['filename']

                        action_col1, action_col2 = st.columns(2)

                        with action_col1:
                            # Activate/Deactivate Button
                            if is_active:
                                if st.button(f"Deactivate '{filename}'", key=f"deact_{selected_doc_id}"):
                                    try:
                                        # Use the existing connection
                                        ac = conn.cursor()
                                        ac.execute("UPDATE documents SET is_active = 0 WHERE doc_id = ?", (selected_doc_id,))
                                        conn.commit()
                                        log_action(st.session_state['username'], "deactivate_document", f"Deactivated document ID: {selected_doc_id} ({filename})", conn)
                                        # Clear cached index on change
                                        if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                                        if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                                        st.success(f"Document '{filename}' deactivated. Rebuild index to apply changes.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deactivating document: {e}")
                            else:
                                if st.button(f"Activate '{filename}'", key=f"act_{selected_doc_id}"):
                                    try:
                                        # Use the existing connection
                                        ac = conn.cursor()
                                        ac.execute("UPDATE documents SET is_active = 1 WHERE doc_id = ?", (selected_doc_id,))
                                        conn.commit()
                                        log_action(st.session_state['username'], "activate_document", f"Activated document ID: {selected_doc_id} ({filename})", conn)
                                         # Clear cached index on change
                                        if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                                        if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                                        st.success(f"Document '{filename}' activated. Rebuild index to apply changes.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error activating document: {e}")

                        with action_col2:
                            # Delete Button (with confirmation)
                            st.error("⚠️ Deleting a document is permanent and removes all associated data.")
                            if st.checkbox(f"Confirm deletion of '{filename}'?", key=f"del_confirm_{selected_doc_id}"):
                                if st.button(f"DELETE '{filename}'", type="primary", key=f"del_btn_{selected_doc_id}"):
                                    try:
                                        # Use the existing connection
                                        conn.execute("PRAGMA foreign_keys = ON")
                                        ac = conn.cursor()
                                        # Deleting from documents should cascade delete chunks due to schema
                                        ac.execute("DELETE FROM documents WHERE doc_id = ?", (selected_doc_id,))
                                        conn.commit()
                                        log_action(st.session_state['username'], "delete_document", f"Deleted document ID: {selected_doc_id} ({filename})", conn)
                                         # Clear cached index on change
                                        if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                                        if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                                        st.success(f"Document '{filename}' and its chunks deleted. Rebuild index if needed.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting document: {e}")
                                        print(f"Delete error for doc {selected_doc_id}: {e}")


            # --- Tab 2: Upload Documents ---
            with tab2:
                st.subheader("Upload New Documents")

                uploaded_files = st.file_uploader(
                    "Select documents (PDF or TXT)",
                    accept_multiple_files=True,
                    type=["pdf", "txt"]
                )

                if uploaded_files:
                    # Use a form for metadata and options
                    with st.form("upload_form"):
                        st.write("--- Document Information ---")
                        category = st.selectbox(
                            "Document Category",
                            ["Clinical Guidelines", "Drug Information", "Policy", "Protocol", "Research", "Educational", "Other"],
                            index=0
                        )
                        description = st.text_area("Description (optional)")
                        expiry_date = st.date_input("Expiry Date (optional)", value=None)

                        st.write("--- Processing Options ---")
                        chunk_size = st.slider("Chunk Size (characters)", 500, 3000, 1000, 100)
                        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 500, 200, 50)
                        st.caption("Adjust chunking for optimal search results. Overlap helps maintain context between chunks.")

                        # Debug mode checkbox
                        debug_mode = st.checkbox("Enable debug output during processing", value=False)

                        upload_submitted = st.form_submit_button("Process and Upload Files", type="primary")

                    if upload_submitted:
                        # Check API key before starting expensive processing
                        client = get_openai_client()
                        if not client:
                             st.error("❌ Cannot process upload: OpenAI API key is missing or invalid.")
                        else:
                            metadata_opts = {
                                "category": category,
                                "description": description,
                                "expiry_date": expiry_date.isoformat() if expiry_date else None,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap
                            }

                            progress_bar_total = st.progress(0)
                            status_text_total = st.empty()
                            files_processed = 0
                            files_failed = 0

                            for i, file in enumerate(uploaded_files):
                                status_text_total.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                                progress_bar_total.progress((i + 1) / len(uploaded_files))

                                if debug_mode:
                                    st.write(f"--- Debug: Processing {file.name} ---")
                                    st.write(f"Size: {file.size} bytes, Type: {file.type}")
                                    st.write(f"Options: {metadata_opts}")

                                try:
                                    # --- 1. Extract Text ---
                                    text = ""
                                    file.seek(0) # Ensure reading from start
                                    if file.name.lower().endswith(".pdf"):
                                        try:
                                             reader = PdfReader(file)
                                             if reader.is_encrypted:
                                                 st.error(f"❌ Skipping encrypted PDF: {file.name}")
                                                 files_failed += 1
                                                 continue
                                             text = "".join(page.extract_text() + "\n\n" for page in reader.pages if page.extract_text())
                                        except Exception as pdf_err:
                                             st.error(f"❌ Error reading PDF {file.name}: {pdf_err}")
                                             files_failed += 1
                                             continue
                                    elif file.name.lower().endswith(".txt"):
                                        try:
                                             text = file.read().decode("utf-8")
                                        except UnicodeDecodeError:
                                             try:
                                                 file.seek(0)
                                                 text = file.read().decode("latin-1") # Try fallback encoding
                                             except Exception as txt_err:
                                                 st.error(f"❌ Error reading TXT {file.name}: {txt_err}")
                                                 files_failed += 1
                                                 continue
                                    else:
                                        st.error(f"❌ Unsupported file type: {file.name}")
                                        files_failed += 1
                                        continue

                                    if not text or not text.strip():
                                        st.warning(f"⚠️ No text extracted from {file.name}. Skipping.")
                                        files_failed += 1
                                        continue

                                    if debug_mode:
                                        st.write(f"Extracted text length: {len(text)}")
                                        st.text_area("Extracted Text Sample", text[:500]+"...", height=100, key=f"debug_text_{i}")


                                    # --- 2. Chunk Text ---
                                    chunks = []
                                    start_index = 0
                                    while start_index < len(text):
                                        end_index = min(start_index + chunk_size, len(text))
                                        chunk_text = text[start_index:end_index].strip()
                                        if chunk_text: # Only add non-empty chunks
                                             chunks.append(chunk_text)
                                        # Move start index for next chunk, considering overlap
                                        next_start = start_index + chunk_size - chunk_overlap
                                        # Prevent infinite loop if step is non-positive
                                        if next_start <= start_index:
                                            next_start = start_index + 1
                                        start_index = next_start


                                    if not chunks:
                                        st.warning(f"⚠️ No chunks created for {file.name}. Skipping.")
                                        files_failed += 1
                                        continue

                                    if debug_mode:
                                        st.write(f"Created {len(chunks)} chunks.")
                                        st.text_area("First Chunk Sample", chunks[0][:500]+"...", height=100, key=f"debug_chunk_{i}")


                                    # --- 3. Store in Database (Transaction) ---
                                    doc_id = str(uuid.uuid4())
                                    chunk_ids = [f"{doc_id}_{j+1}" for j in range(len(chunks))]
                                    uploader = st.session_state.get("username", "admin") # Default to admin if session lost

                                    try:
                                        # Use the existing connection from the 'with' block at the start of the function
                                        uc = conn.cursor()
                                        # Start transaction explicitly if needed within the loop
                                        conn.execute("BEGIN TRANSACTION")

                                        # Insert document record
                                        uc.execute(
                                            "INSERT INTO documents (doc_id, filename, uploader, category, description, expiry_date, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                            (
                                                doc_id, file.name, uploader,
                                                metadata_opts['category'], metadata_opts['description'],
                                                metadata_opts['expiry_date'], 1 # Active by default
                                            )
                                        )

                                        # Insert chunks (without embeddings initially if rebuild happens later)
                                        chunk_data = [(chunk_ids[j], doc_id, chunks[j], None) for j in range(len(chunks))] # Embeddings = None for now
                                        uc.executemany("INSERT INTO chunks (chunk_id, doc_id, text, embedding) VALUES (?, ?, ?, ?)", chunk_data)

                                        conn.commit() # Commit transaction for this file

                                        log_action(uploader, "upload_document", f"Uploaded '{file.name}' (ID: {doc_id}) with {len(chunks)} chunks.", conn)
                                        st.success(f"✅ Successfully processed and stored: {file.name} ({len(chunks)} chunks)")
                                        files_processed += 1
                                        # Clear cached index after successful upload
                                        if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                                        if "index_metadata" in st.session_state: del st.session_state["index_metadata"]

                                    except sqlite3.Error as db_err:
                                        conn.rollback() # Rollback transaction on error for this file
                                        st.error(f"❌ Database error for {file.name}: {db_err}")
                                        print(f"DB error during upload of {file.name}: {db_err}")
                                        files_failed += 1


                                except Exception as proc_err:
                                    st.error(f"❌ Unexpected error processing {file.name}: {proc_err}")
                                    print(f"Processing error for {file.name}: {proc_err}\n{traceback.format_exc()}")
                                    files_failed += 1


                            # --- Final Summary & Rebuild Prompt ---
                            status_text_total.text("File processing complete.")
                            progress_bar_total.progress(1.0)
                            st.info(f"Processing finished: {files_processed} succeeded, {files_failed} failed.")

                            if files_processed > 0:
                                 st.warning("⚠️ Remember to **Rebuild Index** (in the next tab) to make the new documents searchable!")


            # --- Tab 3: Rebuild Index ---
            with tab3:
                st.subheader("Rebuild Search Index")
                st.markdown("""
                Click the button below to regenerate the search index based on all **active** documents currently in the database.
                This process involves generating embeddings for each document chunk using the OpenAI API and can take some time depending on the number of documents.
                """)
                st.warning("Ensure your OpenAI API key is correctly configured in the sidebar settings before rebuilding.")

                if st.button("Rebuild Index Now", type="primary"):
                     # Check API key again just before rebuild
                     client = get_openai_client()
                     if not client:
                          st.error("❌ Cannot rebuild index: OpenAI API key is missing or invalid.")
                     else:
                          with st.spinner("🛠️ Rebuilding search index... This may take a while."):
                              # Clear cached index before rebuild
                              if "faiss_index" in st.session_state: del st.session_state["faiss_index"]
                              if "index_metadata" in st.session_state: del st.session_state["index_metadata"]
                              success, count = rebuild_index_from_db() # Call the rebuild function
                              if success:
                                  st.success(f"✅ Search index rebuilt successfully with {count} document chunks.")
                                  # Attempt to reload index into cache immediately after successful rebuild
                                  load_search_index() # This will update sidebar status
                              else:
                                  st.error("❌ Failed to rebuild search index. Check logs or previous errors.")


    except Exception as e:
        st.error(f"❌ Error on Document Management page: {str(e)}")
        print(f"Admin Docs Page Error: {e}\n{traceback.format_exc()}")
    # Connection is closed automatically by the 'with' statement


def admin_users_page():
    """Admin user management page"""
    if not is_admin():
        st.warning("⛔ Admin access required for this page.")
        return

    st.title("👥 User Management")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Users List", "Create User", "User Activity Logs"])

    try:
        # Use context manager for the connection
        with sqlite3.connect(DB_PATH, timeout=20.0) as conn:

            # --- Tab 1: Users List ---
            with tab1:
                st.subheader("All Users")
                users_df = pd.read_sql_query(
                    "SELECT username, role, email, full_name, created_at, last_login FROM users ORDER BY username", conn
                )

                if users_df.empty:
                    st.info("No users found.")
                else:
                    # Format dates
                    users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                    users_df['last_login'] = pd.to_datetime(users_df['last_login']).dt.strftime('%Y-%m-%d %H:%M').fillna('Never')

                    st.dataframe(users_df, use_container_width=True)

                    st.divider()
                    st.subheader("User Actions")
                    user_options = {row['username']: f"{row['username']} ({row['role']})" for _, row in users_df.iterrows()}
                    selected_username = st.selectbox(
                        "Select user for actions:",
                        options=list(user_options.keys()),
                        format_func=lambda x: user_options.get(x, "Unknown User")
                    )

                    if selected_username:
                        selected_user_info = users_df[users_df['username'] == selected_username].iloc[0]
                        current_role = selected_user_info['role']
                        is_self = (selected_username == st.session_state.get('username')) # Prevent actions on self

                        if is_self:
                             st.info("ℹ️ You cannot perform administrative actions on your own account here. Use the Profile page.")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # --- Reset Password ---
                            st.write("**Reset Password**")
                            reset_pwd_key = f"show_reset_{selected_username}"
                            if st.button("Reset Password", key=f"reset_pwd_btn_{selected_username}", disabled=is_self):
                                st.session_state[reset_pwd_key] = True # Use unique state key

                            if st.session_state.get(reset_pwd_key, False):
                                with st.form(f"reset_pwd_form_{selected_username}"):
                                    new_pwd = st.text_input("New Password", type="password", key=f"new_pwd_{selected_username}")
                                    confirm_pwd = st.text_input("Confirm New Password", type="password", key=f"confirm_pwd_{selected_username}")
                                    reset_submitted = st.form_submit_button("Confirm Reset")

                                    if reset_submitted:
                                        if not new_pwd or len(new_pwd) < 6:
                                            st.error("Password must be at least 6 characters.")
                                        elif new_pwd != confirm_pwd:
                                            st.error("Passwords do not match.")
                                        else:
                                            try:
                                                salt = "salt_key" # Ensure consistency
                                                new_hash = hashlib.sha256(f"{new_pwd}:{salt}".encode()).hexdigest()
                                                # Use existing connection
                                                ac = conn.cursor()
                                                ac.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, selected_username))
                                                conn.commit()
                                                log_action(st.session_state['username'], "reset_password", f"Admin reset password for user: {selected_username}", conn)
                                                st.success(f"Password reset successfully for {selected_username}.")
                                                st.session_state[reset_pwd_key] = False # Hide form
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error resetting password: {e}")


                        with col2:
                             # --- Change Role ---
                            st.write("**Change Role**")
                            current_role_index = 0 if current_role == 'user' else 1
                            new_role = st.selectbox("New Role", ["user", "admin"], index=current_role_index, key=f"role_{selected_username}", disabled=is_self)
                            if st.button("Apply Role Change", key=f"change_role_{selected_username}", disabled=is_self or new_role == current_role):
                                 try:
                                     # Use existing connection
                                     ac = conn.cursor()
                                     ac.execute("UPDATE users SET role = ? WHERE username = ?", (new_role, selected_username))
                                     conn.commit()
                                     log_action(st.session_state['username'], "change_role", f"Admin changed role for {selected_username} from {current_role} to {new_role}", conn)
                                     st.success(f"Role changed to {new_role} for {selected_username}.")
                                     st.rerun()
                                 except Exception as e:
                                     st.error(f"Error changing role: {e}")


                        with col3:
                            # --- Delete User ---
                            st.write("**Delete User**")
                            st.error("⚠️ Deletion is permanent!")
                            delete_confirm_key = f"del_confirm_{selected_username}"
                            if st.checkbox(f"Confirm deletion?", key=delete_confirm_key, disabled=is_self):
                                if st.button("DELETE USER", type="primary", key=f"del_user_{selected_username}", disabled=is_self):
                                    try:
                                        # Use existing connection
                                        conn.execute("PRAGMA foreign_keys = ON")
                                        ac = conn.cursor()
                                        # Cascade delete should handle related data
                                        ac.execute("DELETE FROM users WHERE username = ?", (selected_username,))
                                        conn.commit()
                                        log_action(st.session_state['username'], "delete_user", f"Admin deleted user: {selected_username}", conn)
                                        st.success(f"User {selected_username} deleted successfully.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting user: {e}")


            # --- Tab 2: Create User ---
            with tab2:
                st.subheader("Create New User")
                with st.form("create_user_form"):
                    new_username = st.text_input("Username*")
                    new_password = st.text_input("Password*", type="password")
                    confirm_password = st.text_input("Confirm Password*", type="password")
                    new_role = st.selectbox("Role*", ["user", "admin"], index=0)
                    new_email = st.text_input("Email (optional)")
                    new_full_name = st.text_input("Full Name (optional)")
                    create_submitted = st.form_submit_button("Create User")

                    if create_submitted:
                        if not new_username or not new_password or not confirm_password:
                            st.error("Username, Password, and Confirmation are required.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            try:
                                # Use existing connection
                                create_c = conn.cursor()
                                # Check if username exists
                                create_c.execute("SELECT 1 FROM users WHERE username = ?", (new_username,))
                                if create_c.fetchone():
                                    st.error("Username already exists.")
                                else:
                                    # Hash password and insert
                                    salt = "salt_key" # Ensure consistency
                                    new_hash = hashlib.sha256(f"{new_password}:{salt}".encode()).hexdigest()
                                    create_c.execute(
                                        "INSERT INTO users (username, password_hash, role, email, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                                        (new_username, new_hash, new_role, new_email or None, new_full_name or None, datetime.datetime.now().isoformat())
                                    )
                                    conn.commit()
                                    log_action(st.session_state['username'], "create_user", f"Admin created new user: {new_username} (Role: {new_role})", conn)
                                    st.success(f"User '{new_username}' created successfully!")
                                    st.rerun() # Rerun to update user list
                            except sqlite3.IntegrityError as ie:
                                 # Catch potential unique constraint errors (like email)
                                 st.error(f"Database error: Could not create user. Email might already be in use. ({ie})")
                            except Exception as e:
                                st.error(f"Error creating user: {e}")


            # --- Tab 3: User Activity Logs ---
            with tab3:
                st.subheader("User Activity Logs (Audit Trail)")
                log_limit = st.number_input("Number of recent logs to display:", min_value=10, max_value=500, value=100, step=10)

                audit_df = pd.read_sql_query(
                    f"""
                    SELECT timestamp, user_id, action, details
                    FROM audit_log
                    ORDER BY timestamp DESC
                    LIMIT {log_limit}
                    """, conn
                )

                if audit_df.empty:
                    st.info("No audit logs found.")
                else:
                    audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(audit_df, use_container_width=True)


    except Exception as e:
        st.error(f"❌ Error on User Management page: {str(e)}")
        print(f"Admin Users Page Error: {e}\n{traceback.format_exc()}")
    # Connection closed by 'with' statement


def admin_analytics_page():
    """Admin analytics dashboard page"""
    if not is_admin():
        st.warning("⛔ Admin access required for this page.")
        return

    st.title("📊 Analytics Dashboard")

    # Tabs for different analytics
    tab_names = ["Usage Stats", "Feedback Analysis", "Search Patterns", "Document Stats"]
    tabs = st.tabs(tab_names)

    try:
        # Use context manager for connection
        with sqlite3.connect(DB_PATH, timeout=20.0) as conn:

            # --- Tab 1: Usage Statistics ---
            with tabs[0]:
                st.subheader("System Usage Overview")

                # Fetch counts
                counts_query = """
                SELECT
                    (SELECT COUNT(*) FROM users) as users_count,
                    (SELECT COUNT(*) FROM documents WHERE is_active = 1) as active_docs_count,
                    (SELECT COUNT(*) FROM documents WHERE is_active = 0) as inactive_docs_count,
                    (SELECT COUNT(DISTINCT d.doc_id) FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE d.is_active = 1) as indexed_docs_count,
                    (SELECT COUNT(*) FROM chunks c JOIN documents d ON c.doc_id = d.doc_id WHERE d.is_active = 1) as active_chunks_count,
                    (SELECT COUNT(*) FROM qa_pairs) as qa_count,
                    (SELECT COUNT(*) FROM search_history) as searches_count,
                    (SELECT COUNT(*) FROM feedback) as feedback_count
                """
                counts_df = pd.read_sql_query(counts_query, conn)
                counts = counts_df.iloc[0] if not counts_df.empty else None

                if counts is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Users", int(counts.get('users_count', 0)))
                    col1.metric("Active Documents", int(counts.get('active_docs_count', 0)))
                    col1.metric("Inactive Documents", int(counts.get('inactive_docs_count', 0)))

                    col2.metric("Documents in Index", int(counts.get('indexed_docs_count', 0))) # Docs with active chunks
                    col2.metric("Chunks in Index", int(counts.get('active_chunks_count', 0))) # Active chunks
                    col2.metric("Questions Answered", int(counts.get('qa_count', 0)))

                    col3.metric("Total Searches", int(counts.get('searches_count', 0)))
                    col3.metric("Feedback Entries", int(counts.get('feedback_count', 0)))
                else:
                    st.warning("Could not fetch usage counts.")

                # Activity over time (e.g., searches per day)
                st.divider()
                st.subheader("Activity Over Time (Last 30 Days)")
                activity_df = pd.read_sql_query(
                    """
                    SELECT date(timestamp) as date, COUNT(*) as count
                    FROM search_history
                    WHERE date(timestamp) >= date('now', '-30 days')
                    GROUP BY date
                    ORDER BY date
                    """, conn
                )
                if not activity_df.empty:
                     activity_df['date'] = pd.to_datetime(activity_df['date'])
                     st.line_chart(activity_df.set_index('date')['count'], use_container_width=True)
                else:
                     st.info("No search activity in the last 30 days.")


            # --- Tab 2: Feedback Analysis ---
            with tabs[1]:
                st.subheader("User Feedback Analysis")

                feedback_stats = pd.read_sql_query(
                    "SELECT COUNT(*) as total, AVG(rating) as avg_rating FROM feedback", conn
                )
                total_fb = feedback_stats['total'].iloc[0] if not feedback_stats.empty else 0
                avg_rating = feedback_stats['avg_rating'].iloc[0] if not feedback_stats.empty and pd.notna(feedback_stats['avg_rating'].iloc[0]) else 0


                col1, col2 = st.columns(2)
                col1.metric("Total Feedback Entries", int(total_fb))
                col2.metric("Average Rating", f"{avg_rating:.2f} / 3.00" if total_fb > 0 else "N/A")

                if total_fb > 0:
                    rating_dist = pd.read_sql_query(
                        "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating ORDER BY rating", conn
                    )
                    rating_labels = {1: "1 - Not Helpful", 2: "2 - Somewhat", 3: "3 - Very Helpful"}
                    rating_dist['Rating Label'] = rating_dist['rating'].map(rating_labels)

                    # Ensure all rating categories exist for consistent charting
                    full_rating_dist = pd.DataFrame({'Rating Label': list(rating_labels.values())})
                    rating_dist = pd.merge(full_rating_dist, rating_dist, on='Rating Label', how='left').fillna(0)

                    st.bar_chart(rating_dist.set_index('Rating Label')['count'], use_container_width=True)

                    st.divider()
                    st.subheader("Recent Feedback Comments")
                    recent_comments = pd.read_sql_query(
                        """
                        SELECT f.rating, f.comment, f.timestamp, q.query
                        FROM feedback f JOIN qa_pairs q ON f.question_id = q.question_id
                        WHERE f.comment IS NOT NULL AND f.comment != ''
                        ORDER BY f.timestamp DESC LIMIT 20
                        """, conn
                    )
                    if not recent_comments.empty:
                         for _, row in recent_comments.iterrows():
                              st.markdown(f"**Rating: {row['rating']}/3** on *{pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')}* for Q: `{row['query'][:60]}...`")
                              st.markdown(f"> {row['comment']}")
                              st.markdown("---")
                    else:
                         st.info("No recent feedback with comments.")
                else:
                    st.info("No feedback has been submitted yet.")


            # --- Tab 3: Search Patterns ---
            with tabs[2]:
                st.subheader("Search Query Analysis")

                # Most frequent searches
                popular_searches = pd.read_sql_query(
                    """
                    SELECT query, COUNT(*) as count
                    FROM search_history GROUP BY query ORDER BY count DESC LIMIT 20
                    """, conn
                )
                if not popular_searches.empty:
                     st.write("**Most Frequent Searches:**")
                     st.dataframe(popular_searches, use_container_width=True)
                else:
                     st.info("No search history found.")

                # Searches with no results
                no_results_searches = pd.read_sql_query(
                    """
                    SELECT query, COUNT(*) as count
                    FROM search_history WHERE num_results = 0
                    GROUP BY query ORDER BY count DESC LIMIT 20
                    """, conn
                )
                if not no_results_searches.empty:
                     st.divider()
                     st.write("**Searches Returning No Results:**")
                     st.dataframe(no_results_searches, use_container_width=True)
                else:
                     st.info("No searches recorded with zero results.")


            # --- Tab 4: Document Stats ---
            with tabs[3]:
                st.subheader("Document Statistics")

                # Document count by category
                category_counts = pd.read_sql_query(
                    "SELECT category, COUNT(*) as count FROM documents GROUP BY category ORDER BY count DESC", conn
                )
                if not category_counts.empty:
                     st.write("**Documents per Category:**")
                     st.bar_chart(category_counts.set_index('category')['count'], use_container_width=True)
                else:
                     st.info("No documents found to categorize.")

                # Document upload timeline
                upload_timeline = pd.read_sql_query(
                    """
                    SELECT date(upload_date) as date, COUNT(*) as count
                    FROM documents GROUP BY date ORDER BY date
                    """, conn
                )
                if not upload_timeline.empty:
                     st.divider()
                     st.write("**Document Uploads Over Time:**")
                     upload_timeline['date'] = pd.to_datetime(upload_timeline['date'])
                     st.line_chart(upload_timeline.set_index('date')['count'], use_container_width=True)
                else:
                     st.info("No document upload history found.")


    except Exception as e:
        st.error(f"❌ Error loading analytics data: {str(e)}")
        print(f"Analytics Page Error: {e}\n{traceback.format_exc()}")
    # Connection closed by 'with' statement


# --- Main App Logic ---

# Set page config for wider layout
st.set_page_config(layout="wide", page_title="PharmInsight", page_icon="💊")


# Attempt DB initialization (safe to call multiple times)
if init_database():
    # Initialize session state if DB init is successful
    initialize_session_state()

    # Attempt to load API key early if not already set
    # This helps ensure the API key status is reflected early in the sidebar
    if not st.session_state.get("openai_api_key"):
         get_openai_client() # Check env/secrets

    # Attempt to load index early if authenticated (provides feedback in sidebar)
    # Do this *after* API key check, as rebuild might need the key
    if st.session_state.get("authenticated"):
        load_search_index() # This now handles more feedback and auto-rebuild attempts

    # Render sidebar (content depends on auth state)
    render_sidebar()

    # Page routing
    page = st.session_state.get("page", "login") # Default to login

    # Main content area
    if not st.session_state.get("authenticated"):
        # If not authenticated, force page to login regardless of state
        if page != "login":
             st.session_state.page = "login"
             st.rerun() # Rerun to show login page
        login_form()
    elif page == "main":
        main_page()
    elif page == "profile":
        profile_page()
    elif page == "history":
        history_page()
    elif page == "admin_docs":
        admin_docs_page()
    elif page == "admin_users":
        admin_users_page()
    elif page == "admin_analytics":
        admin_analytics_page()
    else:
        # Fallback to main page if route is unknown and authenticated
        st.warning(f"Unknown page '{page}'. Redirecting to home.")
        st.session_state["page"] = "main"
        st.rerun() # Rerun to show main page

else:
    # Critical DB Error on startup
    st.error("🚨 CRITICAL ERROR: Could not initialize the database. Application cannot start.")
    st.markdown("Please check file permissions for `pharminsight.db`, ensure sufficient disk space, or contact support.")

```

I've applied the necessary changes to the `load_search_index` and `main_page` functions. Now, the main page should provide a clearer explanation if the index isn't loaded and will disable the search functionality in that case. The sidebar should also give better feedback during the index loading process.

You can try running this updated code. Remember to check the sidebar messages and potentially use the "Rebuild Index" button in the Admin panel if the index still seems unavailab

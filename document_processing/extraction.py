"""
Document text extraction utilities for PharmInsight.
"""
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
import fitz as pymupdf  # PyMuPDF

def extract_text_from_pdf(file):
    """
    Extract text from PDF using PyMuPDF with PyPDF2 as fallback
    
    Args:
        file: Uploaded file object
        
    Returns:
        str: Extracted text
    """
    try:
        # First try with PyMuPDF (faster and better quality)
        memory_file = BytesIO(file.read())
        file.seek(0)  # Reset file pointer for potential later use
        
        try:
            # Try to use pymupdf
            doc = pymupdf.open(stream=memory_file, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text("text") + "\n\n"
                
            return text
        except Exception as pymupdf_error:
            st.warning(f"PyMuPDF extraction failed, falling back to PyPDF2: {pymupdf_error}")
            file.seek(0)  # Reset file pointer
            
            # Fall back to PyPDF2
            try:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                return text
            except Exception as pypdf2_error:
                st.error(f"Error extracting text with PyPDF2: {pypdf2_error}")
                return ""
    except Exception as e:
        st.error(f"General error extracting text from PDF: {e}")
        return ""

def extract_text_from_file(file):
    """
    Extract text from various file types
    
    Args:
        file: Uploaded file object
        
    Returns:
        tuple: (text, success, message)
    """
    try:
        # Extract text based on file type
        if file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file)
            if not text.strip():
                return "", False, "No text content could be extracted from the PDF file"
            return text, True, "PDF processed successfully"
            
        elif file.name.lower().endswith((".txt", ".csv", ".md")):
            try:
                file_bytes = file.read()
                text = file_bytes.decode("utf-8")
                file.seek(0)  # Reset file pointer
                return text, True, "Text file processed successfully"
            except UnicodeDecodeError:
                # Try other encodings
                file.seek(0)
                for encoding in ["latin-1", "windows-1252", "iso-8859-1"]:
                    try:
                        text = file.read().decode(encoding)
                        file.seek(0)
                        return text, True, f"Text file processed successfully (using {encoding} encoding)"
                    except:
                        file.seek(0)
                        continue
                return "", False, "Could not decode text file with any supported encoding"
        else:
            return "", False, f"Unsupported file type: {file.name}"
            
    except Exception as e:
        return "", False, f"Error extracting text: {str(e)}"

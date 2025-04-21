# PharmInsight

PharmInsight is a clinical knowledge base application for pharmacists. It allows users to search through clinical documents, ask questions, and get evidence-based answers with citations.

## Features

- **Document Management**: Upload, index, and search PDF and text documents
- **Question Answering**: Ask clinical questions and get answers based on the indexed documents
- **Vector Search**: Advanced semantic search using OpenAI embeddings and FAISS
- **User Management**: Authentication, role-based access control, and user profiles
- **Feedback System**: Collect and analyze user feedback on system responses
- **Analytics**: Track usage patterns and search quality

## Architecture

The application has been modularized into several components:

```
pharminsight/
├── app.py                  # Main application entry point
├── config.py               # Configuration and constants
├── database/               # Database operations
├── auth/                   # Authentication functionality
├── document_processing/    # Document handling
├── search/                 # Vector search functionality
├── qa/                     # Question answering
├── ui/                     # User interface
└── utils/                  # Utility functions
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pharminsight.git
cd pharminsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your_api_key_here"
```

### Running the Application

```bash
streamlit run app.py
```

## Default Login Credentials

- **Admin User**: 
  - Username: admin
  - Password: adminpass

- **Regular User**:
  - Username: user
  - Password: password

## Usage

### Document Management

1. Log in as an admin user
2. Go to "Document Management" in the admin panel
3. Upload PDF or text documents
4. Set document metadata and processing options
5. Process and index the documents

### Asking Questions

1. Log in to the application
2. Enter your clinical question in the search box
3. View the answer, explanation, and sources
4. Provide feedback on the answer quality

## Development

### Adding New Features

1. Identify the appropriate module for your feature
2. Implement the backend functionality
3. Create or update the UI components
4. Update app.py if necessary to include your new feature

### Code Style

- Follow PEP 8 guidelines
- Use docstrings for all functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
